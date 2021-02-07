import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader  
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

from tensorboardX import SummaryWriter


def train(cfg, writer, logger):

    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    augmentations = cfg["training"].get("augmentations", None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    
    tloader_params = {k: v for k, v in cfg["data"]["train"].items()}
    tloader_params.update({'root':cfg["data"]["path"]})

    vloader_params = {k: v for k, v in cfg["data"]["val"].items()}
    vloader_params.update({'root':cfg["data"]["path"]})
    
    t_loader = data_loader(**tloader_params)
    v_loader = data_loader(**vloader_params)
    
    
    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(
        t_loader,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
    )

    valloader = data.DataLoader(
        v_loader, batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["n_workers"]
    )

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)

    # Setup Model
    model = get_model(cfg["model"], n_classes).to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])
    loss_type = cfg["training"]["loss"]["name"]

    if loss_type == 'BalancedCE' or loss_type =='WeightedCE':
        cls_num_list = np.zeros((n_classes,))
        print("=" * 10, "CALCULATING WEIGHTS", "=" * 10)
        # for _, valloader in loaders['val'].items():
        for _, (_, labels_list) in tqdm(enumerate(valloader)):
            for i in range(n_classes):
                cls_num_list[i] = cls_num_list[i] + (labels_list[0] == i).sum()
        
        if loss_type == 'BalancedCE':
            beta = (np.sum(cls_num_list)-1)/np.sum(cls_num_list)
            effective_num = 1.0 - np.power(beta, cls_num_list)
            effective_num[effective_num==0] = 1
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.tensor(per_cls_weights,dtype=torch.float).cuda(device)
            cls_num_list = None
        elif loss_type =='WeightedCE':
            median = np.median(cls_num_list)
            per_cls_weights = median/cls_num_list
            per_cls_weights[per_cls_weights==np.inf] = 0.0
            per_cls_weights = torch.tensor(per_cls_weights,dtype=torch.float).cuda(device)
            cls_num_list = None
        else:
            per_cls_weights = None
    else:
        per_cls_weights = None 
        cls_num_list = None


    
    if cfg["training"]["loss"][loss_type] is not None: 
        loss_params = {k: v for k, v in cfg["training"]["loss"][loss_type].items()}
    else:
        loss_params = {}
    loss_params["n_classes"] = n_classes
    loss_params["ignore_index"] = cfg["training"]["loss"]["ignore_index"]
    

    if per_cls_weights is not None:
        loss_params["weight"] = per_cls_weights 
    if cls_num_list is not None:
        loss_params["cls_num_list"] = cls_num_list 

    loss_fn = get_loss_function(cfg,**loss_params)
    logger.info("Using loss {}".format(loss_fn))


    start_iter = 0
    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
            )
            checkpoint = torch.load(cfg["training"]["resume"])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["training"]["resume"], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

    val_loss_meter = averageMeter()
    time_meter = averageMeter()
 
    best_iou = -100.0
    i = start_iter
    flag = True
    while i <= cfg["training"]["train_iters"] and flag:
        for (images, labels) in trainloader:
            i += 1
            start_ts = time.time()
            
            model.train()
            images = images.to(device)
            labels = labels.to(device).squeeze()
            if labels.shape[0] <= 2:
                continue
            
            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(input=outputs, target=labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            time_meter.update(time.time() - start_ts)
            if (i + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss.item(),
                    time_meter.avg / cfg["training"]["batch_size"],
                )

                print(print_str)
                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                time_meter.reset()

            if (i + 1) % cfg["training"]["val_interval"] == 0 or (i + 1) == cfg["training"][
                "train_iters"
            ]:
                model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device).squeeze()

                        outputs = model(images_val)
                        val_loss = loss_fn(input=outputs, target=labels_val)

                        pred = outputs.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()
                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())

                writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)
                logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))

                score, class_iou,_,_ = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

                for k, v in class_iou.items():
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)

                val_loss_meter.reset()
                running_metrics_val.reset()

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(
                        writer.file_writer.get_logdir(),
                        "{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                    )
                    torch.save(state, save_path)

            if (i + 1) == cfg["training"]["train_iters"]:
                flag = False
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/deeplab_synthia.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    logdir = os.path.join("runs", os.path.basename(args.config)[:-4],cfg['model']['backbone'],cfg['id'])
    writer = SummaryWriter(logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    train(cfg, writer, logger)