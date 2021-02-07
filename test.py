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
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def test(cfg, logdir):

    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            print(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
            )

            checkpoint = torch.load(cfg["training"]["resume"])
            model.load_state_dict(checkpoint["model_state"])
            check_iter = checkpoint["epoch"]
            print(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["training"]["resume"], checkpoint["epoch"]
                )
            )
        else:
            print("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

    # images = []
    if not os.path.exists(logdir):
    	os.makedirs(logdir)
    model.eval()
    with torch.no_grad():
        for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
            images_val = images_val.to(device)
            labels_val = labels_val.to(device).squeeze()


            outputs = model(images_val)
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()
            running_metrics_val.update(gt, pred)

           

    score, class_iou, clss_acc, clss_prec = running_metrics_val.get_scores()

    stats = {'iou':[],'acc':[],'prec':[]}

    for k, v in score.items():
        print(k, v)
        stats[k] = v

    for k, v in class_iou.items():
    	stats['iou'].append(v)
    for k, v in clss_acc.items():
    	stats['acc'].append(v)
    for k, v in clss_prec.items():
    	stats['prec'].append(v)
    df = pd.DataFrame(stats)
    df.to_csv(logdir+'performance.csv')


            
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)
    
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4],'test',cfg['model']['backbone'],cfg['id'])
    print("RUNDIR: {}".format(logdir))

    
    test(cfg, logdir)
