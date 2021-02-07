import logging
from ptsemseg.loss.focal_loss import FocalLoss
from ptsemseg.loss.lovasz_loss import LovaszSoftmax
from ptsemseg.loss.ohem_loss import OhemCrossEntropy2d
from ptsemseg.loss.softiou_loss import SoftIoULoss
from ptsemseg.loss.recall_loss import RecallCrossEntropy
from ptsemseg.loss.balanced_loss import BalancedCrossEntropy

import torch

def get_loss_function(cfg,n_classes,**kwargs):
    
    logger = logging.getLogger("ptsemseg")
    if cfg["training"]["loss"]["name"] is None:
        logger.info("Using default cross entropy loss")
        return torch.nn.CrossEntropyLoss(**kwargs)
    else:
        loss_dict = cfg["training"]["loss"]
        loss_type = loss_dict["name"]
        if loss_type is not None and cfg['training']["loss"][loss_type] is not None:
            loss_params = {k: v for k, v in loss_dict[loss_type].items() if k != "name"}
        else:
            loss_params = {}
        if loss_type == 'CrossEntropy' or loss_type == 'BalancedCE' or loss_type == 'WeightedCE' :
            criterion = torch.nn.CrossEntropyLoss(**kwargs)
        elif loss_type == 'Focal':
            criterion = FocalLoss(**kwargs)
        elif loss_type == 'Lovasz':
            criterion = LovaszSoftmax(**kwargs)
        elif loss_type == 'OhemCrossEntropy':
            criterion = OhemCrossEntropy2d(**kwargs)
        elif loss_type == 'SoftIOU':
            criterion = SoftIoULoss(n_classes,**kwargs)
        elif loss_type == 'RecallCE':
            criterion = RecallCrossEntropy(n_classes,**kwargs)
        else:
            raise NotImplementedError
        logger.info("Using {} with {} params".format(loss_type, loss_params))
        return criterion
