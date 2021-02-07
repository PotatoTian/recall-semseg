import torch
import torch.nn as nn
import torch.nn.functional as F

class BalancedCrossEntropy(torch.nn.Module):
    def __init__(self, ignore_index=255, weight=None):
        super(BalancedCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.per_cls_weight = weight


    def forward(self, input, target): 
        # input (batch,n_classes,H,W)
        # target (batch,H,W)
       
        loss = F.cross_entropy(
            input, target, weight=self.per_cls_weight, ignore_index=self.ignore_index
        )
        return loss


