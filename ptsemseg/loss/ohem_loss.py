import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OhemCrossEntropy2d(nn.Module):
    def __init__(self, thresh=0.6, weight=None, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.top_k = thresh
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index,reduction='none')

    def forward(self, input, target):
        """
            Args:
                input:(n, c, h, w)
                target:(n, h, w)
        """
        loss = self.criterion(input,target)
        if self.top_k == 1:
            return torch.mean(loss)
        else:
            valid_loss, idxs = torch.topk(loss, int(self.top_k * loss.size()[0]))    
            return torch.mean(valid_loss) 


