import torch
import torch.nn.functional as F

class RecallCrossEntropy(torch.nn.Module):
    def __init__(self, n_classes=19, ignore_index=255):
        super(RecallCrossEntropy, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def forward(self, input, target): 
        # input (batch,n_classes,H,W)
        # target (batch,H,W)
        pred = input.argmax(1)
        idex = (pred != target).view(-1) 
        
        #calculate ground truth counts
        gt_counter = torch.ones((self.n_classes,)).cuda() 
        gt_idx, gt_count = torch.unique(target,return_counts=True)
        
        # map ignored label to an exisiting one
        gt_count[gt_idx==self.ignore_index] = gt_count[1]
        gt_idx[gt_idx==self.ignore_index] = 1 
        gt_counter[gt_idx] = gt_count.float()
        
        #calculate false negative counts
        fn_counter = torch.ones((self.n_classes)).cuda() 
        fn = target.view(-1)[idex]
        fn_idx, fn_count = torch.unique(fn,return_counts=True)
        
        # map ignored label to an exisiting one
        fn_count[fn_idx==self.ignore_index] = fn_count[1]
        fn_idx[fn_idx==self.ignore_index] = 1 
        fn_counter[fn_idx] = fn_count.float()
        
        weight = fn_counter / gt_counter
        
        CE = F.cross_entropy(input, target, reduction='none',ignore_index=self.ignore_index)
        loss =  weight[target] * CE
        return loss.mean()

