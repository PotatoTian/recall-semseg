import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def RecallCrossEntropy(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    pred = input.argmax(1)
    idex = (pred != target).view(-1) 
    # recall loss
    gt_counter = torch.ones((c)).cuda() #.to(device)
    gt_idx, gt_count = torch.unique(target,return_counts=True)

    gt_count[gt_idx==255] = gt_count[0]
    gt_idx[gt_idx==255] = 0 #n_classes #0
    gt_counter[gt_idx] = gt_count.float()

    fn_counter = torch.ones((c)).cuda() #.to(device)
    fn = target.view(-1)[idex]
    fn_idx, fn_count = torch.unique(fn,return_counts=True)

    fn_count[fn_idx==255] = fn_count[0]
    fn_idx[fn_idx==255] = 0 #n_classes #0
    fn_counter[fn_idx] = fn_count.float()

    if weight is not None:
        weight = 0.5*(fn_counter / gt_counter) + 0.5 * weight
    else:
        weight = fn_counter / gt_counter

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, ignore_index=255
    )
    return loss,weight



def RecallPreCrossEntropy(input, target, fn_weight=None,fp_weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    pred = input.argmax(1)
    idex = (pred != target).view(-1) 
    # recall loss
    gt_counter = torch.ones((c)).cuda() #.to(device)
    gt_idx, gt_count = torch.unique(target,return_counts=True)

    gt_count[gt_idx==255] = gt_count[0]
    gt_idx[gt_idx==255] = 0 #n_classes #0
    gt_counter[gt_idx] = gt_count.float()

    fn_counter = torch.ones((c)).cuda() #.to(device)
    fn = target.view(-1)[idex]
    fn_idx, fn_count = torch.unique(fn,return_counts=True)

    fn_count[fn_idx==255] = fn_count[0]
    fn_idx[fn_idx==255] = 0 #n_classes #0
    fn_counter[fn_idx] = fn_count.float()
    if fn_weight is not None:
        fn_weight = 0.5*(fn_counter / gt_counter) + 0.5 * fn_weight

    else:
        fn_weight = fn_counter / gt_counter

    # precision loss
    pred_idx, pred_count= torch.unique(pred,return_counts=True)
    pred_counter = torch.ones((c)).cuda()

    pred_count[pred_idx==255] = pred_count[0]
    pred_idx[pred_idx==255] = 0 #n_classes #0
    pred_counter[pred_idx] = pred_count.float()

    fp = pred.view(-1)[idex]
    fp_idx, fp_count= torch.unique(fp,return_counts=True)
    fp_counter = torch.ones((c)).cuda()

    fp_count[fp_idx==255] = fp_count[0]
    fp_idx[fp_idx==255] = 0 #n_classes #0
    fp_counter[fp_idx] = fp_count.float()

    if fp_weight is not None:
        fp_weight = 0.5 * (fp_counter  / pred_counter) + 0.5 * fp_weight
    else:
        fp_weight = fp_counter  / pred_counter


    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    recall_loss = F.cross_entropy(
        input, target, weight=fn_weight, ignore_index=255
    )

    input_fp = input[idex]
    precision_loss = F.nll_loss(
            torch.log(1-torch.nn.Softmax(dim=1)(input_fp)+1e-9), fp, weight= fp_weight)

    return recall_loss + precision_loss,fn_weight,fp_weight


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, ignore_index=255
    )
    return loss




# def focal_loss(input_values, gamma):
#     """Computes the focal loss"""
#     p = torch.exp(-input_values)
#     loss = (1 - p) ** gamma * input_values
#     return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average = True, gamma=1):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
    def forward(self, input, target):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()

        # Handle inconsistent size between input and target
        if h != ht or w != wt:
            # input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
            input = F.upsample(input, size=(ht, wt), mode="bilinear", align_corners=True)
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)

        CE = F.cross_entropy(input, target, reduction='none')
        p = torch.exp(-CE)
        if self.weight is not None:
            loss = (1 - p) ** self.gamma * self.weight[target] * CE
        else: 
            loss = (1 - p) ** self.gamma * CE
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
        # return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight,size_average=self.size_average), self.gamma)

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, size_average = True,max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list[m_list==np.inf] = 0.0
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()

        # Handle inconsistent size between input and target
        if h != ht or w != wt:
            # input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
            input = F.upsample(input, size=(ht, wt), mode="bilinear", align_corners=True)

        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)

        index = torch.zeros_like(input, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = input - batch_m
        output = torch.where(index, x_m, input)

        return F.cross_entropy(self.s*output, target, weight=self.weight,size_average=self.size_average)





def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None):
    if not isinstance(input, tuple):
        return cross_entropy2d(input=input, target=target, weight=weight)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(
            target.device
        )

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss


def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=weight, reduce=False, ignore_index=250
        )

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)
