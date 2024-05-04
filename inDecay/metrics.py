import os, sys
from . import my_utils, PATH
import numpy as np
import torch
import torchmetrics
from scipy.stats import pearsonr


# some transformation
class_557 = list(my_utils.rev_index.values())
frameshift_557 = my_utils.frame_shift

class_912 = my_utils.All_Lindel_class
ndel = 912 - 21
del_frameshift = [int(e.split('+')[1]) for e in class_912[:ndel-1]] + [38]
ins_frameshift = [int(e.split('+')[0]) for e in class_912[ndel:-1]] + [3]
all_frameshift= del_frameshift + ins_frameshift
frameshift_912 = np.array([fs%3!=0 for fs in all_frameshift])



# some metrics

# TODO: Topk coarse
# KL , entropy
# p_i * |pi-pi^pred| ; weight mean aboslute error
class Topk_Event_Overlapping(torchmetrics.Metric):
    def __init__(self, k):
        """
        Metric to quantify the recall of the top 5 frequent events
        """
        super().__init__()
        self.k = k
        self.add_state("overlap", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target = self._input_format(preds, target)  # type: ignore
        assert preds.shape == target.shape

        k = self.k if self.k <= target.shape[1] else target.shape[1]
        pred_idxs = torch.topk(preds, k=k, dim=1).indices.cpu().numpy()
        target_idxs = torch.topk(target, k=k, dim=1).indices.cpu().numpy()

        batch_overlap = 0
        for i_p, i_t in zip(pred_idxs, target_idxs):
            batch_overlap += len(np.intersect1d(i_p, i_t))

        self.overlap += batch_overlap # type: ignore
        self.total += target.shape[0]  # type: ignore

    def compute(self):
        return self.overlap.float() / self.total  # type: ignore

class Frameshift_Rsqure(torchmetrics.Metric):
    def __init__(self, outsize):
        """
        Metric to quantify the recall of the top 5 frequent events
        """
        super().__init__()
        self.outsize = outsize
        self.add_state("overlap", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target = self._input_format(preds, target)  # type: ignore
        assert preds.shape == target.shape

        k = self.k if self.k <= target.shape[1] else target.shape[1]
        pred_idxs = torch.topk(preds, k=k, dim=1).indices.cpu().numpy()
        target_idxs = torch.topk(target, k=k, dim=1).indices.cpu().numpy()

        batch_overlap = 0
        for i_p, i_t in zip(pred_idxs, target_idxs):
            batch_overlap += len(np.intersect1d(i_p, i_t))

        self.overlap += batch_overlap # type: ignore
        self.total += target.shape[0]  # type: ignore

    def compute(self):
        return self.overlap.float() / self.total  # type: ignore
    
def kld_fn(y1, y2, reduction='mean'):
    Y1 = torch.from_numpy(y1+1e-8)
    Y2 = torch.from_numpy(y2+1e-8)
    kld_instance = torchmetrics.KLDivergence(reduction=reduction)
    
    if reduction=='mean':
        return kld_instance(Y1,Y2).numpy().item()
    else:
        return kld_instance(Y1,Y2).numpy()


def top5_recall_fn(y1,y2):
    Y1 = torch.from_numpy(y1)
    Y2 = torch.from_numpy(y2)
    instance = Topk_Event_Overlapping(5)
    return instance(Y1,Y2).numpy().item()

def top10_recall_fn(y1,y2):
    Y1 = torch.from_numpy(y1)
    Y2 = torch.from_numpy(y2)
    instance = Topk_Event_Overlapping(10)
    return instance(Y1,Y2).numpy().item()
    

def frameshift557_r2(y1,y2): 
    Y1 = y1 @ frameshift_557
    Y2 = y2 @ frameshift_557
    return pearsonr(Y1,Y2)[0]**2

def frameshift912_r2(y1,y2): 
    Y1 = y1 @ frameshift_912
    Y2 = y2 @ frameshift_912
    return pearsonr(Y1,Y2)[0]**2

def insratio557_r2(y1,y2): 
    Y1 = y1 @ frameshift_transform
    Y2 = y2 @ frameshift_transform
    return pearsonr(Y1,Y2)[0]**2

def insratio912_r2(y1,y2): 
    Y1 = y1 @ frameshift_transform
    Y2 = y2 @ frameshift_transform
    return pearsonr(Y1,Y2)[0]**2