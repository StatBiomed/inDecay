from multiprocessing import reduction
import os, sys, re
import torch
import torchmetrics
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial
from scipy import special
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import auc
from typing import Dict, List, Union
from sklearn.metrics.pairwise import cosine_similarity
# qrguide
from qrguide.transformation import *
from collections import defaultdict

kld_matcher = re.compile(r"Lindel_pred_test_([\.,\d]{,20})_(\w*).npy")
find_kld = lambda fn : float(kld_matcher.match(fn).group(1))


global ref_lookup
ref_lookup = None

global rename_dict

rename_dict = {
    'BOB': 'iPSC', 'CHO': 'CHO',
    'E14TG2A': 'mESC', 'HAP1': 'HAP1','K562': 'K562'
}

pj = os.path.join

def major_events(x, y, thre=0, reduction='mean'):
    # x_event= np.argwhere(x > 0)
    x_me_idx= np.argwhere((x > thre))
    x_me=x[:, x_me_idx.T[1]]
    y_me=y[:, x_me_idx.T[1]]
    return x_me, y_me
def transform_r2(x,y, transform_matrix): 
    X = x @ transform_matrix
    Y = y @ transform_matrix
    return pearsonr(X,Y)[0]**2

def top_k_overlap(x, y, k, reduction='mean'):

    x_top_idx = x.argpartition(-1*k, axis=1)[:,-1*k:]
    y_top_idx = y.argpartition(-1*k, axis=1)[:,-1*k:]

    overlap_ls = [len(np.intersect1d(idxs1, idxs2)) for (idxs1, idxs2) in zip(x_top_idx,y_top_idx)]
    if reduction == 'mean':
        TopK = np.mean(overlap_ls)
    elif reduction == 'sum':
        TopK = np.sum(overlap_ls)
    elif reduction == 'none':
        TopK = np.array(overlap_ls)
    else:
        raise ValueError("Invalid argument for reduction")

    return TopK

def topk_recall_fn(x, y, k, reduction='mean'):
    if x.shape[1] < k:
        to_pad = k - x.shape[1]
        x = np.concatenate([x, np.zeros((1,to_pad))], axis=1)
        y = np.concatenate([y, np.zeros((1,to_pad))], axis=1)

    fn = partial(top_k_overlap, k=k, reduction=reduction)
    top = fn(x,y)
    
    if "__iter__" in dir(top):
        if len(top) == 1:
            top = top.item()
    return top

def topk_kendall(x,y,k, reduction='mean'):
    if x.shape[1] < k:
        k= x.shape[1]
    fn = partial(topk_kendall_fn, k=k, reduction=reduction)

    top = fn(x,y)
    if "__iter__" in dir(top):
        if len(top) == 1:
            top = top.item()
    return top

def topk_kendall_fn(x, y, k, reduction='mean'):
    from scipy.stats import kendalltau
    # Get indices of top k values for x and y
    idx_x = np.argsort(-x)[:,:k]
    idx_y = np.argsort(-y)[:,:k]
    combined_indices = np.union1d(idx_x, idx_y)
    
    rank_x = [[list(idx_xi).index(i)+1 if i in idx_xi else k+1 for i in combined_indices] for idx_xi in idx_x ]
    rank_y = [[list(idx_yi).index(i)+1 if i in idx_yi else k+1 for i in combined_indices] for idx_yi in idx_y]
    tau, _ = kendalltau(rank_x, rank_y)
    return tau
def kld_fn(x, y, reduction='mean'):
    X = torch.from_numpy(x+1e-8)
    Y = torch.from_numpy(y+1e-8)
    kld_instance = torchmetrics.KLDivergence(reduction=reduction)
    
    if reduction=='mean':
        return kld_instance(X,Y).numpy().item()
    else:
        return kld_instance(X,Y).numpy()
def forecast_frameshift(ya, pre, indels):
    if len(indels) >1: 
        indel_lengths = [tokFullIndel(idfr)[1] if '_' in idfr else int(idfr[1:]) for idfr in indels] # ForeCast's func for getting indel length
    else:
        indel_lengths = [tokFullIndel(idfr)[1] for idfr in indels[0]] # ForeCast's func for getting indel length
    is_frameshift = [idl%3!=0 for idl in indel_lengths]      #  mod 3 != 0
    
    y_fs = ya @ is_frameshift
    pred_fs = pre @ is_frameshift
    
    return y_fs[0], pred_fs[0]
# def forecast_frameshift(ya, pre, indels):
#     indel_lengths = [tokFullIndel(idfr)[1] for idfr in indels[0]] # ForeCast's func for getting indel length
#     is_frameshift = [idl%3!=0 for idl in indel_lengths]      #  mod 3 != 0
    
#     y_fs = ya @ is_frameshift
#     pred_fs = pre @ is_frameshift
    
#     return y_fs[0], pred_fs[0]

def forecast_frame_ratios(ya, pre, indels):
    """
    Compute per-reading-frame (mod 3) ratios for observed and predicted.
    Frame 0 = in-frame (+0), Frame 1 = +1 frameshift, Frame 2 = +2 frameshift.

    Args:
        ya   : ndarray (1, N_events), observed probabilities
        pre  : ndarray (1, N_events), predicted probabilities
        indels: list of indel identifier strings, or list-of-list (single-item)

    Returns:
        dict mapping frame int (0,1,2) -> (y_ratio, pred_ratio)
    """
    if len(indels) > 1:
        indel_lengths = [tokFullIndel(idfr)[1] if '_' in idfr else int(idfr[1:]) for idfr in indels]
    else:
        indel_lengths = [tokFullIndel(idfr)[1] for idfr in indels[0]]

    frames = np.array([l % 3 for l in indel_lengths], dtype=float)
    result = {}
    for frame in [0, 1, 2]:
        mask = (frames == frame).astype(float)
        result[frame] = (float((ya @ mask).flat[0]), float((pre @ mask).flat[0]))
    return result


def assessment_recipe_frame_breakdown(Y_lookup, pred_lookup, reduction='mean'):
    """
    Compute per-reading-frame R² (Pearson r²) between observed and predicted
    frame ratios across all oligos.

    Returns dict with keys 'frame0_r2', 'frame1_r2', 'frame2_r2', and per-oligo
    lists 'frame{i}_y', 'frame{i}_pred'.
    """
    per_frame_y    = {0: [], 1: [], 2: []}
    per_frame_pred = {0: [], 1: [], 2: []}

    for oligo, Y in Y_lookup.items():
        Y = Y.T
        Indel = Y[[0], :]
        y = Y[[1], :].astype("float32")
        pred = pred_lookup[oligo]

        frame_dict = forecast_frame_ratios(y, pred, Indel)
        for frame, (y_r, pred_r) in frame_dict.items():
            per_frame_y[frame].append(y_r)
            per_frame_pred[frame].append(pred_r)

    result = {}
    for frame in [0, 1, 2]:
        y_arr    = np.array(per_frame_y[frame])
        pred_arr = np.array(per_frame_pred[frame])
        if np.std(y_arr) > 0 and np.std(pred_arr) > 0:
            r, _ = pearsonr(y_arr, pred_arr)
            result[f'frame{frame}_r2'] = r ** 2
        else:
            result[f'frame{frame}_r2'] = float('nan')
        result[f'frame{frame}_y']    = per_frame_y[frame]
        result[f'frame{frame}_pred'] = per_frame_pred[frame]

    return result


def forecast_delratio(ya, pre, indels):
    if len(indels) >1: 
        indel_lengths = [tokFullIndel(idfr)[0] if '_' in idfr else idfr[0] for idfr in indels] # ForeCast's func for getting indel length
    else:
        indel_lengths = [tokFullIndel(idfr)[0] for idfr in indels[0]] # ForeCast's func for getting indel length
    is_del = [idl.startswith('D') for idl in indel_lengths]      #  mod 3 != 0
    
    y_fs = ya @ is_del
    pred_fs = pre @ is_del
    
    return y_fs[0], pred_fs[0]
def major_events_recall_ratio(x: np.ndarray, 
                             y: np.ndarray, 
                             thre: float = 0.15, 
                             reduction: str = 'mean') -> float | np.ndarray:
    """
    Calculate overlap ratio between indices of elements > threshold in x and y for each row.
    
    Args:
        x: 2D array where each row is a probability distribution (sums to 1)
        y: 2D array with same shape as x
        thre: Threshold for considering elements as "major events"
        reduction: 'none' returns per-row ratios, 'mean' returns NaN-ignored average
    
    Returns:
        Overlap ratio(s) as float or array, with NaN where x has no elements > thre
    """
    # Input validation
    assert x.shape == y.shape, "x and y must have the same shape"
    assert np.isclose(x.sum(axis=1)[0], 1), f"All rows in x must sum to 1 (normalized):{x.sum(axis=1)[0]}"
    
    overlaps = []
    for x_row, y_row in zip(x, y):
        # Get indices where values exceed threshold
        x_idx = np.where(x_row >= thre)[0]
        y_idx = np.where(y_row >= thre)[0]
        if len(x_idx) == 0:
            overlaps.append(np.nan)  # No major events in x
        else:
            # Calculate overlap ratio
            intersection = len(np.intersect1d(x_idx, y_idx))
            overlaps.append(intersection / len(x_idx))
    
    # Apply reduction
    if len(overlaps)==1 and np.isnan(overlaps).any():
        return np.array([np.nan], dtype=float)
    else:
        if reduction == 'mean':
            return np.nanmean(overlaps)
        elif reduction == 'sum':
            return np.nansum(overlaps)
        elif reduction == 'none':
            return np.array(overlaps)
        else:
            raise ValueError(f"Invalid reduction: {reduction}. Use 'mean', 'sum', or 'none'.")

def reduction_major(majorls, reduction_fn):
    if len (majorls) ==1 and np.isnan(majorls).any():
        majorls = np.nan
    else:
        majorls = reduction_fn(majorls)
    return majorls 
def create_empty_lists(vlist, metric):
    # Create a dictionary to store lists with dynamic names
    lists_dict = {}
    for v in vlist:
        lists_dict[f'{metric}{v}'] = []
    return lists_dict


def assessment_recipe_forecast_zygote(
    df_lookup,
    reduction='mean', top_metric=[1,5], 
    major_metric=[0.25], tau_metric=[5]):
    """
    Compute performance metrics for each sample/model.
    Arguments:
        df_lookup: dict of DataFrames, indexed by oligo.
        reduction: string, reduction method ('mean', 'sum', 'none').
        top_metric: list of ints, k for top-k event recall.
        major_metric: list of thresholds for major event recall.
        tau_metric: list of ints for Kendall calculation.
    Returns:
        Dictionary mapping model name to metrics, per-oligo lists, and summary statistics.
    """

    metrics_name = [f"Top{k} event recall" for k in top_metric]

    # Per-model aggregators
    per_model_metrics = defaultdict(lambda: defaultdict(list))
    per_model_oligoid = defaultdict(list)
    per_model_rep1_frameshift = defaultdict(list)
    per_model_rep1_delratio = defaultdict(list)
    per_model_pred_frameshift = defaultdict(list)
    per_model_pred_delratio = defaultdict(list)
    per_model_major_metric = defaultdict(list)

    # Main calculation loop
    for oligo, df in df_lookup.items():
        Indel = df['Indel'].values if 'Indel' in df else df.index.values
        models = [c for c in df.columns if c not in ['Indel', 'observed']]
        y = np.array([df['observed'].values.tolist()])
        for model in models:
            pred = np.array([df[model].values.tolist()])

            # Compute metrics
            metrics = {name: topk_recall_fn(y, pred, k, reduction) 
                       for name, k in zip(metrics_name, top_metric)}
            metrics['KL Divergence'] = kld_fn(y, pred, reduction)

            for thre in major_metric:
                per_model_major_metric[(model, thre)].append(
                    major_events_recall_ratio(y, pred, thre, reduction)
                )
            for t in tau_metric:
                metrics[f'Top{t} Kendall tau'] = topk_kendall(y, pred, t, reduction)
            y_fs, pred_fs = forecast_frameshift(y, pred, Indel)
            y_dr, pred_dr = forecast_delratio(y, pred, Indel)
            metrics['abs frameshift error'] = abs(y_fs - pred_fs)

            per_model_oligoid[model].append(oligo)
            per_model_rep1_frameshift[model].append(y_fs)
            per_model_pred_frameshift[model].append(pred_fs)
            per_model_rep1_delratio[model].append(y_dr)
            per_model_pred_delratio[model].append(pred_dr)

            # Store all scalar metrics for later mean/aggregation
            for k, v in metrics.items():
                per_model_metrics[model][k].append(v)

    # Compile final per-model output
    output = {}
    reduction_fn = getattr(np, f"nan{reduction}")
    for model in per_model_metrics:
        res = {}
        # Aggregate scalar metrics across oligos (mean or list)
        for k, vlist in per_model_metrics[model].items():
            res[k] = reduction_fn(vlist) if reduction != 'none' else vlist
        # Major event metrics aggregation
        
        for thre in major_metric:
            major_key = f'Major({thre}) event recall'
            vals = per_model_major_metric[(model, thre)]# Complete solution with error handling
            flat_vals = np.array([float(x) if np.isscalar(x) else x.item() for x in vals])
            res[major_key] = reduction_fn(flat_vals) if reduction != 'none' else vals
        
        res['OligoID'] = per_model_oligoid[model]
        
        res['Rep1_frameshift_ratio'] = per_model_rep1_frameshift[model]
        res['Pred_frameshift_ratio'] = per_model_pred_frameshift[model]
        res['Pred_del_ratio'] = per_model_pred_delratio[model]
        res['Rep1_del_ratio'] = per_model_rep1_delratio[model]
        # R2 calculation for frameshift ratio
        yfs = per_model_rep1_frameshift[model]
        pfs = per_model_pred_frameshift[model]
        if len(yfs) > 1 and len(pfs) > 1 and np.std(yfs) > 0 and np.std(pfs) > 0:
            r, _ = pearsonr(yfs, pfs)
            res['R2 of Frameshift Ratio'] = r**2
        else:
            res['R2 of Frameshift Ratio'] = None
        output[model] = res

    return output

def transform_r2(x,y, transform_matrix): 
    X = x @ transform_matrix
    Y = y @ transform_matrix
    return pearsonr(X,Y)[0]**2

def assessment_recipe_32IDL(
    df_lookup,
    reduction='mean', top_metric=[1,5], 
    major_metric=[0.25], tau_metric=[3,5],
    class_names=None):
    """
    Compute 32-binned IDL metrics for each sample/model in zygote forecast.
    Arguments:
        df_lookup: dict of DataFrames, indexed by oligo.
        reduction: string, reduction method ('mean', 'sum', 'none').
        top_metric: list of ints, k for top-k event recall/overlap.
        major_metric: list of thresholds for major event recall.
        tau_metric: list of ints for Kendall calculation (currently returns one value).
        class_names: optional, required for class transformation if input shape >41.
    Returns:
        Dict mapping model name to summary metrics and per-oligo lists.
    """

    # Per-model aggregators
    per_model_metrics = defaultdict(lambda: defaultdict(list))
    per_model_oligoid = defaultdict(list)
    per_model_rep1_frameshift = defaultdict(list)
    per_model_pred_frameshift = defaultdict(list)
    per_model_pred_delratio = defaultdict(list)
    per_model_major_metric = defaultdict(list)
    metrics_name = [f"Top{k}_IDL" for k in top_metric]
    # Main calculation loop
    for oligo, df in df_lookup.items():
        models = [c for c in df.columns if c not in ['Indel','observed']]
        Indel = df.index.values
        Indel_IDL= [ind.split('_')[0] if '_' in ind else ind for ind in Indel]
        Ytrue = df['observed'].values
        Ytrue_IDL = defaultdict(float)
        for k, v in zip(Indel_IDL, Ytrue):
            Ytrue_IDL[k]+=v
        
        Ytrue_IDL = np.array([list(Ytrue_IDL.values())])
        for model in models:
            Ypred = df[model].values
            Ypred_IDL = defaultdict(float)
            for k, v in zip(Indel_IDL, Ypred):
                Ypred_IDL[k]+=v
            Ypred_IDL = np.array([list(Ypred_IDL.values())])
            
            # KLD
            kld = kld_fn(Ytrue_IDL, Ypred_IDL, reduction=reduction)

            # Top-k overlaps and errors
            metrics = {name: topk_recall_fn(Ytrue_IDL, Ypred_IDL, k, reduction) 
                       for name, k in zip(metrics_name, top_metric)}
            for thre in major_metric:
                per_model_major_metric[(model, thre)].append(
                    major_events_recall_ratio(Ytrue_IDL, Ypred_IDL, thre, reduction)
                )
            for t in tau_metric:
                metrics[f'Kendall_Top{t}_IDL'] = topk_kendall(Ytrue_IDL, Ypred_IDL, t, reduction)

            Ktau, _ = kendalltau(Ytrue_IDL, Ypred_IDL)

            # Collect for model
            per_model_oligoid[model].append(oligo)
            per_model_metrics[model]['KLD_IDL'].append(kld)
            per_model_metrics[model]['Kendall_tau_IDL'].append(Ktau)
            for k, v in metrics.items():
                per_model_metrics[model][k].append(v)
            

    # Compile final per-model output
    output = {}
    reduction_fn = getattr(np, f"nan{reduction}")
    for model in per_model_metrics:
        res = {}
        # Aggregate over oligos
        for k, vlist in per_model_metrics[model].items():
            flat_vals = np.array([float(x) if np.isscalar(x) else x.item() for x in vlist])
            if np.isnan(flat_vals).all():
                res[k] = np.nan
            else:
                res[k] = reduction_fn(flat_vals) if reduction != 'none' else flat_vals
        res['OligoID'] = per_model_oligoid[model]
        output[model] = res
    return output

def assessment_recipe_forecast(Y_lookup, pred_lookup, reduction='mean'):
    """
    This function compute the metrics for somatic samples from qrguide
    """

    metrics_fn = [kld_fn, top1_recall_fn, top5_recall_fn, top10_recall_fn]
    metrics_name = ['KL divergence', 'Top1 events recall','Top5 events recall','Top10 events recall']
    perform = []
    y_framshift = []
    pred_framshift = []
    y_delratio = []
    pred_delratio = []
    assert len(Y_lookup) == len(pred_lookup), "samples are not matched"
    for oligo, Y in Y_lookup.items():
        
        Y = Y.T
        pred = pred_lookup[oligo]
        Indel = Y[[0],:]
        y = Y[[1],:].astype("float32")
        # other metrics
        perform.append({name:fn(pred,y,reduction) for name, fn in zip(metrics_name, metrics_fn)})

        # frameshift
        y_fs, pred_fs = forecast_frameshift(y, pred, Indel)
        y_dr, pred_dr = forecast_delratio(y, pred, Indel)

        y_framshift.append(y_fs)
        pred_framshift.append(pred_fs)

        y_delratio.append(y_dr)
        pred_delratio.append(pred_dr)

    perform_df = pd.json_normalize(perform)
    perform_df['OligoID'] = list(Y_lookup.keys())

    perform_df['Rep1_frameshift'] = y_framshift
    perform_df['Pred_frameshift'] = pred_framshift

    perform_df['Rep1_delratio'] = y_delratio
    perform_df['Pred_delratio'] = pred_delratio
    

    if reduction == 'mean':
        perform_json = perform_df[metrics_name].mean(axis=0).to_dict()
    else:
        perform_json = {col:perform_df[col].values for col in perform_df.columns}
    r = pearsonr(perform_df['Rep1_frameshift'].values, perform_df['Pred_frameshift'].values)[0]

    perform_json['Rep1_frameshift']= perform_df['Rep1_frameshift'].values.tolist()
    perform_json['Pred_frameshift']= perform_df['Pred_frameshift'].values.tolist()
    perform_json['R2 of Frameshift ratio'] = r**2

    Coll_I_TopK = Forecast_collapse_Ins_TopK(Y_lookup, pred_lookup, reduction=reduction)
    perform_json.update(Coll_I_TopK)
    
    return perform_json

def Forecast_collapse_Ins_TopK(Y_lookup, pred_lookup, reduction='mean'):
    """
    This function will compute the top K with insertion events all collapse into 1bp. 2bp and 3bp

    Args:
        Y_lookup (dict): Oligos : ndarray
        pred_lookup (dict): Oligos -> ndarray [pred]
    """
    top_1_ls = []
    major25ls= []
    major15ls = []
    top_5_ls = []
    top_10_ls = []
    top_5_tau = []
    top_10_tau = []

    for oligo, Y in Y_lookup.items():
        Y = Y.T
        # collapse events
        Y_df = pd.DataFrame(Y_lookup[oligo],columns=['Identifier', 'Probability'])
        Y_df['Predicted'] = pred_lookup[oligo].flatten()
        Y_df['Collapse_ins'] = Y_df['Identifier'].apply(lambda x: x.split("_")[0] if x.startswith("I") else x)
        # summing ins prop according to ins-length
        collapse_ins_df = Y_df.groupby("Collapse_ins").agg({'Probability':'sum', 'Predicted':'sum'})
        coll_y = collapse_ins_df.values[:,[0]].T
        coll_pred = collapse_ins_df.values[:,[1]].T
        top_1_ls.append( top1_recall_fn(coll_y, coll_pred, reduction) )
        top_5_ls.append( top5_recall_fn(coll_y, coll_pred, reduction) )
        top_10_ls.append( top10_recall_fn(coll_y, coll_pred, reduction) )
        top_5_tau.append( topk_kendall_fn(coll_y, coll_pred, 5, reduction='mean'))
        top_10_tau.append( topk_kendall_fn(coll_y, coll_pred, 10, reduction='mean'))
        major25ls.append( major_events_recall_ratio(coll_y, coll_pred, 0.25, reduction))
        major15ls.append( major_events_recall_ratio(coll_y, coll_pred, 0.15, reduction))
        

    if reduction != 'none':
        reduction_fn = getattr(np, f"nan{reduction}")
        
        top_1_ls = reduction_fn(top_1_ls)
        top_5_ls = reduction_fn(top_5_ls)
        top_10_ls = reduction_fn(top_10_ls)
        uniform_list = [x.item() if hasattr(x, 'item') else x for x in major25ls]
        major25_ls = reduction_fn(uniform_list)
        uniform_list = [x.item() if hasattr(x, 'item') else x for x in major15ls]
        major15_ls = reduction_fn(uniform_list)
        top_5_tau = reduction_fn(top_5_tau)
        top_10_tau = reduction_fn(top_10_tau)


    return {"Top5 events tau": top_5_tau,"Top10 events tau": top_10_tau, "Coll_I_Top1": top_1_ls,"Coll_I_Top5": top_5_ls, "Coll_I_Top10":top_10_ls,"Major(0.15) event recall": major15_ls, "Major(0.25) event recall": major25_ls}


def top1_recall_fn(x,y,reduction='mean'):
    if x.shape[1] < 1:
        to_pad = 1 - x.shape[1]
        x = np.concatenate([x, np.zeros((1,to_pad))], axis=1)
        y = np.concatenate([y, np.zeros((1,to_pad))], axis=1)

    fn = partial(top_k_overlap, k=1, reduction=reduction)
    top1 = fn(x,y)
    
    if "__iter__" in dir(top1):
        if len(top1) == 1:
            top1 = top1.item()
    return top1

def top5_recall_fn(x,y,reduction='mean'):
    if x.shape[1] < 5:
        to_pad = 5 - x.shape[1]
        x = np.concatenate([x, np.zeros((1,to_pad))], axis=1)
        y = np.concatenate([y, np.zeros((1,to_pad))], axis=1)

    fn = partial(top_k_overlap, k=5, reduction=reduction)
    top5 = fn(x,y)
    
    if "__iter__" in dir(top5):
        if len(top5) == 1:
            top5 = top5.item()
    return top5

def top10_recall_fn(x,y,reduction='mean'):
    if x.shape[1] < 10:
        to_pad = 10 - x.shape[1]
        x = np.concatenate([x, np.zeros((1,to_pad))], axis=1)
        y = np.concatenate([y, np.zeros((1,to_pad))], axis=1)

    fn = partial(top_k_overlap, k=10, reduction=reduction)

    top10 = fn(x,y)
    if "__iter__" in dir(top10):
        if len(top10) == 1:
            top10 = top10.item()
    return top10

def assessment_recipe_41IDL(Ytrue_IDL, Ypred_IDL, class_names, reduction='mean'):
    
    if Ytrue_IDL.shape[1] in [557, 912]:
        T_IDL = IndelLen_transform(class_label=class_names)
        Ytrue_IDL = Ytrue_IDL @ T_IDL
        Ypred_IDL = Ypred_IDL @ T_IDL

    elif Ytrue_IDL.shape[1] == 894:
        T_IDL = IndelLen_transform894(class_label=class_names)
        Ytrue_IDL = Ytrue_IDL @ T_IDL
        Ypred_IDL = Ypred_IDL @ T_IDL
    else:
        assert Ytrue_IDL.shape[1] == 41
        assert Ypred_IDL.shape[1] == 41

    kld = kld_fn(Ytrue_IDL, Ypred_IDL, reduction=reduction)
    # overlapping of most frequent events
    top10_overlap = top10_recall_fn(Ytrue_IDL, Ypred_IDL, reduction=reduction)
    top5_overlap = top5_recall_fn(Ytrue_IDL, Ypred_IDL, reduction=reduction)
    top1_overlap = top1_recall_fn(Ytrue_IDL, Ypred_IDL, reduction=reduction)

    Del_transform = Indel_Len_to_InDel_ratio()[:,0]

    # actually del_r2 is always the same as ins_r2
    try:
        del_r2 = transform_r2(Ytrue_IDL, Ypred_IDL, Del_transform)
    except:
        del_r2= 0

    W1 = Fix_class_W1_distance(Ytrue_IDL, Ypred_IDL, reduction=reduction)

    Ktau, p = kendalltau(Ytrue_IDL, Ypred_IDL)

    metric_dict = {
        "KLD_IDL":kld,
        "Top1_IDL":top1_overlap,
        "Top5_IDL":top5_overlap,
        "Top10_IDL":top10_overlap,
        "W1-distance_IDL":W1,
        "delratio_r2":del_r2,
        "Kendall_tau_IDL":Ktau
    }
    return metric_dict
def assessment_recipe_41IDL_forecast(Y_lookup, pred_lookup, reduction='mean'):
    """
    convert Y lookup to 41 dimension Indel size distribution
    then assess several metrices in somatic
    """

    Ytrue_IDL,  Ypred_IDL = Indel_Len_Distribution_All(Y_lookup, pred_lookup)

    Ytrue_IDL = Ytrue_IDL.astype(float)
    Ypred_IDL = Ypred_IDL.astype(float)

    metrics_dict = assessment_recipe_41IDL(Ytrue_IDL, Ypred_IDL, class_557, reduction=reduction)
    return metrics_dict
def Indel_Len_Distribution_All(Y_lookup, Pred_lookup, Oligos=None):
    """
    Get Indel length distribution for all testset Oligos in the lookup objects.
    Input
    --------
    Y_lookup : dict, oligo -> ndarray of shape (n,2). [[Identifier name, frequency]]. The lookup item storing true labels with identifiers.
    Pred_lookup : dict, oligo -> ndarray of shape (1,n). The lookup item storing predicted values. [1, frequency]. 
    
    Return
    --------
    M_IDLen : Matrix of Indel Length distribution. (1133, 41). The indel is ordered like : Deltion [0-37] | Insertion [38-40].
    """
    # use parital func to fix two params
    IDLen_of_ = partial(Indel_Len_Transform, Y_lookup=Y_lookup, Pred_lookup=Pred_lookup)

    # oligos orders
    Oligos = list(Y_lookup.keys()) if Oligos is None else Oligos

    list_IDLen_true = []
    list_IDLen_pred = []
    for oligo in Oligos:
        M_IDLen_true, M_IDLen_pred = IDLen_of_(Oligo=oligo)
        list_IDLen_true.append( M_IDLen_true ) 
        list_IDLen_pred.append( M_IDLen_pred )
    list_IDLen_true = np.stack(list_IDLen_true) 
    list_IDLen_pred = np.stack(list_IDLen_pred) 

    return list_IDLen_true, list_IDLen_pred

def Fix_class_W1_distance(Y1, Y2, reduction='mean'):
    """
    Compute the W1 wasserstain distance with the following form
            W1(P1, P2) = \sum_i p_{1,i} * | p_{1,i} - p_{2,i} |
    Input
    --------
        Y1 : ndarray (n_sample, n_events), frequency matrix. Y1 is the base of wassertain.
        Y2 : ndarray (n_sample, n_events), frequency matrix
        reduction : str in ['mean', 'sum', 'none']
    Return
    --------
        W1 : ndarray or scaler, distance for each sample if reduction is 'none'. Otherwise return a reduced W1 over all samples.
    """
    assert Y1.shape == Y2.shape,  "discordant shape between Array 1 and Array 2"
    assert np.isclose(Y1.sum(axis=1)[0].item(), 1), "Input is not normalized"

    res = np.abs(Y1 - Y2)
    W1_elements = np.multiply(Y1,res).sum(axis=1)

    if reduction == 'mean':
        W1 = W1_elements.mean()
    elif reduction == 'sum':
        W1 = W1_elements.sum()
    elif reduction == 'none':
        W1 = W1_elements
    else:
        raise ValueError("Invalid argument for reduction")

    return W1

