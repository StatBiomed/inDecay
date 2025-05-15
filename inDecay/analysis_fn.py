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


#   ____                _   ____        _         
#  |  _ \ ___  __ _  __| | |  _ \  __ _| |_ __ _  
#  | |_) / _ \/ _` |/ _` | | | | |/ _` | __/ _` | 
#  |  _ <  __/ (_| | (_| | | |_| | (_| | || (_| | 
#  |_| \_\___|\__,_|\__,_| |____/ \__,_|\__\__,_| 
import numpy as np

import numpy as np

def smooth_predictions(predictions, threshold):
    """
    Redistributes prediction values below a given threshold proportionally to other classes.

    Args:
        predictions (list): A list of prediction probabilities (sum should be 1).
        threshold (float): The threshold below which values are redistributed.

    Returns:
        list: Smoothed predictions.
    """
    # Convert predictions to NumPy array for easier manipulation

    predictions = np.array(predictions, dtype=float)

    # Identify indices where predictions are below the threshold
    below_threshold_indices = predictions < threshold

    # Calculate the total probability to redistribute
    redistribute_prob = np.sum(predictions[below_threshold_indices])

    # Zero out the values below the threshold
    predictions[below_threshold_indices] = 0

    # Redistribute proportionally to remaining classes
    remaining_sum = np.sum(predictions)
    if remaining_sum > 0:
        predictions += (redistribute_prob * (predictions / remaining_sum))
    TOL = 1e-6  # 0.0001% tolerance
    if not np.isclose(np.sum(predictions, axis=1), 1, atol=TOL, rtol=0).all():
        predictions = np.array(predictions / np.sum(predictions, axis=1, keepdims=True), dtype=float)
    assert np.isclose(np.sum(predictions, axis=1), 1, atol=TOL, rtol=0).all(), np.sum(predictions, axis=1)
   
    return predictions#.tolist()



    
def read_test_oligos(test_oligo_path=None):
    if test_oligo_path is None:
        test_oligo_path = os.path.join(PATH.main_dir, "result/test_set_oligo_Feb2.txt")
    test_oligos = pd.read_table(test_oligo_path, names=['OligoID'])["OligoID"].values
    return test_oligos
    

def read_Lindel_result(exp):
    """
    test set predicted profile
    """
    if "LV7A" not in exp:
        pass

    test_file_dir = pj(PATH.high_dir, exp)
    test_npy_file = [file for file in os.listdir(test_file_dir) if "Lindel_pred_test_" in file]
                       
    if len(test_npy_file) >1:
        # if more than 1, then take the most accurate one
        klds = [find_kld(fn) for fn in test_npy_file]
        test_npy_file = test_npy_file[np.argmin(klds)]
    elif len(test_npy_file) == 0:
        raise FileNotFoundError
    else:
        test_npy_file = test_npy_file[0]
                       
    test_pred = np.load(pj(test_file_dir, test_npy_file))
    # pred_dict= {"Oligo%d"%o:y for o, y in zip(order, test_pred)}        
    # print(exp, len(order))
    # return [pred_dict[oligo] for oligo in test_oligos]
    return test_pred

def read_labeld_XY_matrix(matrix_path):
    """
    the last column of the matrix record the order of the Guides
    """
    if matrix_path.endswith("npz"):
        raw = np.load(matrix_path)['arr_0']
    else:
        raw = np.load(matrix_path)
    data = raw[:,:-1]
    oligo_order = raw[:,-1]
    return data.astype('float32'), oligo_order

def read_Lindel_TestY(exp, test_oligos):
    """
    test set true profile
    """
    Cellline = exp.split("_")[3]
    rep = exp.split("_")[4]

    Y_file = pj(PATH.high_dir, exp, f"{Cellline}_{rep}.npy")
    Y, order = read_labeld_XY_matrix(Y_file)
    Y_lookup = {"Oligo%d"%o:y for o,y in zip(order, Y)}
    return np.stack([Y_lookup[o] for o in test_oligos])


def get_reference(reference_path):
    """
    get annotaion of the guides
    each value is a list of [Guide, refseq, pamsite, Strand]
    """
    reference =list(SeqIO.parse(reference_path,'fasta'))
    # dict : oligo -> list    
    ref_info_lookup = {}
    for SeqRecord in reference:
        OligoID, Guide = SeqRecord.id.split("_")
        _, pamsite, Strand = SeqRecord.description.split(" ")
        pamsite = int(pamsite)
        refseq = SeqRecord.seq.__str__()
        
        ref_info_lookup[OligoID] = [Guide, refseq, pamsite, Strand]
    return ref_info_lookup





#   __  __      _        _          
#  |  \/  | ___| |_ _ __(_) ___ ___ 
#  | |\/| |/ _ \ __| '__| |/ __/ __|
#  | |  | |  __/ |_| |  | | (__\__ \
#  |_|  |_|\___|\__|_|  |_|\___|___/                          


def kld_fn(x, y, reduction='mean'):
    X = torch.from_numpy(x+1e-8)
    Y = torch.from_numpy(y+1e-8)
    kld_instance = torchmetrics.KLDivergence(reduction=reduction)
    
    if reduction=='mean':
        return kld_instance(X,Y).numpy().item()
    else:
        return kld_instance(X,Y).numpy()


def label_mh(refseq, cutsite, label_df):
    """for ForeCasT data

    Args:
        refseq (_type_): _description_
        cutsite (_type_): _description_
        label_df (_type_): _description_

    Returns:
        _type_: _description_
    """
    # construct
    filtered_map = alignmap.construct_diagonal_map(refseq, cutsite)
    detected_events = alignmap.extract_features_from_map(filtered_map)

    is_mh = np.zeros((label_df.shape[0],1))
    mml_v = np.zeros((label_df.shape[0],))
    for i, locs in enumerate(label_df['loc'].values):
        locs = eval(locs)
        for ss_end in locs:
            left, right = ss_end[:2]
            dl = right - left
            relative_ss = left - cutsite
            event_name = f"{relative_ss}+{dl}"

            if event_name in detected_events.keys():
                is_mh[i] = 1
                mml_v[i] = detected_events[event_name]

    label_df['mh_length'] = mml_v
    return is_mh, label_df


def topk_kendall_fn(x, y, k, reduction='mean'):
    x_top_idx = x.argpartition(-1*k, axis=1)[:,-1*k:]
    y_top= y[:, x_top_idx[0]]
    x_top= x[:, x_top_idx[0]]
    # print("x:",x_top,"y:",y_top)
    Ktau, p = kendalltau(x_top, y_top)

    if reduction == 'mean':
        tau = np.mean(Ktau)
    elif reduction == 'sum':
        tau = np.sum(Ktau)
    elif reduction == 'none':
        tau = np.array(Ktau)
    else:
        raise ValueError("Invalid argument for reduction")
    return Ktau

def topk_kendall(x,y,k, reduction='mean'):
    if x.shape[1] < k:
        to_pad = k - x.shape[1]
        x = np.concatenate([x, np.zeros((1,to_pad))], axis=1)
        y = np.concatenate([y, np.zeros((1,to_pad))], axis=1)

    fn = partial(topk_kendall_fn, k=k, reduction=reduction)

    top = fn(x,y)
    if "__iter__" in dir(top):
        if len(top) == 1:
            top = top.item()
    return top


def minor_events_recall_ratio(x, y, thre=0.2, reduction='mean'):
    # x_event= np.argwhere(x > 0)
    x_me_idx= np.argwhere((x > 0.05) & (x < thre))
    x_me=x[:, x_me_idx.T[1]].sum()
    y_me=y[:, x_me_idx.T[1]].sum()
    diff = x_me - y_me
    return diff

def top_events(x, y, k, reduction='mean'):
    x_top_idx = x.argpartition(-1*k, axis=1)[:,-1*k:]
    y_top= y[:, x_top_idx[0]]
    x_top= x[:, x_top_idx[0]]
    return x_top, y_top


def major_events(x, y, thre=0, reduction='mean'):
    # x_event= np.argwhere(x > 0)
    x_me_idx= np.argwhere((x > thre))
    x_me=x[:, x_me_idx.T[1]]
    y_me=y[:, x_me_idx.T[1]]
    return x_me, y_me

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

def transform_r2(x,y, transform_matrix): 
    X = x @ transform_matrix
    Y = y @ transform_matrix
    return pearsonr(X,Y)[0]**2



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



def compute_topk_overlaps_IDL(
    Ytrue_IDL: Union[List, np.ndarray], 
    Ypred_IDL: Union[List, np.ndarray], 
    klist: List[int], 
    reduction: str = 'mean'
) -> Dict[str, float]:
    """
    Compute top-k overlaps for multiple k values using predefined recall functions.
    
    Args:
        Ytrue_IDL: Ground truth data
        Ypred_IDL: Predicted data
        klist: List of k values to compute (e.g., [1, 2, 3, 5, 10])
        reduction: Reduction method ('mean', 'sum', etc.)
        
    Returns:
        Dictionary with keys like 'top{k}_overlap' and computed values
    """
    # Map k values to their corresponding recall functions
    results={}
    for k in klist:
        # Get the function and compute overlap
        results[f'Top{k}_IDL'] = topk_recall_fn(Ytrue_IDL, Ypred_IDL, k=k, reduction=reduction)
        
    return results

def compute_topk_error_IDL(
    Ytrue_IDL: Union[List, np.ndarray], 
    Ypred_IDL: Union[List, np.ndarray], 
    klist: List[int], 
    reduction: str = 'mean'
) -> Dict[str, float]:
    """
    Compute top-k overlaps for multiple k values using predefined recall functions.
    
    Args:
        Ytrue_IDL: Ground truth data
        Ypred_IDL: Predicted data
        klist: List of k values to compute (e.g., [1, 2, 3, 5, 10])
        reduction: Reduction method ('mean', 'sum', etc.)
        
    Returns:
        Dictionary with keys like 'top{k}_overlap' and computed values
    """
    # Map k values to their corresponding recall functions
    results={}
    for k in klist:
        # Get the function and compute overlap
        y_t, pred_t= top_events(Ytrue_IDL, Ypred_IDL, k)
        results[f'abs_error_top{k}_IDL'] = np.sum(abs(y_t- pred_t))
        
    return results

def compute_major_IDL(
    Ytrue_IDL: Union[List, np.ndarray], 
    Ypred_IDL: Union[List, np.ndarray], 
    threlist: List[float], 
    reduction: str = 'mean'
) -> Dict[str, float]:
    """
    Compute top-k overlaps for multiple k values using predefined recall functions.
    
    Args:
        Ytrue_IDL: Ground truth data
        Ypred_IDL: Predicted data
        klist: List of k values to compute (e.g., [1, 2, 3, 5, 10])
        reduction: Reduction method ('mean', 'sum', etc.)
        
    Returns:
        Dictionary with keys like 'top{k}_overlap' and computed values
    """
    # Map k values to their corresponding recall functions
    results={}
    for thre in threlist:
        # Get the function and compute overlap
        results[f'Major{thre}_IDL'] = reduction_major([major_events_recall_ratio(Ytrue_IDL, Ypred_IDL, thre, reduction=reduction)],reduction_fn= eval(f"np.nan{reduction}"))
        
    return results

def assessment_recipe_41IDL(Ytrue_IDL, Ypred_IDL, class_names, reduction, top_metric, major_metric):
    
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
    # atop= adaptive_topk_recall(Ytrue_IDL, Ypred_IDL)
    y_0, pred_0 = major_events(Ytrue_IDL, Ypred_IDL)
    Cos = cosine_similarity(y_0, pred_0)
    # overlapping of most frequent events

    topkoverlaps = compute_topk_overlaps_IDL(Ytrue_IDL, Ypred_IDL, klist=top_metric, reduction=reduction)
    major_IDL = compute_major_IDL(Ytrue_IDL, Ypred_IDL, threlist= major_metric, reduction=reduction)
    toperror = compute_topk_error_IDL(Ytrue_IDL, Ypred_IDL, klist=top_metric,)
    Del_transform = Indel_Len_to_InDel_ratio()[:,0]

    # actually del_r2 is always the same as ins_r2
    try:
        del_r2 = transform_r2(Ytrue_IDL, Ypred_IDL, Del_transform)
    except:
        del_r2= 0

    # W1 = Fix_class_W1_distance(Ytrue_IDL, Ypred_IDL, reduction=reduction)

    Ktau, p = kendalltau(Ytrue_IDL, Ypred_IDL)
    metric_dict = {
        "KLD_IDL":kld,
        "CosineSimilarity_IDL":Cos[0][0],
        # "adaptive_topk_recall":atop,
        # "W1-distance_IDL":W1,
        "delratio_r2":del_r2,
        "Kendall_tau_IDL":Ktau
    } 
    return metric_dict | major_IDL | major_IDL | topkoverlaps | toperror


#     _____               ____          _     _____                 
#    |  ___|__  _ __ ___ / ___|__ _ ___| |_  |  ___|   _ _ __   ___ 
#    | |_ / _ \| '__/ _ \ |   / _` / __| __| | |_ | | | | '_ \ / __|
#    |  _| (_) | | |  __/ |__| (_| \__ \ |_  |  _|| |_| | | | | (__ 
#    |_|  \___/|_|  \___|\____\__,_|___/\__| |_|   \__,_|_| |_|\___|

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

def Forecast_collapse_Ins_events(Y_lookup, pred_lookup, smooth, reduction, top_metric, major_metric, tau_metric):
    """
    This function will compute the top events with insertion events all collapse into 1bp. 2bp

    Args:
        Y_lookup (dict): Oligos : ndarray
        pred_lookup (dict): Oligos -> ndarray [pred]
    """


    all_tau_ls = []
    cal_cos = []
    major_ls= create_empty_lists(major_metric,'Coll_I_Major')
    top_ls= create_empty_lists(top_metric,'Coll_I_Top')
    toperror_ls= create_empty_lists(top_metric,'Coll_I_abs_error_top')
    tau_ls= create_empty_lists(tau_metric,'Coll_I_Kendall_Top')
    for oligo, Y in Y_lookup.items():

        Y = Y.T
        Y_df = pd.DataFrame(Y_lookup[oligo],columns=['Identifier', 'Probability'])
        Y_df['Predicted'] = pred_lookup[oligo].flatten()
        Y_df['Collapse_ins'] = Y_df['Identifier'].apply(lambda x: x.split("_")[0] if x.startswith("I") else x)
        # summing ins prop according to ins-length
        collapse_ins_df = Y_df.groupby("Collapse_ins").agg({'Probability':'sum', 'Predicted':'sum'})
        coll_y = collapse_ins_df.values[:,[0]].T
        coll_pred = collapse_ins_df.values[:,[1]].T
        coll_pred=smooth_predictions(coll_pred,threshold=smooth)
        all_tau_ls.append(kendalltau(coll_y, coll_pred))
        y_0, pred_0= major_events(coll_y, coll_pred)
        cal_cos.append( cosine_similarity(y_0, pred_0)[0][0])
        for thre in major_metric:
            major_ls[f'Coll_I_Major{thre}'].append(major_events_recall_ratio(coll_y, coll_pred, thre, reduction))
        for k in top_metric:
            top_ls[f'Coll_I_Top{k}'].append(topk_recall_fn(coll_y, coll_pred, k, reduction))
            coll_yt, coll_predt= top_events(coll_y, coll_pred, k)
            toperror_ls[f'Coll_I_abs_error_top{k}'].append(np.sum(abs(coll_yt - coll_predt)))
        for t in tau_metric:  
            tau_ls[f'Coll_I_Kendall_Top{t}'].append(topk_kendall(coll_y, coll_pred, t, reduction))


    if reduction != 'none':
        reduction_fn = eval(f"np.nan{reduction}")
        cal_cos = reduction_fn(cal_cos)
        all_tau_ls = reduction_fn(all_tau_ls)
        for t in tau_metric:  
            tau_ls[f'Coll_I_Kendall_Top{t}']=reduction_fn(tau_ls[f'Coll_I_Kendall_Top{t}'])
        for thre in major_metric:
            major_ls[f'Coll_I_Major{thre}']= reduction_major(major_ls[f'Coll_I_Major{thre}'],reduction_fn)
        for k in top_metric:
            top_ls[f'Coll_I_Top{k}']=reduction_fn(top_ls[f'Coll_I_Top{k}'])
            toperror_ls[f'Coll_I_abs_error_top{k}']=reduction_fn(toperror_ls[f'Coll_I_abs_error_top{k}'])
            

    return {"oligos": Y_lookup.keys(), "Coll_I_CosineSimilarity": cal_cos, "Coll_I_Kendall_tau": all_tau_ls} | major_ls | top_ls | tau_ls | toperror_ls



def forecast_frameshift(ya, pre, indels):
    indel_lengths = [tokFullIndel(idfr)[1] for idfr in indels[0]] # ForeCast's func for getting indel length
    is_frameshift = [idl%3!=0 for idl in indel_lengths]      #  mod 3 != 0
    
    y_fs = ya @ is_frameshift
    pred_fs = pre @ is_frameshift
    
    return y_fs[0], pred_fs[0]

def forecast_delratio(ya, pre, indels):
    indel_lengths = [tokFullIndel(idfr)[0] for idfr in indels[0]] # ForeCast's func for getting indel length
    is_del = [idl.startswith('D') for idl in indel_lengths]      #  mod 3 != 0
    
    y_fs = ya @ is_del
    pred_fs = pre @ is_del
    
    return y_fs[0], pred_fs[0]

def forecast_transform(Y_lookup, pred_lookup, transform=forecast_frameshift):
    """
    Summarize the ratio for each distribution
    """
    Y_fs = []
    Pred_fs = []
    for gene, Y in Y_lookup.items():
        Y = Y.T

        pred = pred_lookup[gene]
        Indel = Y[[0],:]
        y = Y[[1],:].astype("float32")

        # frameshift
        y_fs, pred_fs = transform(y, pred, Indel)
        Y_fs.append(y_fs)
        Pred_fs.append(pred_fs)

    return Y_fs, Pred_fs


def assessment_recipe_forecast(Y_lookup, pred_lookup, smooth, reduction='mean', top_metric=[1,3,5,10], major_metric=[0.2, 0.25, 0.3, 0.35, 0.4], tau_metric=[3,5,10] ):
    """
    This function compute the metrics for every samples
    """

    metrics_name = [f"Top{k} events recall" for k in top_metric]
    perform = []
    y_framshift = []
    pred_framshift = []
    y_delratio = []
    pred_delratio = []
    cal_cos=[]
    assert len(Y_lookup) == len(pred_lookup), "samples are not matched"
    major_ls= create_empty_lists(major_metric,'Major')
    for oligo, Y in Y_lookup.items():
        
        Y = Y.T
        pred = pred_lookup[oligo]
        pred= smooth_predictions(pred,smooth)
        
        Indel = Y[[0],:]
        y = Y[[1],:].astype("float32")

        # other metrics
        metrics={name: topk_recall_fn(y, pred, k, reduction) for name, k in zip(metrics_name, top_metric)}
        metrics.update({'KL Divergence': kld_fn(y, pred, reduction)})
        y_0, pred_0= major_events(y, pred)
        Cos=cosine_similarity(y_0, pred_0)
        cal_cos.append(Cos[0][0])
        metrics.update({'CosineSimilarity':Cos[0][0]})

        for thre in major_metric:
            major_ls[f'Major{thre}'].append(major_events_recall_ratio(y, pred, thre, reduction))

        for t in tau_metric:  
            metrics[f'Kendall_Top{t}']=topk_kendall(y, pred, t, reduction)
        
        # frameshift
        y_fs, pred_fs = forecast_frameshift(y, pred, Indel)
        y_dr, pred_dr = forecast_delratio(y, pred, Indel)

        y_framshift.append(y_fs)
        pred_framshift.append(pred_fs)
        y_delratio.append(y_dr)
        pred_delratio.append(pred_dr)

        metrics.update({'abs_frameshift_error': abs(y_fs-pred_fs)}) 
        perform.append(metrics)

    if reduction != 'none':
        reduction_fn = eval(f"np.nan{reduction}")
        for thre in major_metric:
            major_ls[f'Major{thre}']= reduction_major(major_ls[f'Major{thre}'],reduction_fn)

    perform_df = pd.json_normalize(perform)
    if reduction == 'mean':
        perform_json = perform_df[metrics.keys()].mean(axis=0).to_dict() | major_ls
    else:
        perform_json = {col:perform_df[col].values for col in perform_df.columns}

    perform_json['OligoID'] = list(Y_lookup.keys())

    perform_json['Rep1_frameshift'] = y_framshift
    perform_json['Pred_frameshift'] = pred_framshift

    perform_json['Rep1_delratio'] = y_delratio
    perform_json['Pred_delratio'] = pred_delratio
    if len(list(Y_lookup.keys()))>1:
        r = pearsonr(perform_json['Rep1_frameshift'], perform_json['Pred_frameshift'])[0]
        perform_json['R2 of Frameshift ratio'] = r**2

    Coll_I= Forecast_collapse_Ins_events(Y_lookup, pred_lookup, reduction=reduction, smooth=smooth, top_metric=top_metric, major_metric=major_metric, tau_metric=tau_metric)
    perform_json.update(Coll_I)
    
    return perform_json


def assessment_recipe_IDL_forecast(Y_lookup, pred_lookup, smooth, reduction='mean', top_metric =[1,2,3,5], major_metric=[0.2, 0.25, 0.3, 0.4, 0.5]):
    """
    convert Y lookup to 41 dimension Indel size distribution
    then assess several metrices
    """

    Ytrue_IDL,  Ypred_IDL = Indel_Len_Distribution_All(Y_lookup, pred_lookup)

    Ytrue_IDL = smooth_predictions(Ytrue_IDL.astype(float), threshold=smooth)
    Ypred_IDL = smooth_predictions(Ypred_IDL.astype(float), threshold=smooth)
    
    metrics_dict = assessment_recipe_41IDL(Ytrue_IDL, Ypred_IDL, class_557, reduction, top_metric, major_metric)
    return metrics_dict



def tokFullIndel(indel):
    """
    Function copied from ForeCast 
    Arguments:
        indel : indel identifier. e.g: D10_L-13C2R0
    Return:
        indel_type , indel size , details , muts
    """
    # D10_L-13C2R0
    indel_toks = indel.split('_')
    indel_type, indel_details = indel_toks[0], '' # D10
    if len(indel_toks) > 1:
        indel_details =  indel_toks[1]            # L-13C2R0
    cigar_toks = re.findall(r'([CLRDI]+)(-?\d+)', indel_details)

    details, muts = {'I':0,'D':0,'C':0}, []
    # only count the occurance of Insertion , deletion and complements 
    # L and R not counted
    for (letter,val) in cigar_toks:
        details[letter] = eval(val)

    #  > 2 means location identifier has letter no just 'L' and 'R'
    if len(indel_toks) > 2 or (indel_type == '-' and len(indel_toks) > 1):
        mut_toks = re.findall(r'([MNDSI]+)(-?\d+)(\[[ATGC]+\])?', indel_toks[-1])
        for (letter,val,nucl) in mut_toks:
            if nucl == '':
                nucl = '[]'
            muts.append((letter, eval(val), nucl[1:-1]))

    if indel_type[0] == '-':
        isize = 0
    else:
        isize = eval(indel_type[1:]) # D10 -> 10 , I2 -> 2
    return indel_type[0],isize,details, muts


def read_processed_df(exp, high_dir):
    """
    The processed csv file merging all oligos
    """
    Celline = exp.split("_")[3]
    rep = exp.split("_")[4]
    save_dir = pj(high_dir, exp)
    csv_path = pj(save_dir,f"{Celline}_{rep}.csv")
    processed_df = pd.read_csv(csv_path).astype({"Count":"int"})
    return processed_df

def ForeCast_del_ratio(processed_df,normalize=False):
    """
    Take the processed df and summarize the ratio of deletion events and ins events

    Input
    ----------
    processed_df : DataFrame, high_dir/<exp>/<cell>_<rep>.csv
    normalize : bool, whether the return dataframe will be normalized

    Return
    ----------
    DataFrame (n_oligo, 2), columns: [del , ins]
    """

    # ST_events = processed_df.query("`ForeCast_valid` == True")
    delratio_sum = processed_df.groupby(["OligoID",'Indel_type']).agg({"Count":'sum'}).reset_index(col_level=0)
    pivot_df = delratio_sum.pivot(index='OligoID', columns= 'Indel_type', values='Count')
    
    if normalize:
        pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)

    return pivot_df

def ForeCast_dlen_distribution(processed_df, normalize=False):
    """
    Take the processed df and summarize the dlen distribution for every Oligo, return a new dataframe with n_oligo, 41
    Deletion : 1-38bp , Insertion : 1-3bp

    Input
    ----------
    processed_df : DataFrame, high_dir/<exp>/<cell>_<rep>.csv
    normalize : bool, whether the return dataframe will be normalized

    Return
    ----------
    DataFrame (n_oligo, 41), 
    """
    Column = ["D%d"%i for i in range(1,39)] + ['I%d'%i for i in range(1,4)]

    # ST_events = processed_df.query("`ForeCast_valid` == True")
    processed_df.loc[:,"DI_len"] = processed_df.Identifier.apply(lambda x : x.split("_")[0])
    
    summary_df = processed_df.groupby(["OligoID",'DI_len']).agg({"Count":'sum'}).reset_index(col_level=0)
    pivot_df = summary_df.pivot(index='OligoID', columns='DI_len', values = 'Count').fillna(0)

    if normalize:
        pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)

    # fill the column to 41 categories
    for col in Column:
        if col not in pivot_df.columns:
            pivot_df[col] = 0 

    return pivot_df[Column]


##########################################
#     ForeCast data transformation
##########################################

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

def ST_subset(indel_type, Y_lookup, Pred_lookup, Oligos=None):
    IDType_of_ = partial(subset_by_indeltype, indeltype=indel_type, Y_lookup=Y_lookup, Pred_lookup=Pred_lookup)

    # oligos orders
    Oligos = list(Y_lookup.keys()) if Oligos is None else Oligos

    list_IDtype = {}
    for oligo in Oligos:
        list_IDtype[oligo] = IDType_of_(Oligo=oligo) 

    return list_IDtype



def ForeCast_format_pathway_ratio(test_oligos, processed_df, exp, pred_lookup):
    """
    Compare the ratio for ForeCast format 
    Input:
    ------------
    test_oligos: list of oligos
    precessed_df : pd.DataFrame, output of `read_processed_df`
    exp : str, 
    pred_lookup : str, 
    
    Return:
    ------------
    y_ratio_df, ypred_ratio_df
        pd.DataFrame, with columns ['c-NHEJ ins', 'MMEJ del', 'c-NHEJ del', 'OligoID', 'Cell']
        Each line is a oligo. 
    """
    
    cell = exp.split("_")[3]
    ypred_ratio = []
    y_ratio = []
    
    global ref_lookup
    if ref_lookup is None:
        ref_lookup = get_reference()
    
    
    for oligo in tqdm(test_oligos):
        
        Guide, refseq, pamsite, Strand = ref_lookup[oligo]
        label_df = read_data(oligo, processed_df, exp)
        mh_mask, label_df = label_mh(refseq, int(pamsite)-3, label_df)
        is_ins = label_df['Identifier'].apply(lambda x: x.startswith('I'))
        # compute y 
        y = label_df['Frac Sample Reads'].values
        
        # ins ratio
        ins_ratio = np.dot(y, is_ins).item()
        mh_ratio = np.dot(y*(1-ins_ratio), mh_mask)[0].item()
        nhej_ratio = 1  - mh_ratio - ins_ratio
        y_ratio.append([ins_ratio, mh_ratio, nhej_ratio, oligo, cell])
        
        # compute for y pred
        y_pred = pred_lookup[oligo]
        ins_pred = np.dot(y_pred, is_ins).item()
        mh_pred = np.dot(y_pred * (1-ins_ratio), mh_mask)[0].item()
        nhej_pred = 1  - mh_pred - ins_pred
        
        ypred_ratio.append([ins_pred, mh_pred, nhej_pred, oligo, cell])
        
    ypred_ratio_df = pd.DataFrame(ypred_ratio, columns=['c-NHEJ ins', 'MMEJ del', 'c-NHEJ del', 'OligoID', 'Cell'])
    y_ratio_df = pd.DataFrame(y_ratio, columns=['c-NHEJ ins', 'MMEJ del', 'c-NHEJ del', 'OligoID', 'Cell'])
    
    return y_ratio_df, ypred_ratio_df


def ratio_visualization(y_ratio_df, ypred_ratio_df, pathway):
    """
    
    pathway : 'MMEJ del', 'c-NHEJ del', 'c-NHEJ ins'
    """
    c = {"MMEJ del":np.array([172,137,27])/255,
         "c-NHEJ del":np.array([79,119,51])/255,
         "c-NHEJ ins":np.array([31,119,180])/255
        }[pathway]
    fig, axs = plt.subplots(1, 5, figsize=(20,3), dpi=500)
    axs = axs.flatten()
    for i,cell in enumerate(y_ratio_df['Cell'].unique()): 

        ax = axs[i]
        ydf = y_ratio_df.query('`Cell` == @cell')
        pdf = ypred_ratio_df.query('`Cell` == @cell')
        x, y = ydf[pathway].values, pdf[pathway].values
        ax.scatter(x, y,
                   alpha=0.7,
                   color=c)
        r = pearsonr(x,y)[0]
        ax.text(0.5, 0.05, s=r'$r$ = '+ str(round(r,3)))
        ax.set_xlabel(f"observed {pathway} ratio")
        ax.set_ylabel(f"predicted {pathway} ratio")
        
    return fig, axs


#########################################################
#     ForeCast data format distal and proximal ratio
#########################################################


def list_eval(loc_string):
    return eval(re.sub(r"([A-Z]{1,2})", r"'\1'", loc_string))

def get_distal(label_df, cutsite):
    """
    deletion that ends at cutsite
    """
    ends = [[s_e[1] for s_e in list_eval(ls)] for ls in label_df['loc'].values]
    distal_mask = [cutsite in e for e in ends]
    
    return distal_mask

def get_proximal(label_df, cutsite):
    """
    deletion that start from cutsite
    """
    last_site = cutsite -1
    ends = [[s_e[0] for s_e in list_eval(ls)] for ls in label_df['loc'].values]
    proximal_mask = [last_site in e for e in ends]
    
    return proximal_mask

def get_d(label_df):
    """
    keep deletion indels
    """
    is_d = label_df.Identifier.apply(lambda x: x.startswith("D"))
    
    return label_df[is_d]

def DP_del_ratio(label_df, cutsite, ratio="relative"):
    """
    return the ratio of distal , proximal deletion
    if ratio =="relative", relative ratio , else it is absolute probability
    """
    
    del_df = get_d(label_df)
    y_del = del_df['Frac Sample Reads'].values
    
    distal_mask = get_distal(del_df, cutsite)
    proximal_mask = get_proximal(del_df, cutsite)
    
    distal_ratio, proximal_ratio = np.dot(y_del, distal_mask), np.dot(y_del, proximal_mask)
    summ = distal_ratio + proximal_ratio
    
    if summ == 0:
        return 0,0
    elif ratio !="relative":
        return distal_ratio, proximal_ratio
    else:
        return distal_ratio/summ, proximal_ratio/summ



