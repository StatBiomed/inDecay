import os, io, csv, sys, time
from . import PATH
import numpy as np
import pandas as pd
from scipy import special
import subprocess
import pickle as pkl
# -- selftarget utils --
sys.path.append(PATH.STpyutils_dir)
# -- lindel utils --
sys.path.append(PATH.Lindel_dir)
import matplotlib.pyplot as plt
global data_dir
global combine_train
global combine_test
global Lindel_only


def top_k_overlap(x, y, k, reduction='mean'):

    # the location of top10 events
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


def top5_recall_fn(x,y,reduction='mean'):
    fn = partial(top_k_overlap, k=5, reduction=reduction)
    return fn(x,y)

def top10_recall_fn(x,y,reduction='mean'):
    fn = partial(top_k_overlap, k=10, reduction=reduction)
    return fn(x,y)

def compute_entropy(y_p):
    """
    compute self information H = ∑ p*log(p)
    Arguments:
        y : (n_sample, n_events)
    Reutrn:
        entropy:  (n_sample,)
    """

    y = y_p + 1e-6
    # self information
    H = -1*np.multiply(y, np.log(y)).sum(axis=1)
    return H


def compute_KLD(y_q,y_p):
    """
    compute KL divergence ∑ p*log(p/q)
        y: true label, can include zero value
        y_hat: predicted values , can not include zero
    Return:
        KLD: scaler value 
    refer: `https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kl_div.html`
    """
    assert y_q.shape == y_p.shape, "please make sure the two input vector have aligned dimension"
    
    y = y_q + 1e-6
    y_hat = y_p + 1e-6

    kl_sum =  lambda i, i_hat: special.kl_div(i,i_hat).sum() + special.kl_div(i_hat,i).sum()
    # multiple sample
    if len(y.shape) >1:
        return [kl_sum(i,i_hat) for i,i_hat in zip(y,y_hat)]
    else:
        return kl_sum(y,y_hat)


def dlen_transform(class_label):
    """
    a transform matrix to summarize the indel length distribution

    Input
    ---------
    class_label : list of str, class name of 912 classes or 557 classes

    Return
    ---------
    Transformation matrix : ndarray (557/912, 41)
    """

    n_class = len(class_label)
    Transorm_M = np.zeros((n_class, 41))

    ndel = n_class - 21
    del_frameshift = [int(e.split('+')[1]) for e in class_label[:ndel-1]] + [38] # e: event
    ins_frameshift = [int(e.split('+')[0]) for e in class_label[ndel:-1]] + [3]
    all_frameshift= del_frameshift + ins_frameshift

    for i, fs in enumerate(del_frameshift):
        Transorm_M[i, fs-1] =1
    
    for i, fs in enumerate(ins_frameshift):
        Transorm_M[i+ndel, 37+fs] = 1
    
    return Transorm_M

def Indel_Len_to_InDel_ratio():
    """
    The transformation matrix from 41 indel len distribution to deleltion and insertion ratio
    Deletion : 1-38bp , Insertion : 1-3bp
    Return: ndarray (41,2)
    """
    transofrmation = np.zeros((41,2))
    transofrmation[:38,0] = 1
    transofrmation[38:,1] = 1
    return transofrmation

def DelRatio_transform(class_label):

    dlen_T = dlen_transform(class_label)
    IDL_2_DR = Indel_Len_to_InDel_ratio()

    return dlen_T @ IDL_2_DR



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
    processed_df["DI_len"] = processed_df.Identifier.apply(lambda x : x.split("_")[0])
    
    summary_df = processed_df.groupby(["OligoID",'DI_len']).agg({"Count":'sum'}).reset_index(col_level=0)
    pivot_df = summary_df.pivot(index='OligoID', columns='DI_len', values = 'Count').fillna(0)

    if normalize:
        pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)

    # fill the column to 41 categories
    for col in Column:
        if col not in pivot_df.columns:
            pivot_df[col] = 0 

    return pivot_df[Column]
