import os, sys, re
from Bio import SeqIO
import importlib
import numpy as np
import pandas as pd
import pickle as pkl
from scipy import special
from scipy.signal import convolve2d
from functools import partial
import torch
import torchmetrics
from tqdm import tqdm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


#     _____               ____          _    
#    |  ___|__  _ __ ___ / ___|__ _ ___| |_  
#    | |_ / _ \| '__/ _ \ |   / _` / __| __| 
#    |  _| (_) | | |  __/ |__| (_| \__ \ |_  
#    |_|  \___/|_|  \___|\____\__,_|___/\__| 

global class_557

label,rev_index,features,frame_shift = pkl.load(importlib.resources.open_binary("qrguide.utils", "model_prereq.pkl"))
class_557 = list(rev_index.values())

wb = pkl.load(importlib.resources.open_binary("qrguide.utils", 'Model_weights.pkl'))

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


def IndelLen_transform(class_label):
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
    largest_len = 38 if n_class == 912 else 28
    if n_class == 912:
        del_frameshift = [int(e.split('+')[1]) for e in class_label[:ndel-1]] + [38] # e: event
    else:
        del_frameshift = [int(e.split('+')[1]) for e in class_label[:-21]] 
    ins_frameshift = [int(e.split('+')[0]) for e in class_label[ndel:-1]] + [3]
    all_frameshift= del_frameshift + ins_frameshift

    for i, fs in enumerate(del_frameshift):
        Transorm_M[i, fs-1] =1
    
    for i, fs in enumerate(ins_frameshift):
        Transorm_M[i+ndel, 37+fs] = 1
    
    return Transorm_M

def DelRatio_transform(class_label):

    dlen_T = IndelLen_transform(class_label)
    IDL_2_DR = Indel_Len_to_InDel_ratio()

    return dlen_T @ IDL_2_DR

def get_mh_557(seq):
    

    return mh_mask # 1*557


def Indel_Len_Transform(Y_lookup, Pred_lookup, Oligo, get_transform=False):
    """
    Transform various length ForeCast Prediction result to indel len distribution.
    Deletion : 1-38 , Insertion : 1-3

    Input
    --------
    Y_lookup : dict, oligo -> ndarray of shape (n,2). [[Identifier name, frequency]]. The lookup item storing true labels with identifiers.
    Pred_lookup : dict, oligo -> ndarray of shape (1,n). The lookup item storing predicted values. [1, frequency]. 
    Oligo : str, "OligoXXX"
    get_transform : bool, default False. If True, function will return 2 more transformation matrices

    Return
    --------
    Del_Ins_Dis : ndarray of shape (41,),  0-38 | 1-3
    D_transform : ndarray (n, 38) , only when `get_transform` is set to True
    I_transform : ndarray (n, 3) , only when `get_transform` is set to True
    """
    Indentifiers = Y_lookup[Oligo][:,0]
    y_true = Y_lookup[Oligo][:,1]
    y_pred = Pred_lookup[Oligo][0]

    # construct matrix
    D_transform = np.zeros((len(Indentifiers),38))
    I_transform = np.zeros((len(Indentifiers),3))

    for i, idf in enumerate(Indentifiers):
        indel_type, isize, details, muts = tokFullIndel(idf)

        if indel_type == 'D':
            isize = 38 if isize >= 38 else isize
            D_transform[i, isize-1] = 1
        
        elif indel_type == 'I':
            isize = 3 if isize >= 3 else isize
            I_transform[i, isize-1] = 1

    # compute indel len distribution
    D_len = y_true @ D_transform
    I_len = y_true @ I_transform

    D_len_pred = y_pred @ D_transform
    I_len_pred = y_pred @ I_transform

    if get_transform:
        return np.concatenate([D_len, I_len]), D_transform, I_transform
    else:
        return np.concatenate([D_len, I_len]), np.concatenate([D_len_pred, I_len_pred])   


def subset_by_indeltype(indeltype, Oligo, Y_lookup, Pred_lookup):
    """
    For a given oligo, subset event belonging to the specified indeltype from the prediciton and observation probability vector 
    
    Input
    --------
    indeltype: str, like ['D', 'I', 'D1', 'I1', 'I2']
    Oligo : str
    Y_lookup
    Pred_lookup

    Return
    --------
    Y_true subseted
    Y_pred subseted
    """
    Indentifiers = Y_lookup[Oligo][:,0]
    y_true = Y_lookup[Oligo][:,1]
    y_pred = Pred_lookup[Oligo][0]

    if indeltype in ['D', 'I']:
        event_index = [idf.startswith(indeltype) for idf in Indentifiers]
    else:
        INDELs = np.array([idf.split("_")[0] for idf in Indentifiers])
        event_index = INDELs == indeltype
    
    return y_true[event_index], y_pred[event_index]



def get_fs_transform912(class_912):
    """
    a transform matrix to summarize the ratio of frameshift indel
    Input
    ---------
    class_912: list of str, the label of 912 indel classes, which is adata.var_names
    Return
    --------
    frameshift_912 : ndarray
    """
    ndel = 912 - 21
    del_frameshift = [int(e.split('+')[1]) for e in class_912[:ndel-1]] + [38] # e: event
    ins_frameshift = [int(e.split('+')[0]) for e in class_912[ndel:-1]] + [3]
    all_frameshift= del_frameshift + ins_frameshift
    frameshift_912 = np.array([fs%3!=0 for fs in all_frameshift])
    return frameshift_912

def get_fs_transform557(class_557):
    """
    a transform matrix to summarize the ratio of frameshift indel
    Input
    ---------
    class_557: list of str, the label of 912 indel classes, which is adata.var_names
    Return
    --------
    frameshift_557 : ndarray
    """
    label,rev_index,features,frame_shift  = pkl.load(importlib.resources.open_binary("qrguide.utils", "model_prereq.pkl"))
    return frame_shift

def get_fs_transform894(class_894):
    """
    a transform matrix to masking which indel can cause frameshift (indel size ≠ 3)
    Input
    ---------
    class_894: list of str, saving the order of 894 indel labels. This output format is for U-net model
    Return
    --------
    frame_shift_894 : binary ndarray (894,) 
    """
    frame_shift = np.zeros((894,))
    for i,label in enumerate(class_894):

        if label in ["1+A", "2+AA", ">38"]:
            idl = 4 # set to frameshift
        elif label == '3':
            idl = 4
        else:
            idl = int(label.split('+')[1])

        frame_shift[i] = (idl%3 != 0)
    return frame_shift


def IndelLen_transform894(class_label):
    """
    a transform matrix to summarize the indel length distribution
    in the order of [del1,del2,...,del38, ins1, ins2, ins3] 

    Input
    ---------
    class_label : list of str, class name of 894 label

    Return
    ---------
    Transformation matrix : ndarray (894, 41)
    """
    assert len(class_label) == 894, "class_label is not for 894 format"

    T_IDL894 = np.zeros((894,41))

    for i,label in enumerate(class_label):
        # three insertion class
        if label == '1+A':    
            T_IDL894[i,38] = 1
        elif label == '2+AA':
            T_IDL894[i,39] = 1
        elif label == '3':
            T_IDL894[i,39] = 1

        # deletion
        elif label == ">38":  
            T_IDL894[i,37] = 1
        else:
            del_size = int(label.split('+')[1])
            T_IDL894[i,del_size-1] = 1

    
    return T_IDL894

def label_transform_557to894():
    """
    the transformation that map 557 format output to 894 format output

    Input
    ---------
    class_label : list of str, class name of 912 classes or 557 classes

    Return
    ---------
    Transformation matrix : ndarray (557, 894)
    """
    return Mapper_557_to_894

def get_del_ins_ratio_transform(n_class):
    """
    a transform matrix to summarize the ratio of deletion events and insertion events
    Input
    ---------
    n_class: int: 912 or 557

    Return
    --------
    del_ins_transform : ndarray of shape (n_class ,2)
    """
    n_del_class = n_class - 21
    ratio_transform = np.zeros((n_class,2))
    ratio_transform[:n_del_class, 0] = 1
    ratio_transform[n_del_class:, 1] = 1
    return ratio_transform

def get_del_ins_ratio_transform894():
    """
    Specifically design for 894 format, a transform matrix to summarize the ratio of deletion events and insertion events
    Input
    ---------
    No

    Return
    --------
    del_ins_transform : ndarray of shape (894 ,2)
    """
    ins_labels = ["1+A", "2+AA", "3"]
    ins_label_locs = [780, 779, 819] # precomputed

    ratio_transform = np.zeros((894,2))
    ratio_transform[:,0] = 1    # set all to deletion first

    # find insertin
    ratio_transform[ins_label_locs, 0] = 0
    ratio_transform[ins_label_locs, 1] = 1
    
    return ratio_transform

def label_transform_557to894(class_557, class_894):
    """
    the transformation that map 557 format output to 894 format output

    Input
    ---------
    class_label : list of str, class name of 912 classes or 557 classes

    Return
    ---------
    Transformation matrix : ndarray (557, 894)
    """
    assert len(class_557) == 557
    assert len(class_894) == 894

    loc_lookup894 = {l:i for i,l in enumerate(class_894)}
    overlapped = np.intersect1d(class_557, class_894)

    Mapper_557_to_894 = np.zeros((557,894))
    for i,label in enumerate(class_557):
        if label in overlapped:
            # for label that can be found in 894, 
            loc_i = loc_lookup894[label]
            Mapper_557_to_894[i, loc_i] = 1
            
        elif label.split('+')[1].isalpha():

            # for two other insertion class
            if label.startswith("1+"):
                loc_i = loc_lookup894["1+A"]
            elif label.startswith("2+"):
                loc_i = loc_lookup894['2+AA']
            
            Mapper_557_to_894[i, loc_i] = 1
    
    # assert np.sum(Mapper_557_to_894) == len(overlapped) + 20

    return Mapper_557_to_894

def label_transform_912to894(class_912, class_894):
    """
    the transformation that map 912 format output to 894 format output

    Input
    ---------
    class_label : list of str, class name of 912 classes or 912 classes

    Return
    ---------
    Transformation matrix : ndarray (912, 894)
    """
    assert len(class_912) == 912
    assert len(class_894) == 894

    loc_lookup894 = {l:i for i,l in enumerate(class_894)}
    overlapped = np.intersect1d(class_912, class_894)

    Mapper_912_to_894 = np.zeros((912,894))
    for i,label in enumerate(class_912):
        if label in overlapped:
            # for label that can be found in 894, 
            loc_i = loc_lookup894[label]
            Mapper_912_to_894[i, loc_i] = 1
            
        elif label.split('+')[1].isalpha():

            # for two other insertion class
            if label.startswith("1+"):
                loc_i = loc_lookup894["1+A"]
            elif label.startswith("2+"):
                loc_i = loc_lookup894['2+AA']
            
            Mapper_912_to_894[i, loc_i] = 1
    
    # assert np.sum(Mapper_912_to_894) == len(overlapped) + 20

    return Mapper_912_to_894

# def label_transform_912to894():
#     ndel = 912 - 21
#     Ins_to_IDL = np.zeros((21,3))

#     ins_frameshift = [int(e.split('+')[0]) for e in BOB_adata.var['label'][ndel:-1]] + [3]

#     for i, fs in enumerate(ins_frameshift):
#         Ins_to_IDL[i, fs-1] = 1
#     Transform_912_894 = np.zeros((912, 894))
#     Transform_912_894[:ndel, :ndel] = np.eye(ndel)
#     Transform_912_894[ndel:, ndel:] = Ins_to_IDL
#     return Transform_912_894


###
###

def extract_features_from_map(input_map):
    """
    with the filtered diagonal pairwise alignment matrix, we detect possible mh events and extract their features
    
    Params
    ---------------
    input_map
        np.ndarray, the output of func `construct_diagonal_map` or `diag_conv_filter`, shape [0] sequence before cutsite, shape[1] sequence after cutsite
    
    Returns
    ---------------
    detected_events
        dict, the events are characterized by their deletion start site and deletion length, e.g: `1_7` denotes a deletion event start from 1bp left to the cutsite 
    """
    detected_events = {}

    # TODO: replate 30 with actual cutsite
    max_ws = np.min(input_map.shape)
    for ws in range(2, max_ws):     #window size

        # construct convolution filter
        kernel = np.diag(np.full((ws,), 1))
        conv2d_fn = lambda x: convolve2d(kernel, x , mode='valid').item()

        # go through the input matrix to find MH events
        for i in range(input_map.shape[0]-ws+1):
            for j in range(input_map.shape[1]-ws+1):

                # only if the diagonal can span the kernel
                if (input_map[i,j] == 1) & (input_map[i+ws-1,j+ws-1] == 1):

                    ss = (i + ws) - input_map.shape[0]      # right-mh                             # deletion start site 
                    # ss = i  - input_map.shape[0]            # left-mh                             # deletion start site 
                    mh_length = conv2d_fn(input_map[i:i+ws,j:j+ws])     # compute aligned length, penalized by gap
                    del_length = input_map.shape[0] + j - i                             # deletion length

                    # add events
                    if mh_length >= ws/2:
                        event_name = f"{ss}+{del_length}"
                        
                        # save the mh strength of the same events into a list 
                        # and we will select the longest mh later
                        try:
                            detected_events[event_name].append(mh_length)
                        except KeyError:
                            detected_events[event_name] = [mh_length]
                else:
                    continue


    # finally select the max mh length
    for key, values in detected_events.items():
        detected_events[key] = max(values)
        
    return detected_events

def get_cmatrix(sequence, cut_site, label):
    """
    quick fn combining gen_indel and gen_cmatrix 
    Input
    --------
    sequence : string
    cut_site : int , 
    label : list of string
    
    Return
    --------
    cmatrix : ndarray, (912,912) or (557,557)
    """
    assert len(label) in [557, 912], "invalid label , either 557 or 912 classes"
    if len(label) == 557:
        if len(sequence)!=60 & cut_site !=30:
            sequence = center_seq_at_30bp(sequence, cut_site)
            cut_site = 30
        # assert len(sequence)==60, "the input sequence must be 60bp long"
        # assert cut_site ==30, "only support cutting at 30 bp for 557 class encoding"

    indels = gen_indel(sequence, cut_site)
    cmatrix = gen_cmatrix(indels, label)
    return cmatrix

def gen_cmatrix(indels,label): 
    ''' Combine redundant classes based on microhomology, matrix operation'''
    combine = []
    for s in indels:
        if s[-2] == 'mh':
            tmp = []
            for k in s[-3]:
                try:
                    tmp.append(label['+'.join(list(map(str,k)))])
                except KeyError:
                    pass
            if len(tmp)>1:
                combine.append(tmp)

    temp = np.diag(np.ones((len(label))), 0)
    for key in combine:
        for i in key[1:]:
            temp[i,key[0]] = 1
            temp[i,i]=0    
    return temp


def gen_indel(sequence,cut_site):
    '''This is the function that used to generate all possible unique indels and 
    list the redundant classes which will be combined after'''
    nt = ['A','T','C','G']
    up = sequence[0:cut_site]
    down = sequence[cut_site:]
    dmax = min(len(up),len(down))  # maximum deletion length
    uniqe_seq ={}
    for dstart in range(1,cut_site+3): # deletion start site
        for dlen in range(1,dmax):
            if len(sequence) > dlen+dstart > cut_site-2:
                # 
                seq = sequence[0:dstart]+sequence[dstart+dlen:]
                indel = sequence[0:dstart] + '-'*dlen + sequence[dstart+dlen:]
                array = [indel,sequence,13,'del',dstart-30,dlen,None,None,None]
                # 13 ?
                try: 
                    # if don't raise keyerror, there is a repeative indel
                    uniqe_seq[seq]  
                    if dstart-30 <1:
                        # update the array for repeated indel if dstart <= 30
                        uniqe_seq[seq] = array
                except KeyError: uniqe_seq[seq] = array

    for base in nt:
        # 1bp insertion
        seq = sequence[0:cut_site]+base+sequence[cut_site:]
        indel = sequence[0:cut_site]+'-'+sequence[cut_site:]
        array = [sequence,indel,13,'ins',0,1,base,None,None]
        try: uniqe_seq[seq] = array
        except KeyError: uniqe_seq[seq] = array

        for base2 in nt:
            # 2bp insertion
            seq = sequence[0:cut_site] + base + base2 + sequence[cut_site:]
            indel = sequence[0:cut_site]+'--'+sequence[cut_site:]
            array = [sequence,indel,13,'ins',0,2,base+base2,None,None]
            try: uniqe_seq[seq] = array
            except KeyError:uniqe_seq[seq] = array

    # where's combinatory insertion & deletion  ?

    uniq_align = label_mh(list(uniqe_seq.values()),4)

    for read in uniq_align:
        if read[-2]=='mh':
            merged=[] 
            # merged means these indel will be considered as redundant when generating class matrx
            for i in range(0,read[-1]+1):          # mh length
                merged.append((read[4]-i,read[5])) # start_site - 30 - mh length, dlen
            read[-3] = merged # 0to mh length , dlens

    return uniq_align

def label_mh(sample,mh_len):
    '''
    Function to label microhomology in deletion events
    Arguments:
        sample: list of uniq_indel values, [[indel,sequence,13,'del',dstart-30,dlen,None,None,Nones]]
        mh_len: int, 4
    Returns:
        sample: the same as input `sample` but 
    '''
    for k in range(len(sample)):
        read = sample[k]
        if read[3] == 'del':
            # only for deletion
            idx = read[2] + read[4] +17       # deletion start site
            idx2 = idx + read[5]              # start + len = deletion end point
            x = mh_len if read[5] > mh_len else read[5]  # min(4, del_len)
            for i in range(x,0,-1):
                # They can only detect MH  happened on the left 
                if read[1][idx-i:idx] == read[1][idx2-i:idx2] and i <= read[5]:
                    sample[k][-2] = 'mh'
                    sample[k][-1] = i
                    break  
                    # so the loop iterate with decreasing i, so the it stop at the longest MH
            if sample[k][-2]!='mh':
                sample[k][-1]=0
    return sample


def center_seq_at_30bp(seq, cutsite):
    """
    make sequence to 60bp and pad with `N`
    """
    offset = cutsite - 30
    if offset > 0:
        # pad seq at right
        seq = seq[offset:]
    elif offset < 0:
        # pad seq at left
        seq = "N"*offset + seq
        seq = seq[:60]

    if len(seq) < 60:
        seq = seq+"N"*(60-len(seq))
    else:
        seq = seq[:60]

    return seq