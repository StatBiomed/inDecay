import os, io, re, csv, sys, time
sys.path.append(os.path.abspath("../"))
from . import my_utils, PATH
import torch
import numpy as np
import pandas as pd
from scipy import special
import subprocess
import pickle as pkl
from scipy.signal import convolve2d
from matplotlib import pyplot as plt

ins_wb = None
base_dist_M = None

A,T,G,C = 'A','T','G','C'
AA,AT,AC,AG,CG,CT,CA,CC = 'AA','AT','AC','AG','CG','CT','CA','CC'
GT,GA,GG,GC,TA,TG,TC,TT = 'GT','GA','GG','GC','TA','TG','TC','TT'

def decay(x, k):
    y = 1 / (1+np.exp(x)**k)
    return y

def get_distance_matrix(size=50):

    global base_dist_M
    base_dist_M = np.sqrt(np.broadcast_to(np.arange(0,size), shape=(size,size)).T**2 + \
                np.broadcast_to(np.arange(size-1,-1,-1), shape=(size,size))**2)
    
    base_dist_M = base_dist_M/base_dist_M.max()


def one_hot(seq,complementary=False):
    """
    one_hot encoding on sequence
    complementary: encode nucleatide into complementary one
    """
    # setting
    seq = list(seq.replace("U","T"))
    seq_len = len(seq)
    complementary = -1 if complementary else 1
    # compose dict
    keys = ['A', 'C', 'G', 'T'][::complementary]
    oh_dict = {keys[i]:i for i in range(4)}
    # array
    oh_array = np.zeros((seq_len,4))
    for i,C in enumerate(seq):
        try:
            oh_array[i,oh_dict[C]]=1
        except:
            continue      # for nucleotide that are not in A C G T   
    return oh_array 

def get_ins_weight_bias():
    """
    get prerequisite ins ratio
    """
    weights = my_utils.wb
    w1,b1,w2,b2,w3,b3 = weights
    global ins_wb
    ins_wb = w1, b1

def del_ins_intercept(guide):
    """
    The linear ratio model
    """
    
    if ins_wb is None:
        get_ins_weight_bias()

    w1, b1 = ins_wb
    guide_oh = onehotencoder(guide)
    ds_bias, ins_bias = special.softmax(np.dot(guide_oh, w1)+ b1)
    return ds_bias, ins_bias

# some function 

def list_eval(loc_string):
    return eval(re.sub(r"([A-Z]{1,2})", r"'\1'", loc_string))

def get_distal(label_df, cutsite):
    
    ends = [[s_e[1] for s_e in list_eval(ls)] for ls in label_df['loc'].values]
    distal_mask = [cutsite in e for e in ends]
    
    return distal_mask

def get_proximal(label_df, cutsite):
    
    last_site = cutsite -1
    ends = [[s_e[0] for s_e in list_eval(ls)] for ls in label_df['loc'].values]
    proximal_mask = [last_site in e for e in ends]
    
    return proximal_mask

def compute_gc_ratio(guide):
    C = guide.count("C")
    G = guide.count("G")
    return (C+G) / len(guide)

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

def pair_align_map(seq, cut_site=39, plotout=False):
    """
    we split the sequence at CRISPR cut site and construct the substitution matrix
    set `plotout=True` to visualize the micro-homology alignment
    """    
    
    left = seq[:cut_site]
    right = seq[cut_site:]
    onehot_left = one_hot(left)
    onehot_right = one_hot(right)  # A C G T
    
    align_map = np.zeros((cut_site, len(right)))
    
    for i in range(4):
        outprod = onehot_left[:,i].reshape(-1,1) @ onehot_right[:,i].reshape(1,-1)
        align_map += outprod 
        
    if plotout:
        plt.matshow(align_map, cmap='Blues')
        plt.yticks(range(cut_site), list(left), fontsize=5)
        plt.xticks(range(len(right)), list(right), fontsize=5)
    
    return align_map

def diag_conv_filter(matrix, score=1, panelty=-1):
    """
    for a pair-wise alignment matrix, detect the diagonal line with kernel convolution
    """
    
    conv_f = np.diag((score,score))
    out = np.full_like(matrix, panelty)
    
    for i in range(matrix.shape[0]-1):
        for j in range(matrix.shape[1]-1):
        # move from left to right first
        # then change line
            
            if np.multiply(matrix[i:i+2,j:j+2], conv_f).sum() == 2:
                out[i,j] = 1
                out[i+1,j+1] = 1
                
    return out

def construct_diagonal_map(seq, cut_site=39, score=1, panelty=-1, matrix_size=None):
    """
    a function to simply convert input sequences to filtered alignment map
    
    Params:
    ------------------
    seq:
        str, the input sequence, 
    cut_site:
        int: default 30, the cut site of the sequence and it will determine the second dimension of output
    plotout:
        bool, whether to plot out the matrix for visualization
    
    Return:
    ------------------
    out:
        np.ndarray, filtered matrix only contains diagonal elements
    """
    alignmap = pair_align_map(seq, cut_site)
    filtered_map = diag_conv_filter(alignmap, score, panelty)

    if matrix_size is None: 
        # no padding is required
        left = -1 * matrix_size
        filtered_map = torch.from_numpy(filtered_map).float()[left:, :matrix_size]

        dim1, dim2 = tuple(filtered_map.shape)
        right_pad = max(matrix_size - dim2, 0)
        upper_pad = max(matrix_size - dim1, 0)
        padding = (
            0, right_pad,  # pad at right for dim -1
            upper_pad,0,   # pad at left for dim -2
        )
        filtered_map = torch.nn.functional.pad(filtered_map, padding).numpy()
    
    return filtered_map


# def various_kernel_conv(filtered_map, kernel_size_ls=[3,5,7,9,11]):
    
#     for ws in kernel_size_ls:
#         kernel = np.diag(np.full((ws,), 1))
#         conv2d_fn = lambda x: convolve2d(kernel, x , mode='valid').item()
# s

def label_mh(refseq, cutsite, label_df, mml_name='mh_length',  score=1, panelty=-1):
    # construct
    filtered_map = construct_diagonal_map(refseq, cutsite, score=score, panelty=panelty)
    detected_events = extract_features_from_map(filtered_map)

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

    label_df[mml_name] = mml_v
    return is_mh, label_df

def compute_decay_sum_P50(seq, cut_site=39):
    """
    for each reference sequence  output 4 features
    """
    filtered_map = construct_diagonal_map(seq, cut_site=cut_site, score=1, panelty=-1, matrix_size=50)
    filtered_map_1 = construct_diagonal_map(seq, cut_site=cut_site, score=1, panelty=0, matrix_size=50)

    mh_intensity_sum = []
    decay_term = [
        lambda x : np.power(3, x),
        lambda x : np.power(2, x),
    ]
    for decay_fn in decay_term:
        dist_M = decay_fn(base_dist_M)

        for filter_M in [filtered_map, filtered_map_1]:
            summ  = np.multiply(filter_M, dist_M).sum()
            mh_intensity_sum.append(summ)        
    
    return mh_intensity_sum

def ST_decayfeat_v1(label_df, refseq, cutsite, k1=0.5, k2=0.6, h=1.3):
    """
    Construct 15 features for each indel gen dataframe
    DEL : dl, ss, ss-decay, mml, proximal(left), aproximal(right), dl-decay, del_intcpt, n_events
    INS : insl, C, shift , full_complement ins, n_coevents
    Input
    ------------
    label_df : df by forecast indelgentarget , must contain columns [mh_length, identifier, loc, n_coevent]
    refseq : taraget sequence
    cutsite : pamsite -3
    k1 : ss decay param
    k2 : dl decay param
    h : MH strength scaler


    Return
    ------------
    x : np.ndarray, [df.shape, 15]
    """
    MML_1 = label_df['mh_length'].values
    MML_2 = label_df['mh_length2'].values
    Idfs = label_df['Identifier'].values
    locs = label_df['loc'].values
    coevents = label_df['n_coevent'].values
    X2 = np.zeros((len(Idfs), 18))

    distal_mask = get_distal(label_df, cutsite)
    proximal_mask = get_proximal(label_df, cutsite)

    # prior knowledge  
    guide = refseq[cutsite-17:cutsite+3]
    guide_gc = compute_gc_ratio(guide)
    del_intcpt, ins_intcpt = del_ins_intercept(guide)

    # mh intensity
    mh_pixelsum_list = compute_decay_sum_P50(refseq, cutsite)

    for i,idf in enumerate(Idfs):
        indel_type, indel_size,  details, muts  = my_utils.tokFullIndel(idf)
        ss = details['L'] + details['C']

        # for i, locs in enumerate(label_df['loc'].values):
        loc_ls = locs[i]
        ss = np.max([ss_end[0] for ss_end in list_eval(loc_ls)]) - cutsite

        if indel_type == 'D':
            X2[i, 0] = indel_size
            X2[i, 1] = ss
            X2[i, 2] = decay(ss, k1)
            X2[i, 3] = MML_1[i]**h                   # max mm length
            X2[i, 4] = MML_2[i]**h                   # max mm length
            X2[i, 5] = proximal_mask[i]            # proximal del, this is different from v1 !!
            X2[i, 6] = distal_mask[i]              # distal del, this is different from v1 !!
            X2[i, 7] = decay(indel_size, k2)
            X2[i, 8] = del_intcpt
            X2[i, 9] = coevents[i]

        elif indel_type == 'I':

            # one identifier may contain different inserted 
            inserts = [ss_end[-1] for ss_end in list_eval(loc_ls)]
            right_nt = refseq[cutsite : cutsite+indel_size]
            left_nt =  refseq[cutsite-indel_size:cutsite]

            X2[i, 10] = indel_size
            X2[i, 11] = details['C']
            X2[i, 12] = (ss + indel_size) == 0 
            X2[i, 13] = indel_size == details['C']
            X2[i, 14] = ins_intcpt
            X2[i, 15] = coevents[i]
            X2[i, 16] = 1/len(inserts)*int(right_nt in inserts)
            X2[i, 17] = 1/len(inserts)*int(left_nt in inserts)

        X2[i, 18] = guide_gc
        X2[i, 19] = mh_pixelsum_list[0]
        X2[i, 20] = mh_pixelsum_list[1]
        X2[i, 21] = mh_pixelsum_list[2]
        X2[i, 22] = mh_pixelsum_list[3]

    return X2

def ST_decayfeat_v2(label_df, refseq, cutsite, k1=0.5, k2=0.6, h=1.3):
    """
    Construct 18 features for each indel gen dataframe
    DEL : dl, ss, ss-decay, mml, proximal(left), aproximal(right), dl-decay, del_intcpt, n_events
    INS : insl, C, shift , full_complement ins, n_coevents
    Input
    ------------
    label_df : df by forecast indelgentarget , must contain columns [mh_length, identifier, loc, n_coevent]
    refseq : taraget sequence
    cutsite : pamsite -3
    k1 : ss decay param
    k2 : dl decay param
    h : MH strength scaler


    Return
    ------------
    x : np.ndarray, [df.shape, 18]
    """
    MML = label_df['mh_length'].values
    Idfs = label_df['Identifier'].values
    locs = label_df['loc'].values
    coevents = label_df['n_coevent'].values
    X2 = np.zeros((len(Idfs), 18))

    distal_mask = get_distal(label_df, cutsite)
    proximal_mask = get_proximal(label_df, cutsite)

    # prior knowledge  
    guide = refseq[cutsite-17:cutsite+3]
    guide_gc = compute_gc_ratio(guide)
    del_intcpt, ins_intcpt = del_ins_intercept(guide)

    for i,idf in enumerate(Idfs):
        indel_type, indel_size,  details, muts  = my_utils.tokFullIndel(idf)
        ss = details['L'] + details['C']

        # for i, locs in enumerate(label_df['loc'].values):
        loc_ls = locs[i]
        ss = np.max([ss_end[0] for ss_end in list_eval(loc_ls)]) - cutsite

        if indel_type == 'D':
            X2[i, 0] = indel_size
            X2[i, 1] = ss
            X2[i, 2] = decay(ss, k1)
            X2[i, 3] = MML[i]**h                   # max mm length
            X2[i, 4] = proximal_mask[i]            # proximal del, this is different from v1 !!
            X2[i, 5] = distal_mask[i]              # distal del, this is different from v1 !!
            X2[i, 6] = decay(indel_size, k2)
            X2[i, 7] = del_intcpt
            X2[i, 8] = coevents[i]

        elif indel_type == 'I':

            # one identifier may contain different inserted 
            inserts = [ss_end[-1] for ss_end in list_eval(loc_ls)]
            right_nt = refseq[cutsite : cutsite+indel_size]
            left_nt =  refseq[cutsite-indel_size:cutsite]

            X2[i, 9] = indel_size
            X2[i, 10] = details['C']
            X2[i, 11] = (ss + indel_size) == 0 
            X2[i, 12] = indel_size == details['C']
            X2[i, 13] = ins_intcpt
            X2[i, 14] = coevents[i]
            X2[i, 15] = 1/len(inserts)*int(right_nt in inserts)
            X2[i, 16] = 1/len(inserts)*int(left_nt in inserts)

        X2[i, 17] = guide_gc

    return X2

def K_mer(seq,K):
    """
    one_hot encoding on sequence
    complementary: encode nucleatide into complementary one
    """
    # setting
    seq = list(seq.replace("U","T"))
    seq_len = len(seq) - K

    # lookup position
    oh_dict = {}
    posi = 0
    for nt1 in ['A', 'C', 'G', 'T']:
        for nt2 in ['A', 'C', 'G', 'T']:
            posi += 1
            oh_dict[nt1+nt2] = posi
    # array
    Kmer_array = np.zeros((seq_len,4**K))
    for i in range(seq_len):
        try:
            Kmer_array[i, oh_dict[seq[i:i+K]]]=1
        except:
            continue      # for nucleotide that are not in A C G T   
    return Kmer_array 

def onehotencoder(seq):
    nt= ['A','T','C','G']
    head = []
    l = len(seq)
    for k in range(l):
        for i in range(4):
            head.append(nt[i]+str(k))

    for k in range(l-1):
        for i in range(4):
            for j in range(4):
                head.append(nt[i]+nt[j]+str(k))
    head_idx = {}
    for idx,key in enumerate(head):
        head_idx[key] = idx
    encode = np.zeros(len(head_idx))
    for j in range(l):
        encode[head_idx[seq[j]+str(j)]] =1.
    for k in range(l-1):
        encode[head_idx[seq[k:k+2]+str(k)]] =1.
    return encode