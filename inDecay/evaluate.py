
import os, sys, json,re
import pickle as pkl
from anyio import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from inDecay import PATH, models, my_utils, analysis_fn
from qrguide import transformation
from qrguide.transformation import *
from scripts.STfeatv2_inDecay_finetune import find_ckpt
import warnings
warnings.filterwarnings('ignore')
import shutil
import subprocess
pj=os.path.join

os.chdir(PATH.main_dir)

from inDecay.models import Topk_Event_Overlapping
from tqdm.auto import tqdm
from scipy.stats import pearsonr

from inDecay import analysis_fn
from matplotlib.ticker import PercentFormatter
from io import StringIO
import shutil
from collections import defaultdict
from functools import partial
os.chdir(PATH.main_dir)
def read_pkl(path):
    with open(path, 'rb') as f:
        Y = pkl.load(f)
    f.close()
    return Y

def evalute_fn(Y, Y_pred):
    eval_json = analysis_fn.assessment_recipe_forecast(df_lookup, reduction='mean',top_metric=[1,2,3,5])
    eval_json.update(analysis_fn.assessment_recipe_IDL(df_lookup,reduction='mean',top_metric=[1,2,3,5]))

    eval_df = pd.json_normalize(eval_json)
    return eval_df


def evalute_species_fn(Y_true_path, Y_pred_path, species, smooth):
    """species: {'p':porcine, 's':goat, 'c':cattle}"""
    Y_pred =read_pkl(Y_pred_path)
    Y = read_pkl(Y_true_path)
    if not species is None:
        Y_pred_filter={k: v for k, v in Y_pred.items() if k.startswith(species)}
        Y_filter={k: v for k, v in Y.items() if k.startswith(species)}
        eval_json = analysis_fn.assessment_recipe_forecast(Y_filter, Y_pred_filter, smooth=smooth,reduction='mean',top_metric=[1,2,3,5])
        eval_json.update(analysis_fn.assessment_recipe_IDL_forecast(Y_filter, Y_pred_filter, smooth=smooth,reduction='mean',top_metric=[1,2,3,5]))
    else:
        eval_json = analysis_fn.assessment_recipe_forecast(Y, Y_pred, smooth=smooth,reduction='mean',top_metric=[1,2,3,5])
        eval_json.update(analysis_fn.assessment_recipe_IDL_forecast(Y, Y_pred, smooth=smooth,reduction='mean',top_metric=[1,2,3,5]))

    eval_df = pd.json_normalize(eval_json)
    return eval_df

def evalute_species_pred(Y_true, Y_pred, species, smooth):
    """species: {'p':porcine, 's':goat, 'c':cattle}"""

    if not species is None:
        Y_pred_filter={k: v for k, v in Y_pred.items() if k.startswith(species)}
        Y_true_filter={k: v for k, v in Y_true.items() if k.startswith(species)}
        eval_json = analysis_fn.assessment_recipe_forecast(Y_true_filter, Y_pred_filter, smooth=smooth,reduction='mean',top_metric=[1,2,3,5])
        eval_json.update(analysis_fn.assessment_recipe_IDL_forecast(Y_true_filter, Y_pred_filter, smooth=smooth,reduction='mean',top_metric=[1,2,3,5]))
    else:
        eval_json = analysis_fn.assessment_recipe_forecast(Y_true, Y_pred, smooth=smooth,reduction='mean',top_metric=[1,2,3,5])
        eval_json.update(analysis_fn.assessment_recipe_IDL_forecast(Y_true, Y_pred, smooth=smooth,reduction='mean',top_metric=[1,2,3,5]))

    eval_df = pd.json_normalize(eval_json)
    return eval_df


def evalute_fn_pred(Y_true, Y_pred, smooth):
    eval_json = analysis_fn.assessment_recipe_forecast(Y_true, Y_pred, reduction='mean',top_metric=[1,2,3,5])
    eval_json.update(analysis_fn.assessment_recipe_IDL_forecast(Y_true, Y_pred, reduction='mean',top_metric=[1,2,3,5]))

    eval_df = pd.json_normalize(eval_json)
    return eval_df

def get_ratios(Y_true, Y_pred):

    ratio_json = []
    for oligo, Y in Y_true.items():
        Y = Y.T

        pred = Y_pred[oligo]
        Indel = Y[[0],:]
        y = Y[[1],:].astype("float32")

        # frameshift
        y_fs, pred_fs = analysis_fn.forecast_frameshift(y, pred, Indel)
        y_dr, pred_dr = analysis_fn.forecast_delratio(y, pred, Indel)

        ratio_json.append(
            {'Gene':oligo, "Rep1_frameshift":y_fs, "Pred_frameshift":pred_fs, "Rep_delratio":y_dr, "Pred_delratio":pred_dr}
        )
    
    return pd.json_normalize(ratio_json)

def ratio_error(row, ratio):
    error = row[f"Rep_{ratio}"] - row[f'Pred_{ratio}']
    return np.abs(error).item()

def create(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def find_pkl_and_eval10_addice(temp,  extend_guide, cells, species, lr=0.001, L2=0.3, foldnum=51):
    """
    given the data_archive name, auto find 
    """
    perform = []
        
    ratio_df = []
    for cell in cells:
        archive_folder = f"{PATH.main_dir}/pl_trainer_log/{extend_guide}_{cell}_featv5_c20_ST_DeepDecay_mul_identity_lr{lr}_L2{L2}_T{temp}/{species}/"

        foldnum=foldnum
        create(pj(PATH.main_dir, 'pretrained', f"{species}-{cell}_featv5_c20_{extend_guide}_lr{lr}_L2{L2}_T{temp}"))
        for k_index in range(foldnum):

            def annotate_df(df):
                df['kfold_index'] = k_index
                df['celltype'] = cell
            
            # the directory name

            second_save_path = f"{foldnum}fold_{k_index}"
            # auto find pkl files
            Y_true = pj(archive_folder, second_save_path, "ForeCast_TestY.pkl")
            Y_baseline = pj(archive_folder, second_save_path, "Pretrained_Baseline_TestPred.pkl")
            # print(Y_baseline)
            shutil.copyfile(pj(PATH.main_dir,f'pretrained/{cell}_featv5_c20.ckpt'),pj(PATH.main_dir,'pretrained',f"{species}-{cell}_featv5_c20_{extend_guide}_lr{lr}_L2{L2}_T{temp}", str(foldnum)+'.ckpt'))
            Y_pred_path = my_utils.find_ckpt(pj(archive_folder, second_save_path, "lightning_logs"))
            shutil.copyfile(Y_pred_path,pj(PATH.main_dir,'pretrained',f"{species}-{cell}_featv5_c20_{extend_guide}_lr{lr}_L2{L2}_T{temp}", str(k_index)+'.ckpt'))
            Y_pred_path = Y_pred_path.replace(".ckpt", "TestPred.pkl")
            

            # get all the evaluation metrics            
            df_k = evalute_fn(Y_true, Y_pred_path)
            df_baseline_k = evalute_fn(Y_true, Y_baseline)
            df_baseline_k['celltype'] = cell

            # get frameshift / del ratio for each genes
            ratio_df_k = get_ratios(Y_true, Y_pred_path)
            ratio_df_baseline_k = get_ratios(Y_true, Y_baseline)
            ratio_df_baseline_k['celltype'] = cell

            ratio_df_k['Baseline_frameshift'] = ratio_df_baseline_k['Pred_frameshift']
            ratio_df_k['Baseline_delratio'] = ratio_df_baseline_k['Pred_delratio']

            # annotate
            for df in [df_k,df_baseline_k,ratio_df_k]:
                annotate_df(df)

            df_baseline_k['fix_setting'] = 'baseline'

            perform.append(df_k)
            perform.append(df_baseline_k)
            ratio_df.append(ratio_df_k)
            
    perform  = pd.concat(perform)
    ratio_df = pd.concat(ratio_df)
    

    return perform, ratio_df 

def simple_recut(Y_lookup, pred_lookup, size_allow_recut=1, n_recut=0):
    """
    Simple recutting. Redistribute small indels to other indels by multiplying a transformation matrix

    Args:
    ------
    Y_lookup (dict): {Oligos : [identifier, observed repair probability]}
    pred_lookup (dict):  {Oligos : [predicted probability]}
    size_allow_recut (int) : the indel size allowed for recutting
    n_recut : the times of recutting 

    Output:
    -----
    recut_lookup (dict) : {Oligos : [recut repair probability]}
    """
    recut_lookup = {}
    if n_recut==0:
        return pred_lookup
    else:
    # lookup 
        for oligo, Y in Y_lookup.items():

            oligo_df = pd.DataFrame(Y, columns=['idf','p_obs'])
            oligo_df['p_pred'] = pred_lookup[oligo].flatten()
            oligo_df.loc[:,['type','size']] = np.stack(
                [my_utils.tokFullIndel(idf)[0:2] for idf in oligo_df['idf'].values]
                    )
            N_event = oligo_df.shape[0]

            # recutting
            recut_transform = np.diag( oligo_df[['p_pred']].values )

            # retain
            # recutted indels -> 0, others -> unchanged 
            recut_transform = np.eye(N=N_event)
            for i, row in oligo_df.iterrows():
                if int(row['size']) <= size_allow_recut:
                    # labeld the row location of recutted indels
                    recut_transform[i, :] = oligo_df['p_pred'].values 
                    
            # 
            recut_prob = np.broadcast_to( oligo_df[['p_pred']].values, shape=(N_event,N_event) )
            recut_prob = np.multiply(recut_prob, recut_transform) # element-wise


            # print(recut_prob.shape)
            for n in range(n_recut):
                recut_prob = np.multiply(recut_prob, recut_transform)
                # print(recut_prob.shape)
            
            recut_lookup[oligo] = recut_prob.sum(axis=0,keepdims=True) 

        return recut_lookup#, recut_transform




def check_livestock_frommouse(cell, temp, mean=True, IDL=True, n_recut=0, ckpt_dir=None, return_pkl=False):
    """
    Predict live stock samples with mouse model, prediction is bagged from differrent folds.
    
    Input:
    -------
    cell : pretrained cell type
    temp : temperature of the inDecay model
    mean : bool default True, whether to average the prediction
    IDL : bool default True, whether to evaluate prediction by indel length
    n_recut : int default 0, the iteration number of calling `simple_recut`
    ckpt_dir : str default None, the directory of the fewshot trained model
    return_pkl : bool default False, whether to return the full event probablity dict

    Returns:
    -------
    df_all : dict, {gene-name: df [observed, pretarined, fewshot]}
    """
    ccsv=pd.read_csv(f'data/zygote_fah/gene_seq.csv').set_index('guide')
    Y_true_path=f"pl_trainer_log/zygote_fah-{cell}-T{temp}-k10.5-k20.6-h1.3/0/ForeCast_TestY.pkl"
    # cmd = f"find pl_trainer_log/zygote_fah-{cell}-T{temp}-k10.5-k20.6-h1.3/ -type d -empty -delete"
    # subprocess.run(cmd, shell=True, check=True)
    totalfolds=len(os.listdir(f"pl_trainer_log/zygote_fah-{cell}-T{temp}-k10.5-k20.6-h1.3/"))-1
    
    
    if (ckpt_dir is not None):
        assert os.path.exists(ckpt_dir), "given checkpoint not exists"
        Y_baseline_path = f"{ckpt_dir}/{totalfolds}/Pretrained_Baseline_TestPred.pkl"
    else:
        Y_baseline_path=f"pl_trainer_log/zygote_fah-{cell}-T{temp}-k10.5-k20.6-h1.3/{totalfolds}/Pretrained_Baseline_TestPred.pkl"
        ckpt_dir = f"pl_trainer_log/zygote_fah-{cell}-T{temp}-k10.5-k20.6-h1.3"

    Y_lookup = read_pkl(Y_true_path)
    Y_baseline=read_pkl(Y_baseline_path)
    ratio_json = []
    df_all={}

    for oligo, Y in Y_lookup.items():
        print(f"r2 for sample {oligo} is {ccsv['r2'].loc[oligo]}")
        Y = Y.T
        base= Y_baseline[oligo]
        Indel = Y[[0],:]
        if IDL is True:
            Indel_IDL= [ind.split('_')[0] for ind in Indel[0]]
        else:
            Indel_IDL= Indel[0]
        y = Y[[1],:].astype("float32")
        predlist=[]
    
        if mean is True:
            for f in range(0, totalfolds):
                Y_pred_path=f"{ckpt_dir}/{f}/Pretrained_Baseline_TestPred.pkl"
                Y_pred=read_pkl(Y_pred_path)
                
                Y_pred=simple_recut(Y_lookup, Y_pred, n_recut=n_recut)
                    # print('yes')
                    
                pred_lookup =Y_pred[oligo]
                predlist.append(pred_lookup[0])
            pred=np.mean(predlist,axis=0)
            predstd=np.std(predlist,axis=0)
            predstd_IDL = defaultdict(float)    
            for k, v in zip(Indel_IDL, predstd):
                predstd_IDL[k]+=v
        else:
            Y_pred_path=f"pl_trainer_log/zygote_fah-{cell}-T{temp}-k10.5-k20.6-h1.3/{f}/Pretrained_Baseline_TestPred.pkl"
            pred_lookup =read_pkl(Y_pred_path)
            pred_lookup=simple_recut(Y_lookup, pred_lookup, n_recut=n_recut)
            pred = pred_lookup[oligo][0]
        
        pred_IDL = defaultdict(float)
        base_IDL = defaultdict(float)
        y_IDL = defaultdict(float)
        for k, v in zip(Indel_IDL, pred):
            pred_IDL[k]+=v
        for k, v in zip(Indel_IDL, base[0]):
            base_IDL[k]+=v
        for k, v in zip(Indel_IDL, y[0]):
            y_IDL[k]+=v
        ev=sum(1 for value in y_IDL.values() if value > 0)

        pred_top_idx = np.array(list(pred_IDL.values())).argpartition(-5)[-5:]
        y_top_idx = np.array(list(y_IDL.values())).argpartition(-min(5,ev))[-min(5,ev):]
        b_top_idx = np.array(list(base_IDL.values())).argpartition(-5)[-5:]
        idx=list(set(pred_top_idx).union(set(y_top_idx)).union(set(b_top_idx)))
        kidx=[list(y_IDL.keys())[i] for i in idx]
        data={'Indel': kidx, 'fewshot':[pred_IDL[k] for k in kidx], 'observed':[y_IDL[k] for k in kidx], 'pretrained':[base_IDL[k] for k in kidx]}#'ice': syn[0][idx],
        # print(data)
        df = pd.DataFrame(data=data,columns=['observed','pretrained','fewshot'],index=kidx)
        if mean is True:
            df['fewshot_std']=[predstd_IDL[k] for k in kidx]
        df = df.sort_values('observed', ascending=False)
        df_all.update({oligo:df})
    
    if return_pkl:
        return pred_IDL, base_IDL, y_IDL
    else:
        return df_all