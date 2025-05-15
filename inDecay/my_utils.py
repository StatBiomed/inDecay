import os, io, csv, sys, time, re, shutil
from . import PATH
import numpy as np
import pandas as pd
from scipy import special
import subprocess
import pickle as pkl
from qrguide import transformation, analysis_fn
import matplotlib.pyplot as plt

global data_dir
global combine_train
global combine_test
global Lindel_only


##
## Some processed file
##
data_dir = PATH.high_dir


wb = transformation.wb
label,rev_index,features,frame_shift = transformation.label, transformation.rev_index, transformation.features, transformation.frame_shift
# anything about Y
Lindel_prereq = os.path.join(PATH.main_dir,"Extended_912_class_Feb06.pkl")
All_Lindel_class, class_to_loc_lookup = pkl.load(open(Lindel_prereq,'rb'))

def read_pkl(path):
    with open(path, 'rb') as f:
        Y = pkl.load(f)
    f.close()
    return Y

def find_ckpt(ckpt_version_dir):
    """
    find the latest version, if not finished then return last one
    """
    get_v = lambda s: int(s.replace("version_",""))
    
    versions = [get_v(subdir) for subdir in os.listdir(ckpt_version_dir)]

    for v in versions:
        checkpoint_dir = os.path.join(ckpt_version_dir, 'version_%d'%v, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            try:
                shutil.rmtree(os.path.join(ckpt_version_dir, 'version_%d'%v)) 
            except:
                continue


    versions = [get_v(subdir) for subdir in os.listdir(ckpt_version_dir) if subdir.startswith('version')]
    maxv  = np.max(versions)

    # try:
    #     ckpts = os.listdir()
    # except FileNotFoundError:
    #     maxv -= 1
    ckpts = list(filter(lambda x : x.endswith('.ckpt'), 
                            os.listdir(os.path.join(ckpt_version_dir, 'version_%d'%maxv, 'checkpoints')))
                )
    
    while len(ckpts) == 0:
        maxv -= 1
        ckpts = list(filter(lambda x : x.endswith('.ckpt'), 
                            os.listdir(os.path.join(ckpt_version_dir, 'version_%d'%maxv, 'checkpoints')))
                            )
        if maxv == -1:
            raise FileNotFoundError("no checkpoints found for any versions")
    
    if len(ckpts) >1:
        ckpt = ckpts[-1]
    else:
        ckpt = ckpts[0]
    
    return os.path.join(ckpt_version_dir, 'version_%d'%maxv, 'checkpoints', ckpt)

def loop_allckpt(ckpt_version_dir):
    """
    find the latest version, if not finished then return last one
    """
    get_v = lambda s: int(s.replace("version_",""))
    
    versions = [get_v(subdir) for subdir in os.listdir(ckpt_version_dir)]

    for v in versions:
        checkpoint_dir = os.path.join(ckpt_version_dir, 'version_%d'%v, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            try:
                shutil.rmtree(os.path.join(ckpt_version_dir, 'version_%d'%v)) 
            except:
                continue


    versions = [get_v(subdir) for subdir in os.listdir(ckpt_version_dir) if subdir.startswith('version')]
    maxv  = np.max(versions)

    # try:
    #     ckpts = os.listdir()
    # except FileNotFoundError:
    #     maxv -= 1
    ckpts = list(filter(lambda x : x.endswith('.ckpt'), 
                            os.listdir(os.path.join(ckpt_version_dir, 'version_%d'%maxv, 'checkpoints')))
                )
    
    while len(ckpts) == 0:
        maxv -= 1
        ckpts = list(filter(lambda x : x.endswith('.ckpt'), 
                            os.listdir(os.path.join(ckpt_version_dir, 'version_%d'%maxv, 'checkpoints')))
                            )
        if maxv == -1:
            raise FileNotFoundError("no checkpoints found for any versions")
    
    if len(ckpts) >1:
        ckpt = ckpts
    
    return [os.path.join(ckpt_version_dir, 'version_%d'%maxv, 'checkpoints', ckpt[i]) for i in range(len(ckpt))] 
######################################################################################
#   ____         _   ____                                   _____                    
#  |  _ \   ___ | | |  _ \   ___   ___  __ _  _   _        |  ___|_   _  _ __    ___ 
#  | | | | / _ \| | | | | | / _ \ / __|/ _` || | | |       | |_  | | | || '_ \  / __|
#  | |_| ||  __/| | | |_| ||  __/| (__| (_| || |_| |       |  _| | |_| || | | || (__ 
#  |____/  \___||_| |____/  \___| \___|\__,_| \__, |       |_|    \__,_||_| |_| \___|
#                                             |___/                                  
######################################################################################


def tokFullIndel(indel):
    """
    This function is taken from SelfTarget 
    https://github.com/felicityallen/SelfTarget/selftarget_pyutils/selftarget/indel.py
    """
    indel_toks = indel.split('_')
    indel_type, indel_details = indel_toks[0], ''
    if len(indel_toks) > 1:
        indel_details =  indel_toks[1]
    cigar_toks = re.findall(r'([CLRDI]+)(-?\d+)', indel_details)
    details, muts = {'I':0,'D':0,'C':0}, []
    for (letter,val) in cigar_toks:
        details[letter] = eval(val)
    if len(indel_toks) > 2 or (indel_type == '-' and len(indel_toks) > 1):
        mut_toks = re.findall(r'([MNDSI]+)(-?\d+)(\[[ATGC]+\])?', indel_toks[-1])
        for (letter,val,nucl) in mut_toks:
            if nucl == '':
                nucl = '[]'
            muts.append((letter, eval(val), nucl[1:-1]))
    if indel_type[0] == '-':
        isize = 0
    else:
        isize = eval(indel_type[1:])
    return indel_type[0],isize,details, muts

def get_indelgen_file(OligoID, Guide):
    indelgen_dir = os.path.join(PATH.data_dir, "Indelgen_result")
    genindel = os.path.join(indelgen_dir, f"{OligoID}_{Guide}_genindels.txt")
    return genindel

def get_feature_file(OligoID, Guide):
    indelfeature_dir = os.path.join(PATH.data_dir, "IndelFeature_result")
    gen_feature_file = os.path.join(indelfeature_dir, f"{OligoID}_{Guide}_features.txt")
    return gen_feature_file

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

def ForeCast_gen_indels(target_seq, pam_idx):
    INDELGENTARGET_EXE = os.getenv("INDELGENTARGET_EXE", f"{PATH.toolkit_dir}/indelgentarget")
    
    # 
    random_idx = np.random.randint(0, 1000000)
    tmp_genindels_file = target_seq[:5] + str(random_idx) + time.strftime("%B%d") + ".tempfile"
    cmd = INDELGENTARGET_EXE + ' %s %d %s' % (target_seq, pam_idx, tmp_genindels_file)
    print(cmd); subprocess.check_call(cmd.split())
    
    # read the generated file
    df = pd.read_table(tmp_genindels_file, skiprows=1, names=['Identifier', 'Collapsed', 'Details','Outcome_seq'])

    # remove tmp file
    os.remove(tmp_genindels_file)

    return df.astype({'Collapsed':int})

def load_dlen_matrix():
    return np.load("/home/wergillius/Project/CRISPR_data/class_2_deletion_Len_matrix.npy").astype("float64")

def load_dss_matrix():
    return np.load("/home/wergillius/Project/CRISPR_data/class_2_deletion_site_matrix.npy").astype("float64")

def load_PC_object(n_PC):
    """
    n_PC : 'PC50' or 'PC200', 
    return : pca, `sklearn.decomposition._pca.PCA`
    """
    with open(f"class_2_{n_PC}.PCA",'rb') as handler:
        pca = pkl.load(handler)
        handler.close()
    return pca

def PCA_reconstruct(y, PCA_instance):
    z = PCA_instance.transform(y)
    y_hat =  PCA_instance.inverse_transform(z)
    return y_hat

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

def get_transform():
    all_912_class = All_Lindel_class

    del_frameshift = [int(e.split('+')[1])*-1 for e in all_912_class[:890]] + [-38]
    ins_frameshift = [int(e.split('+')[0]) for e in all_912_class[891:-1]] + [3]
    indel_len_transform = np.array(del_frameshift + ins_frameshift)

    
    # 912 -> 1
    frameshift_transform = np.array([fs%3!=0 for fs in all_frameshift])
    
    
    # 912 -> 42 
    idl_tranform_M = np.zeros((912,42))
    i = 0

    for idl in range(-38, 4):
        loc = np.where(indel_len_transform == idl)[0]
        idl_tranform_M[loc, i] = 1
        i += 1
    
    return fs_transform, idl_tranform_M

#######################################################
#              ╦  ┬┌┐┌┌┬┐┌─┐┬      ┌┬┐┌─┐┌┬┐┌─┐
#              ║  ││││ ││├┤ │       ││├─┤ │ ├─┤
#              ╩═╝┴┘└┘─┴┘└─┘┴─┘    ─┴┘┴ ┴ ┴ ┴ ┴
#######################################################
    


def load_Lindel_data(fname, feature_size=3033):
    """
    load the processed np.txt by Lindel
    return X,y
    """
    feature_size = 3033

    data = np.loadtxt(fname, delimiter="\t", dtype=str)
    Seqs = data[:,0]
    data = data[:,1:].astype('float32')

    # Sum up deletions and insertions to 
    X = data[:,:feature_size]  # the extracted features are not used
    y = data[:, feature_size:]
    return Seqs, X, y

def load_Lindel_seq_y(fname):
    """
    load the processed np.txt by Lindel
    return X,y
    """
    feature_size = 0

    data = np.loadtxt(fname, delimiter="\t", dtype=str)
    Seqs = data[:,0]
    y = data[:,1:].astype('float32')

    return Seqs, y


def load_Lindel_TrainValset_Deletion(data="LF"):
    """
    Load the Lindel train and validation set, either Lidel only or combining Lindel and FOREcasT
    Arguments:
        data: str, "LF" or "LO"
    Returns:
        Seqs:
        data:
        train_size:
        valid_size:
        Seq_train:
    """
    # determind data path
    LF_path = "/home/wergillius/Project/Lindel/Lindel_data_analysis/data/Lindel_ForeCasT_combined_training.txt"
    LO_path = "/home/wergillius/Project/Lindel/Lindel_data_analysis/data/Lindel_training.txt"
    fname = LO_path if data == "LO" else LF_path

    feature_size = 3033

    Seqs, X, y = load_Lindel_data(fname)
    
    np.random.seed(121)
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    train_size = round(len(y) * 0.9) if 'ForeCasT' in fname else 3900
    valid_size = round(len(y) * 0.1) if 'ForeCasT' in fname else 450 

    Seq_train = Seqs[idx]
    x_train,x_valid = [],[]
    y_train,y_valid = [],[]

    for i in range(train_size):
        if 1> sum(y[i,:536])> 0 :
            norm_class = y[i,:536]/sum(y[i,:536])
            y_train.append(norm_class)
            x_train.append(X[i])
    for i in range(train_size,len(Seq_train)):
        if 1> sum(y[i,:536])>0 : 
            norm_class = y[i,:536]/sum(y[i,:536])
            y_valid.append(norm_class)
            x_valid.append(X[i])
    x_train,x_valid = np.array(x_train),np.array(x_valid)
    y_train,y_valid = np.array(y_train),np.array(y_valid)

    return Seq_train, x_train,x_valid, y_train,y_valid

# filter out gRNAs that did not generate deletion

def filter_deletion_free_events(loaded_data, return_filterd=False):
    """
    Arguments:
        loaded_data : the outcome of func : `my_utils.load_Lindel_data`
        return_filterd : bool, whether we return deletion free samples in another list
    Returns:
        filtered_data : [seq, X, y]
    """
    n_deletion_events = 536

    Seqs, X, y = loaded_data

    y_delete = y[:,:536]
    print("number of raw samples :", len(y))
    print("number of deletion free samples :", np.sum(y_delete.sum(axis=1)==0))

    # select clean result
    withDe_index = np.where(y_delete.sum(axis=1)!=0)[0]
    out = [Seqs[withDe_index], X[withDe_index], y[withDe_index]]

    # if we want to retrival what's filterd out
    if return_filterd:
        Deletion_free_index = np.where(y_delete.sum(axis=1)==0)[0]
        out.append([Seqs[Deletion_free_index], X[Deletion_free_index], y[Deletion_free_index]])

    return out

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

#convert
def convert_SelfTarget_identifier_2_Lindel_class(identifier,Outcome_seq, WT_seq=None, cutsite=None):
    """
    mannully map the outcome format of SelfTarget to lindel outcome class
    Arugments:
        identifier:
        Outcome_seq:
    """
    indel_type, indel_size,  details, muts = tokFullIndel(identifier)
    # deletion , dss and dlen
    if cutsite is None:
        cutsite = 30
    if WT_seq is None:
        WT_seq = ''

    if indel_type == 'D':
        dss = details['L'] + details['C'] + details['I'] + 1
        dlen = int(identifier.split("_")[0].replace("D",""))
        Lindel_class_key = f"{dss}+{dlen}"
        if Lindel_class_key in label.keys():
            return Lindel_class_key
        elif (dlen > 38) & (len(WT_seq)-indel_size==len(Outcome_seq)):
            return ">38"
        else:
            return Lindel_class_key+", Null Indel for Lindel"
    # insertion
    elif indel_type == 'I':
        # PAM Index -3 + insertion (negative value)
        Iss = cutsite + details['L'] + details['C'] 
        Inserted_nucleotids = Outcome_seq[Iss : Iss+indel_size]
        if indel_size < 3:
            Lindel_class_key = f"{indel_size}+{Inserted_nucleotids}"
        else:
            Lindel_class_key = '3'
        return Lindel_class_key

def convert_all_outcome_for_OligoID(OligoID, processed_df, WT_seq, cutsite, verbose=False):

    df = processed_df.query("`OligoID` == @OligoID")
    converted_keys = []
    for idtfy,outcome_seq in df[['Identifier','Outcome_seq']].values:
        converted_keys.append(convert_SelfTarget_identifier_2_Lindel_class(idtfy,outcome_seq, WT_seq=WT_seq, cutsite=cutsite))

    df['labeled'] = converted_keys

    unconsidered = lambda x: ("Null Indel" in x) or (">30" in x)

    unexpected_indels = df[df['labeled'].apply(unconsidered)]
    p_unconsider = unexpected_indels['Identifier'].nunique() / df['Identifier'].nunique()
    p_count = unexpected_indels['Count'].sum() /  df['Count'].sum()

    if verbose:
        print("{:3f} % of indel not considered, taking up {:3f} % of the reads".format(p_unconsider*100, p_count*100))
    return df, (p_unconsider,p_count)

    

##############################################
#         ╦  ┬┌┐┌┌┬┐┌─┐┬      ┌─┐┬ ┬┌┐┌┌─┐
#         ║  ││││ ││├┤ │      ├┤ │ │││││  
#         ╩═╝┴┘└┘─┴┘└─┘┴─┘    └  └─┘┘└┘└─┘
##############################################

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

def readTheta(theta_file):
    """
    a function copied from 
    `/home/wergillius/Project/SelfTarget/indel_prediction/predictor/model.py`
    """
    f = io.open(theta_file)
    train_set = f.readline()[:-1].split(',')
    feature_columns, theta = [], []
    for toks in csv.reader(f, delimiter='\t'):
        feature_columns.append(toks[0])
        theta.append(eval(toks[1]))
    return theta, train_set, feature_columns


