import os, io, csv, sys, time
sys.path.append(os.path.abspath("../"))
from . import my_utils, PATH
import numpy as np
import pandas as pd
import pickle as pkl
import torch
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pytorch_lightning as pl
from inDecay import alignmap
from Bio import SeqIO

pj = os.path.join
global reference_path
reference_path = pj(PATH.data_dir, "SelfTarget_NewScaffold.fasta")

A,T,G,C = 'A','T','G','C'
AA,AT,AC,AG,CG,CT,CA,CC = 'AA','AT','AC','AG','CG','CT','CA','CC'
GT,GA,GG,GC,TA,TG,TC,TT = 'GT','GA','GG','GC','TA','TG','TC','TT'

def get_reference():
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

class ST_dataset(Dataset):
    def __init__(self, Oligos, processed_df, experiments, read_data_fn, feat_ext_fn, transformation=lambda x: x, padding=False, normalize=True):
        """
        reading in the data for each indels, extract the features for each indels
        
        Arguments
        --------------
        Oligos: list of OligoIDs
        processed_df: pd.DataFrame object
        experiments: string
        read_data_fn: callable, how to get the indel details of a oligo
        feat_ext_fn: callable, how to get features from the read_data output
        transformation: callable, transformation function
        padding: bool, default False.
        
        Return
        --------------
        """
        self.Oligos = Oligos
        self.experiments = experiments
        self.processed_df = processed_df # this is also the ref_dict for sanger data
        self.padding = padding
        self.Identifiers = {}
        self.transformation = transformation
        self.ins_wb = alignmap.get_ins_weight_bias()
        self.feat_ext_fn = feat_ext_fn
        self.read_data_fn = read_data_fn
        self.label_col = 'Frac Sample Reads' if normalize else 'Count'

        if not experiments.startswith("ST"):
            self.ref_lookup = processed_df
        else:
            self.ref_lookup = get_reference()

    def __len__(self):
        return len(self.Oligos)
    
    
    def __getitem__(self,i):
        oligo = self.Oligos[i]
        Guide, refseq, pamsite, Strand = self.ref_lookup[oligo]
        cutsite = int(pamsite) - 3 

        assert Strand == 'FORWARD'

        label_df = self.read_data_fn(oligo, self.processed_df, self.experiments)
        # x = label_df[feature_columns].values

        mh_mask, label_df = alignmap.label_mh(refseq, cutsite, label_df, mml_name='mh_length')
        mh_mask, label_df = alignmap.label_mh(refseq, cutsite, label_df, mml_name='mh_length2', panelty=0)

        x5 = self.feat_ext_fn(label_df, refseq, cutsite)
        x5 = self.transformation(x5)

        y = label_df[self.label_col].values

        # self.Identifiers[self.Oligos[i]] = label_df['Identifier']

        return torch.from_numpy(x5).float(),torch.from_numpy(y).float()

class ST_datasetv5(ST_dataset):
    def __init__(self, *args, feature_slice, **kwargs):
        super().__init__(*args, **kwargs)

        self.ft_slice  = feature_slice # a slice object to subset features
    
    def __getitem__(self,i):
        oligo = self.Oligos[i]
        Guide, refseq, pamsite, Strand = self.ref_lookup[oligo]
        cutsite = int(pamsite) - 3 

        assert Strand == 'FORWARD'

        label_df, feat_df = self.read_data_fn(oligo, self.processed_df, self.experiments)
        # x = label_df[feature_columns].values

        mh_mask, label_df = alignmap.label_mh(refseq, cutsite, label_df, mml_name='mh_length')
        mh_mask, label_df = alignmap.label_mh(refseq, cutsite, label_df, mml_name='mh_length2', panelty=0)

        x4 = self.feat_ext_fn(label_df, refseq, cutsite)
        x4 = self.transformation(x4)

        ForeCast_feat = feat_df.values[:,self.ft_slice]
        assert x4.shape[0] == ForeCast_feat.shape[0]
        x5 = np.concatenate([x4, ForeCast_feat], axis=1).astype(np.float32)

        y = label_df[self.label_col].values

        return torch.from_numpy(x5).float(),torch.from_numpy(y).float()


class Sanger_dataset(ST_dataset):
    """
    Sanger sequencing dat
    """
    def __init__(self, *args, weight_lookup, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_lookup = weight_lookup

    def __getitem__(self,i):
        x,y = super().__getitem__(i)
        oligo = self.Oligos[i]
        weight = self.weight_lookup[oligo] # sample weight
        return x, y, torch.tensor(weight)

class Sanger_dataset_count10_r2(ST_dataset):
    """
    Sanger sequencing dat
    """
    def __init__(self, *args, r2_lookup, count_lookup, **kwargs):
        super().__init__(*args, **kwargs)
        self.r2_lookup = r2_lookup
        self.count_lookup = count_lookup

    def __getitem__(self,i):
        x,y = super().__getitem__(i)
        oligo = self.Oligos[i]
        wr2= self.r2_lookup[oligo] # sample weight
        wcount = self.count_lookup[oligo]
        return x, y, torch.tensor(wr2), torch.tensor(wcount)
class inDecay_DataModule(pl.LightningDataModule):
    def __init__(self, experiment, indel_type, DS_class, test_oligo_file, batch_size=32, n_worker=8, seed=0,  threshold=1000):
        """
        The pl DataModule, a flexible wrapper for inDecay dataset. It support all training tasks including deletion, insertion, ratio.
        The structure of inDelcay datareader is like : DataModule takes in experiment and then decide oligos. Dataset takes in the oligos
        If we want to apply transform for nhej / mmej, one can first pass the transform func as the args of Dataset using partial function
        Params
        --------------
        experiment:
            str , the experiment that should be one of the following:
            [ST_June_2017_BOB_LV7A_DPI7, ST_June_2017_CHO_LV7A_DPI7, ST_June_2017_E14TG2A_LV7A_DPI7 , ST_June_2017_HAP1_LV7A_DPI7, ST_June_2017_K562_800x_LV7A_DPI7]
        indel_type:
            str, specify the training task. inDecay is a multi-stage model comprised of del, ins and ratio model. Each of them is trained separately.  
        DS_class:
            callable, the defined inDecay Dataset class. This Dataset takes in arg `experiment` and arg `Oligos` which will be generated later in `self.setup`.
            If the Dataset class has more arguments to pass, please use the `functools.partial` to define the other arguments first. 
            Example: 
            ```
            from functools import partial
            transform_class = partial(nhej_p_del_DS, nhej_transform=funcXXX, mmej_transform=funcxxx)
            ```
        batch_size:
            int, the size of minibatch. How many Guides to use for each iteration
        n_worker:
            int, the number of threads to used to process data reader
        test_oligo_fle:
            str, path of txt. It defines the test set
        seed:
            int, random seed 
        threshold:
            int, the threshold for the total count. Only guides passing this threshold will be used in training and testing.
        Return
        --------------
            DataModule     
        """
        super().__init__()
        self.experiment = experiment
        self.Cellline = experiment.split("_")[3]
        self.rep = experiment.split("_")[4]
        self.indel_type = indel_type
        self.batch_size= batch_size
        self.n_worker = n_worker
        self.threshold = threshold
        self.seed = seed
        self.test_oligo_file = test_oligo_file

        # the df here decides what oligos will be used
        self.df = self.read_df()

        self.ddDS_class = DS_class
    
    def setup(self, stage):
        """
        split the train, val, test and then instanize the Dataset
        """
        oligos = get_Train_Val_Test(self.df, self.test_oligo_file, self.seed, self.threshold)
        self.Train_oligos, self.Val_oligos, self.Test_oligos = oligos

        self.DS_train = self.DS_class(self.Train_oligos, self.experiment)
        self.DS_val = self.DS_class(self.Val_oligos, self.experiment)
        self.DS_test = self.DS_class(self.Test_oligos, self.experiment)

    def read_df(self):
        """the dataset that only contain deletion results"""
        csv_path = pj(PATH.high_dir, self.experiment, f"{self.Cellline}_{self.rep}_{self.indel_type}df.csv")
        assert os.path.exists(csv_path), "deletion df csv not found, please check the result of `compute_nhej_mmej_features.py` or make sure you have put them under the correct path"
        del_df = pd.read_csv(csv_path,low_memory=False)
        return del_df
    
    def train_dataloader(self):
        return DataLoader(self.DS_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_worker)
    def val_dataloader(self):
        return DataLoader(self.DS_val, batch_size=self.batch_size, shuffle=False, num_workers=self.n_worker)
    def test_dataloader(self):
        return DataLoader(self.DS_test, batch_size=self.batch_size, shuffle=False, num_workers=self.n_worker)


class Base_del_Dataset(Dataset):
    def __init__(self, Oligos, experiment):
        """
        Data Reader that return the featurs for InDecay deletion model. 
        This is the Base class that load in all the prerequisite for the convenience of later inheritance.
        Params
        --------------
        Oligos:
            int [str], the OligoIDs of the guide-target pair that are going to return in this dataset
        experiment:
            str , the experiment that should be in 
            [ST_June_2017_BOB_LV7A_DPI7, ST_June_2017_CHO_LV7A_DPI7, ST_June_2017_E14TG2A_LV7A_DPI7 , ST_June_2017_HAP1_LV7A_DPI7, ST_June_2017_K562_800x_LV7A_DPI7]
        Return
        --------------
            Dataset        
        """
        super().__init__()
        self.Oligos = Oligos
        self.experiment = experiment
        self.Cellline = experiment.split("_")[3]
        self.rep = experiment.split("_")[4]
        
        # Read X
        # load the requirements
        # self.df = self.read_del_df()          # in case we need it
        self.check_prerequisite()  
        lookups = load_lookup(self.experiment)
        self.nhej_p_lookup = lookups[0]      
        self.nhej_feature_lookup = lookups[1]
        self.mmej_feature_lookup = lookups[2]
        self.mhratio_lookup = lookups[3]
        self.ins_feature_lookup = lookups[4]

        # Read Y
        self.Y_lookup = self.load_Y()


    def check_prerequisite(self):
        """make sure the file exist"""
        pkl_files = ['nhej_p_lookup.pkl', 'nhej_feature_lookup.pkl', 'mmej_feature_lookup.pkl',
                    'mhratio_lookup.pkl', "Ins_feature_lookup.pkl"]
        
        for file in pkl_files:
            pkl_path = pj(PATH.high_dir, self.experiment, file)
            assert os.path.exists(pkl_path), f"Prerequisite file {pkl_path} not found, please rerun `compute_nhej_mmej_features.py` or make sure you have placed them under the correct path"

    def read_del_df(self):
        """the dataset that only contain deletion results"""
        csv_path = pj(PATH.high_dir, self.experiment, f"{self.Cellline}_{self.rep}_deldf.csv")
        assert os.path.exists(csv_path), "deletion df csv not found, please check the result of `compute_nhej_mmej_features.py` or make sure you have put them under the correct path"
        del_df = pd.read_csv(csv_path,low_memory=False)
        return del_df
    
    def read_cmatrix(self, oligo):
        """read the cmatrix, this is guide specific. It records which are duplicating products"""
        cm_path = pj(PATH.data_dir, '912_cmatrix', f"{oligo}_cmatrix.npy")
        assert os.path.exists(cm_path), "cmatrix of %s not found" %oligo
        raw_cm = np.load(cm_path, allow_pickle=True)
        if raw_cm.dtype == 'O': # if loading csr sparse matrix
            cmatrix = raw_cm.item().todense()
        else:  # if loading dense matrix
            cmatrix = raw_cm
        return cmatrix.astype('float32')
    
    def check_oligos(self):
        """make sure the oligos are those with featurs"""
        # df_oligoset = self.df.OligoID.unique()
        # missed_in_df = [oligo for oligo in self.Oligos if oligo not in df_oligoset]
        # assert len(missed_in_df) == 0, f"{len(missed_in_df)} oligos are not found in deletion event dataframe"

        # basically the 4 lookup object have the same key, so randomly pick one
        featured_oligos = self.nhej_feature_lookup.keys() 
        missed_in_lookup = [oligo for oligo in self.Oligos if oligo not in featured_oligos] 
        assert len(missed_in_lookup) == 0, f"{len(missed_in_lookup)} oligos are not found in feature lookup"
    
    def load_Y(self):
        Ymatrix_path = pj(PATH.high_dir, self.experiment, f"{self.Cellline}_{self.rep}.npy")
        assert os.path.exists(Ymatrix_path), "Y matrix not found"
        Y_matrix, Y_order = read_labeld_XY_matrix(Ymatrix_path)
        Y_lookup = {"Oligo%d"%oligo : y for y,oligo in zip(Y_matrix, Y_order)}
        return Y_lookup

    def __len__(self):
        return len(self.Oligos)
    
    def __getitem__(self,i):
        oligo = self.Oligos[i]
        return oligo
    
class nhej_p_del_DS(Base_del_Dataset):
    def __init__(self, Oligos, experiment, n_classes=891, nhej_transform=None, mmej_transform=None):
        """
        For nhej part, we use the precomputed p 
        """
        super().__init__(Oligos=Oligos, experiment=experiment)
        # transform
        self.nhej_transform = nhej_transform 
        self.mmej_transform = mmej_transform
        self.n_classes = n_classes

        # read and normalize
        
        self.norm_Y_lookup = self.normalize_y()
        self.Cmatrix_lookup = {oligo:self.read_cmatrix(oligo)[:n_classes,:n_classes] for oligo in self.Oligos}
    
    def normalize_y(self):
        norm_Y_lookup = {}
        for oligo in self.Oligos:
            y = self.Y_lookup[oligo]
            norm_y = y[:self.n_classes]/y[:self.n_classes].sum()
            norm_Y_lookup[oligo] = norm_y
        return norm_Y_lookup

    def __getitem__(self, i):
        oligo = self.Oligos[i]
        # retrieve y
        y = self.norm_Y_lookup[oligo].astype("float32")
        cmatrix = self.Cmatrix_lookup[oligo]
        # retrieve x
        nhej_p = self.nhej_p_lookup[oligo].astype("float32")
        mmej_feat = self.mmej_feature_lookup[oligo].astype("float32")
        mh_ratio = self.mhratio_lookup[oligo]

        # apply transformation
        if self.nhej_transform is not None:
             nhej_p = self.nhej_transform(nhej_p)
        if self.mmej_transform is not None:
             mmej_feat = self.mmej_transform(mmej_feat)

        return (nhej_p, mmej_feat, mh_ratio), y, cmatrix
    
class nhej_feat_del_DS(Base_del_Dataset):
    def __init__(self, Oligos, experiment, n_classes=891, nhej_transform=None, mmej_transform=None):
        """
        For nhej part, we use the extracted features
        """
        super().__init__(Oligos=Oligos, experiment=experiment)
        self.nhej_transform = nhej_transform 
        self.mmej_transform = mmej_transform
        self.n_classes = n_classes
        self.norm_Y_lookup = self.normalize_y()
        self.Cmatrix_lookup = {oligo:self.read_cmatrix(oligo)[:n_classes,:n_classes] for oligo in self.Oligos}

    def normalize_y(self):
        norm_Y_lookup = {}
        for oligo in self.Oligos:
            y = self.Y_lookup[oligo]
            norm_y = y[:self.n_classes]/y[:self.n_classes].sum()
            norm_Y_lookup[oligo] = norm_y
        return norm_Y_lookup

    def __getitem__(self, i):
        oligo = self.Oligos[i]
        # retrieve y
        y = self.norm_Y_lookup[oligo].astype("float32")
        cmatrix = self.Cmatrix_lookup[oligo]
        # retrieve x
        nhej_feat = self.nhej_feature_lookup[oligo].astype("float32")
        mmej_feat = self.mmej_feature_lookup[oligo].astype("float32")
        mh_ratio = self.mhratio_lookup[oligo] #.astype("float32")

        # apply transformation
        if self.nhej_transform is not None:
             nhej_feat = self.nhej_transform(nhej_feat)
        if self.mmej_transform is not None:
             mmej_feat = self.mmej_transform(mmej_feat)

        return (nhej_feat, mmej_feat, mh_ratio), y, cmatrix

class ins_feat_DS(Base_del_Dataset):
    def __init__(self, Oligos, experiment, insfeat_transform=None):
        """
        For nhej part, we use the extracted features
        """
        super().__init__(Oligos=Oligos, experiment=experiment)
        self.insfeat_transform = insfeat_transform

        # read and normalize
        self.Y_lookup = {k:v[-21:]/v[-21:].sum() for k,v in self.Y_lookup.items()}

    def __getitem__(self, i):
        oligo = self.Oligos[i]
        # retrieve y
        y = self.Y_lookup[oligo].astype("float32")
        # retrieve x
        ins_feat = self.ins_feature_lookup[oligo].astype("float32")

        # apply transformation
        if self.insfeat_transform is not None:
             ins_feat = self.insfeat_transform(ins_feat)

        return ins_feat, y
    

def read_labeld_XY_matrix(matrix_path):
    """
    the last column of the matrix record the order of the Guides
    """
    raw = np.load(matrix_path)
    data = raw[:,:-1]
    oligo_order = raw[:,-1]
    return data.astype('float32'), oligo_order


def load_lookup(experiment):
    """
    for the experiment, load the saved lookup to retrieve the extracted features
    Input
    ---------
    experiment:
        str , the experiment that should be in
        [ST_June_2017_BOB_LV7A_DPI7, ST_June_2017_CHO_LV7A_DPI7, ST_June_2017_E14TG2A_LV7A_DPI7 , ST_June_2017_HAP1_LV7A_DPI7, ST_June_2017_K562_800x_LV7A_DPI7]
    Return
    ---------
    lookups
        list: [dict] len=5, in the order of [nhej_p_lookup, nhej_feature_lookup, mmej_feature_lookup,mhratio_lookup, Ins_feature_lookup]
    """
    lookups = []
    for file in ['nhej_p_lookup.pkl', 'nhej_feature_lookup.pkl', 'mmej_feature_lookup.pkl', 'mhratio_lookup.pkl', 'Ins_feature_lookup.pkl']:
        with open(pj(PATH.high_dir, experiment, file) , 'rb') as f:
            lookup = pkl.load(f)
            f.close()
        lookups.append(lookup)
    return lookups



def get_Train_Val_Test(df, test_oligo_file:str, seed:int=0, threshold=1000):
    """
    For all the oligo (guide-target pair) in the dataframe, we first select valid oligos passing the total count threshold.
    Then we 
    Input
    ---------
    df
        pandas.DataFrame: must contain column ["OligoID", "Count"]
    test_oligo_fle
        str, path of txt. It defines the test set
    seed:
        int, random seed 
    threshold:
        int, the threshold for the total count. Only guides passing this threshold will be used in training and testing.
    Return
    ---------
    Oligos:
        tuple of list, each list contain the name of oligos. They are return in the order of train, val, test.
    """
    agg_sum_df = df.groupby("OligoID").agg({"Count":"sum"})
    Passed = agg_sum_df.query("`Count` >= @threshold").index
    print(len(Passed))
    # TEST
    # Test_O_file = os.path.join(PATH.main_dir, "result/test_set_oligo_Feb2.txt")
    Test_Oligos = pd.read_table(test_oligo_file, names=['OligoID'])["OligoID"].values # list of str
    N_test = len(Test_Oligos)

    N_passed_test = np.sum([(oligo in Test_Oligos) for oligo in Passed])
    print(f"Valid TestSet :{N_passed_test}, {round(N_passed_test/N_test,3)}")

    # TRAIN-VAL : the remaining valid guides
    TrainVal_Oligos = [oligo for oligo in Passed if oligo not in Test_Oligos]
    empty_oligos = ['Oligo48008','Oligo17384','Oligo33698','Oligo17541','Oligo48644','Oligo17195','Oligo17189','Oligo48756','Oligo18899','Oligo46611','Oligo17887', 'Oligo46958']
    TrainVal_Oligos = [oligo for oligo in TrainVal_Oligos if oligo not in empty_oligos]

    # Split Train Val 
    np.random.seed(seed) # setting different seeds among repeats 
    np.random.shuffle(TrainVal_Oligos)
    train_size = int(len(TrainVal_Oligos)*0.9)
    Train_Oligos = TrainVal_Oligos[:train_size]
    Val_Oligos = TrainVal_Oligos[train_size:]
    
    return Train_Oligos, Val_Oligos, Test_Oligos

    # n_splits=len(genes)
# def get_Sanger_train_test(genes, seed=0):
#     """
#     K-fold cross validation spliting for sanger data.
#     The number of fold depends on the available sample size.
#     The more data we have , the fewer fold , more sample for testing.

#     Args:
#         genes (list): a list of gene names
#         seed (int, optional): random seed. fixed to 0.

#     Returns:
#         kf_index_ls (list of tuple): 
#         [(train_idx, val_idx, test_idx) * n_splits]
#     """
#     # decide the number of testing by # of genes
#     n_splits = 5


#     # split k-fold 
#     kf = KFold(n_splits, shuffle=True, random_state=seed)
#     kf_splits = kf.split(genes)

#     # split train val
#     kf_index_ls = []
#     for trainval_idx, test_idx in kf_splits:
#         val_size = int(0.2*len(trainval_idx))
#         train_idx, val_idx = train_test_split(trainval_idx,
#                                       test_size=val_size)
#         kf_index_ls.append(
#             (train_idx, val_idx, test_idx)
#         )
#     return kf_index_ls
def get_Sanger_train_test(genes, seed=0):
    """
    K-fold cross validation spliting for sanger data.
    The number of fold depends on the available sample size.
    The more data we have , the fewer fold , more sample for testing.

    Args:
        genes (list): a list of gene names
        seed (int, optional): random seed. fixed to 0.

    Returns:
        kf_index_ls (list of tuple): 
        [(train_idx, val_idx, test_idx) * n_splits]
    """
    # decide the number of testing by # of genes

    n_splits=len(genes)

    # split k-fold 
    kf = KFold(n_splits, shuffle=True, random_state=seed)
    kf_splits = kf.split(genes)

    # split train val
    kf_index_ls = []
    for trainval_idx, test_idx in kf_splits:
        val_size = int(0.2*len(trainval_idx))
        train_idx, val_idx = train_test_split(trainval_idx,
                                      test_size=val_size)
        kf_index_ls.append(
            (train_idx, val_idx, test_idx)
        )
        # print(genes[test_idx])
    return kf_index_ls
def get_Sanger_train_test_WZ(genes, seed=0):
    """
    K-fold cross validation spliting for sanger data.
    The number of fold depends on the available sample size.
    The more data we have , the fewer fold , more sample for testing.

    Args:
        genes (list): a list of gene names
        seed (int, optional): random seed. fixed to 0.

    Returns:
        kf_index_ls (list of tuple): 
        [(train_idx, val_idx, test_idx) * n_splits]
    """
    # decide the number of testing by # of genes
    if len(genes) > 60:
        n_splits = 3
    elif len(genes) > 30:
        n_splits = 5
    else:
        n_splits = 10


    # split k-fold 
    kf = KFold(n_splits, shuffle=True, random_state=seed)
    kf_splits = kf.split(genes)

    # split train val
    kf_index_ls = []
    for trainval_idx, test_idx in kf_splits:
        val_size = int(0.1*len(trainval_idx))
        train_idx, val_idx = train_test_split(trainval_idx,
                                      test_size=val_size)
        kf_index_ls.append(
            (train_idx, val_idx, test_idx)
        )
    return kf_index_ls
#######
# Old functions
#######
Lo_trainval_pkl = "/home/wergillius/Project/CRISPR_data/models/Lo_trainval_with_del.pkl"

class Tensor_DataModule(pl.LightningDataModule):
    
    def __init__(self, XY, batch_size, norm_Y=False, n_worker=4):
        """
        pl DataModule class for compile tensor. e.g the Lindel data
        
        Arguments:
        ------------
        XY: list of tuple, len = 3
        
        Returns:
        -----------
        """
        super().__init__()
        self.XY = XY
        self.batch_size = batch_size
        self.n_worker = n_worker 
        if norm_Y:
            self.XY = [(X, Y/Y.sum(axis=1).reshape(-1,1)) for X, Y in XY]

    def setup(self, stage):
        """
        stage determined which set to use
        """
        train_X, train_Y = self.XY[0]
        val_X, val_Y = self.XY[1]
        test_X, test_Y = self.XY[2]

        self.DS_train = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_Y))
        self.DS_val = TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_Y))
        self.DS_test = TensorDataset(torch.from_numpy(test_X), torch.from_numpy(test_Y))

    def train_dataloader(self):
        return DataLoader(self.DS_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_worker)
    def val_dataloader(self):
        return DataLoader(self.DS_val, batch_size=self.batch_size, shuffle=False, num_workers=self.n_worker)
    def test_dataloader(self):
        return DataLoader(self.DS_test, batch_size=self.batch_size, shuffle=False, num_workers=self.n_worker)


class Lo_dataset(Dataset):
    def __init__(self, pkl_path:str=Lo_trainval_pkl, label_loc_lookup=None,
                    which_set=0, 
                    seed=41):
        
        self.which_set = which_set
        self.transform = ToTensor()
        self.pkl_path = pkl_path
        self.label_loc_lookup = label_loc_lookup
        with open(pkl_path, 'rb') as f:
            decay_mmej = pkl.load(f)
            f.close()
        
        # there should be 4341 samples for train val
        # 438 for test 
        train_val_idx = train_test_split(np.arange(len(decay_mmej[1])), 
                                              test_size=0.1, 
                                              random_state=seed)

        idxx = np.arange(len(decay_mmej[1])) if which_set == 2 else train_val_idx[which_set]
        self.set_idex = idxx

        self.Guides  = decay_mmej[0][idxx]
        self.mmej_feat = decay_mmej[1][idxx]
        self.X = decay_mmej[2][idxx]
        self.Y = decay_mmej[3][idxx]

        self.cmatrix = decay_mmej[4][idxx]
        self.mh_strength = decay_mmej[5][idxx]
        self.grlookup =decay_mmej[6]
        self.nhej_tabular = decay_mmej[7]

        self.generate_nhej_p()

    def generate_nhej_p(self):
        copmiled_nhej_p_pkl = self.pkl_path.replace(".pkl","compiled_nhej.pkl")
        if os.path.exists(copmiled_nhej_p_pkl):
            with open(copmiled_nhej_p_pkl, 'rb') as f:
                self.nhej_p = pkl.load(f)
                f.close()
        else:
            self.nhej_p = []
            for guide in self.Guides:
                target = self.grlookup[guide]
                indels = my_utils.gen_indel(target, 30)
                p = alignmap.nhej_p_from_indels(indels, self.label_loc_lookup)
                self.nhej_p.append(p)
            with open(copmiled_nhej_p_pkl, 'wb') as f:
                pkl.dump(self.nhej_p, f)
                f.close()

    def __len__(self):
        return len(self.set_idex)
    
    def __getitem__(self, i):
        # the input for our model
        mmej = torch.from_numpy(self.mmej_feat[i]).float()
        nhej = torch.from_numpy(self.nhej_p[i]).float()
        mh_strength = torch.tensor(self.mh_strength[i]).float()

        # X
        X = torch.from_numpy(self.X[i]).float()

        # the output
        y = self.Y[i, :536]
        y = torch.from_numpy( y / y.sum()) # normalize to all deletion
        cmatrix = torch.from_numpy(self.cmatrix[i][:536,:536]).float()
        return (nhej, mmej, mh_strength), X , y.float(), cmatrix



class Lo_raw_data(Lo_dataset):
    def __init__(self, pkl_path: str = Lo_trainval_pkl, which_set=0, seed=41):
        super().__init__(pkl_path, which_set, seed)

    def __getitem__(self, i):
        # the input for our model
        X = torch.from_numpy(self.X[i]).float()

        # the output
        y = self.Y[i,:536]
        y = torch.from_numpy( y / y.sum()) # normalize to all deletion
        # cmatrix = torch.from_numpy(self.cmatrix[i][:536,:536]).float()
        return X , y.float()

class Lo_raw_from_numpy(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, i):
        x = self.X[i]
        y = self.Y[i]
        return x , y
