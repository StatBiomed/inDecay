import os, sys, subprocess, argparse, time, shutil, math 
import numpy as np
import pandas as pd 
import torch
from torch.nn import functional as F
torch.set_num_threads(4)
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import callbacks 
import pickle as pkl
# from models. import Base_del_model, ST_Decay, ST_DeepDecay, ST_Decay_Scaler, ST_DeepDecay_dropout, ST_DeepDecay_Multinomial
from inDecay import my_utils, alignmap, models, reader, PATH
sys.path.append(PATH.main_dir)
from tqdm.contrib.concurrent import process_map

to_train = True
to_predict = True
to_write_y = True
num_workers = 12

ndel = 14
nins = 8
nshare = 39

# model params
k1 = 0.5 
k2 = 0.6
h = 1.3
hidden = [128, 64]
L2_Lambda = 1e-4
L1_Lambda = 0
lr = 1e-4

# Torch Device
device = 'gpu' if torch.cuda.is_available() else 'cpu'

# some path and requisite files
pj = os.path.join
data_dir = PATH.data_dir

reference_path = pj(data_dir, "SelfTarget_NewScaffold.fasta")


def check_dir(DIR_NAME):
    if not os.path.exists(DIR_NAME):
        os.mkdir(DIR_NAME)

def readFeaturesData(features_file):
    
    feature_data = pd.read_csv(features_file, skiprows=2, sep='\t', dtype={'Inserted Seq':str})
    feature_cols = [x for x in feature_data.columns if x not in ['Oligo ID','Indel','Left','Right','Inserted Seq']]
    indel_feature_data = 1*feature_data[['Indel'] + feature_cols].groupby('Indel').any()
    # indel_feature_data['Indel'] = indel_feature_data.index
    return indel_feature_data.reset_index(), feature_cols

def find_ckpt(ckpt_version_dir):
    """
    find the latest version, if not finished then return last one
    """
    get_v = lambda s: int(s.replace("version_",""))
    
    versions = [get_v(subdir) for subdir in os.listdir(ckpt_version_dir)]

    for v in versions:
        checkpoint_dir = pj(ckpt_version_dir, 'version_%d'%v, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            try:
                shutil.rmtree(pj(ckpt_version_dir, 'version_%d'%v)) 
            except:
                continue


    versions = [get_v(subdir) for subdir in os.listdir(ckpt_version_dir)]
    maxv  = np.max(versions)

    try:
        ckpts = os.listdir()
    except FileNotFoundError:
        maxv -= 1
    ckpts = list(filter(lambda x : x.endswith('.ckpt'), 
                            os.listdir(pj(ckpt_version_dir, 'version_%d'%maxv, 'checkpoints')))
                )
    
    while len(ckpts) == 0:
        maxv -= 1
        ckpts = list(filter(lambda x : x.endswith('.ckpt'), 
                            os.listdir(pj(ckpt_version_dir, 'version_%d'%maxv, 'checkpoints')))
                            )
    
    if len(ckpts) >1:
        ckpt = ckpts[-1]
    else:
        ckpt = ckpts[0]
    
    return pj(ckpt_version_dir, 'version_%d'%maxv, 'checkpoints', ckpt)


ref_lookup = reader.get_reference()


def read_data(OligoID, processed_df, experiments):
    """Read the precompute features according to the OligoID"""
    # read features
    Guide, refseq, pamsite, Strand = ref_lookup[OligoID]
    idfgen_file = my_utils.get_indelgen_file(OligoID, Guide)
    idfgen = pd.read_table(idfgen_file, skiprows=1, names=['Identifier', 'n_coevent', 'loc'])

    def merging(OligoID,idfgen=idfgen):
        oligo_df = processed_df.query("`OligoID` == @OligoID")

        label_df = idfgen.merge(oligo_df[['OligoID','Identifier', 'Count']], 
                        left_on=['Identifier'], right_on=['Identifier'], suffixes=['', '_filled'], how='left') # type: ignore 
        # label_df['Count']= label_df['Count']
        # label_df.replace([np.inf, -np.inf], 0, inplace=True)
        label_df=label_df.fillna(0)# make indels that are not capture with count=0
        # print(label_df[label_df.isna().any(axis=1)])
        return label_df
    
    label_df = merging(OligoID)

    total_sum = label_df['Count'].sum()
    label_df['Frac Sample Reads'] = label_df['Count']/total_sum
    label_df['Frac Sample Reads'] = label_df['Frac Sample Reads'] if total_sum!=0 else 0
    return label_df


def interaction_transform(X, ndel, nins):
    """
    Data transformation
    """
    n_features = ndel + nins + math.comb(ndel,2) + math.comb(nins,2)
    n_total = X.shape[1]
    n_shared = n_total - ndel - nins
    interaction_del = []
    interaction_ins = []
    
    for i in range(0,ndel):
        for j in range(0,ndel):
            if i>=j:
                continue
            interaction_del.append(np.multiply(X[:,i],X[:,j]).reshape(-1,1))
    for i in range(ndel,ndel + nins):
        for j in range(ndel,ndel + nins): 
            if i>=j:
                continue
            interaction_ins.append(np.multiply(X[:,i],X[:,j]).reshape(-1,1))

    X_inter = np.concatenate([X] + interaction_del + interaction_ins, axis=1)
    
    if n_shared > 0:
        interaction_shared = []
        for i in range(ndel + nins, n_total):
            for j in range(ndel + nins):
                interaction_shared.append(np.multiply(X[:,i],X[:,j]).reshape(-1,1))

        X_inter = np.concatenate([X_inter]+ interaction_shared, axis=1)

    # assert X_inter.shape[1] == n_features 
    return X_inter

def decay_transform(X):
    """
    Data transformation
    """
    interaction_del = []
    interaction_ins = []
    for i in range(0,5):
        for j in range(0,5):
            if i==j:
                continue
            interaction_del.append(np.multiply(X[:,i],X[:,j]))
    for i in range(5,9):
        for j in range(5,9): 
            interaction_ins.append(np.multiply(X[:,i],X[:,j]))

    X_del = np.hstack([X[:,:5], interaction_del])
    X_ins = np.hstack([X[:,:5], interaction_ins])
    return np.hstack([X_del, X_ins])

def my_collect_fn(batch_list):
    features = [item[0].requires_grad_() for item in batch_list]
    ys = [item[1].requires_grad_() for item in batch_list]
    return features, ys

# def my_collect_fn(batch_list):
    
#     features = []
#     ys = []

#     most_events = np.max([max(item[1].shape) for item in batch_list])

#     for item in batch_list:
#         X = item[0]
#         y = item[1]

#         n_event = max(y.shape)  # y shape can be [b, e, 1] or [e,1] or [e,]
#         pad_len = most_events - n_event
#         X_padding = (0,0, 0, pad_len)

#         y_padding = X_padding if len(y.shape) > 1 else (0,pad_len)

#         features.append( F.pad(X, pad=X_padding) )
#         ys.append( F.pad(y, pad=y_padding) )

#     features = torch.stack(features).requires_grad_()
#     ys = torch.stack(ys).requires_grad_()

    # return features, ys

if __name__ == "__main__":
    parser = argparse.ArgumentParser("The script to train inDecay")
    # parser.add_argument("--Set", required=True, type=str, help="either `TestSet1` or `TestSet2`")
    parser.add_argument("-E","--experiment", type=str, required=True, help='The dir name of dataset')
    parser.add_argument("-C","--read_cutoff", type=int, default=500, help='The threshold of total count. Only Guides having total read count over this threshold are used')
    parser.add_argument("-T","--test_oligos", type=str, default="results/test_set_oligo_Feb2.txt", help='The file deciding which oligos are used in the training set')
    parser.add_argument("-G","--GPU_devices", type=int, default=None, help='The gpu to use')
    parser.add_argument("-P","--Pretrain", required=False, type=str, default=None, help="the pretrained parameter theta")
    parser.add_argument("-M","--Model_Class", required=False, type=str, default="ST_DeepDecay", help="inDecay / DeepDecay")
    parser.add_argument("-D","--Data_transform", required=False, type=str, default="identity", help="the name of data transformation")
    parser.add_argument("-O","--Mode", required=False, type=str, default="Train", help="the action of this script, can be `Train`, `Evaluate`, `Evaluate_only`, `Baseline` and `Write_Y`")
    parser.add_argument("--progress_bar", required=False, type=str, default="True", help="boolen, whether to show progress bar")
    args = parser.parse_args()

        ## Mode ##
    if args.Mode == 'Train':
        to_train = to_write_y = to_predict = True

    elif args.Mode == 'Evaluate':
        to_train = False
        to_write_y = to_predict = True
    
    elif args.Mode == 'Evaluate_only':
        to_predict = True
        to_train = to_write_y = False
        
    elif args.Mode == 'Write_Y':
        to_train = to_predict = False
        to_write_y = True

    elif args.Mode == 'Baseline':
        to_train = to_predict = to_write_y = False 
        to_baseline = True
    else:
        raise ValueError("Invalide action combination")

    # some save file settings
    experiments = args.experiment
    Cellline = experiments.split("_")[3]
    rep = experiments.split("_")[4]
    trainer_log = pj(PATH.main_dir, 'pl_trainer_log')
    save_dir = pj(data_dir, 'somatic')
    csv_path = pj(save_dir,f"{Cellline}_{rep}.csv")

    gpu_device = 0 if device=='gpu' else 10

    if args.GPU_devices is not None:
        gpu_device= args.GPU_devices
    
    print(f"Runing {experiments} using cud: {gpu_device}")
    pth_save_dir = os.path.join(PATH.pth_dir, f"ST_featv51e4lr_{args.Model_Class}_{args.Data_transform}_C{args.read_cutoff}")
    
    # Temp Theta file
    date = time.strftime("%B%d")
    pth_save_path = pj(pth_save_dir, experiments)

    for DIR in [trainer_log, data_dir, pth_save_dir, save_dir, pth_save_path]:
        check_dir(DIR)  

    processed_df = pd.read_csv(csv_path).query("`in_LdGen` == True").astype({"Count":"int"})
    processed_df = processed_df.query("`Strand` == 'FORWARD'")
    

    # split training and testing data
    Train_Oligos, Val_Oligos, Test_Oligos = reader.get_Train_Val_Test(
        processed_df, 
        test_oligo_file = os.path.join(PATH.main_dir, args.test_oligos),
        seed = 0,
        threshold = args.read_cutoff
        )

    # DATA TRANSFORMATION
    
    if args.Data_transform == "identity":
        transform = lambda x: x
        n_features = ndel + nins + nshare
    elif args.Data_transform == "interaction":
        transform = lambda x: interaction_transform(x, ndel, nins)
        n_features = ndel + nins + nshare + math.comb(ndel,2) + math.comb(nins,2) + nshare*(ndel+nins)
    elif ":" in args.Data_transform:
        # decay transform
        raise ValueError("Invalid transform name")
    else:
        raise ValueError("Invalid transform name")
    
    
    # Modeling and Training
    model_class = eval("models.%s" %args.Model_Class)
    model_parsms = dict(inputsize=n_features, outputsize=1,  lr=lr,
                        L1_lambda=L1_Lambda, L2_lambda=L2_Lambda)
    if 'Deep' in args.Model_Class:
        model_parsms['hidden'] = hidden
        
    model = model_class(**model_parsms)
    if args.Pretrain is not None:
        model = model_class.load_from_checkpoint(args.Pretrain)


    # dataset    
    normalize = ('Multinomial' not in args.Model_Class) & ('weight' not in args.Model_Class)
    # normalize = 'weight' not in args.Model_Class 
    feature_extraction_fn = lambda label_df, refseq, cutsite : alignmap.ST_decayfeat_v5(label_df, refseq, cutsite, k1, k2, h)

    Train_DS = reader.ST_dataset(Train_Oligos,processed_df, experiments, 
                          read_data_fn = read_data,
                          transformation=transform,
                          feat_ext_fn = feature_extraction_fn,
                          normalize=normalize)
    Val_DS = reader.ST_dataset(Val_Oligos,processed_df, experiments, 
                               read_data_fn = read_data,
                               transformation=transform , 
                               feat_ext_fn = feature_extraction_fn,
                               normalize=normalize)
    Test_DS = reader.ST_dataset(Test_Oligos,processed_df, experiments, 
                                read_data_fn = read_data,
                                transformation=transform,
                                feat_ext_fn = feature_extraction_fn,
                                normalize=normalize)

    Train_DL = DataLoader(Train_DS, shuffle=True, batch_size=32, num_workers=num_workers, collate_fn=my_collect_fn)
    Val_DL = DataLoader(Val_DS, shuffle=False, batch_size=32, num_workers=num_workers, collate_fn=my_collect_fn)
    Test_DL = DataLoader(Test_DS, shuffle=False, batch_size=32, num_workers=num_workers, collate_fn=my_collect_fn)

    trainer = pl.Trainer(
			auto_lr_find=True,
            accelerator=device,
            # fast_dev_run=True,
            enable_progress_bar=eval(args.progress_bar),
			default_root_dir=pth_save_path,
            devices = [gpu_device],
			max_epochs=100,
			callbacks=[ callbacks.ModelCheckpoint(filename='{epoch}-{val_cre:.8f}',
                                                  monitor="val_cre", mode="min", save_top_k=2),
                        callbacks.EarlyStopping(monitor="val_cre", mode="min", patience=20),])
    

    if to_train:
        model.train()
        trainer.fit(model, Train_DL, val_dataloaders=Val_DL)
        print(trainer.ckpt_path)

        model.eval()
        trainer.validate(model, Test_DL)

    # only save once for Y
    if to_write_y:
        Forecast_Y = pj(pth_save_path, "ForeCast_TestY.pkl")
        if not os.path.exists(Forecast_Y):
            get_identifiers = lambda oligo : read_data(oligo, processed_df, experiments)[['Identifier', 'Frac Sample Reads']].values

            Y_ls = [get_identifiers(oligo) for oligo in Test_Oligos]
            # Y_ls = process_map(get_identifiers, Test_Oligos, max_workers=8)
            Y_lookup = {o:Y_ls[i] for i,o in enumerate(Test_Oligos)}

            handle = open(Forecast_Y, 'wb')
            pkl.dump(Y_lookup, handle)    
            handle.close()
            print('finished write to ',  Forecast_Y)


    if to_predict:
        if args.Pretrain is not None:
            ckpt_abspath = os.path.join(PATH.main_dir, args.Pretrain)
        else:
            ckpt_abspath = find_ckpt(pj(pth_save_path, 'lightning_logs'))
        assert os.path.exists(ckpt_abspath)

        try:
            model.eval()
        except:
            model = model_class.load_from_checkpoint(ckpt_abspath).to(gpu_device)
            model.eval()
        model.eval()
        predict_y = trainer.predict(model, Test_DL)
        if isinstance(predict_y[0], list):
            predict_y = sum(predict_y, [])  # to join lists 
        
        pred_lookup = {o:predict_y[i].cpu().numpy() for i,o in enumerate(Test_Oligos)} # type: ignore

        TestPred = ckpt_abspath.replace(".ckpt", "TestPred.pkl")
        pred_f = open(TestPred, 'wb')
        pkl.dump(pred_lookup, pred_f)
        pred_f.close()
        print("prediction writed to %s" %TestPred)