import os, sys, argparse, time,  math 
import re, json
import numpy as np
import pandas as pd 
import torch
from torch.nn import functional as F
torch.set_num_threads(4)
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import callbacks 
import pickle as pkl
from inDecay import my_utils, alignmap, models, reader, PATH
sys.path.append(PATH.main_dir)
from inDecay import analysis_fn
import random
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
lr = 1e-3


# Torch Device
device = 'gpu' if torch.cuda.is_available() else 'cpu'

# some path and requisite files
pj = os.path.join
data_dir = PATH.data_dir

reference_path = pj(data_dir, "SelfTarget_NewScaffold.fasta")


def check_dir(DIR_NAME):
    if not os.path.exists(DIR_NAME):
        os.mkdir(DIR_NAME)


ref_lookup = reader.get_reference()


def read_data(OligoID, processed_df, experiments):
    """Read the precompute features according to the OligoID"""
    # read features
    Guide, refseq, pamsite, Strand = ref_lookup[OligoID]
    idfgen_file = my_utils.get_indelgen_file(OligoID, Guide)
    idfgen = pd.read_table(idfgen_file, skiprows=1, header=None, sep='\t').iloc[:, :3]
    idfgen.columns = ['Identifier', 'n_coevent', 'loc']

    def merging(OligoID,idfgen=idfgen):
        oligo_df = processed_df.query("`OligoID` == @OligoID")
        label_df = idfgen.merge(oligo_df[['OligoID','Identifier', 'Count']], 
                        left_on=['Identifier'], right_on=['Identifier'], suffixes=['', '_filled'], how='left') # type: ignore 
        label_df['OligoID']=label_df['OligoID'].fillna(OligoID) # make indels that are not capture with count=0
        label_df['Count']=label_df['Count'].fillna(0)
        return label_df
    
    label_df = merging(OligoID)
    total_sum = label_df['Count'].sum()
    label_df['Frac Sample Reads'] = label_df['Count']/total_sum
    return label_df

def transform_pkl(model, Y_lookup, pred_lookup):

    dicts={}
    for oligo, Y in Y_lookup.items():
        Y = Y.T
        indel = Y[0, :]
        y = Y[1, :].astype('float32')
        pred= pred_lookup[oligo][0]
        df = pd.DataFrame({
        'Indel': indel,
        'observed': y,
        model: pred,
    })
        df = df.sort_values('observed', ascending=False)
        df.set_index('Indel', inplace=True)
        dicts[oligo] = df
    return dicts

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

import numbers
def mean_numeric_dict(dict_list):
    # Collect sums and counts for each key
    sums = {}
    counts = {}

    for d in dict_list:
        for key, value in d.items():
            # Check if value is numeric (int, float)
            if isinstance(value, numbers.Number):
                sums[key] = sums.get(key, 0) + value
                counts[key] = counts.get(key, 0) + 1
            else:
                # Ignore non-numeric values
                pass

    # Compute mean for each key
    mean_dict = {}
    for key in sums:
        if counts[key] > 0:
            mean_dict[key] = sums[key] / counts[key]
        else:
            mean_dict[key] = None  # or float('nan'), or skip key

    return mean_dict

def write_evaluate_json(Y_lookup, pred_lookup, model, ckpt_abspath, args, prefix=""):
    
    date = time.strftime("%b%d")

    performance_json={}
    performance_json.update(analysis_fn.assessment_recipe_forecast(Y_lookup, pred_lookup))
    performance_json.update(analysis_fn.assessment_recipe_41IDL_forecast(Y_lookup, pred_lookup))
    performance_json['End_date'] = time.strftime("%b%d-%H:%-M")
    performance_json['ckpt_path'] = ckpt_abspath
    
    L1_Lambda = args.L1_Lambda
    L2_Lambda = args.L2_Lambda
    
    # # model params 
    training_params = {}
    for pm in ["ndel", "nins", "nshare", "k1", "k2", "h", "hidden", "L2_Lambda", "L1_Lambda", "lr", "args.Fix_params"]:
        training_params[pm] = eval(pm)

    performance_json['training_params'] = training_params
    
    # save the metrics
    result_dir = f"{PATH.main_dir}/results/Transfer/C{args.read_cutoff}/{date}_V{args.Val_size}_{args.Model_Class}_{Cellline}_{L2_Lambda}_randinit/" 
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if prefix == "":
        json_path = pj(result_dir, f"N{args.N_Finetune}_rep{args.repeat}-{date}.json")
    else:
        json_path = pj(result_dir, f"N{args.N_Finetune}_rep{args.repeat}_{prefix}-{date}.json")
    new_performance_json = performance_json.copy()

    array_data = {}
    for key in ['Rep1_frameshift', 'Pred_frameshift']:
        if key in new_performance_json:
            array_data[key] = new_performance_json.pop(key)
    new_performance_json.update(array_data)
    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(json_path, "w") as write_file:
        json.dump(new_performance_json, write_file, indent=4, cls=_NpEncoder)
    
    print("\n"+"="*20)
    print("performance json saved to %s" %json_path)
    print("="*20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("The script to extract SelfTarget proccessed txt file and map to Lindel classes")
    parser.add_argument("-E","--experiment", type=str, required=True, help='The dir name of dataset')
    parser.add_argument("-C","--read_cutoff", type=int, default=100, help='The threshold of total count. Only Guides having total read count over this threshold are used')
    parser.add_argument("-T","--test_oligos", type=str, default="results/test_set_oligo_Feb2.txt", help='The file deciding which oligos are used in the training set')
    parser.add_argument("-G","--GPU_devices", type=int, default=None, help='The gpu to use')
    parser.add_argument("-P","--Pretrain", required=False, type=str, default=None, help="the pretrained parameter theta")
    parser.add_argument("-M","--Model_Class", required=False, type=str, default="ST_DeepDecay", help="inDecay / DeepDecay")
    parser.add_argument("-F","--Fix_params", required=False, type=str, default=None, help="the layers to fix, eg. del_regressor[:2]")
    parser.add_argument("-D","--Data_transform", required=False, type=str, default="identity", help="the name of data transformation")
    parser.add_argument("-O","--Mode", required=False, type=str, default="Train", help="the action of this script, can be `Train`, `Evaluate`, `Evaluate_only`, `Baseline` and `Write_Y`")
    parser.add_argument("-N","--N_Finetune", required=False, type=str, default="50", help="the number of oligos to use in the finetuning process")
    parser.add_argument("-R","--Rounds", required=False, type=int, default=1, help="Max epoches for finetuning") #100
    parser.add_argument("-U","--LearningRate", required=False, type=str, default="3e-4", help="the learning rate")
    parser.add_argument("-V","--Val_size", type=float, default=0.1, help='The ratio of sample used for validation')
    parser.add_argument("--L1_Lambda", required=False, type=float, default="0", help="the weight to regulate the L1 loss")
    parser.add_argument("--L2_Lambda", required=False, type=float, default="1e-4", help="the weight to regulate the L2 loss")
    parser.add_argument("--progress_bar", required=False, type=str, default="True", help="boolen, whether to show progress bar")
    parser.add_argument("--repeat", required=False, type=int, default=1, help="Repeat index (1-based). Sets the random seed for finetune oligo sampling and is encoded in the output filename.")
    args = parser.parse_args()

    lr = float(args.LearningRate)

    ## Mode ##
    if args.Mode == 'Train':
        to_train = to_write_y = to_predict = True
        to_baseline = True

    elif args.Mode == 'Evaluate':
        to_train = False
        to_baseline = True
        to_write_y = to_predict = True
    
    elif args.Mode == 'Evaluate_only':
        to_predict = True
        to_baseline = False
        to_train = to_write_y = False
        
    elif args.Mode == 'Write_Y':
        to_train = to_predict = False
        to_write_y = True

    elif args.Mode == 'Baseline':
        to_train = to_predict = False
        to_write_y = to_baseline = True
    else:
        raise ValueError("Invalide action combination")

    # N=0 means no finetuning — evaluate pretrained baseline only
    if int(args.N_Finetune) == 0:
        to_train = to_predict = False
        to_write_y = to_baseline = True

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
    
    print(f"Runing {experiments} using cuda: {gpu_device}")
    pth_save_dir = os.path.join(PATH.pth_dir, f"v5_fintune_{args.Model_Class}_{args.Data_transform}_C{args.read_cutoff}")
    for DIR in [trainer_log, data_dir, pth_save_dir, save_dir]:
        check_dir(DIR)
    # Temp Theta file
    date = time.strftime("%B%d")
    pth_save_path = pj(pth_save_dir, f"{experiments}_V{args.Val_size}_N{args.N_Finetune}_rep{args.repeat}_{args.Fix_params}_L2{args.L2_Lambda}")
    os.makedirs(pth_save_path, exist_ok=True)

    processed_df = pd.read_csv(csv_path).query("`in_LdGen` == True").astype({"Count":"int"})
    processed_df = processed_df.query("`Strand` == 'FORWARD' & `Identifier` != 'Not Present' ")

    # filter to oligos with valid NGG PAM (indelgentarget asserts seq[pam_idx+1:pam_idx+3] == "GG")
    def _has_ngg_pam(oligo_id):
        _, refseq, pamsite, _ = ref_lookup[oligo_id]
        return refseq[pamsite + 1 : pamsite + 3].upper() == "GG"
    valid_oligos = [o for o in processed_df["OligoID"].unique() if _has_ngg_pam(o)]
    n_before = processed_df["OligoID"].nunique()
    processed_df = processed_df[processed_df["OligoID"].isin(valid_oligos)]
    print(f"Oligos with NGG PAM: {len(valid_oligos)}/{n_before} (dropped {n_before-len(valid_oligos)})")

    # split training and testing data
    Train_Oligos, Val_Oligos, Test_Oligos = reader.get_Train_Val_Test(
        e = experiments,
        df=processed_df,
        test_oligo_file = os.path.join(PATH.main_dir, args.test_oligos),
        seed = 0,
        threshold = args.read_cutoff,
        count_predictable=False
        )

    # filter test oligos to those with valid NGG PAM
    n_test_before = len(Test_Oligos)
    Test_Oligos = [o for o in Test_Oligos if o in ref_lookup and _has_ngg_pam(o)]
    print(f"Test oligos with NGG PAM: {len(Test_Oligos)}/{n_test_before} (dropped {n_test_before-len(Test_Oligos)})")
    # load predefined finetuning set
    if "R" in args.N_Finetune:
        # random 
        size = int(args.N_Finetune[1:])
        Finetune_Oligos = np.load(f'{PATH.main_dir}/results/random_finetune_oligo_list/BOB_{size}_finetune_list.npy')

    else:
        size = int(args.N_Finetune)
        Finetune_df=pd.read_csv(f"{PATH.main_dir}/results/Finetune_OligoIndex_Jul19.csv", index_col=0)
        finetune_set = 'FinetuneSet_n%s'%args.N_Finetune                        # For example, select 30 oligos
        

        ## ONLY USE THE RANDOM

        np.random.seed(args.repeat)
        Finetune_Oligos = np.random.choice(Train_Oligos, size=size, replace=False)


    # train-val for finetuning
    val_size = int(args.Val_size * size)                               # num. of oligos for finetuning validatoin 
    Finetune_Oligos_val = np.random.choice(Finetune_Oligos, size=val_size)
    Finetune_Oligos_train = [oligo for oligo in Finetune_Oligos if oligo not in Finetune_Oligos_val]



    # DATA TRANSFORMATION
    if args.Data_transform == "identity":
        transform = lambda x: x
        n_features = ndel + nins + nshare
    elif args.Data_transform == "interaction":
        transform = lambda x: alignmap.interaction_transform(x, ndel, nins)
        n_features = ndel + nins + nshare + math.comb(ndel,2) + math.comb(nins,2) + nshare*(ndel+nins)
    elif ":" in args.Data_transform:
        # decay transform
        raise ValueError("Invalid transform name")
    else:
        raise ValueError("Invalid transform name")
    
    
    # Modeling and Training
    model_class = eval("models.%s" %args.Model_Class)
    model_parsms = dict(inputsize=n_features, outputsize=1,  lr=lr,
                        L1_lambda=args.L1_Lambda, L2_lambda=args.L2_Lambda)
    if 'Deep' in args.Model_Class:
        model_parsms['hidden'] = hidden
        
    model = model_class(**model_parsms)
    
    if args.Pretrain is None:
        raise ValueError("Finetuning jobs must start with a trained model")
    elif not os.path.exists(args.Pretrain): 
        raise FileNotFoundError("Invalid path to Pretrained model")
    else:
        model = model_class.load_from_checkpoint(args.Pretrain, map_location='cpu')
        # ckpt = torch.load(args.Pretrain)
        # model.load_state_dict(ckpt['state_dict'])
        last_layer_w = model.del_regressor[2].weight.data
        model.del_regressor[2].weight.data += torch.randn_like(last_layer_w)

        last_layer_b = model.del_regressor[2].bias.data
        model.del_regressor[2].bias.data += torch.randn_like(last_layer_b)

    if args.Fix_params is not None:
        for p in eval(f"model.{args.Fix_params}").parameters():
            p.require_grad = False

        print(args.Fix_params, "is fixed")

    # dataset    
    normalize = ('Multinomial' not in args.Model_Class) & ('weight' not in args.Model_Class)
    # normalize = 'weight' not in args.Model_Class 
    feature_extraction_fn = lambda label_df, refseq, cutsite : alignmap.ST_decayfeat_v5(label_df, refseq, cutsite, k1, k2, h)

    Test_DS = reader.ST_dataset(Test_Oligos,processed_df, experiments,
                                read_data_fn = read_data,
                                transformation=transform,
                                feat_ext_fn = feature_extraction_fn,
                                normalize=normalize)
    Test_DL = DataLoader(Test_DS, shuffle=False, batch_size=1, num_workers=num_workers, collate_fn=my_collect_fn)

    if to_train:
        Train_DS = reader.ST_dataset(Finetune_Oligos_train,processed_df, experiments,
                              read_data_fn = read_data,
                              transformation=transform,
                              feat_ext_fn = feature_extraction_fn,
                              normalize=normalize)
        Val_DS = reader.ST_dataset(Finetune_Oligos_val,processed_df, experiments,
                                   read_data_fn = read_data,
                                   transformation=transform ,
                                   feat_ext_fn = feature_extraction_fn,
                                   normalize=normalize)
        Train_DL = DataLoader(Train_DS, shuffle=True, batch_size=3, num_workers=num_workers, collate_fn=my_collect_fn)
        Val_DL = DataLoader(Val_DS, shuffle=False, batch_size=3, num_workers=num_workers, collate_fn=my_collect_fn)

    trainer = pl.Trainer(
			# auto_lr_find=True,
            accelerator=device,
            # fast_dev_run=True,
            enable_progress_bar=eval(args.progress_bar),
			default_root_dir=pth_save_path,
            devices = [gpu_device],
			max_epochs = args.Rounds,
			callbacks=[ callbacks.ModelCheckpoint(filename='{epoch}-{val_cre:.8f}',
                                                  monitor="val_cre", mode="min", save_top_k=2),
                        callbacks.EarlyStopping(monitor="val_cre", mode="min", patience=20),])
    
    Forecast_Y = pj(pth_save_path, "ForeCast_TestY.pkl")
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
        # if args.Pretrain is not None:
        #     ckpt_abspath = os.path.join(PATH.main_dir, args.Pretrain)
        # else:
        Forecast_Y = pj(pth_save_path, "ForeCast_TestY.pkl")
        ckpt_abspath = my_utils.find_ckpt(pj(pth_save_path, 'lightning_logs'))
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


        ## evaluate in the test set
        f = open(Forecast_Y, 'rb')
        Y_lookup = pkl.load(f)  # forecast : ST
        f.close()
        write_evaluate_json(Y_lookup, pred_lookup,'', ckpt_abspath, args, prefix="")

    if to_baseline:
        #  to generate baseline for the pretrained model
        Forecast_Y = pj(pth_save_path, "ForeCast_TestY.pkl")
        pmodel = model_class.load_from_checkpoint(args.Pretrain, map_location='cpu')

        pmodel.eval()
        predict_y = trainer.predict(pmodel, Test_DL)

        if isinstance(predict_y[0], list):
            predict_y = sum(predict_y, [])  # to join lists 
        pred_lookup = {o:predict_y[i].cpu().numpy() for i,o in enumerate(Test_Oligos)}
        print(len(pred_lookup))
        TestPred = Forecast_Y.replace("ForeCast_TestY.pkl", "Pretrained_Baseline_TestPred.pkl")
        pred_f = open(TestPred, 'wb')
        pkl.dump(pred_lookup, pred_f)
        pred_f.close()
        print("prediction writed to %s" %TestPred)


        ## evaluate in the test set
        f = open(Forecast_Y, 'rb')
        Y_lookup = pkl.load(f)  # forecast : ST
        f.close()

        write_evaluate_json(Y_lookup, pred_lookup, '', args.Pretrain, args, prefix="Baseline")

        

        