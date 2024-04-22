import os, sys, subprocess, argparse, time, PATH, shutil, math 
import my_utils
sys.path.append(PATH.main_dir)
# from features import readFeaturesData
# from model import calcThetaX, computeRegularisers,  assessFit
import numpy as np
import pandas as pd 
import torch
torch.set_num_threads(4)
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import callbacks 
import pickle as pkl
from Bio import SeqIO
from scripts.reimplement_forecast import Forecast_model
from inDecay import alignmap, models, reader
from tqdm.contrib.concurrent import process_map
from scripts.STfeatv2_inDecay_finetune import check_dir, readFeaturesData, find_ckpt, interaction_transform, decay_transform

num_workers = 12

ndel = 9 
nins = 8
nshare = 1

# model params
k1 = 0.5 
k2 = 0.6
h = 1.3
hidden = [128, 64]
L2_Lambda = 1e-4
L1_Lambda = 0
lr = 3e-4

# Torch Device
device = 'gpu' if torch.cuda.is_available() else 'cpu'

# some path and requisite files
pj = os.path.join
SelfTarget_data_dir = PATH.data_dir
high_dir = PATH.high_dir


def get_sanger_training():
    """
    get all the sanger sequencing samples
    """
    # find the dir
    indelgen_dir = os.path.join(PATH.data_dir, "Sanger_training")
    assert os.path.exists(indelgen_dir), "the sanger sequencing data is not correctly deposited"

    # look for processed tables
    dfs  = [file for file in os.listdir(indelgen_dir)  if file.endswith("_SelfTarget.csv")]
    genes = np.array([f.split("_SelfTarget")[0] for f in dfs])
    assert len(dfs) != 0, f"no samples found under {indelgen_dir}\nplese make sure you have put data in the right dir"

    # create a lookup table for easy retrival of reference
    def_table= os.path.join(indelgen_dir, "Sanger_ref.csv")
    assert os.path.exists(def_table)
    gene_ref_tab = pd.read_csv(def_table)

    gene_ref_dict = {}
    for i, row in gene_ref_tab.iterrows():
        gene_ref_dict[row['gene']] = '-', row['ref'], 42, "FORWARD"

    return genes, gene_ref_dict


def save_spliting(genes, kf_indeces):
    """ save the training genes and testing genes    

    Args:
        genes (list): result of func get_sanger_training
        kf_indeces (list): list of tuple (idxs)
    """
    gene_by_fold = [[','.join(genes[idx]) for idx in idxs] for idxs in kf_indeces]
    split_df = pd.DataFrame(gene_by_fold,
                            columns=['Train_gene', 'Val_gene', 'Test_gene'])
    # save
    split_df.to_csv("result/Sanger_spliting.csv") # save to result (will updated in github repo)
    split_df.to_csv(os.path.join(indelgen_dir, f"Sanger_spliting_{date}.csv")) # backup locally

def read_sanger_data(gene, gene_ref_lookup, exp):
    """
    read the identifier and read count of the given genes
    """
    # read table of the gene
    idgen_df_path = os.path.join(PATH.data_dir, "Sanger_training", f"{gene}_SelfTarget.csv")
    label_df = pd.read_csv(idgen_df_path).query("Identifier != 'Not Present'")

    # normalize
    total_sum = label_df['Count'].sum()
    label_df['n_coevent'] = label_df['loc'].apply(lambda x: x.count("("))
    label_df['Frac Sample Reads'] = label_df['Count']/total_sum
    return label_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser("`inDecay` few shot learning on Sanger sequencing data")
    parser.add_argument("-G","--GPU_devices", type=int, default=None, help='The gpu to use')
    parser.add_argument("-P","--Pretrain", required=False, type=str, default=None, help="the pretrained parameter theta")
    parser.add_argument("-d","--L2_Lambda", required=False, type=float, default=3e-5, help="the regularization strength")
    parser.add_argument("-L","--L1_Lambda", required=False, type=float, default=0, help="the regularization strength")
    parser.add_argument("-M","--Model_Class", required=False, type=str, default=0, help="the regularization strength")
    parser.add_argument("-D","--Data_transform", required=False, type=str, default="interaction", help="the name of data transformation")
    parser.add_argument("-T","--test_split", required=True, type=int, help='which fold to used to split train test genes')
    parser.add_argument("-O","--Mode", required=False, type=str, default="Train", help="the action of this script, can be `Train`, `Evaluate`, `Evaluate_only` and `Write_Y`")
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
    else:
        raise ValueError("Invalide action combination")


    # if args.GPU_devices is not None:
    gpu_device= args.GPU_devices
    print(f"Runing  using cud: {gpu_device}")
    
    # Temp Theta file
    date = time.strftime("%B%d")
    indelgen_dir = os.path.join(PATH.data_dir, "Sanger_training")

    # get gene list
    # and also a gene to table look up
    genes, gene_ref_dict  = get_sanger_training()

    # split genes
    kf_indeces = reader.get_Sanger_train_test(genes)
    train_idx, val_idx, test_idx = kf_indeces[args.test_split]
    train_genes = genes[train_idx]
    val_genes = genes[val_idx]
    test_genes = genes[test_idx]


    # save train test splits
    if not os.path.exists("result/Sanger_spliting.csv"):
        save_spliting(genes, kf_indeces) # create a 2-dimensional list  then to df
    else:
        # sanity check
        spliting_df = pd.read_csv("result/Sanger_spliting.csv")
        assert len(kf_indeces) == spliting_df.shape[0], 'the num of fold has changed'
        
        record_test_genes = spliting_df.iloc[args.test_split]['Test_gene']
        record_test_genes = np.array(record_test_genes.split(","))
        assert np.all(record_test_genes == np.array(test_genes)), "the testing gene has changed"
        

    # some checkpoint settings
    save_dir = pj(high_dir, "Sanger")
    exp_name = args.Pretrain.split("ST_June_2017_")[-1].split("_LV7A_DPI7")[0]

    pth_save_dir = os.path.join(PATH.pth_dir, f"ST_featv2_{args.Model_Class}_{args.Data_transform}")
    pth_save_path = pj(pth_save_dir, f"Sanger_{exp_name}_{len(kf_indeces)}fold_{args.test_split}")
    
    for DIR in [SelfTarget_data_dir, pth_save_dir, high_dir, save_dir]:
        check_dir(DIR)  


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
    model_class = eval("inDecay.%s"%args.Model_Class)
    model_parsms = dict(inputsize=n_features, outputsize=1,  lr=lr,
                        L1_lambda=L1_Lambda, L2_lambda=L2_Lambda)
    if 'Deep' in args.Model_Class:
        model_parsms['hidden'] = hidden
        
    model = model_class(**model_parsms)
    if args.Pretrain is not None:
        model = model_class.load_from_checkpoint(args.Pretrain)


    # dataset    
    normalize = 'Multinomial' not in args.Model_Class 
    feature_extraction_fn = lambda label_df, refseq, cutsite : alignmap.ST_decayfeat_v2(label_df, refseq, cutsite, k1, k2, h)

    Train_DS = reader.ST_dataset(train_genes, gene_ref_dict, "Sanger", 
                          read_data_fn = read_sanger_data,
                          transformation=transform,
                          feat_ext_fn = feature_extraction_fn,
                          normalize=normalize)
    Val_DS = reader.ST_dataset(val_genes, gene_ref_dict, "Sanger", 
                               read_data_fn = read_sanger_data,
                               transformation=transform , 
                               feat_ext_fn = feature_extraction_fn,
                               normalize=normalize)
    Test_DS = reader.ST_dataset(test_genes, gene_ref_dict, "Sanger", 
                                read_data_fn = read_sanger_data,
                                transformation=transform,
                                feat_ext_fn = feature_extraction_fn,
                                normalize=normalize)

    Train_DL = DataLoader(Train_DS, shuffle=True, batch_size=1, num_workers=num_workers)
    Val_DL = DataLoader(Val_DS, shuffle=False, batch_size=1, num_workers=num_workers)
    Test_DL = DataLoader(Test_DS, shuffle=False, batch_size=1, num_workers=num_workers)

    trainer = pl.Trainer(
			auto_lr_find=True,
            accelerator=device,
            # fast_dev_run=True,
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

        # TODO: comment this later
        # Forecast_Y = "/home/wergillius/data/CRISPR_data/pl_trainer_log/ST_featv2_ST_DeepDecay_interaction/Sanger_Y.pkl"
        # test_genes = genes 

        if not os.path.exists(Forecast_Y):
            get_identifiers = lambda gene : read_sanger_data(gene, gene_ref_dict, "Sanger")[['Identifier', 'Frac Sample Reads']].values

            Y_ls = [get_identifiers(gene) for gene in test_genes]
            # Y_ls = process_map(get_identifiers, Test_Oligos, max_workers=8)
            Y_lookup = {o:Y_ls[i] for i,o in enumerate(test_genes)}

            handle = open(Forecast_Y, 'wb')
            pkl.dump(Y_lookup, handle)    
            handle.close()
            print('finished write to ',  Forecast_Y)


    if to_predict:
        
        ckpt_abspath = find_ckpt(pj(pth_save_path, 'lightning_logs'))

        assert os.path.exists(ckpt_abspath)

        try:
            model.eval()
        except:
            model = model_class.load_from_checkpoint(ckpt_abspath).to(gpu_device)
            model.eval()
        model.eval()
        
        
        # TODO: comment this 2 lines later
        # DS = reader.ST_dataset(genes, gene_ref_dict, "Sanger", 
        #                         read_data_fn = read_sanger_data,
        #                         transformation=transform,
        #                         feat_ext_fn = feature_extraction_fn,
        #                         normalize=normalize)
        # Test_DL = DataLoader(DS, shuffle=False, batch_size=1, num_workers=num_workers)

        predict_y = trainer.predict(model, Test_DL)
        pred_lookup = {o:predict_y[i].cpu().numpy() for i,o in enumerate(test_genes)} # type: ignore

        TestPred = ckpt_abspath.replace(".ckpt", "TestPred.pkl")
        
        # TODO: comment this 2 lines later
        # exp_name = args.Pretrain.split("ST_June_2017_")[-1].split("_LV7A_DPI7")[0]
        # TestPred = os.path.join(pth_save_dir, exp_name+"_TestPred.pkl")

        pred_f = open(TestPred, 'wb')
        pkl.dump(pred_lookup, pred_f)
        pred_f.close()
        print("prediction writed to %s" %TestPred)
    
    


