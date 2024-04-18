#!/usr/bin/env python
import os, PATH,argparse
import numpy as np
import scanpy as sc
from inDecay import alignmap, models
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import callbacks 


# Define useful functions

def one_hot_60bp(seq):
    pad_len = 60 - len(seq)
    Oh_X = alignmap.one_hot(seq)
    one_hot_seq = np.concatenate([Oh_X, np.zeros((pad_len,4))], axis=0)
    return one_hot_seq

# Load data
def trunc_seq(row):
    ref = row['Refseq']
    cut = row['Pamsite'] - 3

    seq = ref[cut - 30:cut + 30] 
    seq += 'N' * (60 - len(seq))

    return seq

def read_Anndata(experiment, read_cutoff):
    """
    read adata
    """
    exp = experiment
    Cellline = exp.split("_")[3]
    rep = exp.split("_")[4]
    rep2 = exp.split("_")[5] if "800x" in exp else exp.split("_")[4]

    h5ad_file = os.path.join("/home/wergillius/data/CRISPR_data/Adata", f"{Cellline}_{rep2}_557class.h5ad")
    adata = sc.read_h5ad(h5ad_file)
    adata.obs['60bp'] = adata.obs.apply(trunc_seq, axis=1)

    # filtering samplings
    sc.pp.filter_cells(adata, min_counts=read_cutoff)
    sc.pp.normalize_total(adata, target_sum=1)

    ##   by ratio
    adata.obs['del_ratio'] = adata.X[:,:-21].sum(axis=1)
    adata.obs['ins_ratio'] = adata.X[:,-21:].sum(axis=1)
    adata = adata[adata.obs.ins_ratio > 0].copy()
    adata = adata[adata.obs.del_ratio > 0].copy()

    ##   by strand
    if "Strand" in adata.obs_keys():
        adata = adata[adata.obs.Strand  == 'FORWARD'].copy()
    else:
        adata = adata

    return adata

def read_ratio_data(experiment, read_cutoff):
    """_summary_

    Args:
        experiment (str): the experiment
        read_cutoff (int): minimum total count

    Returns:
        X_Y pair of train, val, test,
    """
    adata = read_Anndata(experiment, read_cutoff)

    # train test
    trainval = adata[adata.obs.TestSet == False].copy()
    test_ad = adata[adata.obs.TestSet == True].copy()
    val_ad = sc.pp.subsample(trainval, fraction=0.1, copy=True)
    train_ad = trainval[~trainval.obs_names.isin(val_ad.obs_names)]

    # generate X
    X_Y = []
    for ad in [train_ad, val_ad, test_ad]:
        # DelX, insX, ratioX = my_utils.get_Lindel_input(ad.obs['60bp'].values)
        X_ = ad.obs['60bp'].apply(one_hot_60bp)
        Y = np.stack(ad.obs[['del_ratio','ins_ratio']].values)
        X = np.stack(X_)
        X_Y.append([ torch.from_numpy(X).float(), torch.from_numpy(Y).float() ])

    return X_Y   # a list of X-Y pair


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Lindel model on different cell types")
    parser.add_argument("-E","--experiment", type=str, required=True, help='The dir name of dataset')
    parser.add_argument("-C","--read_cutoff", type=int, default=500, help='The threshold of total count. Only Guides having total read count over this threshold are used')
    parser.add_argument("-T","--test_oligos", type=str, default="result/test_set_oligo_Feb2.txt", help='The file deciding which oligos are used in the training set')
    parser.add_argument("-G","--GPU_devices", type=int, default="0", help='The gpu to use')
    args = parser.parse_args()


    # data    
    ratio_XY = read_ratio_data(args.experiment, args.read_cutoff)
    train_ds, val_ds, test_ds = [TensorDataset(*x_y) for x_y in ratio_XY]
    train_dl = DataLoader(train_ds, batch_size=22, shuffle=True, num_workers=8)
    val_dl = DataLoader(val_ds, batch_size=22, shuffle=False, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=22, shuffle=False, num_workers=4)


    # Hyper-params 
    out_channel = 32
    hidden = 128
    lr = 3e-4
    L1 = 0
    L2 = 1e-5
    # optimizer = 'LBFGS'
    optimizer = "RMSprop"

    # define model
    # model = inDecay.inDecay_ratio(lr=lr, L1_lambda=L1, L2_lambda=L2, optim_class=optimizer)
    model = models.DeepDecay_ratio(out_channel=out_channel, hidden=hidden,
                    lr=lr, L1_lambda=L1, L2_lambda=L2, optim_class=optimizer)
    
    # set up trainer
    device = 'gpu' if torch.cuda.is_available() else 'cpu' 
    workdir = os.path.join(PATH.pth_dir, "DeepDecay_ratio", args.experiment)

    if args.GPU_devices is not None:
        gpu_device= args.GPU_devices
    else:
        gpu_device = {
        "ST_June_2017_BOB_LV7A_DPI7":0,
        "ST_June_2017_CHO_LV7A_DPI7":1,
        "ST_June_2017_E14TG2A_LV7A_DPI7":2,
        "ST_June_2017_HAP1_LV7A_DPI7":3,
        "ST_June_2017_K562_800x_LV7A_DPI7":3,
        }[args.experiments]

    trainer = pl.Trainer(
			auto_lr_find=True,
            accelerator=device,
            # fast_dev_run=True,
			default_root_dir=workdir,
            devices = [gpu_device],
			max_epochs=100,
			callbacks=[ callbacks.ModelCheckpoint(filename='{epoch}-{val_loss:.8f}',
                                                  monitor="val_loss", mode="min", save_top_k=2),
                        callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=20),])
    
    model.train();
    trainer.fit(model, train_dl, val_dataloaders=val_dl)
    
    model.eval();
    print("\ncheckpoint saved to : %s \n"%trainer.ckpt_path)
    trainer.validate(model, test_dl, ckpt_path="best")
    trainer.test(model, test_dl, ckpt_path="best")