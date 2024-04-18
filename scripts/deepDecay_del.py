import os, sys, subprocess, argparse, time, PATH
sys.path.append(PATH.main_dir)
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
from inDecay import reader
from inDecay import models

# Torch Device
device = 'gpu' if torch.cuda.is_available() else 'cpu'

# some path and requisite files
pj = os.path.join
SelfTarget_data_dir = PATH.data_dir
high_dir = PATH.high_dir
pth_save_dir = os.path.join(PATH.pth_dir, "ForeCast")
reference_path = pj(SelfTarget_data_dir, "SelfTarget_NewScaffold.fasta")

def check_dir(path):
    """
    check the existence of 3 order of parent dir
    """
    dir_1st = os.path.dirname(path)
    dir_2nd = os.path.dirname(dir_1st)
    dir_3rd = os.path.dirname(dir_2nd)
    for DIR in [dir_3rd, dir_2nd, dir_1st]:
        if not os.path.exists(DIR):
            os.mkdir(DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("The script to extract SelfTarget proccessed txt file and map to Lindel classes")
    # parser.add_argument("--Set", required=True, type=str, help="either `TestSet1` or `TestSet2`")
    parser.add_argument("-E","--experiment", type=str, required=True, help='The dir name of dataset')
    parser.add_argument("-C","--read_cutoff", type=int, default=500, help='The threshold of total count. Only Guides having total read count over this threshold are used')
    parser.add_argument("-T","--test_oligos", type=str, default="result/test_set_oligo_Feb2.txt", help='The file deciding which oligos are used in the training set')
    parser.add_argument("-G","--GPU_devices", type=str, default="0", help='The gpu to use')
    parser.add_argument("-P", "--Pretrain", required=False, type=str, default=None, help="the ckpt of pretrained model")
    parser.add_argument("-L","--L1", required=False, type=float, default=3e-4, help="the L1 regularization strength")
    parser.add_argument("-R","--L2", required=False, type=float, default=1e-4, help="the L2 regularization strength")
    args = parser.parse_args()

    # some save file settings
    experiments = args.experiment
    Cellline = experiments.split("_")[3]
    rep = experiments.split("_")[4]
    save_dir = pj(high_dir, experiments)
    csv_path = pj(save_dir,f"{Cellline}_{rep}.csv")

    # Temp Theta file
    date = time.strftime("%B%d")
    if args.Pretrain is not None:
        assert os.path.exists(args.Pretrain), "pretrain file not found"


    # instantiate the DataModule 
    test_oligo_abspath = pj(PATH.main_dir, args.test_oligos)
    NHEJ_feat_DM = reader.inDecay_DataModule(experiments, 'del', 
                                   DS_class=reader.nhej_feat_del_DS,         # How to control this ?
                                   test_oligo_file=test_oligo_abspath,
                                   threshold=args.read_cutoff) 
    
    # updatable_param = "kapp1, sigma, mu" # c not updated
    # del_model = inDecay.nhej_p_profiler(lr=1e-3, 
    #                                     kappa1=1.1, sigma=4, mu=3, 
    #                                     kappa2=0.2, c=2, 
    #                                     updatable_param=updatable_param,
    #                                     L1_lambda=args.L1,
    #                                     L2_lambda=args.L2,
    #                                     optim_class= "Adam"
    #                                     )
    
    del_model = models.Deep_inDecay(
        hidden =[16],
        lr=3e-4, 
        n_mmejfeat = 3,
        n_nhejfeat = 6,
        L1_lambda = 0,
        L2_lambda = 1e-9,
    )

    log_dir = pj(PATH.pth_dir, 'DeepDecay_debugging_', experiments)
    check_dir(log_dir)
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=log_dir,
        filename='{epoch}-{val_loss:.2f}'
        )

    trainer = pl.Trainer(
        accelerator = device,
        devices = [int(s) for s in args.GPU_devices.split(",")],
        auto_lr_find=True,
        default_root_dir=log_dir,
        max_epochs=100, 
        callbacks=[checkpoint_callback,
                callbacks.EarlyStopping(monitor="val_cre", mode="min", patience=15)]
        )

    NHEJ_feat_DM.setup(stage='fit')
    Train_loader = NHEJ_feat_DM.train_dataloader()
    Val_loader = NHEJ_feat_DM.val_dataloader()
    trainer.fit(del_model, train_dataloaders=Train_loader, val_dataloaders=Val_loader)
    trainer.validate(del_model, NHEJ_feat_DM.val_dataloader())

    NHEJ_feat_DM.setup(stage='test')
    Test_loader = NHEJ_feat_DM.test_dataloader()
    test_result = trainer.validate(del_model, dataloaders=Test_loader, ckpt_path='best')
    test_result = trainer.predict(del_model, dataloaders=Test_loader, ckpt_path='best')
    del_ckpt = trainer.ckpt_path
    print("saved to", del_ckpt)


