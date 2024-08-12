import os, sys, subprocess, argparse, time, shutil, math 
import numpy as np
import pandas as pd 
import torch
from torch import nn

torch.set_num_threads(4)
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import callbacks 

from inDecay import my_utils, alignmap, ratio_models, reader, PATH
sys.path.append(PATH.main_dir)
from tqdm.contrib.concurrent import process_map
from STfeatv2_inDecay import  read_data, ref_lookup

to_train = True
to_predict = True
to_write_y = True
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
data_dir = PATH.data_dir

reference_path = pj(data_dir, "SelfTarget_NewScaffold.fasta")


def compute_ratios(Oligo, processed_df):

    Guide, refseq, pamsite, Strand = ref_lookup[Oligo]
    cutsite = int(pamsite) - 3
    
    label_df = read_data(Oligo, processed_df, None)
    mh_mask, label_df = alignmap.label_mh(refseq, cutsite, label_df)
    
    # label indel type
    label_df['indel_type'] = label_df.Identifier.apply(lambda x: my_utils.tokFullIndel(x)[0])

    # compute ins del ratio
    del_ins_dict = label_df.groupby("indel_type").agg({"Frac Sample Reads":"sum"}).to_dict()['Frac Sample Reads']

    ins_ratio = del_ins_dict['I']
    ins_ratio = ins_ratio if ins_ratio>0 else 0.001  # for numerical stability
    del_ratio = del_ins_dict['D']

    # compute mh ratio
    del_df = label_df.query("`indel_type` == 'D'")
    mh_del_frac = del_df.query("`mh_length` > 0")['Frac Sample Reads'].sum()

    mh_ratio  = mh_del_frac / del_ratio if del_ratio>0 else 0.001


    return ins_ratio, mh_ratio


def compute_dig_map(Oligo, matrix_size=50, score=1, panelty=-1):

    Guide, refseq, pamsite, Strand = ref_lookup[Oligo]
    cutsite = int(pamsite) - 3

    left = -1 * matrix_size
    filtered_map = alignmap.construct_diagonal_map(refseq, score=score, cut_site=cutsite, panelty=panelty)[left:, :matrix_size]
    filtered_map = torch.from_numpy(filtered_map).float().unsqueeze(0)
    # C, H, W

    N, dim1, dim2 = tuple(filtered_map.shape)
    right_pad = max(matrix_size - dim2, 0)
    upper_pad = max(matrix_size - dim1, 0)
    padding = (
        0, right_pad,  # pad at right for dim -1
        upper_pad,0,   # pad at left for dim -2
        0,0            # no padding for dim -3
    )
    filtered_map = nn.functional.pad(filtered_map, padding).numpy()

    return filtered_map[0]  # remove channel




class ratio_dataset(Dataset):
    def __init__(self, Oligos, ref_lookup, ins_lookup , mh_lookup, matrix_size):
        super().__init__()

        self.Oligos = Oligos
        self.ref_lookup = ref_lookup
        self.ins_lookup = ins_lookup
        self.mh_lookup = mh_lookup

        self.matrix_size = matrix_size

    def __len__(self):
        return len(self.Oligos)

    def __getitem__(self,i):
        oligo = self.Oligos[i]
        Guide, refseq, pamsite, Strand = self.ref_lookup[oligo]
        cutsite = int(pamsite) - 3 

        assert Strand == 'FORWARD'

        # the input : 1.onehot encoded guide, 2.the dignonal filtered substitution matrix
        guide_oh = alignmap.one_hot(Guide).T[:,  :20] # L,C -> C,L 
        guide_oh = torch.from_numpy(guide_oh).float()

        if guide_oh.shape[1] <20:
            oh_padding = (
                20-guide_oh.shape[1],0,  # dim -1,pad left
                0,0                      # no padding for dim -2
            )
            guide_oh = nn.functional.pad(guide_oh, oh_padding)

        left = -1 * self.matrix_size
        filtered_map = alignmap.construct_diagonal_map(refseq, cut_site=cutsite,panelty=0)[left:, :self.matrix_size]
        filtered_map = torch.from_numpy(filtered_map).float().unsqueeze(0)
        # C, H, W

        N, dim1, dim2 = tuple(filtered_map.shape)
        right_pad = max(self.matrix_size - dim2, 0)
        upper_pad = max(self.matrix_size - dim1, 0)
        padding = (
            0, right_pad,  # pad at right for dim -1
            upper_pad,0,   # pad at left for dim -2
            0,0            # no padding for dim -3
        )
        filtered_map = nn.functional.pad(filtered_map, padding)

        # return the ratio
        ins_ratio = self.ins_lookup[oligo]
        mh_ratio = self.mh_lookup[oligo]

        return guide_oh, filtered_map, ins_ratio, mh_ratio



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Lindel model on different cell types")
    parser.add_argument("-E","--experiment", type=str, required=True, help='The dir name of dataset')
    parser.add_argument("-P", "--matrix_size", type=int, default=50, help="The matrix size used as the second input to the ratio model")
    parser.add_argument("-C","--read_cutoff", type=int, default=500, help='The threshold of total count. Only Guides having total read count over this threshold are used')
    parser.add_argument("-T","--test_oligos", type=str, default="data/test_set_oligo_Feb2.txt", help='The file deciding which oligos are used in the training set')
    parser.add_argument("-G","--GPU_devices", type=int, default="0", help='The gpu to use')
    parser.add_argument("--lr", type=float, default=3e-4, help='The learning rate')
    args = parser.parse_args(["-E" ,"ST_June_2017_BOB_LV7A_DPI7"])
    
    # some save file settings
    experiments = args.experiment
    Cellline = experiments.split("_")[3]
    rep = experiments.split("_")[4]
    trainer_log = pj(PATH.main_dir, 'pl_trainer_log')
    save_dir = pj(data_dir, 'processed_df')
    csv_path = pj(save_dir,f"{Cellline}_{rep}.csv")

    # data    
    processed_df = pd.read_csv(csv_path).query("`in_LdGen` == True").astype({"Count":"int"})
    processed_df = processed_df.query("`Strand` == 'FORWARD'")

    Train_Oligos, Val_Oligos, Test_Oligos = reader.get_Train_Val_Test(
        processed_df, 
        test_oligo_file = os.path.join(PATH.main_dir, args.test_oligos),
        seed = 0,
        threshold = args.read_cutoff
        )
    
    # precompute the ratios
    Oligos = Train_Oligos + Val_Oligos + Test_Oligos.tolist()


    base_dist_M = np.sqrt(np.broadcast_to(np.arange(0,50), shape=(50,50)).T**2 + \
                np.broadcast_to(np.arange(49,-1,-1), shape=(50,50))**2)
    
    base_dist_M = base_dist_M/base_dist_M.max()


    def precompute_fn(x): 
        return compute_ratios(x, processed_df)

    def compute_decay_sum1_P50(Oligo):
        filtered_map = compute_dig_map(Oligo, score=1, panelty=-1)
        filtered_map_0 = compute_dig_map(Oligo, score=1, panelty=0)
        filtered_map_2 = compute_dig_map(Oligo, score=2, panelty=-1)

        mh_intensity_sum = []
        decay_term = [
            lambda x : np.power(3, x),
            lambda x : np.power(2, x),
            lambda x : x**3
        ]
        for decay_fn in decay_term:
            dist_M = decay_fn(base_dist_M)

            for filter_M in [filtered_map, filtered_map_0, filtered_map_2]:
                summ  = np.multiply(filter_M, dist_M).sum()
                mh_intensity_sum.append(summ)        
        
        return mh_intensity_sum


    def compute_decay_sum1_P20(Oligo):
        filtered_map = compute_dig_map(Oligo, score=1, panelty=-1, matrix_size=20)
        filtered_map_0 = compute_dig_map(Oligo, score=1, panelty=0, matrix_size=20)
        

        base_dist_M = base_dist_M[-20:,20]
        mh_intensity_sum = []
        decay_term = [0.2,0.5,0.8]
        for decay in decay_term:
            dist_M = base_dist_M
            dist_M /= dist_M.max()
            dist_M = 1 - dist_M

            dist_M = np.power(dist_M,4)

            for filter_M in [filtered_map, filtered_map_0]:
                summ  = np.multiply(filter_M, dist_M).sum()
                mh_intensity_sum.append(summ)        
        
        return mh_intensity_sum
    
    # process_map(precompute_fn,Oligos,max_workers=8, chunksize=500)
    ratios = np.stack(process_map(precompute_fn, Oligos,max_workers=20, chunksize=10), dtype=np.float32)

    Decay_sum = np.stack(process_map(compute_decay_sum1_P50, Oligos, max_workers=20, chunksize=10), dtype=np.float32)

    # linear regression
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(Decay_sum[:6694], ratios[:6694,1])
    model.score(Decay_sum[:6694], ratios[:6694,1])
    model.score(Decay_sum[6694:], ratios[6694:,1])

    # construct lookup dict for each ratio
    ins_ratio_lookup = dict(zip(Oligos, ratios[:,0]))
    mh_ratio_lookup = dict(zip(Oligos, ratios[:,1]))
    dataset_fn = lambda x : ratio_dataset(x,
                             ref_lookup=ref_lookup,
                             ins_lookup=ins_ratio_lookup,
                             mh_lookup=mh_ratio_lookup,
                             matrix_size=args.matrix_size
                             )


    # dataset
    train_dl = DataLoader(dataset_fn(Train_Oligos), batch_size=22, shuffle=True, num_workers=10)
    val_dl = DataLoader(dataset_fn(Val_Oligos), batch_size=22, shuffle=False, num_workers=10)
    test_dl = DataLoader(dataset_fn(Test_Oligos), batch_size=22, shuffle=False, num_workers=10)


    # Hyper-params 
    out_channel = 32
    channel_2d = 4
    # optimizer = 'LBFGS'
    optimizer = "RMSprop"
    # optimizer = "Adam"

    # define model

    Model = eval(f"ratio_models.Ratio_Model_size{args.matrix_size}") if args.matrix_size !=50 else ratio_models.Ratio_Model
    model = Model(channel_1d=out_channel, 
                    channel_2d=channel_2d,
                    lr=args.lr, optim_class=optimizer)
    
    # set up trainer
    device = 'gpu' if torch.cuda.is_available() else 'cpu' 
    workdir = os.path.join(PATH.pth_dir, "DeepDecay_ratio", f"{Cellline}_MixedConv_P{args.matrix_size}", )

    if args.GPU_devices is not None:
        gpu_device= args.GPU_devices
    else:
        gpu_device = {
        "ST_June_2017_BOB_LV7A_DPI7":6,
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
			max_epochs=500,
			callbacks=[ callbacks.ModelCheckpoint(filename='{epoch}-{val_loss:.8f}',
                                                  monitor="val_loss", mode="min", save_top_k=2),
                        callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=20),])
    
    model.train();
    trainer.fit(model, train_dl, val_dataloaders=val_dl)
    # ckpt_path = "pl_trainer_log/DeepDecay_ratio/BOB_MixedConv_P20/lightning_logs/version_1/checkpoints/epoch=46-val_loss=0.82054460.ckpt"
    
    model.eval();
    print("\ncheckpoint saved to : %s \n"%trainer.ckpt_path)
    trainer.validate(model, test_dl, ckpt_path="best")
    trainer.test(model, test_dl, ckpt_path="best")