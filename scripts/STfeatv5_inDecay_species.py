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
from scripts.STfeatv2_inDecay_finetune import check_dir, decay_transform
import warnings
warnings.filterwarnings('ignore')
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
L2_Lambda = 3e-1
L1_Lambda = 0


# Torch Device
device = 'gpu' if torch.cuda.is_available() else 'cpu'

# some path and requisite files
pj = os.path.join
SelfTarget_data_dir = PATH.data_dir
data_dir = PATH.data_dir




def get_idfgen_file(Gene, Refseq, idgen_dir):

    Guide = Refseq[22:42] # with default cutsize 39

    gen_feature_file = os.path.join(idgen_dir, f"{Gene}_{Guide}_features.txt")

    if not os.path.exists(idgen_dir):
        os.mkdir(idgen_dir)

    if not os.path.exists(gen_feature_file):
        os.system(f"{PATH.Indelgen} {Refseq} {42} {gen_feature_file}")

    return gen_feature_file


def get_sanger_training(data_archive="Sanger_training"):
    """
    get all the sanger sequencing samples
    """
    # find the dir
    sanger_dir = os.path.join(PATH.data_dir, data_archive)
    assert os.path.exists(sanger_dir), "the sanger sequencing data is not correctly deposited"

    # look for processed tables
    dfs  = [file for file in os.listdir(sanger_dir)  if file.endswith("_SelfTarget.csv")]
    genes = np.array([f.split("_SelfTarget")[0] for f in dfs])
    assert len(dfs) != 0, f"no samples found under {sanger_dir}\nplese make sure you have put data in the right dir"

    # create a lookup table for easy retrival of reference
    def_table= os.path.join(sanger_dir, "gene_seq.csv")
    assert os.path.exists(def_table)
    gene_ref_tab = pd.read_csv(def_table)

    gene_ref_dict = {}
    for i, row in gene_ref_tab.iterrows():
        guide = row['seq'][22:42]
        gene_ref_dict[row['guide']] = guide, row['seq'], 42, "FORWARD"

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
    split_df.to_csv(pj(PATH.data_dir, args.data_archive, f"leave{species_dict[args.species]}_spliting.csv")) # save to result (will updated in github repo)
    # split_df.to_csv(os.path.join(sanger_dir, f"{args.data_archive}_spliting_{date}.csv")) # backup locally

def read_sanger_data(gene, gene_ref_lookup, data_archive='Sanger_training'):
    """
    read the identifier and read count of the given genes
    """
    Guide, refseq, pamsite, Strand = gene_ref_lookup[gene]

    # read table of the gene
    sanger_df_path = os.path.join(PATH.data_dir, data_archive, f"{gene}_SelfTarget.csv")
    label_df = pd.read_csv(sanger_df_path).query("Identifier != 'Not Present'")

    # normalize
    total_sum = label_df['Count'].sum()
    label_df['n_coevent'] = label_df['loc'].apply(lambda x: x.count("("))
    label_df['Frac Sample Reads'] = label_df['Count']/total_sum

    # read indelgen
    idgen_dir = os.path.join(PATH.data_dir, data_archive, 'Indelgen_result')
    idfgen_file = get_idfgen_file(gene, refseq, idgen_dir)
    
    idfgen = pd.read_table(idfgen_file, index_col=0, skiprows=1, names=['Identifier', 'n_coevent', 'loc', 'indels'])
    

    labelmerge_df = idfgen.merge(label_df[['Identifier', 'Count', 'Frac Sample Reads']], left_on=['Identifier'], right_on=['Identifier'], how='left')
    labelmerge_df = labelmerge_df.fillna(0)
    return labelmerge_df

def my_collect_fn(batch_list):
    features = [item[0].requires_grad_() for item in batch_list]
    ys = [item[1].requires_grad_() for item in batch_list]
    return features, ys

if __name__ == "__main__":
    parser = argparse.ArgumentParser("The script for few shot learning with embryonic sanger sesquencing data")
    # parser.add_argument("--Set", required=True, type=str, help="either `TestSet1` or `TestSet2`")
    parser.add_argument("-E","--data_archive", required=True, type=str, default='zygote', help='the folder name of processed sanger data')
    parser.add_argument("-C","--threshold", required=False, type=int, default=3, help="the minimum number of events for a sample to be considered valid")
    parser.add_argument("-G","--GPU_devices", type=int, default=None, help='The gpu to use')
    parser.add_argument("-P","--Pretrain", required=False, type=str, default=None, help="the pretrained parameter theta")
    parser.add_argument("-M","--Model_Class", required=False, type=str, default="ST_DeepDecay_Weight", help="inDecay / DeepDecay")
    parser.add_argument("-F","--Fix_params", required=False, type=str, default=None, help="the layers to fix, eg. del_regressor[:2]")
    parser.add_argument("-D","--Data_transform", required=False, type=str, default="identity", help="the name of data transformation")
    parser.add_argument("-T","--test_split", required=False, type=int, help='which fold to used to split train test genes')
    parser.add_argument("-O","--Mode", required=False, type=str, default="Train", help="the action of this script, can be `Train`, `Evaluate`, `Evaluate_only`, `Baseline` and `Write_Y`")
    parser.add_argument("--progress_bar", required=False, type=str, default="True", help="boolen, whether to show progress bar")
    parser.add_argument("--modelnote", required=False, type=str, default=None, help="the dir to save model")
    parser.add_argument("--temperature", required=False, type=float, default=1, help="the softmax temperature")
    parser.add_argument("--L", required=False, type=float, default=1e-3, help="the lr")
    parser.add_argument("--L2", required=False, type=float, default=3e-1, help="the l2")
    parser.add_argument("--species", required=True, type=str, default='p', help="leave which species out, current dict{'s': goat, 'c': cattle, 'p': porcine}")
    args = parser.parse_args()
    lr = args.L
    L2_Lambda =args.L2
    species_dict={'m': 'mouse', 's':'goat', 'p':'porcine', 'c':'cattle'}
    
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
        to_train = to_predict = to_baseline= False
        to_write_y = True

    elif args.Mode == 'Baseline':
        to_train = to_predict =  False 
        to_write_y = to_baseline = True
    else:
        raise ValueError("Invalide action combination")




 # if args.GPU_devices is not None:
    gpu_device= args.GPU_devices
    print(f"Runing {args.data_archive} fold {args.test_split}  using cuda: {gpu_device}")

    # Temp Theta file
    date = time.strftime("%B%d")
    sanger_dir = os.path.join(PATH.data_dir, args.data_archive)

    # get gene list
    # and also a gene to table look up
    genes, gene_ref_dict  = get_sanger_training(args.data_archive)
    
    valided_genes = []
    test_genes=[]
    unused_genes = ''
    for g in genes:
        if not g.startswith(args.species):
            label_df = read_sanger_data(g, gene_ref_dict, args.data_archive)
            n_event = label_df.query('`Identifier` != "Identifier"')['Identifier'].nunique()
            if n_event < args.threshold:
                unused_genes += f', {g}'
            else:
                valided_genes.append(g)
        else:
            test_genes.append(g)

    # print("Unused genes : \n" + unused_genes)
    print("number_valided genes: \n" + str(len(valided_genes)))

    genes = np.array(valided_genes)

    # split genes
    kf_indeces = reader.get_Sanger_train_test(genes)
    train_idx, val_idx, test_idx = kf_indeces[args.test_split]
    train_genes = genes[train_idx]
    val_genes = np.concatenate((genes[val_idx],genes[test_idx]))
    test_genes = np.array(test_genes)
    print("test_genes:",test_genes)


    # # save train test splits
    # if not os.path.exists(pj(PATH.data_dir, args.data_archive, f"leave{species_dict[args.species]}_spliting.csv")):
    #     save_spliting(genes, kf_indeces) # create a 2-dimensional list  then to df
    # else:
    #     # sanity check
    #     spliting_df = pd.read_csv(pj(PATH.data_dir, args.data_archive, f"leave{species_dict[args.species]}_spliting.csv"))
    #     assert len(kf_indeces) == spliting_df.shape[0], 'the num of fold has changed'
        
    #     record_test_genes = spliting_df.iloc[args.test_split]['Test_gene']
    #     record_test_genes = np.array(record_test_genes.split(","))
    #     assert np.all(record_test_genes == np.array(test_genes)), "the testing gene has changed"
        

    # some checkpoint settings
    save_dir = pj(data_dir, "Sanger")
    exp_name = args.Pretrain
    if "/" in exp_name:
        exp_name = os.path.basename(args.Pretrain).replace(".ckpt","")
    if not os.path.exists(os.path.join(PATH.pth_dir, f"{args.modelnote}_{exp_name}_{args.Model_Class}_{args.Data_transform}_lr{args.L}_L2{args.L2}_T{args.temperature}")):
        os.mkdir(os.path.join(PATH.pth_dir, f"{args.modelnote}_{exp_name}_{args.Model_Class}_{args.Data_transform}_lr{args.L}_L2{args.L2}_T{args.temperature}"))
    pth_save_dir = os.path.join(PATH.pth_dir,f"{args.modelnote}_{exp_name}_{args.Model_Class}_{args.Data_transform}_lr{args.L}_L2{args.L2}_T{args.temperature}",f"leave_{species_dict[args.species]}")
    pth_save_path = pj(pth_save_dir, f"{len(kf_indeces)}fold_{args.test_split}")
    
    for DIR in [PATH.pth_dir, pth_save_dir, pth_save_path]:
        check_dir(DIR)  

    
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
     # Modeling and Training
    model_class = eval("models.%s"%args.Model_Class)
    model_parsms = dict(inputsize=n_features, outputsize=1,  lr=lr,
                        L1_lambda=L1_Lambda, L2_lambda=L2_Lambda)
    if 'Deep' in args.Model_Class:
        model_parsms['hidden'] = hidden
        
    model = model_class(**model_parsms)
    if args.Pretrain is not None:
        model = model_class.load_from_checkpoint(args.Pretrain)
        pmodel = model_class.load_from_checkpoint(args.Pretrain)

    if args.Fix_params is not None:
        for p in eval(f"model.{args.Fix_params}").parameters():
            p.require_grad = False

        print(args.Fix_params, "is fixed")

    # dataset    
    # dataset    
    normalize = ('Multinomial' not in args.Model_Class) & ('weight' not in args.Model_Class)

    feature_extraction_fn = lambda label_df, refseq, cutsite : alignmap.ST_decayfeat_v5(label_df, refseq, cutsite, k1, k2, h)

    Train_DS = reader.ST_dataset(train_genes, gene_ref_dict, 
                            experiments=args.data_archive, 
                            read_data_fn = read_sanger_data,
                            transformation=transform,
                            feat_ext_fn = feature_extraction_fn,
                            normalize=normalize)
    Val_DS = reader.ST_dataset(val_genes, gene_ref_dict, 
                            experiments=args.data_archive, 
                            read_data_fn = read_sanger_data,
                            transformation=transform , 
                            feat_ext_fn = feature_extraction_fn,
                            normalize=normalize)
    Test_DS = reader.ST_dataset(test_genes, gene_ref_dict, 
                                experiments=args.data_archive, 
                                read_data_fn = read_sanger_data,
                                transformation=transform,
                                feat_ext_fn = feature_extraction_fn,
                                normalize=normalize)

    Train_DL = DataLoader(Train_DS, shuffle=True, batch_size=3, num_workers=num_workers, collate_fn=my_collect_fn)
    Val_DL = DataLoader(Val_DS, shuffle=False, batch_size=50, num_workers=num_workers, collate_fn=my_collect_fn)
    Test_DL = DataLoader(Test_DS, shuffle=False, batch_size=1, num_workers=num_workers, collate_fn=my_collect_fn)

    trainer = pl.Trainer(
			auto_lr_find=True,
            accelerator=device,
            # fast_dev_run=True,
            enable_progress_bar=eval(args.progress_bar),
			default_root_dir=pth_save_path,
            devices = [gpu_device],
			max_epochs=100,
            check_val_every_n_epoch=1,
			callbacks=[ callbacks.ModelCheckpoint(filename='{epoch}-{val_cre:.8f}',
                                                  monitor="val_cre", mode="min", save_top_k=-1, every_n_epochs=1),
                        callbacks.EarlyStopping(monitor="val_cre", mode="min", patience=20),])
    

    if to_train:
        model.train()
        trainer.fit(model, Train_DL, val_dataloaders=Val_DL)
        print(trainer.ckpt_path)

        model.eval()
        print(test_genes, args.modelnote, trainer.validate(model, Test_DL))

    # only save once for Y
    # only save once for Y
    if to_write_y:
        Forecast_Y = pj(pth_save_path, "ForeCast_TestY.pkl")

        # if not os.path.exists(Forecast_Y):
        get_identifiers = lambda gene : read_sanger_data(gene, gene_ref_dict, args.data_archive)[['Identifier', 'Frac Sample Reads']].values

        Y_ls = [get_identifiers(gene) for gene in test_genes]
        # Y_ls = process_map(get_identifiers, Test_Oligos, max_workers=8)
        Y_lookup = {o:Y_ls[i] for i,o in enumerate(test_genes)}

        handle = open(Forecast_Y, 'wb')
        pkl.dump(Y_lookup, handle)    
        handle.close()
        print('finished write to ',  Forecast_Y)


    if to_predict:
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
        
        pred_lookup = {o:predict_y[i].cpu().numpy() for i,o in enumerate(test_genes)} # type: ignore

        TestPred = ckpt_abspath.replace(".ckpt", "TestPred.pkl")
        pred_f = open(TestPred, 'wb')
        pkl.dump(pred_lookup, pred_f)
        pred_f.close()
        print("prediction writed to %s" %TestPred)
    
    
    if to_baseline:
        #  to generate baseline for the pretrained model

        pmodel = model_class.load_from_checkpoint(args.Pretrain)
        pmodel.eval()
        predict_y = trainer.predict(pmodel, Test_DL)

        if isinstance(predict_y[0], list):
            predict_y = sum(predict_y, [])  # to join lists 
        pred_lookup = {o:predict_y[i].cpu().numpy() for i,o in enumerate(test_genes)}
        print(len(pred_lookup))
        TestPred = Forecast_Y.replace("ForeCast_TestY.pkl", "Pretrained_Baseline_TestPred.pkl")

        pred_f = open(TestPred, 'wb')
        pkl.dump(pred_lookup, pred_f)
        pred_f.close()
        print("prediction writed to %s" %TestPred)