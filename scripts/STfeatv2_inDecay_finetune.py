import os, sys, subprocess, argparse, time, shutil, math ,  re, json
import numpy as np
import pandas as pd 
import torch
torch.set_num_threads(4)
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import callbacks
import pickle as pkl
from inDecay import alignmap, models, reader, my_utils, PATH
from tqdm.contrib.concurrent import process_map
from qrguide import analysis_fn, transformation
from scripts.STfeatv2_inDecay import check_dir,readFeaturesData ,find_ckpt, read_data, interaction_transform, decay_transform


num_workers = 12

ndel = 9 
nins = 8
nshare = 1

# model params
k1 = 0.5 
k2 = 0.6
h = 1.3
hidden = [128, 64]
L2_Lambda = 0
L1_Lambda = 0
fix_params = False

# Torch Device
device = 'gpu' if torch.cuda.is_available() else 'cpu'

# some path and requisite files
pj = os.path.join
data_dir = PATH.data_dir

reference_path = pj(data_dir, "SelfTarget_NewScaffold.fasta")
ref_lookup = reader.get_reference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("The script to extract SelfTarget proccessed txt file and map to Lindel classes")
    # parser.add_argument("--Set", required=True, type=str, help="either `TestSet1` or `TestSet2`")
    parser.add_argument("-E","--experiment", type=str, required=True, help='The dir name of dataset')
    parser.add_argument("-C","--read_cutoff", type=int, default=500, help='The threshold of total count. Only Guides having total read count over this threshold are used')
    parser.add_argument("-T","--test_oligos", type=str, default="result/test_set_oligo_Feb2.txt", help='The file deciding which oligos are used in the training set')
    parser.add_argument("-G","--GPU_devices", type=int, default=None, help='The gpu to use')
    parser.add_argument("-P","--Pretrain", required=False, type=str, default=None, help="the pretrained parameter theta")
    parser.add_argument("-d","--L2_Lambda", required=False, type=float, default=3e-5, help="the regularization strength")
    parser.add_argument("-L","--L1_Lambda", required=False, type=float, default=0, help="the regularization strength")
    parser.add_argument("-M","--Model_Class", required=False, type=str, default="ST_DeepDecay", help="the regularization strength")
    parser.add_argument("-D","--Data_transform", required=False, type=str, default="interaction", help="the name of data transformation")
    parser.add_argument("-O","--Mode", required=False, type=str, default="Train", help="the action of this script, can be `Train`, `Evaluate`, `Evaluate_only` and `Write_Y`")
    parser.add_argument("-N","--N_Finetune", required=False, type=str, default="50", help="the number of oligos to use in the finetuning process")
    parser.add_argument("-R","--Rounds", required=False, type=int, default=50, help="Max epoches for finetuning")
    parser.add_argument("-U","--LearningRate", required=False, type=str, default="3e-4", help="the learning rate")
    args = parser.parse_args()

    lr = float(args.LearningRate)

    # some save file settings
    experiments = args.experiment
    Cellline = experiments.split("_")[3]
    rep = experiments.split("_")[4]
    save_dir = pj(data_dir, 'processed_df')
    csv_path = pj(save_dir,f"{Cellline}_{rep}.csv")

    
    gpu_device = {
        "ST_June_2017_BOB_LV7A_DPI7":0,
        "ST_June_2017_CHO_LV7A_DPI7":1,
        "ST_June_2017_E14TG2A_LV7A_DPI7":2,
        "ST_June_2017_HAP1_LV7A_DPI7":3,
        "ST_June_2017_K562_800x_LV7A_DPI7":3,
        }[experiments]

    if args.GPU_devices is not None:
        gpu_device= args.GPU_devices
    
    print(f"Runing {experiments} using cuda: {gpu_device}")
    pth_save_dir = os.path.join(PATH.pth_dir, f"STfeatv2_{args.Model_Class}_{args.Data_transform}_finetune")
    for DIR in [data_dir, pth_save_dir,  save_dir]:
        check_dir(DIR)  
    # Temp Theta file
    date = time.strftime("%b%d")
    pth_save_path = pj(pth_save_dir, f"{experiments}_N{args.N_Finetune}")

    processed_df = pd.read_csv(csv_path).query("`in_LdGen` == True").astype({"Count":"int"})
    processed_df = processed_df.query("`Strand` == 'FORWARD'")
    
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

    ## split training and testing data ##

    Train_Oligos, Val_Oligos, Test_Oligos = reader.get_Train_Val_Test(
        processed_df, 
        test_oligo_file = os.path.join(PATH.main_dir, args.test_oligos),
        seed = 0,
        threshold = args.read_cutoff
        )                                                                   # Only test set is used
    
    "result/yulu_random_oligo_list"
    
    # load predefined finetuning set
    if "R" in args.N_Finetune:
        # random 
        size = int(args.N_Finetune[1:])
        Finetune_Oligos = np.load(f'{PATH.main_dir}/result/yulu_random_oligo_list/BOB_{size}_finetune_list.npy')

    else:
        size = int(args.N_Finetune)
        Finetune_df=pd.read_csv(f"{PATH.main_dir}/result/Finetune_OligoIndex_Jul19.csv", index_col=0)
        finetune_set = 'FinetuneSet_n%s'%args.N_Finetune                        # For example, select 30 oligos
        Finetune_Oligos=Finetune_df.query('`%s` == True'%finetune_set).index    # finetune oligos

    # train-val for finetuning
    val_size = int(0.1 * size)                               # num. of oligos for finetuning validatoin 
    Finetune_Oligos_val = np.random.choice(Finetune_Oligos, size=val_size)
    Finetune_Oligos_train = [O for O in Finetune_Oligos if O not in Finetune_Oligos_val]



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
    
    
    ## Resume and Training ##
    model_class = eval("inDecay.%s"%args.Model_Class)
    model_parsms = dict(inputsize=n_features, outputsize=1,  lr=lr,
                        L1_lambda=L1_Lambda, L2_lambda=L2_Lambda)
    
    if 'Deep' in args.Model_Class:
        model_parsms['hidden'] = hidden # type: ignore
        
    model = model_class(**model_parsms)

    if args.Pretrain is None:
        raise ValueError("Finetuning jobs must start with a trained model")
    elif not os.path.exists(args.Pretrain): 
        raise FileNotFoundError("Invalid path to Pretrained model")
    else:
        model = model_class.load_from_checkpoint(args.Pretrain)
        # ckpt = torch.load(args.Pretrain)
        # model.load_state_dict(ckpt['state_dict'])

        if fix_params:
            for p in model.del_regressor[0:2].parameters():
                p.requires_grad = False


    # dataset    
    normalize = 'Multinomial' not in args.Model_Class 
    feature_extraction_fn = lambda label_df, refseq, cutsite : alignmap.ST_decayfeat_v2(label_df, refseq, cutsite, k1, k2, h)

    Train_DS = reader.ST_dataset(Finetune_Oligos_train, processed_df, experiments, 
                          read_data_fn = read_data,
                          transformation=transform,
                          feat_ext_fn = feature_extraction_fn,
                          normalize=normalize)
    Val_DS = reader.ST_dataset(Finetune_Oligos_val, processed_df, experiments, 
                               read_data_fn = read_data,
                               transformation=transform, 
                               feat_ext_fn = feature_extraction_fn,
                               normalize=normalize)
    Test_DS = reader.ST_dataset(Test_Oligos, processed_df, experiments, 
                                read_data_fn = read_data,
                                transformation=transform,
                                feat_ext_fn = feature_extraction_fn,
                                normalize=normalize)

    Train_DL = DataLoader(Train_DS, shuffle=True, batch_size=1, num_workers=num_workers)
    Val_DL = DataLoader(Val_DS, shuffle=False, batch_size=1, num_workers=num_workers)
    Test_DL = DataLoader(Test_DS, shuffle=False, batch_size=1, num_workers=num_workers)

    trainer = pl.Trainer(
			auto_lr_find = True,
            accelerator = device,
			default_root_dir = pth_save_path,
            devices = [gpu_device],
			max_epochs = args.Rounds,
			callbacks = [ callbacks.ModelCheckpoint(filename='{epoch}-{val_cre:.8f}',
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
        
        ckpt_abspath = find_ckpt(pj(pth_save_path, 'lightning_logs'))
        assert os.path.exists(ckpt_abspath)

        if to_train:
            try:
                model.eval()
                predict_y = trainer.predict(model, Test_DL, ckpt_path=ckpt_abspath)
                print("PL trainer auto-resume best model")
            except:
                print("{} was used to predict".format('/'.join(ckpt_abspath.split('/')[-3:])))
                model = model_class.load_from_checkpoint(ckpt_abspath).to(gpu_device)
                model.eval()
                predict_y = trainer.predict(model, Test_DL)

        else:
            # load the pretrain model
            print("Pretained was used to predict !!!")
            model = model_class.load_from_checkpoint(args.Pretrain)
            model.eval()
            predict_y = trainer.predict(model, Test_DL, ckpt_path=args.Pretrain)
            ckpt_abspath = args.Pretrain
        
        
        pred_lookup = {o:predict_y[i].cpu().numpy() for i,o in enumerate(Test_Oligos)} # type: ignore

        TestPred = ckpt_abspath.replace(".ckpt", "TestPred.pkl")
        pred_f = open(TestPred, 'wb')
        pkl.dump(pred_lookup, pred_f)
        pred_f.close()
        print("prediction writed to %s" %TestPred)


        
        #   ______             _                _        
        #  |  ____|           | |              | |       
        #  | |__ __   __ __ _ | | _   _   __ _ | |_  ___ 
        #  |  __|\ \ / // _` || || | | | / _` || __|/ _ \
        #  | |____\ V /| (_| || || |_| || (_| || |_|  __/
        #  |______|\_/  \__,_||_| \__,_| \__,_| \__|\___|
                                               
                                            
        ## evaluate in the test set
        Forecast_Y = pj(pth_save_path, "ForeCast_TestY.pkl")
        f = open(Forecast_Y, 'rb')
        Y_lookup = pkl.load(f)  # forecast : ST
        f.close()

        # the metric dict
        performance_json = analysis_fn.assessment_recipe_forecast(Y_lookup, pred_lookup)
        IDL_performance = analysis_fn.assessment_recipe_IDL_forecast(Y_lookup, pred_lookup)

        performance_json.update(IDL_performance)
        performance_json['End_date'] = time.strftime("%b%d-%H:%-M")
        performance_json['ckpt_path'] = ckpt_abspath

        # # model params 
        training_params = {}
        for pm in ["ndel", "nins", "nshare", "k1", "k2", "h", "hidden", "L2_Lambda", "L1_Lambda", "lr", "fix_params"]:
            training_params[pm] = eval(pm)

        performance_json['training_params'] = training_params
        
        # save the metrics
        result_dir = f"{PATH.main_dir}/result/Transfer/{args.Model_Class}_{Cellline}"
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        version = re.match(r".*version_(\d{1,2}).*", ckpt_abspath).group(1)
        json_path = pj(result_dir, f"N{args.N_Finetune}-V{version}-{date}.json")
        if args.Mode == "Evaluate":
            json_path = pj(result_dir, f"N0-V0-{date}.json")

        with open(json_path, "w") as write_file:
            json.dump(performance_json, write_file, indent=4)
        
        print("\n"+"="*20)
        print("performance json saved to %s" %json_path)
        print("="*20)
### some pretrained models: 

## CHO:
# /home/wergillius/data/CRISPR_data/pl_trainer_log/ST_featv2_ST_DeepDecay_interaction/ST_June_2017_CHO_LV7A_DPI7/lightning_logs/version_0/checkpoints/epoch=39-val_loss=0.00000000.ckpt
#  -E ST_June_2017_BOB_LV7A_DPI7 -C 1000 -G 3 -L 0 -N 100 -P /home/wergillius/data/CRISPR_data/pl_trainer_log/ST_featv2_ST_DeepDecayback_interaction/ST_June_2017_CHO_LV7A_DPI7/lightning_logs/version_1/checkpoints/epoch=49-val_cre=3.26941109.ckpt -R 20 --Mode Train