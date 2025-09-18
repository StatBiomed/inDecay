import os, sys, subprocess, argparse, time, PATH, re
sys.path.append(PATH.main_dir)
import numpy as np
import pandas as pd 
import torch
torch.set_num_threads(4)
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import callbacks 
import pickle as pkl
from inDecay import my_utils, alignmap, models, reader, PATH
from tqdm.contrib.concurrent import process_map

def indelgen(Refseq, cutsite, output_path=None):

    Guide = Refseq[cutsite-17:cutsite+3] # with default cutsize 39

    if output_path is None:
        indelgen_dir = os.path.join(PATH.main_dir,'predict_ouput')
        gen_feature_file = os.path.join(indelgen_dir, f"{Guide}_predict_out.csv")
    else:
        indelgen_dir = os.path.dirname(output_path) if "/" in output_path else "./"
        gen_feature_file = output_path

    if not os.path.exists(indelgen_dir):
        os.mkdir(indelgen_dir)

    # if not os.path.exists(gen_feature_file):
    os.system(f"{PATH.Indelgen} {Refseq} {cutsite+3} {gen_feature_file}")

    return gen_feature_file

def read_indelgen(indelgen_path):
    df = pd.read_table(indelgen_path, index_col=0, skiprows=1, names=['Identifier', 'n_coevent', 'loc', 'indels'])
    return df.reset_index()


def preprocess_seq(raw_seq, cutsite):
    return raw_seq[cutsite - 39:cutsite+40]  



if __name__ == "__main__":
    parser = argparse.ArgumentParser("The script to extract SelfTarget proccessed txt file and map to Lindel classes")
    # parser.add_argument("--Set", required=True, type=str, help="either `TestSet1` or `TestSet2`")
    parser.add_argument("-M","--Model_checkpoint", type=str, required=True, help='the model weight')
    parser.add_argument("-S","--Sequence", type=str, required=True, help='the sequence to predict')
    parser.add_argument("-C","--Cutsite", type=str, required=True, help='the sequence to predict')
    parser.add_argument("-O","--Output", type=str, required=False, help='the abs path of the output file')
    args = parser.parse_args()
    
    
    ##  load model
    # model path sanity check
    assert os.path.exists(args.Model_checkpoint), "model checkpoint not found"
    
    if "ST_Decay" in args.Model_checkpoint:
        model_class = models.ST_Decay 
    elif ("weight" in args.Model_checkpoint) or ('dropout' in args.Model_checkpoint):
        model_class = models.ST_DeepDecay_dropout
    else:
        # default ST_DeepDecay
        model_class = models.ST_DeepDecay

    model = model_class.load_from_checkpoint(args.Model_checkpoint, map_location='cpu')
    
    # feature version
    version = re.match(r".*feat\W?v(\d).*", args.Model_checkpoint).group(1)
    feature_fn = eval(f"alignmap.ST_decayfeat_v{version}")


    # data transformation
    if 'interaction' in args.Model_checkpoint:
        transform_fn = alignmap.interaction_transform  
    else:
        transform_fn = lambda x : x

    def get_input(truc_seq, cutsite, indelgen_path):
        """
        Transform the sequence into input tensor
        using the found feature extraction func and transformation func 
        
        Input
        -------
        truc_seq : preprocessed sequence, centering at cutsite at 39

        Return
        -------
        input_x : the tensor (1, n_event, n_features), compiled features of each indel (event)
        """
        label_df = read_indelgen(indelgen_path)
        
        ##  extract and transform to input
        mh_mask, label_df = alignmap.label_mh(truc_seq, cutsite, label_df, mml_name='mh_length')
        mh_mask, label_df = alignmap.label_mh(truc_seq, cutsite, label_df, mml_name='mh_length2', panelty=0)

        input_x = feature_fn(label_df, truc_seq, cutsite)
        input_x = transform_fn(input_x)
        input_x = torch.from_numpy(input_x).float().unsqueeze(0)
        return input_x, label_df

    
    # Sequence sanity check
    cutsite = int(args.Cutsite)
    raw_seq = args.Sequence
    assert raw_seq[cutsite +4: cutsite+6] == 'GG', "PAM is not detected for given sequence and cutsite"

    # truncate sequence
    # so that the sequence is cutsite is 39
    truc_seq = preprocess_seq(raw_seq, cutsite)

    ##  read input txt
    indelgen_path = indelgen(truc_seq, cutsite, args.Output)
    input_x, label_df = get_input(truc_seq, cutsite, indelgen_path)

    ## use model.predicts
    with torch.no_grad():
        pred_p = model(input_x)
        pred_p = pred_p.squeeze().numpy()

    assert len(pred_p) == label_df.shape[0], "prediction shape error"
    label_df['predict_frequency'] = pred_p

    # write output
    output_df = label_df[['Identifier', 'indels', 'predict_frequency']]
    output_df = output_df.sort_values(by='predict_frequency', ascending=False)
    output_df.to_csv(indelgen_path, index=False)
    print("\n"+"="*10)
    print(f"Done! \nprediction saved to : \n {indelgen_path}")
    print("="*10)
