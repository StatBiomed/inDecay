import os, sys, subprocess, argparse, time, PATH, shutil, math 
import my_utils
sys.path.append(PATH.main_dir)
import numpy as np
import pandas as pd 
import torch
torch.set_num_threads(4)
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import callbacks 
import pickle as pkl
# from models. import Base_del_model, ST_Decay, ST_DeepDecay, ST_Decay_Scaler, ST_DeepDecay_dropout, ST_DeepDecay_Multinomial
from inDecay import alignmap, models, reader
from tqdm.contrib.concurrent import process_map

def indelgen():
    time  = time.
    os.system("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("The script to extract SelfTarget proccessed txt file and map to Lindel classes")
    # parser.add_argument("--Set", required=True, type=str, help="either `TestSet1` or `TestSet2`")
    parser.add_argument("-M","--Model_checkpoint", type=str, required=True, help='the model weight')
    parser.add_argument("-S","--Sequence", type=str, required=True, help='the sequence to predict')