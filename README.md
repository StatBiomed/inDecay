# **inDecay** : *Predicting CRISPR-induced <ins>in</ins>dels by frequency <ins>Decay</ins>*

This library provides the inDecay package, the script for using our model to predict indels for your own input sequence, the script for training, finetuning the inDecay model.

![image](results/model.jpg)

## Understanding how inDecay works
We provide a demonstrating notebook containing the most necessary code to re-implement and inDecay work flow. You can follow the demo to build an idea of how the features were extracted and designed. It also recodes the a very small inDecay model and how to train the model with pytorch_lightning trainer.  

To unlock full power of inDecay, please follow the installation and training steps.


## Installation
To run inDecay model, please install the package by 
```shell
git clone https://github.com/StatBiomed/inDecay.git
cd inDecay

# create an new environment and install the dependencies
conda env create -f environment.yml

# install the python package
conda activate inDecay
pip install -e ./  
```

And we also encourage users to install indelgen toolkits from SelfTarget(https://github.com/felicityallen/SelfTarget).

```shell
cd ../
git clone https://github.com/felicityallen/SelfTarget.git

# the python dependent
pip install -r requirements.txt
cd selftarget_pyutils
pip install -e .
cd ../indel_prediction
pip install -e .

# compile predictor
cd indel_analysis/indelmap
cmake . -DINDELMAP_OUTPUT_DIR=../inDecay/tool
make && make install
export INDELGENTARGET_EXE=../inDecay/tool/bin/indelgentarget
```
&nbsp;

## Data download
To get the data for re-producing the model or developing related tools, you can easily download the processed data via
```shell
bash script/Data_download.sh
```

The script will ask for the directory to place the data. You the script will create the folder if not existed. An example below:

```shell
bash script/Data_download.sh
[out] Enter the path of folder you wanted to : 
[in] data 
```
&nbsp;  

## Predict with the specified model weights

To predict the editing profile for a collection of sequences, put all your sequence in a `.txt` file (e.g. `INPUTE_SEQUENCES.txt` below). 

Under the main directory , run
```shell
python scripts/STfeatV2_predict.py -S <INPUTE_SEQUENCES.txt> -M <MODEL_WEIGHT.pt>
```

&nbsp;  
## Train the model from scratch

To reproduce the result, you can 
Under the main directory , run
```shell
python scripts/STfeatv2_inDecay.py --experiment --read_cutoff 500 --Model_Class ST_DeepDecay --Data_transform interaction
```
