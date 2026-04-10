# ***inDecay*** : *Predicting CRISPR-induced <ins>in</ins>dels by frequency <ins>Decay</ins>*

This library provides the inDecay package, the script for using our model to predict indels for your own input sequence, the script for training, finetuning the inDecay model.


## Understand the workflow of inDecay  
We provide a demonstrating notebook ([demo/inDecay_demo.ipynb](https://github.com/StatBiomed/inDecay/blob/main/demo/inDecay_demo.ipynb)) containing the most necessary code to re-implement the inDecay work flow. You can follow the demo to get an idea of how the features were extracted and designed. It also records the a simplified inDecay model and the training process using `pytorch_lightning` `Trainer`.  

<img src="results/model.jpg" alt="drawing" width="400"/>  


&nbsp;  
To unlock the full power of inDecay, please follow the installation and training steps below.
&nbsp;  

## Installation  
To run inDecay model, please install the package by 
```shell
git clone https://github.com/StatBiomed/inDecay.git
cd inDecay

# create an new environment and install the dependencies
conda create -n inDecay python=3.10.4 pip

# install the python package
conda activate inDecay
pip install -r requirements.txt
pip install -e ./  
```
&nbsp;  

## Data download
To get the data for re-producing the model or developing related tools, you can easily download the processed data via

```shell
# Enter the a path where you want to save the data: 
bash scripts/Data_download.sh
```




## Set up PATH.py
After you have downloaded the data and install the SelfTarget toolkits, please runn the following script under the main directories. 

```shell
bash scripts/setup_path.sh
```

Please **change the directories mannually** in PATH.py **if you did not download them with default directorial setting** !!

&nbsp;  

And we also encourage users to install indelgen toolkits from SelfTarget(https://github.com/felicityallen/SelfTarget).

```shell
conda activate inDecay
bash scripts/selftarget.sh
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
python scripts/STfeatv5_inDecay.py --experiment ST_June_2017_BOB_LV7A_DPI7 --read_cutoff 500 --Model_Class ST_DeepDecay --Data_transform interaction
```
&nbsp;  

## Finetune model with Sanger sequencing data

To finetune inDecay on your own Sanger sequencing (`.ab1`) files, see the step-by-step tutorial in [demo/zygote_demo.ipynb](https://github.com/StatBiomed/inDecay/blob/main/demo/zygote_demo.ipynb). It walks you through processing raw `.ab1` traces into the labeled training data used by inDecay.


When you finished, run the following command to finetune the model with your own sanger sequencing data
```shell
python scripts/STfeatv5_inDecay_mouse.py --data_archive species -G 0  -P pretrained/mESC_featv5_c20.ckpt -T 1 
```