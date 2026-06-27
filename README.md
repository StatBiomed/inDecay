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

## Data and model-weight download

All reproduction data and model weights are archived under permanent DOIs and pulled with checksum verification by a single script (run from the repo root):

```shell
python scripts/fetch_data.py            # data + weights
# python scripts/fetch_data.py --data    # figshare data only
# python scripts/fetch_data.py --weights # Zenodo Fig 4/5 checkpoints only
```

| Artifact | DOI | Lands in |
|----------|-----|----------|
| Training / somatic data | [10.6084/m9.figshare.25133564](https://doi.org/10.6084/m9.figshare.25133564) | `data/` |
| Figure 4 & 5 fine-tuned checkpoints | [10.5281/zenodo.20977675](https://doi.org/10.5281/zenodo.20977675) | `pl_trainer_log/` |

The legacy `bash scripts/Data_download.sh` still works but is not checksum-verified; `fetch_data.py` is preferred.

## Set up PATH.py

`inDecay/PATH.py` now auto-detects the repository root, so no manual editing is needed for a standard clone. On an HPC or a custom layout, override the defaults with environment variables:

```shell
export INDECAY_MAIN_DIR=/path/to/inDecay     # repo root
export INDECAY_USER_DIR=/path/to/scratch     # optional, for SelfTarget container
```

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


## Finetune model with Sanger sequencing data


&nbsp;  

For example, to finetune the model with livestock data, run
```shell
python scripts/STfeatv5_inDecay_mouse.py --data_archive species -G 0  -P pretrained/mESC_featv5_c20.ckpt -T 1 
```

&nbsp;
## Reproducing the paper figures

The figure notebooks live in [`notebooks/`](notebooks/) and should be **run from the repository root**. Figures 2, 3 and the supplementary panels plot from precomputed tables already in the repo; Figures 4 & 5 additionally require the fine-tuned checkpoints fetched above.

```shell
python scripts/fetch_data.py --weights     # only needed for Figures 4 & 5
jupyter lab notebooks/Figures4_and_5.ipynb
```

| Notebook | Figure |
|----------|--------|
| `notebooks/Figure2.ipynb` | Fig 2 — benchmarking |
| `notebooks/Figure3.ipynb` | Fig 3 — transfer-learning sample-size sweep |
| `notebooks/Figures4_and_5.ipynb` | Figs 4 & 5 — mouse & cross-species transfer |
| `notebooks/SHAP_K562.ipynb` | SHAP feature attribution |
| `notebooks/Supp_FrameBreakdown.ipynb` | Supplementary frame breakdown |