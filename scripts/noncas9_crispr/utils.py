import os
import PATH
import pandas as pd

def indelgen(Refseq, cutsite, output_path=None):

    Guide = Refseq[cutsite-17:cutsite+3] # with default cutsize 39

    if output_path is None:
        indelgen_dir = os.path.join(PATH.trex2_dir,'indelgen_result')
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