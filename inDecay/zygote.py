
import os, sys
import glob
import subprocess
import shutil
from Bio import SeqIO, Seq
import numpy as np
import pandas as pd
from snapgene_reader import snapgene_file_to_dict, snapgene_file_to_seqrecord
from bs4 import BeautifulSoup 
import requests
import time
from tqdm import tqdm
import re
from inDecay import PATH
main_dir= PATH.main_dir

def create(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)
        
def copy_if_empty(src_dir, target_dir, basename, save_name):
    """
    Copy a file from src_dir to target_dir if the file does not already exist in target_dir.
    After copying, rename the file to include save_name and sanitize the filename.

    Args:
        src_dir (str): Source directory path.
        target_dir (str): Target directory path.
        basename (str): Name of the file to copy.
        save_name (str): Prefix to add to the copied file's name.
    """
    if not os.path.exists(f"{target_dir}/{basename}"):
        shutil.copy2(f"{src_dir}/{basename}", target_dir)
        name1 = save_name + '---' + basename
        # Replace spaces and special characters for compatibility
        newname = name1.replace(' ', '-').replace('&', 'and').replace('(', '').replace(')', '-')
        os.rename(f"{target_dir}/{basename}", f"{target_dir}/{newname}")

def trunc_filename(ab1_file):
    """
    Truncate and sanitize an ab1 filename for use as a sample label.

    Args:
        ab1_file (str): The filename to process.

    Returns:
        str: Sanitized filename without extension and special characters.
    """
    return ab1_file.replace(".ab1", "").replace('&', 'and').replace('(', '').replace(')', '-')

def read_control(success_folder, control):
    """
    Read a control sequence file (ab1 or txt), returning the forward and reverse complement sequences.

    Args:
        success_folder (str): Directory containing the control file.
        control (str): Filename of the control file.

    Returns:
        tuple: (FW, RC) where FW is the forward sequence, RC is the reverse complement.
    """
    if control.endswith("ab1"):
        seqRecord = SeqIO.read(f"{success_folder}/{control}", "abi")
        FW, RC = str(seqRecord.seq), str(seqRecord.seq.reverse_complement())
    elif control.endswith("txt"):
        with open(f"{success_folder}/{control}", 'r') as f:
            FW = f.readlines()[0]
        RC = str(Seq.Seq(FW).reverse_complement())
    return FW, RC

def process_guide(FW, RC, guide):
    """
    Locate the guide sequence in the forward or reverse complement sequence,
    extract a 79-bp reference region centered on the guide, and check for PAM.

    Args:
        FW (str): Forward sequence.
        RC (str): Reverse complement sequence.
        guide (str): Guide RNA sequence.

    Returns:
        tuple: (Strand, shorten_ref)
            - Strand (str): 'FW' or 'RC' if found, else False.
            - shorten_ref (str): 79-bp reference region if found and contains PAM, else False.
    """
    for j, seq in enumerate([FW, RC]):
        if guide in seq:
            g_index = seq.index(guide)
            x = (g_index + len(guide)) - 42
            shorten_ref = seq[x:x+79]
            Strand = ['FW', 'RC'][j]
            if shorten_ref[43:45] == "GG":
                return Strand, shorten_ref
            else:
                shorten_ref = ""
        else:
            shorten_ref = ""
    if shorten_ref == "":
        return False, False

def find_ab1_and_control(folder, Guide="", requirements=[], save_name=None, stype=""):
    """
    Locate and process control and experiment ab1 files for a given folder and guide.
    Copies files to intermediate folders, generates definition files for downstream analysis.

    Args:
        folder (str): Folder name containing raw ab1 files.
        Guide (str): Guide RNA sequence to search for.
        requirements (list): List of substrings to filter experiment files.
        save_name (str): Name to use for saving files and definition sheets.
        stype (str): Sample type annotation for definition file.

    Returns:
        tuple: (abi, definition_ice, definition_dec)
            - abi (list): List of experiment ab1 files processed.
            - definition_ice (pd.DataFrame): Definition sheet for decodr analysis.
            - definition_dec (pd.DataFrame): Definition sheet for Decodr analysis.
    """
    # Define paths for raw and intermediate files
    success_folder = f"{rawfile_dir}/{folder}"
    control_exp_folder = f"process/synthego/{save_name}"
    control_exp_decodr = f"process/decodr/{save_name}"
    # Create process directories if they do not exist
    if not os.path.exists(control_exp_folder):
        os.mkdir(control_exp_folder)
        print("make new dir")
    else:
        print("save to ", control_exp_folder)
    if not os.path.exists(control_exp_decodr):
        os.mkdir(control_exp_decodr)
        print("make new dir")
    else:
        print("save to ", control_exp_decodr)

    # Locate control (WT) ab1 file
    controls = [r for r in os.listdir(success_folder) if '.ab1' in r and 'WT' in r]
    assert len(controls) == 1, controls
    control = controls[0]
    copy_if_empty(success_folder, control_exp_folder, control, save_name)
    copy_if_empty(success_folder, control_exp_decodr, control, save_name)
    WT_seq, RC_WT = read_control(success_folder, control)
    strand, shorten_ref = process_guide(WT_seq, RC_WT, Guide)
    if strand == 'FW' and len(shorten_ref) == 79:
        print("find sgRNA in forward strand with PAM")
    elif strand == 'RC' and len(shorten_ref) == 79:
        print("find sgRNA in reverse strand with PAM")
    else:
        print("Can not locate right sgRNA along the reference")

    # Locate and process experiment ab1 files
    abi = [r for r in os.listdir(success_folder) if '.ab1' in r and 'WT' not in r]
    for cond in requirements:
        abi = list(filter(lambda x: cond in x, abi))
    for r in abi:
        copy_if_empty(success_folder, control_exp_folder, r, save_name)
        copy_if_empty(success_folder, control_exp_decodr, r, save_name)

    print("expect %d file" % (len(abi)+1))
    print("found %d under control_and_experiment" % len(os.listdir(control_exp_folder)))
    print("found %d under control_and_experiment" % len(os.listdir(control_exp_decodr)))

    # Prepare definition files for decodr and Decodr
    merged_files = [i for i in os.listdir(control_exp_folder) if 'WT' not in i]
    controls = [i for i in os.listdir(control_exp_folder) if 'WT' in i]
    assert len(controls) == 1, controls
    control = controls[0]
    define_dict_ice = {
        "Label": [trunc_filename(f) for f in merged_files],
        "Control File": [control] * len(merged_files),
        "Experiment File": merged_files,
        "Guide Sequence": [Guide] * len(merged_files),
        "Donor Sequence": [''] * len(merged_files),
    }
    definition_ice = pd.DataFrame(define_dict_ice)
    define_dict_dec = {
        "Sample Title": [trunc_filename(f) for f in merged_files],
        "Sample Type": [stype] * len(merged_files),
        "Guide Sequence(s)": [Guide] * len(merged_files),
        "Nuclease": ['Cas9'] * len(merged_files),
        "Donor Template (Optional)": ["None"] * len(merged_files),
        "Control Data": [control] * len(merged_files),
        "Experiment File(s)": merged_files,
    }
    definition_dec = pd.DataFrame(define_dict_dec)
    definition_ice.to_excel(f"intermediate/synthego/{save_name}.xlsx", index=False)
    definition_dec.to_excel(f"intermediate/decodr/{save_name}.xlsx", index=False)
    print(definition_dec)
    return abi, definition_ice, definition_dec



def analyze_html_file(file_path, date):
    """
    Parse an HTML file to extract experiment names and result URLs for a specific date.

    Args:
        file_path (str): Path to the HTML file to analyze.
        date (str): Date string to filter experiments (must match the date in the HTML).

    Returns:
        list of tuple: Each tuple contains (experiment name, result URL) for the given date.
    """
    with open(file_path, 'r') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    terms = soup.select('div.sc-iujRgT.fXTcfL.col-lg-4')
    term_list = []
    for term in terms:
        name = term.select_one('div.sc-cLQEGU.dowVLF').text.strip()
        adate = term.select_one('p.sc-bdVaJa.sc-htpNat.gsCcyq').text[0:]
        if str(adate) == str(date):
            urls = [element['href'] for element in term.select('a.sc-hSdWYo.sc-eHgmQL.dtSiJv')]
            for url in urls:
                if url.startswith('/results/'):
                    term_list.append((name, url))
    return term_list

def request_by_link(link):
    """
    Fetch result data from DecodR API given a result link.

    Args:
        link (str): Result URL (e.g., '/results/xxxxxx').

    Returns:
        list: List of result dictionaries from the API response.
    """
    url = f'https://decodr.org/api/query/result/{link.split("/")[-1]}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.3'
    }
    res = requests.get(url, headers=headers).json()
    return res['results']['results']['results']

def getContent(results):
    """
    Parse DecodR API results into a list of dictionaries with key experiment information.

    Args:
        results (list): List of result dictionaries from DecodR API.

    Returns:
        list of dict: Each dict contains label, analysisType, rSquared, ratio, N_gt, pValue, and TG_ice.
    """
    datas = []
    for result in results:
        label = result['filenames'][0]['displayData']
        atype = result['filenames'][0]['analysisType']
        r2 = result['rSquared']
        rows = result['proposals']
        for row in rows:
            list1 = {}
            list1['label'] = label
            list1['analysisType'] = atype
            list1['rSquared'] = r2
            list1['ratio'] = '%.2f' % row['contribution']
            list1['N_gt'] = str(row['netIndel']) + '[g1]'
            list1['pValue'] = row['pValue']
            numbers = row['previewSequence']
            TG_ice = ''
            for number in numbers:
                if len(number) == 1:
                    value = '|'
                else:
                    value = number[1]
                TG_ice = TG_ice + value
            list1['TG_ice'] = TG_ice
            datas.append(list1)
    return datas

def download(link):
    """
    Download and parse DecodR results for a given link.

    Args:
        link (str): Result URL (e.g., '/results/xxxxxx').

    Returns:
        list of dict: Parsed result data for the given link.
    """
    return getContent(request_by_link(link))
             
def get_data_from_df(link_df, save_dir=None):
    """
    Download and parse DecodR results for all samples in a DataFrame.

    Args:
        link_df (pd.DataFrame): DataFrame with at least columns ['Sample Title', 'DecodR link'].
        save_dir (str, optional): Directory to save individual sample CSVs. If None, files are not saved.

    Returns:
        tuple: (df_dicst, fails)
            - df_dicst (dict): Mapping from sample name to DataFrame of results.
            - fails (list): List of sample names that failed to download or parse.
    """
    fails = []
    df_dicst = {}
    for i, row in tqdm(link_df.iterrows()):
        sample = row['Sample Title']
        time.sleep(1)  # Avoid hammering the server
        results = request_by_link(row['DecodR link'])
        datas = getContent(results)
        if datas == []:
            fails.append(sample)
            continue
        for d in datas:
            d['sample'] = sample
        df = pd.DataFrame(datas)
        df_dicst[sample] = df
        if save_dir is not None:
            df.to_csv(f"{save_dir}/{sample}.csv", index=False)
    return df_dicst, fails

# def read_and_merge_del(file):
#     ice_1 = pd.read_csv(os.path.join(ICE_dir, file))[["indel_size","Identifier","Count"]].query("`Identifier`!='Not Present' & `indel_size` < 0")
#     dec_1 = pd.read_csv(os.path.join(sanger_dir, file))[["indel_size","Identifier","Count"]].query("`Identifier`!='Not Present' & `indel_size` < 0")

#     merged = ice_1.merge(dec_1, left_on="Identifier", right_on="Identifier", how='outer', suffixes=['_ice', '_dec'])

#     merged = merged.fillna(0)
#     if merged.shape[0]>1:
#         cosine= cosine_similarity(merged['Count_ice'].values.reshape(1, -1),merged['Count_dec'].values.reshape(1, -1)) 
#     elif merged.shape[0] == 0:
#         cosine=1
#     else:
#         cosine=abs(merged['Count_ice'].iloc[0]-merged['Count_dec'].iloc[0])
#     return merged, cosine
# def read_and_merge_len(file):
#     ice_1 = pd.read_csv(os.path.join(ICE_dir, file))[["indel_size","Count","Identifier"]].groupby(["indel_size"]).agg('sum').query("`indel_size`!=0")
#     dec_1 = pd.read_csv(os.path.join(sanger_dir, file))[["indel_size","Count","Identifier"]].groupby(["indel_size"]).agg('sum')

#     merged0 = dec_1.merge(ice_1, left_on="indel_size", right_on="indel_size", how='outer', suffixes=['_dec', '_ice'])
#     merged = merged0.fillna(0).replace('Not Present',0)

#     merged = merged.query("`Identifier_ice`!=0 | `indel_size`>0")
#     merged = merged.query("`Identifier_dec`!= 0 | `Identifier_ice`!=0 ")
#     if merged.shape[0]>1:
#         cosine= cosine_similarity(merged['Count_ice'].values.reshape(1, -1),merged['Count_dec'].values.reshape(1, -1)) 
#     elif merged.shape[0] == 1:
#         cosine=abs(merged['Count_ice'].iloc[0]-merged['Count_dec'].iloc[0])
#     elif merged.shape[0] == 0:
#         cosine= None
    
#     return merged, cosine

# def read_and_merge_insert(file):
#     ice_1 = pd.read_csv(os.path.join(ICE_dir, file))[["indel_size","Count"]].groupby(["indel_size"]).agg('sum').query("`indel_size`>0")
#     dec_1 = pd.read_csv(os.path.join(sanger_dir, file))[["indel_size","Count"]].groupby(["indel_size"]).agg('sum').query("`indel_size`>0")

#     merged = ice_1.merge(dec_1, left_on="indel_size", right_on="indel_size", how='outer', suffixes=['_ice', '_dec'])

#     merged = merged.fillna(0)
#     if merged.shape[0]>1:
#         cosine= cosine_similarity(merged['Count_ice'].values.reshape(1, -1),merged['Count_dec'].values.reshape(1, -1)) 
#         # merged_inter = merged.query("`Count_dec` != 0 & `Count_ice` != 0 ")
#     elif merged.shape[0] == 0:
#         cosine=1
#     else:
#         cosine=abs(merged['Count_ice'].iloc[0]-merged['Count_dec'].iloc[0])
#     return merged, cosine

class getselftarget:
    """
    A class to identify and process self-targeting regions around a guide RNA sequence
    in CRISPR-edited gene data. Designed for integration with the indelgentarget tool.

    Attributes:
        gene (str): Gene identifier (format: "directory---gene_name").
        dir (str): Directory name.
        guidedisc (dict)
        guide (str): Guide RNA sequence (retrieved from guidedisc dictionary).
        FW (str): Forward sequence from control file.
        RC (str): Reverse complement sequence from control file.
        shorten_ref (str): 79-bp reference region around guide RNA with PAM.
        selftarget (str): Command string for indelgentarget analysis.
    """
    
    def __init__(self, guidedisc, gene):
        """
        Initialize self-targeting analysis for a gene.

        Args:
            gene (str): Gene identifier in "directory---gene_name" format.
                        Requires guidedisc dictionary containing guide sequences.
        """
        self.gene = gene
        self.guidedisc = guidedisc
        self.dir = gene.split('---')[0]
        self.guide = guidedisc[self.gene]  # guidedisc must be predefined
        self.FW, self.RC = self.read_control()
        self.shorten_ref = self.process_guide()
        self.selftarget = self.getst()

    def read_control(self):
        """
        Read control sequence file (.ab1 or .txt) for the gene.

        Returns:
            tuple: (FW, RC) forward and reverse complement sequences.

        Raises:
            AssertionError: If multiple control files found.
        """
        raw_dir = f"{PATH.embryo_raw_dir}/process/decodr/"
        controlfiles=[e for e in os.listdir(os.path.join(raw_dir,self.dir)) if 'WT' in e or '.txt' in e]
        assert len(controlfiles)==1, [self.gene, controlfiles]
        self.control= controlfiles[0]
        if self.control.endswith("ab1"):
            seqRecord = SeqIO.read(f"{raw_dir+self.dir+'/'+self.control}", "abi")
            self.FW, self.RC = str(seqRecord.seq), str(seqRecord.seq.reverse_complement())
            
        elif self.control.endswith("txt"):
            with open(f"{raw_dir+self.dir+'/'+self.control}", 'r') as f:
                self.FW = f.readlines()[0]
            self.RC = str(Seq.Seq(self.FW).reverse_complement())
        return self.FW, self.RC

    def process_guide(self):
        """
        Locate guide sequence in reference and extract 79-bp region with PAM check.

        Returns:
            str: 79-bp reference region if guide found with GG PAM, else empty string.
        """
        for j, seq in enumerate([self.FW, self.RC]):
            if self.guide in seq:
                g_index = seq.index(self.guide)
                x = (g_index + len(self.guide)) - 42
                self.shorten_ref = seq[x:x+79]
                # self.shorten_ref = seq[self.cutsite-39:self.cutsite] + seq[self.cutsite:self.cutsite+40]
                self.Strand = ['FW', 'RC'][j]
                if self.shorten_ref[43:45] == "GG":
                    return self.shorten_ref
                else:
                    self.shorten_ref=""
            else:
                self.shorten_ref=""

        if self.shorten_ref=="":
            return False

        # ... (rest of method)

    def getst(self):
        """
        Generate indelgentarget command for self-targeting analysis.

        Returns:
            str: Command string formatted for indelgentarget tool:
                 "indelmap/indelgentarget [reference] [position] [output_path]"
        """
        self.selftarget=f"{'indelmap/indelgentarget'} {self.shorten_ref} {str(42)} {os.path.join(PATH.embryo_raw_dir,'process/SelfTarget',self.dir)}.txt"
        return self.selftarget  


from collections import defaultdict
def complement(c):
    if c == 'A': return 'T'
    if c == 'T': return 'A'
    if c == 'C': return 'G'
    if c == 'G': return 'C'
    else: return c
def remove_dash(seq):
    seq = seq.replace("|","")
    seq_ls = seq.split("–")
    not_aligned = seq_ls[0] + seq_ls[-1]
    return not_aligned
def def_value(): 
    return "Not Present"



class genepro_dec(getselftarget):
    """
    Class for integrating Sanger sequencing decodr analysis results with self-targeting indel predictions.
    This class processes decodr result files, annotates indels with genomic positions, matches them to
    computational predictions (from indelgentarget), and aggregates the results for downstream analysis.

    Attributes:
        sanger_dir (str): Output directory for aggregated results.
        ana_dir (str): Input from decodr.
        selftargetpath (str): Input from indelgen.
        result (str): Path to decodr analysis CSV file.
        selftarget (str): Path to indelgentarget prediction file.
        finallist (list): Processed decodr results with strand-corrected sequences.
        finallist_dash (list): decodr results annotated with genomic positions for indels.
        indel_gen_df (pd.DataFrame): DataFrame of predicted indels from indelgentarget.
        agg_df (pd.DataFrame): Aggregated DataFrame of validated indels and their frequencies.
    """

    def __init__(self, guidedisc, gene, ana_dir, selftargetpath, sanger_dir):
        """
        Initialize the genepro_dec pipeline for a given gene.

        Args:
            gene (str): Gene identifier (e.g., 'dir---GENE').
            sanger_dir (str): Output directory for aggregated results.
        """
        super().__init__(guidedisc, gene)
        self.sanger_dir = sanger_dir
        self.result = ana_dir + '/' + self.gene + '.csv'
        self.dir = self.gene.split('---')[0]
        self.selftarget = selftargetpath + self.dir + '.txt'
        self.finallist = self.read_ice()
        self.finallist_dash = self.count_by_dash()
        self.indel_gen_df = self.read_indel_gen_df()
        self.agg_df = self.sanger_training()

    def read_ice(self):
        """
        Parse decodr analysis results and adjust sequences for strand orientation.

        Returns:
            list: Each entry is a dict with keys:
                - 'label': Gene name
                - 'ratio': Event frequency (float)
                - 'N_gt': Indel size (int)
                - 'TG_ice': Sequence context (reverse-complemented if on RC strand)
        """
        df = pd.read_csv(self.result)
        self.finallist = []
        for row in range(df.shape[0]):
            list1 = {}
            list1['label'] = self.gene
            list1['ratio'] = float(df['ratio'].iloc[row])
            list1['N_gt'] = int(df['N_gt'].iloc[row].split('[')[0])
            if self.Strand == 'FW':
                list1['TG_ice'] = str(df['TG_ice'].iloc[row])
            else:
                list1['TG_ice'] = ''.join(map(complement, reversed(df['TG_ice'].iloc[row])))
            self.finallist.append(list1)
        return self.finallist

    def count_by_dash(self):
        """
        Annotate indels with genomic positions using dash notation.

        Returns:
            list: Each entry is a dict with an added key:
                - 'Dash_range': Genomic position in format:
                    * Deletions: (start,end)
                    * Insertions: (position,inserted_bases)
        Raises:
            AssertionError: If dash alignment validation fails.
        """
        self.finallist_dash = []
        for row in self.finallist.__iter__():
            seq = row['TG_ice']
            cut_site = seq.index("|")
            if int(row['N_gt']) < 0:
                fist_dash = seq.replace("|", "").index("–")
                dash_len = seq.replace("|", "").count("–")
                dash_start = fist_dash - cut_site + 39
                assert seq.replace("|", "")[fist_dash:fist_dash+dash_len] == "–"*dash_len, 'FALSE ALIGN'
                dash_range = "(" + str(dash_start - 1) + "," + str(dash_start + dash_len) + ',' + ")"
            else:
                # insertion
                dash_range = "(38,39," + seq[cut_site+1:cut_site+int(row['N_gt']) +1] + ")"
            row['Dash_range'] = dash_range
            self.finallist_dash.append(row)
        return self.finallist_dash

    def read_indel_gen_df(self):
        """
        Load indelgentarget predictions into a DataFrame.

        Returns:
            pd.DataFrame: Columns:
                - 'Identifier': Unique indel ID
                - 'n_collapse': Collapsed count
                - 'loc': Genomic location
                - 'indelgen_seq': Predicted sequence
        """
        self.indel_gen_df = pd.read_csv(
            self.selftarget,
            sep='\t',
            names=['Identifier', 'n_collapse', 'loc', 'indelgen_seq'],
            skiprows=1
        )
        return self.indel_gen_df

    def sanger_training(self):
        """
        Integrate decodr results with predicted indels and aggregate frequencies.

        Returns:
            pd.DataFrame: Aggregated results with columns:
                - 'Identifier': Matched indel ID
                - 'N_gt'/'indel_size': Indel size
                - 'Indelgen_seq': Predicted sequence
                - 'loc': Genomic location
                - 'Count': Aggregated frequency

        Outputs:
            {sanger_dir}/{gene}_SelfTarget.csv: Aggregated results file.
        """
        df = pd.DataFrame(self.finallist_dash)
        df['indel_size'] = df['N_gt']
        df['not_aligned'] = df['TG_ice'].apply(remove_dash)
        idf_map = defaultdict(def_value)
        idf_loc_map = defaultdict(def_value)
        idf_seq_map = defaultdict(def_value)
        # Annotate
        for i, row in df.iterrows():
            for j, indel_gen_row in self.indel_gen_df.iterrows():
                if row['Dash_range'] in indel_gen_row['loc']:
                    idf_map[row['Dash_range']] = indel_gen_row['Identifier']
                    idf_seq_map[row['Dash_range']] = indel_gen_row['indelgen_seq']
                    idf_loc_map[row['Dash_range']] = indel_gen_row['loc']
        df['Identifier'] = df['Dash_range'].map(idf_map)
        df['Indelgen_seq'] = df['Dash_range'].map(idf_seq_map)
        df['loc'] = df['Dash_range'].map(idf_loc_map)
        agg_df = df.groupby(["Identifier", "N_gt", "indel_size", "Indelgen_seq", "loc"]).agg({"ratio": "sum"})
        agg_df.rename({"ratio": "Count"}, axis=1).to_csv(f"{self.sanger_dir}/{self.gene}_SelfTarget.csv")
        return agg_df

def decide_duplicate(dfolder, seq):
    current_files=pd.read_csv('DEC_sum/NUM_'+seq+'.csv',index_col=0)
    rts={}
    for fd in set(dfolder['datemark']):
        current_check= current_files[current_files.index.str.startswith(fd)]
        # print(fd, current_check)
        compare=[]
        for i in range(current_check.shape[0]):
            extracted_values = re.findall(r'0\.\d+', current_check['files'][i])
            extracted_values = [float(value) for value in extracted_values]
            compare.append(sum(extracted_values))


        rts.update({current_check.index[np.argmax(compare)]: fd})
        print(current_check.index[np.argmax(compare)])

    return rts      

def decide_r2(seq, r2):
    current_files=pd.read_csv('DEC_sum/SUM_'+seq+'.csv',index_col=0)
    rts={}
    remain={}

    for i in range(current_files.shape[0]):
        extracted_values = re.findall(r'0\.\d+', current_files['files'][i])
        extracted_values = [float(value) for value in extracted_values]
        if np.mean(extracted_values)<r2:
            rts[current_files.index[i]]=np.mean(extracted_values)
        else:
            remain[current_files.index[i]]=np.mean(extracted_values)
    return rts
