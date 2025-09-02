import os
import shutil
from Bio import SeqIO, Seq
import pandas as pd
pj=os.path.join
os.getcwd()
import warnings
# Ignore all warnings
warnings.filterwarnings('ignore')
import shutil
from bs4 import BeautifulSoup 
import requests
import time
from tqdm import tqdm
rawfile_dir='raw_all'
def copy_if_empty(src_dir, target_dir, basename, save_name):
    if not os.path.exists(f"{target_dir}/{basename}"):
        shutil.copy2(f"{src_dir}/{basename}", target_dir)
        name1=save_name+'---'+basename
        newname= name1.replace(' ','-').replace('&','and').replace('(','').replace(')','-')
        os.rename(f"{target_dir}/{basename}",f"{target_dir}/{newname}")
def trunc_filename(ab1_file):
    return ab1_file.replace(".ab1","").replace('&','and').replace('(','').replace(')','-')


def process_guide(FW, RC, guide):
    for j, seq in enumerate([FW, RC]):
        if guide in seq:
            g_index = seq.index(guide)
            x = (g_index + len(guide)) - 42
            shorten_ref = seq[x:x+79]
            # self.shorten_ref = seq[self.cutsite-39:self.cutsite] + seq[self.cutsite:self.cutsite+40]
            Strand = ['FW', 'RC'][j]
            if shorten_ref[43:45] == "GG":
                return Strand, shorten_ref
            else:
                shorten_ref=""
        else:
            shorten_ref=""

    if shorten_ref=="":
        return False, False
def read_control(control):
    if control.endswith("ab1"):
        seqRecord = SeqIO.read(control, "abi")
        FW, RC = str(seqRecord.seq), str(seqRecord.seq.reverse_complement())
        
    elif control.endswith("txt"):
        with open(control, 'r') as f:
            FW = f.readlines()[0]
        RC = str(Seq.Seq(FW).reverse_complement())
    return FW, RC
def complement(c):
  if c == 'A': return 'T'
  if c == 'T': return 'A'
  if c == 'C': return 'G'
  if c == 'G': return 'C'
  else: return c
  
def find_ab1_and_control(folder, stype, Guide="", requirements=[], save_name=None):
    """
    Process AB1 files and generate control and experiment definitions.
    
    Parameters:
        folder (str): The directory name containing raw AB1 files.
        stype (str): Sample type (e.g., 'Clonal' or 'Bulk').
        Guide (str): sgRNA guide sequence used for alignment.
        requirements (list): List of strings to filter filenames.
        save_name (str): Name to use when saving output files.
    
    Returns:
        tuple: Lists of ABI files, ICE definition DataFrame, DECODR definition DataFrame.
    """
    # Define paths
    success_folder = f"{rawfile_dir}/{folder}"
    control_exp_ice = f"intermediate/synthego/raw_syn/{save_name}"
    control_exp_decodr = f"intermediate/decodr/raw_dec/{save_name}"

    # Make directories if they don't exist
    os.makedirs(control_exp_ice, exist_ok=True)
    os.makedirs(control_exp_decodr, exist_ok=True)

    # Find control file (WT contains 'WT')
    control = [r for r in os.listdir(success_folder) if '.ab1' in r and 'WT' in r]
    assert len(control) == 1, f"Please check WT in the folder, {control}"
    control_path = os.path.join(success_folder, control[0])

    # Process guide RNA to determine strand and reference sequence
    if Guide:
        WT_seq, RC_WT = read_control(control_path)
        strand, shorten_ref = process_guide(WT_seq, RC_WT, Guide)
        if strand == 'FW' and len(shorten_ref) == 79:
            print("find sgRNA in forward strand with PAM")
        elif strand == 'RC' and len(shorten_ref) == 79:
            print("find sgRNA in reverse strand with PAM")

    # Copy control file to destination folders
    copy_if_empty(success_folder, control_exp_ice, control[0], save_name)
    copy_if_empty(success_folder, control_exp_decodr, control[0], save_name)

    # Filter experiment files based on requirements
    abi = [r for r in os.listdir(success_folder) if '.ab1' in r and 'WT' not in r and '.DS_Store' not in r]
    for cond in requirements:
        abi = list(filter(lambda x: cond in x, abi))

    # Copy filtered experiment files
    for r in abi:
        copy_if_empty(success_folder, control_exp_ice, r, save_name)
        copy_if_empty(success_folder, control_exp_decodr, r, save_name)
    WT_abi= [r for r in os.listdir(control_exp_ice) if 'WT' in r]
    exp_abi= [r for r in os.listdir(control_exp_ice) if '.ab1' in r and 'WT' not in r and '.DS_Store' not in r]

    # Generate ICE and DECODR definition files
    define_dict_ice = {
        "Label" : [trunc_filename(f) for f in exp_abi],
        "Control File" : [WT_abi[0]] * len(exp_abi),
        "Experiment File":exp_abi,
        "Guide Sequence" : [Guide] * len(exp_abi),
        "Donor Sequence" : [''] * len(exp_abi),}
    
    definition_ice = pd.DataFrame(define_dict_ice)

    define_dict_dec = {
        "Sample Title" : [trunc_filename(f) for f in exp_abi],
        "Sample Type" : [stype] * len(exp_abi),
        "Guide Sequence(s)" : [Guide] * len(exp_abi),
        "Nuclease" : ['Cas9'] * len(exp_abi),
        "Donor Template (Optional)" : ["None"] * len(exp_abi),
        "Control Data" : [WT_abi[0]] * len(exp_abi),
        "Experiment File(s)":exp_abi,
                   }
    definition_dec = pd.DataFrame(define_dict_dec)

    # Save definition files
    definition_ice.to_excel(f"{control_exp_ice}.xlsx",index=False)
    definition_dec.to_excel(f"{control_exp_decodr}.xlsx",index=False)

    return exp_abi, definition_ice, definition_dec

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
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'
        
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
    wrong=[]
    df_dicst = {}
    for i, row in tqdm(link_df.iterrows()):
        try:
            sample = row['Sample Title']
            # time.sleep(1)  # Avoid hammering the server
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
        except:
            wrong.append(sample)
    return df_dicst, fails, wrong
