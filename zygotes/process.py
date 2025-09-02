#!/usr/bin/env python3
"""
CRISPR Zygote Data Processing Pipeline

This pipeline organizes and analyzes CRISPR gene editing data from multiple sources (DECODR, ICE, SelfTarget).
It takes raw data files, processes gene sequences, runs SelfTarget analysis, transforms and quality-controls results,
aggregates and summarizes outputs, and finally generates standardized files for downstream use.
"""

from __future__ import annotations

import csv
import glob
import logging
import os
import shutil
import sys
from datetime import date
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from Bio import SeqIO  # noqa: F401
from snapgene_reader import snapgene_file_to_dict, snapgene_file_to_seqrecord  # noqa: F401

from inDecay import PATH
from inDecay.zygote import (
    create,
    process_duplicates,
    decide_r2,
    exp_to_guide,
    genepro_dec,
    genepro_ice,
    getselftarget,
    complement,
)

import warnings
warnings.filterwarnings("ignore")
pj=os.path.join
# ────────────────── Configuration & Constants ───────────────
TODAY_STR = date.today().strftime("%Y%m%d")

ROOT_DIR      = Path.cwd()
DECODR_DIR    = ROOT_DIR / "decodr"
RAW_DIR       = DECODR_DIR / "raw_dec"
OUTPUTS_DIR  = DECODR_DIR / "decodr_outputs"
SELFTARGET_DIR = ROOT_DIR / "SelfTarget"
LBD_DIR    = DECODR_DIR / "decodr_labeled"
AGG_DIR     = DECODR_DIR / "decodr_agg"
SUM_DIR = DECODR_DIR / "decodr_sum"

SPECIES_MAP = dict(m="mouse", p="porcine", s="sheep", c="cattle", g="goat")
R2_THRESHOLD = 0.0
seq='Sanger'
excludes=['mHV5-sg2','mKif27']
# excludes=['mELF5','mH2K1-sg3','mH2K1-sg2','mHV5-sg2','mULK-sg2','mKif27']
logging.basicConfig(
    level=logging.INFO,
    format="[{levelname:.1} @{asctime}]: {message}",
    style="{",
)

def parent_dir(path: Path | str, levels: int = 1) -> Path:
    """Return ancestor directory `levels` above *path*."""
    p = Path(path)
    for _ in range(levels):
        p = p.parent
    return p


# ────────────────────── Step 1: Parse Guides ────────────────
logging.info("Extracting guide information …")
guide_files = glob.glob(str(RAW_DIR / "*.xlsx"))
guide_map: dict[str, str] = {}
for xl_path in guide_files:
    guide_map.update(exp_to_guide(xl_path))
logging.info("Total guides extracted: %d", len(guide_map))

def get_selftarget(gene):
    return getselftarget(
        raw_dir=RAW_DIR,
        cwd_dir=ROOT_DIR ,
        guidedisc=guide_map,
        gene=gene,
    )

# ────────────────────── Step 2: SelfTarget Prep ─────────────
logging.info("Preparing SelfTarget commands …")

create(SELFTARGET_DIR)
def generate_self_target_script(guide_map, output_dir=ROOT_DIR):
    """Generate SelfTarget execution script without redundant commands"""
    # Use set to track unique commands and avoid duplicates
    unique_st = set()
    
    bash_script = output_dir / f"{TODAY_STR}_run_selftarget.sh"
    out_csv = output_dir / "gene_seq_all.csv"
    with out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["folder", "seq", "gDNA"])
        with bash_script.open("w") as sh:
            # Write header
            sh.write("#!/usr/bin/env bash\n")
            sh.write(f"cd {PATH.Indelana}\n\n")
            
            # Collect and write unique commands
            for file in guide_map.keys():
                target=get_selftarget(file)
                st = target.st.strip()
                if st and st not in unique_st:  # Skip empty/duplicate commands
                    unique_st.add(st)
                    sh.write(f"{st}\n")
                    writer.writerow([target.dir, target.shorten_ref, target.guide])
            
            # Write footer
            sh.write("\ncd -\n")
    
    # Set permissions and execute
    bash_script.chmod(0o755)
    logging.info("Executing %s (contains %d unique st)", bash_script, len(unique_st))
    
    try:
        subprocess.run([str(bash_script)], check=True)
    except subprocess.CalledProcessError as e:
        logging.error("Script execution failed: %s", e)
        raise


generate_self_target_script(guide_map)
# ──────────────────── Step 3: Transform Results ─────────────

logging.info("Transforming decodr outputs …")
create(LBD_DIR)
for gene in os.listdir(OUTPUTS_DIR):
    genepro_dec(
        raw_dir=RAW_DIR,
        cwd_dir=ROOT_DIR,
        guidedisc=guide_map,
        gene=gene.replace('.csv', '')
    )


# ────────────── Step 3.5: ICE Results Parsing (NEW SECTION) ──────────────

logging.info("Transforming ice outputs …")
ICE_DIR = ROOT_DIR / "synthego"
ICE_RAW = ICE_DIR / "raw_syn"
ICE_RAWRS= ICE_DIR / "ice_rawresults"
ICE_OUTPUTS = ICE_DIR / "ice_outputs"
ICE_LBD = ICE_DIR / "ice_labeled"

create(ICE_OUTPUTS)
create(ICE_LBD)
reverse_seq = []
head = ['ratio', 'event', 'seq', 'guide', 'r2', 'ori_seq']



genes = glob.glob(pj(ICE_RAWRS, '*/results/*/contribs.txt'))
for file in genes:
    data = pd.read_csv(file, sep=" ", header=None)
    batchfolder = Path(file).parts[-4]
    genename = Path(file).parts[-2]
    resultpath = ICE_RAWRS / batchfolder / f'summary_{batchfolder.split("_")[-1]}.csv'
    guidedf = pd.read_csv(resultpath, index_col=0).loc[str(genename)]
    df = data[[0, 2, 4]].iloc[2:]
    df.columns = head[0:3]

    strand = getselftarget(
        raw_dir=ICE_RAW,
        cwd_dir= ROOT_DIR,
        guidedisc=guide_map,
        gene=genename
    ).Strand
    if strand == 'FW':
        conc_dict = {
            "ratio": list(data[0].iloc[2:]),
            "event": list(data[2].iloc[2:]),
            "seq": list(data[4].iloc[2:]),
            "guide": [guidedf['Guide Sequences']] * len(data[0].iloc[2:]),
            "r2": [guidedf['R Squared']] * len(data[0].iloc[2:]),
            "ori_seq": list(data[4].iloc[2:])
        }
    elif strand == 'RC':
        conc_dict = {
            "ratio": list(data[0].iloc[2:]),
            "event": list(data[2].iloc[2:]),
            "seq": [''.join(map(complement, reversed(seq))) for seq in data[4].iloc[2:]],
            "guide": [guidedf['Guide Sequences']] * len(data[0].iloc[2:]),
            "r2": [guidedf['R Squared']] * len(data[0].iloc[2:]),
            "ori_seq": list(data[4].iloc[2:])
        }
        reverse_seq.append(genename)

    conc = pd.DataFrame(conc_dict)
    conc.to_csv(ICE_OUTPUTS / f'{genename}.csv', index=False)

    genepro_ice(
        raw_dir=ICE_RAW,
        cwd_dir=ROOT_DIR,
        guidedisc=guide_map,
        gene=genename,
    )


# ─────────────── Step 4: Sanger / ICE Aggregation ───────────
logging.info("Aggregating Sanger & ICE folders …")
folder_r2_map: dict[str, list[tuple[str, float]]] = {
    f: [
        (
            csv_name,
            pd.read_csv(OUTPUTS_DIR / csv_name.replace("_SelfTarget", ""))["rSquared"].iat[-1],
        )
        for csv_name in os.listdir(LBD_DIR)
        if f in csv_name and not csv_name.startswith(".")
    ]
    for f in {n.split("---")[0] for n in os.listdir(LBD_DIR) if '.DS_Store' not in n }
}
# ICE fallback if DEC run failed


folder_r2_map_ice: dict[str, list[tuple[str, float]]] = {
    f: [
        (
            csv_name,
            pd.read_csv(ICE_OUTPUTS / csv_name.replace("_SelfTarget", ""))["r2"].iat[-1],
        )
        for csv_name in os.listdir(ICE_LBD)
        if f in csv_name and not csv_name.startswith(".")
    ]
    for f in {n.split("---")[0] for n in os.listdir(ICE_LBD) if '.DS_Store' not in n }
}

for folder in folder_r2_map_ice.keys() - folder_r2_map.keys():
    r2_list = []
    for csv_name in os.listdir(ICE_LBD):
        if folder in csv_name:
            r2 = pd.read_csv(ICE_OUTPUTS / csv_name.replace("_SelfTarget", ""))["r2"].iat[-1]
            if r2 > 0.9:
                shutil.copy2(ICE_LBD / csv_name, LBD_DIR / csv_name)
                r2_list.append((csv_name, r2))
    if r2_list:
        folder_r2_map[folder] = r2_list


# ─────────── Step 5: Gene Summary CSV (genesummary.csv) ─────
# logging.info("Building gene summary …")
# summary_2024 = (
#     pd.read_csv("gene2024_archive_SampleType.csv", index_col=0)
#     .drop(columns="guide")
# )
# summary_2025 = pd.read_csv("gene2025.csv")
# summary_2025["SampleType"] = summary_2025["SampleType"].str.capitalize()

geneall = pd.read_csv("geneall.csv")
gene_seq_df = pd.read_csv("gene_seq_all.csv", index_col=0)

summary = (
    geneall.assign(
        guide=lambda df: df.folder.str.split("_").str[0],
        date=lambda df: df.folder.str.split("_").str[-1],
        count=lambda df: df["count"].fillna(0).astype(int),
        species=lambda df: df.folder.str[0].map(SPECIES_MAP),
        seq=lambda df: df.folder.map(gene_seq_df["seq"]),
        duplicate=lambda df: df.date.str[-1].isin(["R", "F", "M"]).astype(int),
    )
)
summary.to_csv("genesummary.csv", index=False)

# ───────── Step 6: Merge Individual Counts → DEC_sum/* ──────
def merge_counts_by_seq(seq: str = "Sanger") -> None:
    out_root = AGG_DIR
    create(out_root)
    seq_dir = out_root / seq
    create(seq_dir)
    with (out_root / f"NUM_{seq}.csv").open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["folder", "num", "files"])
        for folder, r2_pairs in folder_r2_map.items():
            dfs, files, r2_vals = [], [], []
            for csv_name, r2_val in r2_pairs:
                df = pd.read_csv(LBD_DIR / csv_name)
                dfs.append(df)
                files.append(csv_name.replace("_SelfTarget", ""))
                r2_vals.append(r2_val)
            if dfs:
                (pd.concat(dfs)
                   .groupby(["N_gt", "loc", "indel_size", "Indelgen_seq", "Identifier"])["Count"]
                   .sum()
                   .to_csv(seq_dir / f"{folder}.csv"))
                writer.writerow([folder, len(dfs), list(zip(files, r2_vals))])
merge_counts_by_seq("Sanger")

# ────────────── Step 7: Duplicate Handling & Final Summary ───────
logging.info("Removing rejected duplicates …")
num_table = pd.read_csv(AGG_DIR / f"NUM_{seq}.csv", index_col=0)
clean_idx = process_duplicates(num_table, summary)
num_table = num_table.loc[clean_idx]
summary.set_index('folder',inplace=True)
merge_summ = pd.merge(num_table, summary, left_index=True, right_index=True)
merge_summ['num'] = merge_summ['num'].astype('str')
merge_summ['count'] = merge_summ['count'].astype('str')
merge_summ.reset_index(inplace=True)
num_table.reset_index(inplace=True)

final_summary = merge_summ.groupby("guide").agg({
        "count":  ",".join,
        "files":  ",".join,
        "num":    ",".join,
        "SampleType": ",".join,
        "date":   ",".join,
        "folder": ",".join,
        "seq":    lambda x: x.unique(),
        "gDNA":   lambda x: x.unique(),
    })
final_summary[["seq", "gDNA"]] = final_summary[["seq", "gDNA"]].applymap(
    lambda v: ",".join(map(str, v))
)
final_summary.to_csv(AGG_DIR / "SUM_Sanger.csv")

# ────────────── Step 8: Grouped Summary & Count Table Generation ──────────────


gene_file_map = {}
gene_sample_num = {}

summary_path = AGG_DIR / f"SUM_{seq}.csv"
summary = pd.read_csv(summary_path, index_col=0)

for gene in summary.index:
    file_list = [f"{fname}.csv" for fname in summary.loc[gene, 'folder'].split(',')]
    gene_file_map[gene] = file_list
    sample_type = summary.loc[gene, 'SampleType']
    if sample_type == 'Clonal':
        num = [int(summary.loc[gene, 'num'])]
    elif sample_type == 'Bulk':
        num = [int(summary.loc[gene, 'count'])]
    else:
        types = sample_type.split(',')
        nums = summary.loc[gene, 'num'].split(',')
        counts = summary.loc[gene, 'count'].split(',')
        num = [int(counts[k]) if t == 'Bulk' else int(nums[k]) for k, t in enumerate(types)]
    gene_sample_num[gene] = num

dict_csv_path = AGG_DIR / f"DICT_{seq}.csv"
with open(dict_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Item', 'Files', 'SampleType', 'Nums', 'Events'])
    for gene, files in gene_file_map.items():
        if files:
            writer.writerow([
                gene,
                files,
                summary.loc[gene, 'SampleType'],
                gene_sample_num[gene]
            ])

thre_r2 = 0  # R² threshold, can be adjusted as needed

create(SUM_DIR)
group_count_dir = SUM_DIR / f"Count_{seq}_{thre_r2}"
create(group_count_dir)

dict_csv_path = AGG_DIR / f"DICT_{seq}.csv"
dict_df = pd.read_csv(dict_csv_path, index_col=0)

for i in range(dict_df.shape[0]):
    combined_dfs = []
    file_str = dict_df.iloc[i]['Files']
    file_list = [f.strip().replace("'", "") for f in file_str.strip("[]").split(',')]
    num_str = dict_df.iloc[i]['Nums']
    num_list = [float(n.strip().replace("'", "")) for n in num_str.strip("[]").split(',')]
    sample_types = dict_df.iloc[i]['SampleType'].split(',')

    for idx, file_name in enumerate(file_list):
        file_path = AGG_DIR / seq / file_name
        dfi = pd.read_csv(file_path)
        if sample_types[idx].strip() == 'Bulk':
            dfi['Count'] = dfi['Count'] * num_list[idx]
        
        combined_dfs.append(dfi)

    merged_df = pd.concat(combined_dfs)
    merged_df = (
        merged_df.groupby(['N_gt', 'loc', 'indel_size', 'Indelgen_seq', 'Identifier'])['Count']
        .sum()
        .reset_index()
        .sort_values(by='Count', ascending=False)
    )
    out_file = group_count_dir / f"{dict_df.index[i]}_SelfTarget.csv"
    merged_df.to_csv(out_file, index=False)

# ───────────── Step 9: Export Training CSV for inDecay ──────
logging.info("Exporting training CSV …")
train_df = (
    final_summary.reset_index()[["guide", "seq"]]
    .assign(
        r2=list(decide_r2(current_files=final_summary, seq= seq, r2=1-R2_THRESHOLD).values()),
        count=[pd.read_csv(SUM_DIR / f"Count_Sanger_{int(R2_THRESHOLD)}" / f"{g}_SelfTarget.csv")
    .loc[lambda df: df['Identifier'] != 'Not Present', 'Count'].sum()
    for g in final_summary.index]
    )
)
train_df.to_csv(pj(PATH.data_dir, "gene_seq.csv"), index=False)
train_df.set_index('guide', inplace=True)

# Copy species-specific files
for i in SPECIES_MAP:
    create(pj(PATH.data_dir, SPECIES_MAP[i]))
for guide in train_df.index:
    species_dir = Path(PATH.data_dir) / SPECIES_MAP[guide[0]]
    if not guide in excludes:
            # shutil.copy2(pj(PATH.data_dir, "gene_seq.csv"), species_dir)
        shutil.copy2(
                SUM_DIR / f"Count_Sanger_{int(R2_THRESHOLD)}" / f"{guide}_SelfTarget.csv",
                species_dir / f"{guide}_SelfTarget.csv",
            )
        

logging.info("Pipeline complete.")
