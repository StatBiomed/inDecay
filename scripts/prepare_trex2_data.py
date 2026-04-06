"""
prepare_trex2_data.py
=====================
Data preparation pipeline for finetuning inDecay on TREX2 non-Cas9 data.

This script converts the raw TREX2 preprocessed outputs into the file formats
expected by STfeatv5_inDecay_finetune.py.

Steps
-----
1. Parse TREX2 reference.fasta and extract 20bp guide from pamsite.
2. Write a reformatted reference FASTA: >OligoID_GuideID pamsite Strand
3. Run indelgentarget (via Singularity) for every unique OligoID.
4. Cross-reference indel_profiles.csv with indelgen universe -> add `in_LdGen`.
5. Add `Strand = FORWARD` column and write final somatic CSV.
6. Create a random 10% hold-out test set file.

Usage
-----
    python scripts/prepare_trex2_data.py \\
        --trex2_dir /rds/user/wz369/hpc-work/crispr_nonCas9_preprocess/data/processed/trex2 \\
        --indecay_dir /rds/user/wz369/hpc-work/inDecay \\
        --sif /rds/user/wz369/hpc-work/containers/selftarget.sif \\
        --cellline TREX2 --rep R1 \\
        --test_fraction 0.1 --seed 42 --n_jobs 8
"""

import os
import sys
import argparse
import subprocess
import random
import math
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

pj = os.path.join


# ---------------------------------------------------------------------------
# 1. Parse TREX2 reference FASTA and extract 20-bp guide upstream of cut site
# ---------------------------------------------------------------------------

def extract_guide(refseq: str, pamsite: int, guide_len: int = 20) -> str:
    """
    Extract guide sequence from the reference.

    For a Cas9-style design: guide = seq[pamsite - guide_len - 3 : pamsite - 3]
    i.e. the 20 bp immediately upstream of the PAM (cutsite = pamsite - 3).

    We strip any trailing 'N' characters and take the last `guide_len` clean bp.
    """
    cutsite = pamsite - 3
    start = max(0, cutsite - guide_len)
    guide = refseq[start:cutsite].replace("N", "")
    return guide[-guide_len:] if len(guide) >= guide_len else guide


def parse_trex2_fasta(fasta_path: str):
    """
    Parse the TREX2 reference.fasta.

    TREX2 header format: >OligoID pamsite Strand
    Returns list of dicts: {OligoID, pamsite, Strand, refseq, guide}
    """
    records = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        parts = rec.description.split()
        # parts: [OligoID, pamsite, Strand]
        oligo_id = parts[0]
        pamsite = int(parts[1])
        strand = parts[2]
        refseq = str(rec.seq)
        guide = extract_guide(refseq, pamsite)
        records.append({
            "OligoID": oligo_id,
            "pamsite": pamsite,
            "Strand": strand,
            "refseq": refseq,
            "guide": guide,
        })
    return records


# ---------------------------------------------------------------------------
# 2. Write reformatted reference FASTA  >OligoID_GuideID pamsite Strand
# ---------------------------------------------------------------------------

def write_inDecay_fasta(records, out_path: str):
    """
    Write a reference FASTA compatible with reader.get_reference().

    Header format expected by inDecay:
        >OligoID_GuideID pamsite Strand
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bio_records = []
    for r in records:
        seq_id = f"{r['OligoID']}_{r['guide']}"
        description = f"{r['pamsite']} {r['Strand']}"
        bio_rec = SeqRecord(Seq(r["refseq"]), id=seq_id, description=description)
        bio_records.append(bio_rec)
    with open(out_path, "w") as f:
        SeqIO.write(bio_records, f, "fasta")
    print(f"[✓] Reference FASTA written: {out_path} ({len(bio_records)} entries)")


# ---------------------------------------------------------------------------
# 3. Run indelgentarget via Singularity
# ---------------------------------------------------------------------------

def _find_indelgentarget_in_sif(sif_path: str) -> str:
    """
    Probe the Singularity image to find the indelgentarget binary path.
    Falls back to 'indelgentarget' (assumes it's on PATH inside container).
    """
    candidates = [
        "/SelfTarget/indel_prediction/bin/indelgentarget",
        "/usr/local/bin/indelgentarget",
        "indelgentarget",
    ]
    for c in candidates:
        result = subprocess.run(
            ["singularity", "exec", sif_path, "test", "-x", c],
            capture_output=True,
        )
        if result.returncode == 0:
            return c
    return "indelgentarget"


def run_indelgentarget(args_tuple):
    """
    Worker function: run indelgentarget for a single oligo.

    args_tuple = (sif_path, exe_in_sif, refseq, pamsite, out_path)
    Returns (out_path, success, error_msg)
    """
    sif_path, exe_in_sif, refseq, pamsite, out_path = args_tuple

    if os.path.exists(out_path):
        return out_path, True, "already_exists"

    cmd = [
        "singularity", "exec", sif_path,
        exe_in_sif, refseq, str(pamsite), out_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return out_path, False, result.stderr.strip()

    # Normalise to 3-column format expected by reader.py:
    #   col0: Identifier, col1: n_coevent, col2: loc
    # indelgentarget may produce 4 columns (Identifier, Collapsed, Details, Outcome_seq)
    # We keep only the first 3 and rename if needed.
    _normalise_indelgen_file(out_path)
    return out_path, True, ""


def _normalise_indelgen_file(path: str):
    """
    Ensure indelgen file has exactly 3 tab-separated columns after the header.
    reader.py reads: skiprows=1, names=['Identifier', 'n_coevent', 'loc']
    pandas raises ParserError if file has != 3 columns when names has 3 entries.

    indelgentarget may produce 4 columns: Identifier, Collapsed, Details, Outcome_seq
    We truncate/pad to exactly 3 (keeping Identifier as col 0).
    """
    try:
        with open(path) as f:
            lines = f.readlines()
        if not lines:
            return
        header = lines[0]
        data_lines = lines[1:]
        if not data_lines:
            return
        ncols = len(data_lines[0].rstrip("\n").split("\t"))
        if ncols == 3:
            return  # already correct
        # Truncate to 3 or pad to 3
        new_data = []
        for line in data_lines:
            parts = line.rstrip("\n").split("\t")
            while len(parts) < 3:
                parts.append("0")
            new_data.append("\t".join(parts[:3]) + "\n")
        with open(path, "w") as f:
            f.write(header)
            f.writelines(new_data)
    except Exception:
        pass  # leave file as-is if parsing fails


def generate_indelgen_files(records, indelgen_dir: str, sif_path: str, n_jobs: int = 8):
    """
    Run indelgentarget for all oligos and save to indelgen_dir.
    """
    os.makedirs(indelgen_dir, exist_ok=True)
    exe_in_sif = _find_indelgentarget_in_sif(sif_path)
    print(f"[i] Using indelgentarget binary inside container: {exe_in_sif}")

    tasks = []
    for r in records:
        out_file = pj(indelgen_dir, f"{r['OligoID']}_{r['guide']}_genindels.txt")
        tasks.append((sif_path, exe_in_sif, r["refseq"], r["pamsite"], out_file))

    n_success, n_fail, n_skip = 0, 0, 0
    errors = []

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(run_indelgentarget, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="indelgentarget"):
            out_path, ok, msg = fut.result()
            if msg == "already_exists":
                n_skip += 1
            elif ok:
                n_success += 1
            else:
                n_fail += 1
                errors.append((out_path, msg))

    print(f"[✓] indelgentarget: {n_success} generated, {n_skip} skipped, {n_fail} failed")
    if errors:
        print(f"[!] First 5 failures:")
        for p, e in errors[:5]:
            print(f"    {p}: {e}")
    return n_fail == 0


# ---------------------------------------------------------------------------
# 4 & 5. Build somatic CSV with `in_LdGen` and `Strand` columns
# ---------------------------------------------------------------------------

def build_somatic_csv(indel_profiles_path: str, records, indelgen_dir: str, out_csv: str):
    """
    Enrich indel_profiles.csv with:
      - Strand: constant 'FORWARD'
      - in_LdGen: True if the Identifier appears in indelgentarget output for that oligo

    Saves to out_csv.
    """
    print("[i] Loading indel_profiles.csv …")
    df = pd.read_csv(indel_profiles_path, sep="\t")

    # Build a lookup: OligoID -> set of indelgen identifiers
    ref_lookup = {r["OligoID"]: r["guide"] for r in records}

    print("[i] Building in_LdGen lookup …")
    in_ldgen_lookup = {}
    missing_files = 0
    for oligo_id, guide in tqdm(ref_lookup.items(), desc="in_LdGen"):
        gen_file = pj(indelgen_dir, f"{oligo_id}_{guide}_genindels.txt")
        if not os.path.exists(gen_file):
            missing_files += 1
            in_ldgen_lookup[oligo_id] = set()
            continue
        try:
            idfgen = pd.read_table(gen_file, skiprows=1, header=None, sep='\t').iloc[:, :3]
            idfgen.columns = ["Identifier", "loc", "sequence"]
            in_ldgen_lookup[oligo_id] = set(idfgen["Identifier"].dropna().values)
        except Exception as e:
            in_ldgen_lookup[oligo_id] = set()

    if missing_files:
        print(f"[!] {missing_files} indelgen files not found — in_LdGen=False for those oligos")

    df["Strand"] = "FORWARD"
    df["in_LdGen"] = df.apply(
        lambda row: row["Identifier"] in in_ldgen_lookup.get(row["OligoID"], set()),
        axis=1,
    )

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    n_ldgen = df["in_LdGen"].sum()
    print(f"[✓] Somatic CSV written: {out_csv}")
    print(f"    Total rows: {len(df)}, in_LdGen=True: {n_ldgen} ({100*n_ldgen/len(df):.1f}%)")


# ---------------------------------------------------------------------------
# 6. Create random 10% test set
# ---------------------------------------------------------------------------

def create_test_set(records, out_txt: str, test_fraction: float = 0.1, seed: int = 42):
    """
    Hold out `test_fraction` of OligoIDs as test set.
    Writes one OligoID per line to out_txt.
    """
    all_oligos = [r["OligoID"] for r in records]
    random.seed(seed)
    random.shuffle(all_oligos)
    n_test = max(1, int(len(all_oligos) * test_fraction))
    test_oligos = all_oligos[:n_test]

    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w") as f:
        for oligo in test_oligos:
            f.write(oligo + "\n")
    print(f"[✓] Test set written: {out_txt} ({n_test}/{len(all_oligos)} oligos = {100*test_fraction:.0f}%)")
    return test_oligos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trex2_dir", required=True,
                        help="Path to processed TREX2 data dir (contains indel_profiles.csv, reference.fasta)")
    parser.add_argument("--indecay_dir", required=True,
                        help="Root of inDecay repo (data/ and results/ subdirs will be created here)")
    parser.add_argument("--sif", required=True,
                        help="Path to selftarget Singularity .sif image")
    parser.add_argument("--cellline", default="TREX2",
                        help="Cell line label used in output file naming (default: TREX2)")
    parser.add_argument("--rep", default="R1",
                        help="Replicate label (default: R1)")
    parser.add_argument("--test_fraction", type=float, default=0.1,
                        help="Fraction of oligos held out for test set (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for test/train split (default: 42)")
    parser.add_argument("--n_jobs", type=int, default=8,
                        help="Parallel workers for indelgentarget (default: 8)")
    parser.add_argument("--skip_indelgen", action="store_true",
                        help="Skip running indelgentarget (use if files already exist)")
    args = parser.parse_args()

    # Paths
    trex2_fasta = pj(args.trex2_dir, "reference.fasta")
    trex2_profiles = pj(args.trex2_dir, "indel_profiles.csv")
    data_dir = pj(args.indecay_dir, "data")
    somatic_dir = pj(data_dir, "somatic")
    indelgen_dir = pj(somatic_dir, "Indelgen_result")
    out_fasta = pj(data_dir, "SelfTarget_NewScaffold.fasta")
    out_csv = pj(somatic_dir, f"{args.cellline}_{args.rep}.csv")
    out_test = pj(args.indecay_dir, "results", f"trex2_test_set.txt")

    print("=" * 60)
    print("inDecay TREX2 Data Preparation")
    print("=" * 60)
    print(f"  TREX2 data dir : {args.trex2_dir}")
    print(f"  inDecay dir    : {args.indecay_dir}")
    print(f"  Singularity    : {args.sif}")
    print(f"  Output CSV     : {out_csv}")
    print(f"  Output FASTA   : {out_fasta}")
    print(f"  Test set       : {out_test}")
    print("=" * 60)

    # Validate inputs
    for path, label in [(trex2_fasta, "reference.fasta"), (trex2_profiles, "indel_profiles.csv")]:
        if not os.path.exists(path):
            sys.exit(f"[ERROR] {label} not found: {path}")
    if not os.path.exists(args.sif) and not args.skip_indelgen:
        sys.exit(f"[ERROR] Singularity image not found: {args.sif}\n"
                 f"  Pull it with: singularity pull {args.sif} docker://quay.io/felicityallen/selftarget")

    # Step 1: Parse TREX2 reference
    print("\n[Step 1] Parsing TREX2 reference FASTA …")
    records = parse_trex2_fasta(trex2_fasta)
    print(f"         {len(records)} oligos found")

    # Step 2: Write reformatted reference FASTA
    print("\n[Step 2] Writing inDecay-compatible reference FASTA …")
    # If SelfTarget_NewScaffold.fasta already exists (original Cas9 data), append TREX2 entries
    if os.path.exists(out_fasta):
        existing = list(SeqIO.parse(out_fasta, "fasta"))
        existing_ids = {r.id.split("_")[0] for r in existing}
        new_records = [r for r in records if r["OligoID"] not in existing_ids]
        print(f"         Appending {len(new_records)} new TREX2 entries to existing {out_fasta}")
        bio_new = [
            SeqRecord(Seq(r["refseq"]),
                      id=f"{r['OligoID']}_{r['guide']}",
                      description=f"{r['pamsite']} {r['Strand']}")
            for r in new_records
        ]
        with open(out_fasta, "a") as f:
            SeqIO.write(bio_new, f, "fasta")
    else:
        write_inDecay_fasta(records, out_fasta)

    # Step 3: Run indelgentarget
    if not args.skip_indelgen:
        print(f"\n[Step 3] Generating indelgen files ({len(records)} oligos, {args.n_jobs} workers) …")
        generate_indelgen_files(records, indelgen_dir, args.sif, n_jobs=args.n_jobs)
    else:
        print("\n[Step 3] Skipping indelgentarget (--skip_indelgen set)")

    # Steps 4 & 5: Build somatic CSV
    print("\n[Steps 4-5] Building somatic CSV with in_LdGen and Strand …")
    build_somatic_csv(trex2_profiles, records, indelgen_dir, out_csv)

    # Step 6: Create test set
    print("\n[Step 6] Creating test set …")
    create_test_set(records, out_test, test_fraction=args.test_fraction, seed=args.seed)

    print("\n" + "=" * 60)
    print("Data preparation complete.")
    print("\nNext: pull the Singularity image (if not done yet):")
    print(f"  singularity pull {args.sif} docker://quay.io/felicityallen/selftarget")
    print("\nThen run finetuning:")
    print(f"  python scripts/STfeatv5_inDecay_finetune.py \\")
    print(f"    -E ST_trex2_2024_{args.cellline}_{args.rep} \\")
    print(f"    -P pretrained/K562_featv5_pretrained.ckpt \\")
    print(f"    -M ST_DeepDecay -N 50 -R 100 -C 100 \\")
    print(f"    -T results/trex2_test_set.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
