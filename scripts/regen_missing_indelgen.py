"""
Finds TREX2 oligos whose indelgen files are missing and regenerates them.
Run after prepare_trex2_data.py if some files failed.

Usage:
    /rds/user/wz369/hpc-work/LIBS/mamba/envs/inDecay/bin/python scripts/regen_missing_indelgen.py
"""
import os, subprocess
from Bio import SeqIO
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

INDECAY_DIR  = "/rds/user/wz369/hpc-work/inDecay"
FASTA        = f"{INDECAY_DIR}/data/SelfTarget_NewScaffold.fasta"
INDELGEN_DIR = f"{INDECAY_DIR}/data/somatic/Indelgen_result"
SIF          = "/rds/user/wz369/hpc-work/containers/selftarget.sif"
N_JOBS       = 8

def run_one(args):
    sif, refseq, pamsite, out_path = args
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path, True, "exists"
    cmd = ["singularity", "exec", sif, "indelgentarget", refseq, str(pamsite), out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return out_path, r.returncode == 0, r.stderr.strip()

records = list(SeqIO.parse(FASTA, "fasta"))
tasks = []
for rec in records:
    oligo_id, guide = rec.id.split("_", 1)
    out = f"{INDELGEN_DIR}/{oligo_id}_{guide}_genindels.txt"
    parts = rec.description.split()
    pamsite = int(parts[1])
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        tasks.append((SIF, str(rec.seq), pamsite, out))

print(f"Missing indelgen files: {len(tasks)}/{len(records)}")
if not tasks:
    print("Nothing to do.")
    exit(0)

ok = fail = 0
with ProcessPoolExecutor(max_workers=N_JOBS) as pool:
    futures = {pool.submit(run_one, t): t for t in tasks}
    for fut in tqdm(as_completed(futures), total=len(tasks)):
        _, success, msg = fut.result()
        if success:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL: {msg}")

print(f"Done: {ok} regenerated, {fail} still failed")
