"""
Quick script to rebuild data/somatic/TREX2_R1.csv with correct in_LdGen values.
Run this after indelgen files already exist (skips indelgentarget step).

Usage:
    /rds/user/wz369/hpc-work/LIBS/mamba/envs/inDecay/bin/python scripts/rebuild_trex2_csv.py
"""
import os, random
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

INDECAY_DIR  = "/rds/user/wz369/hpc-work/inDecay"
TREX2_DIR    = "/rds/user/wz369/hpc-work/crispr_nonCas9_preprocess/data/processed/trex2"
INDELGEN_DIR = f"{INDECAY_DIR}/data/somatic/Indelgen_result"
OUT_CSV      = f"{INDECAY_DIR}/data/somatic/TREX2_R1.csv"
OUT_TEST     = f"{INDECAY_DIR}/results/trex2_test_set.txt"
TEST_FRAC    = 0.1
SEED         = 42

# --- Load reference to get guide per OligoID ---
print("Loading reference FASTA ...")
ref_lookup = {}
for rec in SeqIO.parse(f"{INDECAY_DIR}/data/SelfTarget_NewScaffold.fasta", "fasta"):
    oligo_id, guide = rec.id.split("_", 1)
    ref_lookup[oligo_id] = guide

print(f"  {len(ref_lookup)} oligos in reference")

# --- Build in_LdGen lookup ---
print("Building in_LdGen lookup from indelgen files ...")
in_ldgen_lookup = {}
missing = 0
for oligo_id, guide in tqdm(ref_lookup.items()):
    gen_file = f"{INDELGEN_DIR}/{oligo_id}_{guide}_genindels.txt"
    if not os.path.exists(gen_file):
        missing += 1
        in_ldgen_lookup[oligo_id] = set()
        continue
    try:
        df = pd.read_table(gen_file, skiprows=1, header=None, sep='\t').iloc[:, :3]
        df.columns = ["Identifier", "n_coevent", "loc"]
        in_ldgen_lookup[oligo_id] = set(df["Identifier"].dropna().values)
    except Exception as e:
        in_ldgen_lookup[oligo_id] = set()

print(f"  {missing} indelgen files missing")
n_with_ids = sum(1 for s in in_ldgen_lookup.values() if len(s) > 0)
print(f"  {n_with_ids}/{len(ref_lookup)} oligos have indelgen identifiers")

# --- Load and enrich indel_profiles ---
print("Loading indel_profiles.csv ...")
df = pd.read_csv(f"{TREX2_DIR}/indel_profiles.csv", sep="\t")
print(f"  {len(df)} rows, {df['OligoID'].nunique()} oligos")

df["Strand"] = "FORWARD"
df["in_LdGen"] = df.apply(
    lambda row: row["Identifier"] in in_ldgen_lookup.get(row["OligoID"], set()),
    axis=1,
)

n_true = df["in_LdGen"].sum()
print(f"  in_LdGen=True: {n_true}/{len(df)} rows ({100*n_true/len(df):.1f}%)")

df.to_csv(OUT_CSV, index=False)
print(f"  Saved: {OUT_CSV}")

# --- Check passing oligos ---
filt = df[df["in_LdGen"] == True].copy()
filt["Count"] = filt["Count"].astype(float).astype(int)
filt = filt[(filt["Strand"] == "FORWARD") & (filt["Identifier"] != "Not Present")]
agg = filt.groupby("OligoID")["Count"].sum()
print(f"\nOligos with count >= 100: {(agg >= 100).sum()}")
print(f"Oligos with count >= 10:  {(agg >= 10).sum()}")

# --- Recreate test set ---
all_oligos = list(ref_lookup.keys())
random.seed(SEED)
random.shuffle(all_oligos)
n_test = max(1, int(len(all_oligos) * TEST_FRAC))
test_oligos = all_oligos[:n_test]
os.makedirs(os.path.dirname(OUT_TEST), exist_ok=True)
with open(OUT_TEST, "w") as f:
    f.write("\n".join(test_oligos) + "\n")
print(f"\nTest set: {n_test}/{len(all_oligos)} oligos -> {OUT_TEST}")
print("\nDone. Re-run the finetune script.")
