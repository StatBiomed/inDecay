#!/bin/bash
# Full TREX2 data preparation job.
# Runs prepare_trex2_data.py to produce all files needed by the finetune script.
#
# Submit: sbatch scripts/run_trex2_prep.sh
# Or run interactively: bash scripts/run_trex2_prep.sh

#SBATCH --job-name=trex2_prep
#SBATCH --output=logs/trex2_prep_%j.log
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

PYTHON="python"  # TODO: set this path for your environment
INDECAY_DIR="/PATH/TO/inDecay"  # TODO: set this path for your environment
TREX2_DIR="/PATH/TO/trex2/processed"  # TODO: set this path for your environment
SIF="/PATH/TO/selftarget.sif"  # TODO: set this path for your environment

mkdir -p "${INDECAY_DIR}/logs"
cd "${INDECAY_DIR}"

"${PYTHON}" scripts/prepare_trex2_data.py \
    --trex2_dir   "${TREX2_DIR}" \
    --indecay_dir "${INDECAY_DIR}" \
    --sif         "${SIF}" \
    --cellline    TREX2 \
    --rep         R1 \
    --test_fraction 0.1 \
    --seed        42 \
    --n_jobs      "${SLURM_CPUS_PER_TASK:-8}"

echo "Preparation complete."
