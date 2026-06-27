#!/bin/bash
# Build the selftarget Singularity image from Docker Hub.
# Run this once before prepare_trex2_data.py.
#
# Usage:
#   bash scripts/build_selftarget_sif.sh
#
# Or submit as a SLURM job:
#   sbatch scripts/build_selftarget_sif.sh

#SBATCH --job-name=build_selftarget_sif
#SBATCH --output=logs/build_selftarget_sif_%j.log
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail

SIF_PATH="/PATH/TO/selftarget.sif"  # TODO: set this path for your environment
DOCKER_URI="docker://quay.io/felicityallen/selftarget"

# Use a writable tmp dir on scratch for the build cache
APPTAINER_TMPDIR="${TMPDIR:-/tmp}/apptainer_build_$$"
mkdir -p "$APPTAINER_TMPDIR"
export APPTAINER_TMPDIR

echo "Pulling ${DOCKER_URI} -> ${SIF_PATH}"
echo "APPTAINER_TMPDIR: ${APPTAINER_TMPDIR}"

singularity pull --force "${SIF_PATH}" "${DOCKER_URI}"

echo "Done. Image saved to ${SIF_PATH}"
echo "Size: $(du -sh ${SIF_PATH})"

# Quick sanity check: find indelgentarget inside the image
echo ""
echo "Checking for indelgentarget binary inside container..."
singularity exec "${SIF_PATH}" find / -name "indelgentarget" -type f 2>/dev/null | head -5 || true

rm -rf "$APPTAINER_TMPDIR"
