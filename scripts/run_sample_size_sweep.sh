#!/bin/bash
# Sample-size sweep: N in [0,10,30,50,100,300,500,1000], 5 repeats each.
# Jobs run in parallel; concurrency is capped by CPU and GPU resources.
#
# CPU: each job uses num_workers=12 DataLoader threads + 1 main → 13 CPUs/job
# GPU: CUDA context overhead ~2 GB/job on the available A100-80GB
# Effective limit = min(floor(NCPU/13), floor(GPU_FREE_MiB/2048))

set -euo pipefail

PYTHON=/rds/user/wz369/hpc-work/LIBS/mamba/envs/inDecay/bin/python
SCRIPT=scripts/STfeatv5_inDecay_finetune.py
BASE_ARGS="-E ST_trex2_2024_TREX2_R1 \
           -P pretrained/K562_featv5_pretrained.ckpt \
           -M ST_DeepDecay -C 100 \
           -T results/trex2_test_set.txt \
           -G 0 -R 100"

# ── Resource detection ────────────────────────────────────────────────────────
NCPU=$(grep -c "^processor" /proc/cpuinfo 2>/dev/null || nproc)
GPU_FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
               2>/dev/null | head -1 || echo 4096)

CPU_LIMIT=$(( NCPU / 13 ))
GPU_LIMIT=$(( GPU_FREE_MIB / 2048 ))
MAX_JOBS=$(( CPU_LIMIT < GPU_LIMIT ? CPU_LIMIT : GPU_LIMIT ))
MAX_JOBS=$(( MAX_JOBS > 1 ? MAX_JOBS : 1 ))

echo "Resources: ${NCPU} CPUs | ${GPU_FREE_MIB} MiB GPU free"
echo "Parallelism: CPU cap=${CPU_LIMIT}  GPU cap=${GPU_LIMIT}  → using ${MAX_JOBS} parallel jobs"
echo

# ── Per-job runner ────────────────────────────────────────────────────────────
mkdir -p logs

run_job() {
    local N=$1 REP=$2
    local logfile="logs/sweep_N${N}_rep${REP}.log"
    echo "[START] N=${N} rep=${REP}"
    if bash --noprofile --norc -c \
            "$PYTHON $SCRIPT $BASE_ARGS -N $N --repeat $REP" \
            > "$logfile" 2>&1; then
        echo "[DONE]  N=${N} rep=${REP}"
    else
        echo "[FAIL]  N=${N} rep=${REP}  (see $logfile)"
    fi
}
export -f run_job
export PYTHON SCRIPT BASE_ARGS

# ── Sweep ────────────────────────────────────────────────────────────────────
SAMPLE_SIZES=(0 10 30 50 100 300 500 1000)
REPEATS=(1 2 3 4 5)

for N in "${SAMPLE_SIZES[@]}"; do
    for REP in "${REPEATS[@]}"; do
        # Throttle: wait until a slot is free
        while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
            sleep 2
        done
        run_job "$N" "$REP" &
    done
done

wait
echo
echo "=== Sweep complete ==="
echo "Results in: results/Transfer/C100/"
ls results/Transfer/C100/*/N*_rep*.json 2>/dev/null | wc -l | xargs -I{} echo "JSON files written: {}"
