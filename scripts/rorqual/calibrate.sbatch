#!/bin/bash
# Phase 2 calibration on one full Rorqual H100: times 50 training steps per registered mechanism
# at 32 px / P4, 160 px / P4 and 160 px / P8, writes results/calibration.csv in the repository
# clone and prints the 160 px block projection against the 40 GPU-hour decision rule.
# A full H100 (not a MIG slice): the plan takes every time and memory number on one H100 type.
#
# Submit from a login node (the account comes from env.sh):
#     source ~/SYS819/scripts/rorqual/env.sh
#     sbatch "$SYS819_REPO/scripts/rorqual/calibrate.sbatch"
# The log is sys819-calibrate-<jobid>.out in the submit directory.
#SBATCH --job-name=sys819-calibrate
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# shellcheck disable=SC1091
source "${SYS819_REPO:-$SLURM_SUBMIT_DIR}/scripts/rorqual/env.sh"
sys819_activate
# CIFAR-10 on the node-local disk, for the 32 px loader throughput check only (timing is synthetic).
sys819_stage_data cifar10

cd "$SYS819_REPO"
nvidia-smi
python scripts/calibrate.py --out results/calibration.csv \
    --loader-check --set "data.num_workers=$((SLURM_CPUS_PER_TASK - 1))"
