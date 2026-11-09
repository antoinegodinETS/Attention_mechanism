#!/bin/bash
# Phase 2 sanity run: the 100-epoch full-attention baseline on CIFAR-10 (configs/queue/phase2_sanity.txt).
# Target: roughly 80-85 % validation accuracy. Output in $SYS819_RUNS/cifar10-32-p4-full-sanity-s0/.
#
# GPU size: a 3g.40gb MIG slice (3/7 of an H100's compute, 40 GB), not a full H100. The model
# has 2.7 M parameters and 64 tokens at 32 px; a batch of 128 needs well under 2 GB and does not
# fill even this slice, and a 32 px epoch is bound by the CPU augmentation pipeline as much as by
# the GPU. A slice usually waits less in the queue (plan, "Machines"). This run checks accuracy
# only: its times are not used in the report, whose time and memory numbers come from full H100s.
# 6 CPU cores is a 3/8 share of an H100's 16 cores per GPU on a Rorqual node (5 loader workers).
#
# Submit from a login node (the account comes from env.sh):
#     source ~/SYS819/scripts/rorqual/env.sh
#     sbatch "$SYS819_REPO/scripts/rorqual/sanity.sbatch"
# Slurm sends USR1 5 minutes before the time limit; the run checkpoints, exits with status 3 and
# this script resubmits itself, so the next job resumes from last.pt (up to SYS819_MAX_CHUNKS jobs).
#SBATCH --job-name=sys819-sanity
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=%x-%j.out
set -euo pipefail

# shellcheck disable=SC1091
source "${SYS819_REPO:-$SLURM_SUBMIT_DIR}/scripts/rorqual/env.sh"
sys819_activate
sys819_stage_data cifar10

cd "$SYS819_REPO"
nvidia-smi
echo "runs root: $SYS819_RUNS"
status=0
sys819_run_queue configs/queue/phase2_sanity.txt "$SYS819_REPO/scripts/rorqual/sanity.sbatch" \
    --set "data.num_workers=$((SLURM_CPUS_PER_TASK - 1))" || status=$?
python scripts/status.py configs/queue/phase2_sanity.txt
exit "$status"
