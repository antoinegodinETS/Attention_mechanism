#!/bin/bash
# Phase 6 final training on CIFAR-10-HR at 160 px / P4 (1,600 tokens): the 3 seeds x 100 epochs
# of ONE mechanism, in order through configs/queue/phase6_hr160_<mechanism>.txt. Output in
# $SYS819_RUNS/cifar10_hr-160-p4-<mechanism>-final-s<seed>/, each with summary.json.
#
# GPU size: a full H100, like every phase 6 job, so the reported time and memory are comparable.
# Batch 128 fits without accumulation (calibration peak: full 10.2, windows 10.4, axial 13.8 GB).
# 16 CPU cores is an H100's share of a Rorqual node: 15 loader workers keep up with the GPU
# (RandAugment at 160 px costs about 0.7 ms per image per core; the fastest mechanism needs
# about 1,400 images/s). CIFAR-10-HR's 160 px array (4.6 GB) is copied to the node-local disk.
# Time: calibration puts a 100-epoch run at 0.96 h (windows), 1.03 h (axial) and 1.36 h (full)
# of training steps; with validation every epoch and the final test pass, about 1.0 / 1.1 /
# 1.45 h, so 3.0 / 3.3 / 4.4 h per job. The 8 h limit leaves at least 1.8 times that.
#
# Submit from a login node (the account comes from env.sh), cheapest mechanism first:
#     source ~/SYS819/scripts/rorqual/env.sh
#     for m in windows axial full; do
#         sbatch --job-name=sys819-final-hr160-$m --export=ALL,SYS819_MECH=$m "$SYS819_REPO/scripts/rorqual/final160.sbatch"
#     done
# Resumable: Slurm sends USR1 5 minutes before the time limit; the current run checkpoints, the
# queue exits with status 3 and this script resubmits itself with the same SYS819_MECH (up to
# SYS819_MAX_CHUNKS jobs). Finished runs are skipped and a partial one resumes from last.pt, so
# resubmitting by hand after any failure is also safe.
#SBATCH --job-name=sys819-final160
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=%x-%j.out
set -euo pipefail

case "${SYS819_MECH:-}" in
    windows | axial | full) ;;
    *)
        echo "error: set SYS819_MECH to windows, axial or full (sbatch --export=ALL,SYS819_MECH=windows ...)" >&2
        exit 2
        ;;
esac

# shellcheck disable=SC1091
source "${SYS819_REPO:-$SLURM_SUBMIT_DIR}/scripts/rorqual/env.sh"
sys819_activate
sys819_stage_data cifar10_hr:160

cd "$SYS819_REPO"
queue="configs/queue/phase6_hr160_${SYS819_MECH}.txt"
nvidia-smi
echo "mechanism: $SYS819_MECH, queue: $queue, runs root: $SYS819_RUNS"
status=0
sys819_run_queue "$queue" "$SYS819_REPO/scripts/rorqual/final160.sbatch" \
    --set "data.num_workers=$((SLURM_CPUS_PER_TASK - 1))" || status=$?
python scripts/status.py "$queue"
exit "$status"
