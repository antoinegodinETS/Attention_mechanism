#!/bin/bash
# Phase 6 final training at 32 px / P4: the 9 runs (3 mechanisms x 3 seeds x 100 epochs) of ONE
# condition, in order through its queue:
#     SYS819_COND=cifar10   configs/queue/phase6_cifar10.txt  (CIFAR-10)
#     SYS819_COND=hr32      configs/queue/phase6_hr32.txt     (CIFAR-10-HR, 32 px copy)
# Output in $SYS819_RUNS/<dataset>-32-p4-<mechanism>-final-s<seed>/, each with summary.json
# (test accuracy, training time, time per epoch, peak memory).
#
# GPU size: a full H100, like every phase 6 job, so the reported time and memory are comparable
# across conditions. 16 CPU cores is an H100's share of a Rorqual node (15 loader workers).
# Time: the full H100 trains a 32 px epoch in about 4-6 s (calibration: 3.9 / 4.6 / 5.7 s of
# compute for full / windows / axial; the phase 2 baseline took 4.6 s per epoch on a 3g.40gb
# slice), so a 100-epoch run takes 8-12 min and the 9 runs about 1.5 h. The 3 h limit leaves
# twice that.
#
# Submit from a login node (the account comes from env.sh), CIFAR-10 first:
#     source ~/SYS819/scripts/rorqual/env.sh
#     sbatch --job-name=sys819-final-cifar10 --export=ALL,SYS819_COND=cifar10 "$SYS819_REPO/scripts/rorqual/final32.sbatch"
#     sbatch --job-name=sys819-final-hr32 --export=ALL,SYS819_COND=hr32 "$SYS819_REPO/scripts/rorqual/final32.sbatch"
# Resumable: Slurm sends USR1 5 minutes before the time limit; the current run checkpoints, the
# queue exits with status 3 and this script resubmits itself with the same SYS819_COND (up to
# SYS819_MAX_CHUNKS jobs). Finished runs are skipped and a partial one resumes from last.pt, so
# resubmitting by hand after any failure is also safe.
#SBATCH --job-name=sys819-final32
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=%x-%j.out
set -euo pipefail

case "${SYS819_COND:-}" in
    cifar10) data=cifar10 ;;
    hr32) data=cifar10_hr:32 ;;
    *)
        echo "error: set SYS819_COND to cifar10 or hr32 (sbatch --export=ALL,SYS819_COND=cifar10 ...)" >&2
        exit 2
        ;;
esac

# shellcheck disable=SC1091
source "${SYS819_REPO:-$SLURM_SUBMIT_DIR}/scripts/rorqual/env.sh"
sys819_activate
sys819_stage_data "$data"

cd "$SYS819_REPO"
queue="configs/queue/phase6_${SYS819_COND}.txt"
nvidia-smi
echo "condition: $SYS819_COND, queue: $queue, runs root: $SYS819_RUNS"
status=0
sys819_run_queue "$queue" "$SYS819_REPO/scripts/rorqual/final32.sbatch" \
    --set "data.num_workers=$((SLURM_CPUS_PER_TASK - 1))" || status=$?
python scripts/status.py "$queue"
exit "$status"
