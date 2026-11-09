#!/bin/bash
# Phase 3 smoke runs: 2 epochs of full, windows and axial attention (configs/queue/phase3_smoke.txt),
# on CIFAR-10 at 32 px / P4 and on synthetic 160 px / P4 images. Phase 3 is done when all six
# finish. Output in $SYS819_RUNS/<run_id>/ with run ids cifar10-32-p4-<mech>-smoke-s0 and
# synthetic-160-p4-<mech>-smoke-s0.
#
# GPU size: a 3g.40gb MIG slice, as for the sanity run. Full attention at 160 px / P4 peaks at
# about 10 GB (calibration); these runs check that training completes, not its speed.
#
# Submit from a login node (the account comes from env.sh):
#     source ~/SYS819/scripts/rorqual/env.sh
#     sbatch "$SYS819_REPO/scripts/rorqual/smoke.sbatch"
# The log is sys819-smoke-<jobid>.out in the submit directory.
#SBATCH --job-name=sys819-smoke
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@120
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
sys819_run_queue configs/queue/phase3_smoke.txt "$SYS819_REPO/scripts/rorqual/smoke.sbatch" \
    --set "data.num_workers=$((SLURM_CPUS_PER_TASK - 1))" || status=$?
python scripts/status.py configs/queue/phase3_smoke.txt
exit "$status"
