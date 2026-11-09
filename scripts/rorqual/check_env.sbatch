#!/bin/bash
# Phase 0 environment check on one Rorqual H100: runs scripts/check_env.py and pytest.
#
# Submit after scripts/rorqual/setup_env.sh (the account comes from env.sh):
#     source ~/SYS819/scripts/rorqual/env.sh
#     sbatch "$SYS819_REPO/scripts/rorqual/check_env.sbatch"
#SBATCH --job-name=sys819-check-env
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# shellcheck disable=SC1091
source "${SYS819_REPO:-$SLURM_SUBMIT_DIR}/scripts/rorqual/env.sh"
sys819_activate

cd "$SYS819_REPO"
nvidia-smi
python scripts/check_env.py
python -m pytest -q
