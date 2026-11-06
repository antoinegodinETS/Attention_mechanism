#!/bin/bash
# Phase 0 environment check on one Rorqual H100: runs scripts/check_env.py and pytest.
#
# Submit from the repository root after scripts/rorqual/setup_env.sh:
#     sbatch --account=def-<sponsor> scripts/rorqual/check_env.sbatch
#
# PLACEHOLDER: replace def-<sponsor> with your allocation account (`sshare -U` lists it),
# either on the command line as above or in the #SBATCH --account line below.
#SBATCH --job-name=sys819-check-env
#SBATCH --account=def-<sponsor>
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out
set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"   # must match setup_env.sh
VENV_DIR="${VENV_DIR:-$HOME/venvs/sys819-attention}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

module load StdEnv/2023 "$PYTHON_MODULE"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

cd "$REPO_DIR"
nvidia-smi
python scripts/check_env.py
python -m pytest -q
