#!/bin/bash
# Phase 4 cost benchmark on one full Rorqual H100 (no training, no dataset): full, windows and axial
# at 32, 64, 96, 128 and 160 px / P4 plus 160 px / P8, batch 128. Writes results/cost_benchmark.csv
# in the repository clone (replacing the previous table) and draws figures/cost_vs_tokens.png from it.
# A full H100 (not a MIG slice): the plan takes every time and memory number on one H100 type, and
# the naive-kernel memory run needs the whole 80 GB (an out-of-memory case is recorded, not fatal).
# Expected run time is well under 30 minutes; the 1-hour limit follows the cluster's job-length guidance.
#
# Submit from a login node (the account comes from env.sh):
#     source ~/SYS819/scripts/rorqual/env.sh
#     sbatch "$SYS819_REPO/scripts/rorqual/benchmark_cost.sbatch"
# The log is sys819-cost-<jobid>.out in the submit directory.
#SBATCH --job-name=sys819-cost
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# shellcheck disable=SC1091
source "${SYS819_REPO:-$SLURM_SUBMIT_DIR}/scripts/rorqual/env.sh"
sys819_activate

cd "$SYS819_REPO"
nvidia-smi
python scripts/benchmark_cost.py --out results/cost_benchmark.csv
python scripts/make_cost_figure.py --csv results/cost_benchmark.csv --out figures/cost_vs_tokens.png
