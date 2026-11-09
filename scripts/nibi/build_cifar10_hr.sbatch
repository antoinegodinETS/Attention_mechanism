#!/bin/bash
# Build CIFAR-10-HR on Nibi (Digital Research Alliance of Canada) as a CPU job, reading
# ImageNet-21K winter21 from the Alliance copy in /datashare/imagenet/winter21_whole.
#
# Prerequisites (README, "CIFAR-10-HR"): an image-net.org account, the Alliance ImageNet
# opt-in, access to Nibi, and the project venv (bash scripts/rorqual/setup_env.sh works on Nibi
# too; add imagehash with `pip install imagehash` if the wheelhouse lacks it, Nibi nodes have
# internet access).
#
# Submit from the repository root on a Nibi login node:
#     sbatch --account=def-<sponsor> scripts/nibi/build_cifar10_hr.sbatch
#     sbatch --account=def-<sponsor> --export=ALL,STEPS=mapping scripts/nibi/build_cifar10_hr.sbatch
# The second form only lists the candidates and prints per-class counts (a quick check).
# Rerunning resumes: finished steps are skipped.
#
# Outputs: arrays in $SYS819_DATA/cifar10_hr (about 4.8 GB, plus about 5 GB of work files),
# data_splits/cifar10_hr_manifest.csv.gz and results/, figures/ QA files in this clone.
#
# NOT VERIFIED on Nibi: the module versions, the default output location ($SCRATCH, falling back
# to /scratch/$USER), and the time/memory estimates below. Check `module avail python` and
# `echo $SCRATCH` first.
#SBATCH --job-name=sys819-build-hr
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=sys819-build-hr-%j.out
set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"      # placeholder, verify with `module avail python`
VENV_DIR="${VENV_DIR:-$HOME/venvs/sys819-attention}"
SOURCE="${SOURCE:-/datashare/imagenet/winter21_whole}"
RELEASE="${RELEASE:-winter21}"
STEPS="${STEPS:-all}"
export SYS819_DATA="${SYS819_DATA:-${SCRATCH:-/scratch/$USER}/sys819-data}"

module load StdEnv/2023 "$PYTHON_MODULE"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
cd "${SLURM_SUBMIT_DIR:-.}"

echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  cpus ${SLURM_CPUS_PER_TASK:-?}  $(date -Is)"
echo "source $SOURCE  release $RELEASE  data root $SYS819_DATA  steps $STEPS"
python -c "import imagehash, attnvit" || { echo "imagehash or attnvit missing in $VENV_DIR"; exit 1; }
# /datashare is an NFS3 mount: a user in more than 16 Unix groups may be refused access.
ls "$SOURCE/n02691156" >/dev/null || { echo "cannot read $SOURCE (opt-in done? group limit?)"; exit 1; }
mkdir -p "$SYS819_DATA"

python scripts/build_cifar10_hr.py \
    --source "$SOURCE" --release "$RELEASE" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" --steps "$STEPS" "$@"

echo "finished $(date -Is)"
ls -la "$SYS819_DATA/cifar10_hr" || true
