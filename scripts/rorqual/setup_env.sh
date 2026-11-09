#!/bin/bash
# Create the project virtual environment on Rorqual (Digital Research Alliance of Canada).
#
# Run once on a LOGIN node (compute nodes have no internet), from the repository root:
#     bash scripts/rorqual/setup_env.sh
#
# Follows the Alliance convention: load a Python module, create a virtualenv with
# --no-download, and install the Alliance's prebuilt wheels with --no-index.
# The venv lives outside the repository (default ~/venvs/sys819-attention).
# The one exception is imagehash: if the wheelhouse has none, or one older than 4.3, it is
# upgraded from PyPI (login nodes have internet access). Works on Nibi too.
#
# PLACEHOLDER: PYTHON_MODULE is a guess at a current module version. Check what is
# available with `module avail python` (or `module spider python`) and override it:
#     PYTHON_MODULE=python/3.12 bash scripts/rorqual/setup_env.sh
set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"   # placeholder, verify with `module avail python`
VENV_DIR="${VENV_DIR:-$HOME/venvs/sys819-attention}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

module load StdEnv/2023 "$PYTHON_MODULE"

if [ ! -d "$VENV_DIR" ]; then
    mkdir -p "$(dirname "$VENV_DIR")"
    virtualenv --no-download "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --no-index --upgrade pip setuptools wheel
# Alliance wheels: the cluster's torch build already targets its CUDA and the H100s.
pip install --no-index torch torchvision numpy pillow pyyaml pandas matplotlib einops pytest
# timm may or may not exist in the Alliance wheelhouse (`avail_wheels timm`); it is a dev-only
# test oracle, so do not fail the setup if it is missing.
pip install --no-index timm || echo "note: timm wheel not available (dev-only test oracle)"

# imagehash (CIFAR-10-HR dedupe, the data extra) must be >= 4.3: older releases call
# PIL.Image.ANTIALIAS, which Pillow 10 removed, and the Alliance wheelhouse ships 4.1.0.
# Wheelhouse first (with its dependencies), then PyPI for imagehash alone if still too old.
imagehash_ok() {
    python - <<'PY'
import importlib.metadata as md, sys
try:
    v = md.version("imagehash")
except md.PackageNotFoundError:
    sys.exit(1)
sys.exit(0 if tuple(int(p) for p in v.split(".")[:2] if p.isdigit()) >= (4, 3) else 1)
PY
}
pip install --no-index imagehash || echo "note: imagehash wheel not available in the wheelhouse"
pip install --no-index PyWavelets scipy || true   # imagehash dependencies, if the line above failed
if ! imagehash_ok; then
    echo "imagehash is missing or older than 4.3 (PIL.Image.ANTIALIAS bug): upgrading it from PyPI"
    pip install --upgrade "imagehash>=4.3"
fi
python -c "import importlib.metadata as md; print('imagehash', md.version('imagehash'))"
# Install the project itself without touching dependencies (no network on PyPI needed).
pip install --no-index --no-deps --no-build-isolation -e "$REPO_DIR"

python -c "import torch, attnvit; print('torch', torch.__version__, '| attnvit', attnvit.__version__)"
echo "Virtualenv ready: $VENV_DIR"
echo "Next: source scripts/rorqual/env.sh && sbatch scripts/rorqual/check_env.sbatch"
