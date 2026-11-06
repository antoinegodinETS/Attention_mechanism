# SYS819 Attention Mechanism

A solo study for SYS819 *Apprentissage profond* (ÉTS, fall 2026). It compares three attention
mechanisms inside one Vision Transformer trained from scratch: full attention (the ViT baseline),
Swin-style shifted windows, and axial attention. Only the attention block changes between models.
The study measures top-1 accuracy (mean ± std over 3 seeds), FLOPs, training time and peak GPU
memory on CIFAR-10 at 32 px and on CIFAR-10-HR (built from ImageNet-21K) at 32 px and 160 px.

The full plan, phases and decisions are in [`documentation/project-plan.md`](documentation/project-plan.md).

## Layout

```
src/attnvit/       package: models/, data/ (filled in from phase 1 on)
configs/           experiment configs
scripts/           check_env.py and other entry points; rorqual/ holds the cluster scripts
data_splits/       committed split indices and manifests
tests/             CPU-only pytest suite
results/ figures/  committed summaries and generated figures
documentation/     proposal, Report 1 and the project plan
```

`/data/`, `/runs/`, `/checkpoints/` (at the repo root), `*.pt`, `*.npy` and `.venv/` are git-ignored.

**Keep data and virtual environments outside OneDrive.** Syncing a multi-GB array or a venv is
slow and can conflict. On WSL, put them on the ext4 disk (for example under `~`), not under `/mnt/c`.

## Setup

Python 3.11–3.13; use the same minor version on both PCs. `pyproject.toml` does not pin a CPU or
CUDA build of torch, so install the right torch wheel first, then the project.

The editable install (`pip install -e .`) points at the copy of the repository you run it from.
If the venv was first set up from another checkout, re-run `pip install -e ".[dev]"` from your own copy.

### Main PC (WSL, CPU only)

For tests, smoke runs, FLOP counting, building the dataset and analysis.

```bash
python3 -m venv ~/.venvs/sys819-attention          # outside the repo and outside OneDrive
source ~/.venvs/sys819-attention/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"                             # add ,data for imagehash: ".[dev,data]"
pytest
python scripts/check_env.py                         # reports CUDA unavailable, exits 0
```

If `python3 -m venv` fails because `ensurepip` is missing, install `python3.X-venv` with apt or use
another Python 3.11–3.13 interpreter.

### GPU PC (WSL, RTX 2060)

Install only the Windows NVIDIA driver; do not install a Linux driver inside WSL.
Clone the repository onto the WSL ext4 disk (for example `~/src/Attention_mechanism`), **not**
under `/mnt/c`, and keep `data/`, `runs/` and `checkpoints/` there too.

```bash
nvidia-smi                                          # the driver is visible from WSL
git clone <repo-url> ~/src/Attention_mechanism && cd ~/src/Attention_mechanism
python3 -m venv ~/.venvs/sys819-attention
source ~/.venvs/sys819-attention/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"
pytest
python scripts/check_env.py
```

Pick the CUDA wheel (`cu126`, `cu128`, ...) from the selector on pytorch.org; it must be one the
Windows driver supports (`nvidia-smi` shows the highest CUDA version). `check_env.py` must end
with `RESULT ok`: CUDA available, capability 7.5, fp16 step ok and `EFFICIENT_ATTENTION` ok.
Flash attention is expected to be unavailable on Turing.

### Rorqual (Digital Research Alliance of Canada, H100)

On a login node, from the repository root (keep the clone in `$HOME` or project space, and large
data in project or scratch space; scratch is purged):

```bash
bash scripts/rorqual/setup_env.sh                   # venv in ~/venvs/sys819-attention, Alliance wheels
sbatch --account=def-<sponsor> scripts/rorqual/check_env.sbatch
```

Placeholders to replace: `def-<sponsor>` is your allocation account (see `sshare -U`), and the
`python/3.11` module version in both scripts should be checked with `module avail python`
(override with `PYTHON_MODULE=python/3.X`). The job requests one H100 (`--gpus=h100:1`) for 15
minutes and writes `sys819-check-env-<jobid>.out` in the submit directory.
