# SYS819 Attention Mechanism

A solo study for SYS819 *Apprentissage profond* (ÉTS, fall 2026). It compares three attention
mechanisms inside one Vision Transformer trained from scratch: full attention (the ViT baseline),
Swin-style shifted windows, and axial attention. Only the attention block changes between models.
The study measures top-1 accuracy (mean ± std over 3 seeds), FLOPs, training time and peak GPU
memory on CIFAR-10 at 32 px and on CIFAR-10-HR (built from ImageNet-21K) at 32 px and 160 px.

The full plan, phases and decisions are in [`documentation/project-plan.md`](documentation/project-plan.md).

## Layout

```
src/attnvit/       package: data/ (datasets, get_dataset), models/ (from phase 2 on)
configs/           experiment configs
scripts/           check_env.py, dataset builders; rorqual/ and nibi/ hold the cluster scripts
data_splits/       committed split indices, synset mapping and manifests (see its README)
tests/             CPU-only pytest suite
results/ figures/  committed summaries and generated figures
documentation/     proposal, Report 1, the project plan and data.md (datasets, ImageNet-21K access)
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
pip install -e ".[dev]"                             # dev includes imagehash (dataset dedupe)
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

## Datasets

```python
from attnvit.data import get_dataset
ds = get_dataset("cifar10", 32, "train")          # also "cifar10_hr" and "imagenette" at 32 or 160
x, y = ds[0]                                       # uint8 tensor (3, 32, 32), int label
norm = ds.normalizer()                             # float32, per-channel mean/std of the train split
```

Splits are `train`, `val`, `trainval` (train + val, for final retraining) and `test`. Details,
the CIFAR-10-HR pipeline and the ImageNet-21K access route are in
[`documentation/data.md`](documentation/data.md); the committed split files are described in
[`data_splits/README.md`](data_splits/README.md).

### Where the data lives: `SYS819_DATA`

Every script and loader reads the data root from the environment variable `SYS819_DATA`
(default `~/data/sys819`). Keep it outside the repository and outside OneDrive:

```bash
export SYS819_DATA=~/data/sys819                                   # main PC and GPU PC (WSL ext4)
export SYS819_DATA=~/links/projects/<account>/$USER/sys819-data    # Rorqual (project space)
```

Layout and disk usage:

| Path under `$SYS819_DATA` | Contents | Size |
| --- | --- | --- |
| `cifar10/` | torchvision download (archive + `cifar-10-batches-py/`) | 340 MB |
| `imagenette/` | `raw/imagenette2-160.tgz`, `images_160.npy`, `images_32.npy`, labels, manifest | 1.1 GB |
| `cifar10_hr/` | `images_160.npy` 4.6 GB, `images_32.npy` 184 MB, labels, manifest | 4.8 GB |
| `cifar10_hr/work/` | build intermediates (crop pool, candidate list); only where it is built | about 5 GB |

### Building each dataset

On the **main PC** (or the GPU PC), with the venv active, from the repository root:

```bash
python scripts/make_cifar10_split.py --download --qa   # CIFAR-10: download, verify the committed split, QA
python scripts/build_imagenette.py --download          # Imagenette-160: download (95 MB), build, QA (~30 s)
```

Both verify the committed split files against the data and fail instead of rewriting them.
QA outputs go to `results/data_qa/` (CSV, JSON) and `figures/data_qa/` (PNG).

**CIFAR-10-HR** is built on **Nibi**, which hosts ImageNet-21K (winter21) in
`/datashare/imagenet/winter21_whole`. Follow the steps in `documentation/data.md`:

```bash
sbatch --account=def-<sponsor> --export=ALL,STEPS=mapping scripts/nibi/build_cifar10_hr.sbatch  # counts only
sbatch --account=def-<sponsor> scripts/nibi/build_cifar10_hr.sbatch                             # full build
```

Then commit `data_splits/cifar10_hr_manifest.csv.gz` and the `cifar10_hr` QA files, and pull the
arrays to each machine with rsync (`--exclude work`).

**Rorqual** compute nodes have no internet access. Do not build there. Copy finished datasets
into project space from a login node, either from Nibi (see `documentation/data.md`) or from a PC:

```bash
# on the PC: push CIFAR-10 and Imagenette (rorqual.alliancecan.ca is also the transfer node)
rsync -av --progress ~/data/sys819/cifar10 ~/data/sys819/imagenette --exclude raw \
    <user>@rorqual.alliancecan.ca:links/projects/<account>/<user>/sys819-data/
```

CIFAR-10 at its new location needs only the `cifar-10-batches-py/` folder; the loaders never
download anything.

### Tests and data

`pytest` runs the whole CIFAR-10-HR pipeline on a synthetic ImageNet tree (it needs `imagehash`,
installed with the `dev` extra). It also checks the committed split files. The `get_dataset`
tests on real CIFAR-10 and Imagenette are skipped when `$SYS819_DATA` lacks those datasets.
