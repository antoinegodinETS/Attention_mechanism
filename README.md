# SYS819 Attention Mechanism

A solo study for SYS819 *Apprentissage profond* (ÉTS, fall 2026). It compares three attention
mechanisms inside one Vision Transformer trained from scratch: full attention (the ViT baseline),
Swin-style shifted windows, and axial attention. Only the attention block changes between models.
The study measures top-1 accuracy (mean ± std over 3 seeds), FLOPs, training time and peak GPU
memory on CIFAR-10 at 32 px and on CIFAR-10-HR (built from ImageNet-21K) at 32 px and 160 px.

The full plan, phases and decisions are in [`documentation/project-plan.md`](documentation/project-plan.md).

## Layout

```
src/attnvit/       package: data/ (datasets, get_dataset, augment), models/ (ViT, attention),
                   train.py, queue.py, config.py, cost.py (analytic and counted FLOPs)
configs/           base.yaml (the reference recipe) and queue/ (one run per line)
scripts/           check_env, calibrate, benchmark_cost, make_cost_figure, status, dataset builders;
                   rorqual/ and nibi/ hold the cluster scripts
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

On a login node, from the repository clone (`~/SYS819`; keep large data in project space,
scratch is purged after 60 days):

```bash
bash scripts/rorqual/setup_env.sh                   # venv in ~/venvs/sys819-attention, Alliance wheels
                                                    # (imagehash >= 4.3 from PyPI if the wheelhouse is older)
source scripts/rorqual/env.sh                       # account def-chdesa, paths, helpers
sbatch scripts/rorqual/check_env.sbatch
```

`scripts/rorqual/env.sh` is the one place the allocation account is set (`SBATCH_ACCOUNT`), along
with the venv, the data store (`$SYS819_DATA_STORE`, project space) and the runs root
(`$SYS819_RUNS`, project space). Source it before every `sbatch`. Check the `python/3.11` module
version with `module avail python` (override with `PYTHON_MODULE=python/3.X`). Jobs write
`<job-name>-<jobid>.out` in the submit directory.

## Datasets

```python
from attnvit.data import get_dataset
ds = get_dataset("cifar10", 32, "train")          # also "cifar10_hr" at 32 or 160
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
| `cifar10_hr/` | `images_160.npy` 4.6 GB, `images_32.npy` 184 MB, labels, manifest | 4.8 GB |
| `cifar10_hr/work/` | build intermediates (crop pool, candidate list); only where it is built | about 5 GB |

### Building each dataset

On the **main PC** (or the GPU PC), with the venv active, from the repository root:

```bash
python scripts/make_cifar10_split.py --download --qa   # CIFAR-10: download, verify the committed split, QA
```

The script verifies the committed split file against the data and fails instead of rewriting it.
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
into project space from a login node: CIFAR-10-HR from Nibi (see `documentation/data.md`), and
CIFAR-10 from a PC:

```bash
# on the PC: push CIFAR-10 (rorqual.alliancecan.ca is also the transfer node)
rsync -av --progress ~/data/sys819/cifar10 \
    <user>@rorqual.alliancecan.ca:links/projects/<account>/<user>/sys819-data/
```

CIFAR-10 at its new location needs only the `cifar-10-batches-py/` folder; the loaders never
download anything.

### Tests and data

`pytest` runs the whole CIFAR-10-HR pipeline on a synthetic ImageNet tree (it needs `imagehash`,
installed with the `dev` extra). It also checks the committed split files. The `get_dataset`
tests on real CIFAR-10 are skipped when `$SYS819_DATA` lacks it.

## Training

Runs are configured by `configs/base.yaml` (the reference model and the fixed augmentation and
optimisation recipe) plus `key=value` overrides. Each run gets the id
`<dataset>-<res>-p<P>-<mech>-<tag>-s<seed>` and a directory under `$SYS819_RUNS` (default
`~/runs/sys819`, outside the repository) holding `config.yaml`, `metrics.jsonl`, `last.pt` and,
once complete, `summary.json`. Running the same command again resumes from `last.pt`.
`model.mechanism` picks the attention: `full` (CLS head), `windows` (shifted windows, M = 4, set
by `model.attn.window_size`, which must divide the patch grid) or `axial` (column then row pass),
the last two with average pooling. `axial_shared`, axial with one projection set shared by both
passes, is the optional parameter-matched ablation.
Training time is split into `data_wait_s` (blocked on the DataLoader) and `compute_time_s`; compare
mechanisms on the compute time. `peak_mem_mib` is the allocator's allocated peak, with the reserved
peak in `peak_mem_reserved_mib`. `summary.json` also records the CPU side of the run (`num_workers`,
`usable_cpus`, `slurm_cpus_per_task`, `slurm_job_id`, `hostname`).

```bash
python -m attnvit.train configs/base.yaml tag=try seed=0 optim.epochs=2     # one run
python -m attnvit.queue configs/queue/phase2_sanity.txt                    # every line of a queue
python scripts/status.py configs/queue/phase2_sanity.txt                   # done / partial / pending
python scripts/calibrate.py                                                # step times -> results/calibration.csv
python scripts/benchmark_cost.py                                           # phase 4 -> results/cost_benchmark.csv
python scripts/make_cost_figure.py                                         # CSV -> figures/cost_vs_tokens.png
```

The cost benchmark (phase 4) needs no training and no dataset. Per mechanism and setting (32 to
160 px at P4, plus 160 px / P8) it records the forward FLOPs counted by `FlopCounterMode` next to
the analytic count (Report 1 attention formulas plus the MLP, patch embedding and head), the
forward plus backward time per image (median of 50 synchronised steps after 10 warm-up steps,
fp16, batch 128 for every case), the eval forward time, and the peak memory under SDPA and under
the naive kernel that materialises the attention matrix. Out of memory is recorded as `oom`.

The queue smoke-tests every pending run (one optimizer step) before training, skips runs that
have a `summary.json` and resumes partial ones. On SIGUSR1 (Slurm's time-limit warning) a run
stops cleanly with its last checkpoint and the queue exits with status 3.
`data.name=synthetic` trains on random images without any dataset (tests and smoke runs).

On Rorqual (after `source scripts/rorqual/env.sh`):

```bash
sbatch scripts/rorqual/calibrate.sbatch   # full H100, ~5 min: results/calibration.csv + 160 px decision
sbatch scripts/rorqual/benchmark_cost.sbatch  # full H100: results/cost_benchmark.csv + figures/cost_vs_tokens.png
sbatch scripts/rorqual/sanity.sbatch      # 3g.40gb MIG slice: 100-epoch CIFAR-10 baseline, resubmits itself
sbatch scripts/rorqual/smoke.sbatch       # 3g.40gb MIG slice: 2-epoch smoke runs of full, windows and axial
```

Each except the cost benchmark copies CIFAR-10 from `$SYS819_DATA_STORE` to the node-local `$SLURM_TMPDIR` first.
