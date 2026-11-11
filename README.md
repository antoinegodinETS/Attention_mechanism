# SYS819 Attention Mechanism

A solo study for SYS819 *Apprentissage profond* (ÉTS, fall 2026). It compares three attention
mechanisms inside one Vision Transformer trained from scratch: full attention (the ViT baseline),
Swin-style shifted windows, and axial attention. Only the attention block changes between models.
The study measures top-1 accuracy (mean ± std over 3 seeds), FLOPs, training time and peak GPU
memory on CIFAR-10 at 32 px and on CIFAR-10-HR (built from ImageNet-21K) at 32 px and 160 px.

All experiments are finished. The plan, what happened against it and the remaining work (the
presentation and the final report) are in [`documentation/project-plan.md`](documentation/project-plan.md).

## Results

Test accuracy (%), mean ± std over seeds 0–2, 100 epochs, evaluated once on the test split.
Every run used one Rorqual H100.

| Condition | Full | Windows | Axial |
| --- | --- | --- | --- |
| CIFAR-10 32 px | **88.81 ± 0.16** | 86.13 ± 0.19 | 86.85 ± 0.08 |
| CIFAR-10-HR 32 px | **71.44 ± 0.22** | 69.90 ± 0.03 | 70.45 ± 0.37 |
| CIFAR-10-HR 160 px | 81.01 ± 0.71 | 80.84 ± 0.55 | **81.67 ± 1.08** |

At 160 px (1,600 tokens) full / windows / axial need 26.2 / 8.7 / 13.7 forward GFLOPs per image,
77 / 56 / 60 min per 100-epoch run, and 10.0 / 10.1 / 13.5 GiB peak memory. Axial has 4.0 M
parameters (4.3 M at 160 px) against 2.7 M (3.0 M) for the other two.

Hypotheses (Report 1 p.10):

- **H1** (at 32 px all three cost about the same, axial the most): supported. Windows 0.93×,
  axial 1.38× full in FLOPs.
- **H2** (at 160 px windows and axial cut FLOPs a lot): supported. FLOPs −67 % and −48 %, but
  training is only 27 % and 22 % faster, and peak memory does not drop with the memory-efficient
  SDPA kernel.
- **H3** (windows beat full attention): not supported at 32 px (full ahead by 2.7 and 1.5 points),
  inconclusive at 160 px (within the seed spread).
- **H4** (axial close to full attention, within 1 point): not supported on CIFAR-10 (−1.95),
  supported on HR-32 (−0.99, borderline) and HR-160 (+0.66).

The hyperparameter search also found that windows and axial collapse at higher learning rates
(windows in 2 of 8 trials, axial in 5 of 8: every axial trial with lr ≥ 1e-3 but one), while full
attention trained in all 8.

Details: [`results/main_table.md`](results/main_table.md) (with time, memory, parameters, FLOPs
and the selected hyperparameters), [`results/hypotheses.md`](results/hypotheses.md) (thresholds
and the numbers behind each verdict) and the figures in [`figures/`](figures/):
`accuracy_by_mechanism.png`, `accuracy_vs_cost.png`, `cost_vs_tokens.png`, `val_curves.png`,
`attention_keys.png`.

## Layout

```
src/attnvit/       package: data/ (datasets, get_dataset, augment), models/ (ViT, attention),
                   train.py, queue.py, config.py, cost.py (analytic and counted FLOPs)
configs/           base.yaml (the reference recipe), queue/ (one run per line), search/ (the phase 5
                   trial list) and final/ (the configuration chosen per mechanism)
scripts/           check_env, calibrate, benchmark_cost, make_cost_figure, hpsearch, collect_runs,
                   aggregate_results, status, dataset builders;
                   rorqual/ and nibi/ hold the cluster jobs
data_splits/       committed split indices, synset mapping and manifest (see its README)
tests/             CPU-only pytest suite
results/ figures/  committed CSV summaries, tables and figures
documentation/     proposal, Report 1, the project plan and data.md (datasets, ImageNet-21K access)
```

`/data/`, `/runs/`, `/checkpoints/` (at the repo root), `*.pt`, `*.npy` and `.venv/` are git-ignored.

## Reproducing the study

The steps below go in order from a fresh clone: environment, data, the GPU jobs on Rorqual, then
collecting and aggregating the results. Steps 4 and 5 need no GPU, and step 5 alone regenerates
every table and figure from the committed `results/*.csv`.

### 1. Environment

Python 3.11 or later (`requires-python = ">=3.11"`; 3.11 is the version used on the clusters).
`pyproject.toml` does not pin a CPU or CUDA build of torch, so install the right torch wheel
first, then the project. The `dev` extra adds pytest, `timm` (a test oracle) and `imagehash`
(CIFAR-10-HR dedupe).

**Keep data and virtual environments outside OneDrive** (syncing a multi-GB array or a venv is slow
and can conflict). On WSL, put them on the ext4 disk (for example under `~`), not under `/mnt/c`.
The editable install points at the checkout you run it from; re-run `pip install -e ".[dev]"`
if the venv was set up from another copy.

**Local PC (WSL, CPU)**: tests, smoke runs, FLOP counting, collection and aggregation.

```bash
python3 -m venv ~/.venvs/sys819-attention          # outside the repo and outside OneDrive
source ~/.venvs/sys819-attention/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest
python scripts/check_env.py                         # reports CUDA unavailable, exits 0
```

For a local CUDA GPU, install the CUDA wheel instead (pick `cu126`, `cu128`, ... on pytorch.org,
one the driver supports); `check_env.py` must then end with `RESULT ok`. On WSL install only the
Windows NVIDIA driver. No reported number comes from a local GPU.

**Rorqual (Digital Research Alliance of Canada, H100)**: every training run and every time and
memory measurement. On a login node, from the clone (`~/SYS819`; keep large data in project space,
scratch is purged after 60 days):

```bash
bash scripts/rorqual/setup_env.sh                   # venv in ~/venvs/sys819-attention, Alliance wheels
                                                    # (imagehash >= 4.3 from PyPI if the wheelhouse is older)
source scripts/rorqual/env.sh                       # account def-chdesa, paths, helpers
sbatch scripts/rorqual/check_env.sbatch
```

`scripts/rorqual/env.sh` is the one place the allocation account is set (`SBATCH_ACCOUNT`), along
with the venv, the data store (`$SYS819_DATA_STORE`) and the runs root (`$SYS819_RUNS`), both in
project space. Source it before every `sbatch`. Override the Python module with
`PYTHON_MODULE=python/3.X` if `python/3.11` is not available. Jobs write `<job-name>-<jobid>.out`
in the submit directory; the training jobs copy their dataset to the node-local `$SLURM_TMPDIR`
first.

### 2. Data

```python
from attnvit.data import get_dataset
ds = get_dataset("cifar10", 32, "train")          # also "cifar10_hr" at 32 or 160
x, y = ds[0]                                       # uint8 tensor (3, 32, 32), int label
norm = ds.normalizer()                             # float32, per-channel mean/std of the train split
```

Splits are `train` (45,000), `val` (5,000), `trainval` (both) and `test` (10,000). Every script
and loader reads the data root from `SYS819_DATA` (default `~/data/sys819`), outside the
repository:

| Path under `$SYS819_DATA` | Contents | Size |
| --- | --- | --- |
| `cifar10/` | torchvision download (archive + `cifar-10-batches-py/`) | 340 MB |
| `cifar10_hr/` | `images_160.npy` 4.6 GB, `images_32.npy` 184 MB, labels, manifest | 4.8 GB |
| `cifar10_hr/work/` | build intermediates; only on Nibi, where it is built | about 5 GB |

**CIFAR-10** (on a PC, which has internet access):

```bash
python scripts/make_cifar10_split.py --download --qa
```

It verifies the committed split (`data_splits/cifar10_val_idx.json`) against the data and fails
instead of rewriting it. QA outputs go to `results/data_qa/` and `figures/data_qa/`.

**CIFAR-10-HR** is built on **Nibi** from the Alliance's ImageNet-21K copy (winter21,
`/datashare/imagenet/winter21_whole`), which needs an image-net.org account and the CCDB opt-in.
The full procedure is in [`documentation/data.md`](documentation/data.md):

```bash
sbatch --account=def-<sponsor> --export=ALL,STEPS=mapping scripts/nibi/build_cifar10_hr.sbatch  # counts only
sbatch --account=def-<sponsor> scripts/nibi/build_cifar10_hr.sbatch                             # full build
```

A rebuild is checked against the committed `data_splits/cifar10_hr_manifest.csv.gz`: the loader
refuses arrays whose manifest differs.

**Copy to Rorqual.** Rorqual compute nodes have no internet access, so nothing is built there. From
a Rorqual login node, pull CIFAR-10-HR from Nibi (`rsync ... --exclude work`, see
`documentation/data.md`); push CIFAR-10 from a PC:

```bash
rsync -av --progress ~/data/sys819/cifar10 \
    <user>@rorqual.alliancecan.ca:links/projects/<account>/<user>/sys819-data/
```

The loaders never download anything; CIFAR-10 needs only its `cifar-10-batches-py/` folder.

### 3. Runs on Rorqual

**How a run works.** A run is `configs/base.yaml` (the reference model and the fixed augmentation
and optimisation recipe) plus `key=value` overrides. Its id is
`<dataset>-<res>-p<P>-<mech>-<tag>-s<seed>` and its directory under `$SYS819_RUNS` holds
`config.yaml`, `metrics.jsonl`, `last.pt` and, once complete, `summary.json`; running the same
command again resumes from `last.pt`. `model.mechanism` is `full` (CLS head), `windows` (shifted
windows, M = 4 set by `model.attn.window_size`, which must divide the patch grid) or `axial`
(column then row pass), the last two with average pooling. `axial_shared` (one projection set for
both passes) is a parameter-matched ablation that was not run. `data.name=synthetic` trains on
random images (tests and smoke runs).

```bash
python -m attnvit.train configs/base.yaml tag=try seed=0 optim.epochs=2     # one run
python -m attnvit.queue configs/queue/phase3_smoke.txt                     # every line of a queue
python scripts/status.py configs/queue/phase3_smoke.txt                    # done / partial / pending
```

The queue smoke-tests every pending run (one optimizer step) before training, skips runs that
have a `summary.json`, resumes partial ones and frees each run before the next. On SIGUSR1
(Slurm's time-limit warning) a run stops cleanly at its last checkpoint and the queue exits with
status 3; the job scripts then resubmit themselves, and resubmitting one by hand is always safe.

`summary.json` splits training time into `data_wait_s` (blocked on the DataLoader) and
`compute_time_s`. `peak_mem_mib` is the allocator's allocated peak (reserved peak in
`peak_mem_reserved_mib`). It also records the CPU side (`num_workers`, `usable_cpus`,
`slurm_cpus_per_task`, `slurm_job_id`, `hostname`).

All jobs below are submitted from a login node after `source ~/SYS819/scripts/rorqual/env.sh`.

**Calibration and cost benchmark (phases 2 and 4, full H100, timing only).**

```bash
sbatch "$SYS819_REPO/scripts/rorqual/calibrate.sbatch"       # step times -> results/calibration.csv, 160 px decision
sbatch "$SYS819_REPO/scripts/rorqual/benchmark_cost.sbatch"  # -> results/cost_benchmark.csv, figures/cost_vs_tokens.png
```

The cost benchmark covers 32 to 160 px at P4 plus 160 px / P8. Per mechanism and setting it
records the forward FLOPs counted by `FlopCounterMode` next to the analytic count (Report 1
attention formulas plus MLP, patch embedding and head), the forward plus backward time per image
(median of 50 synchronised steps after 10 warm-up steps, fp16, batch 128), the eval forward time,
and the peak memory under SDPA and under the naive kernel that materialises the attention matrix
(out of memory is recorded as `oom`). `scripts/rorqual/sanity.sbatch` (100-epoch CIFAR-10
baseline) and `smoke.sbatch` (2-epoch runs of each mechanism) are the phase 2–3 checks.

**Hyperparameter search (phase 5, one 1g.10gb MIG slice per mechanism).** Random search on
CIFAR-10 (45k/5k split): 8 trials per mechanism, 30 epochs each; learning rate and weight decay
log-uniform on [3e-4, 3e-3] and [0.01, 0.1], drop-path in {0, 0.05, 0.1}, warmup in {5, 10}
epochs. Trial `i` has the same hyperparameters and seed for every mechanism. Search runs (tag
`search.t<i>`) never evaluate the test split.

```bash
for m in full windows axial; do
    sbatch --job-name=sys819-hps-$m --export=ALL,SYS819_MECH=$m "$SYS819_REPO/scripts/rorqual/hpsearch.sbatch"
done
python scripts/hpsearch.py collect      # every trial's run folder -> results/hpsearch_cifar10.csv
python scripts/hpsearch.py select       # best val accuracy at epoch 30 -> configs/final/<mech>.yaml
```

The trial list and queues are committed (`python scripts/hpsearch.py generate --check` verifies
them). `select` keeps the trial with the highest validation accuracy at epoch 30 (ties: higher best
validation accuracy, then lower trial index) and writes it with 100 epochs and a final test
evaluation.

**Final training (phase 6, full H100 with 16 cores per job).** 27 runs of 100 epochs: 3
mechanisms × seeds 0, 1, 2 × CIFAR-10, CIFAR-10-HR 32 px and CIFAR-10-HR 160 px. Each starts from
`configs/final/<mechanism>.yaml`; a queue line changes only the seed and the dataset. Each run
trains on the 45k split, validates on the 5k split every epoch and evaluates the test split once
after the last epoch. Run ids are `<dataset>-<res>-p4-<mechanism>-final-s<seed>`.

```bash
sbatch --job-name=sys819-final-cifar10 --export=ALL,SYS819_COND=cifar10 "$SYS819_REPO/scripts/rorqual/final32.sbatch"
sbatch --job-name=sys819-final-hr32 --export=ALL,SYS819_COND=hr32 "$SYS819_REPO/scripts/rorqual/final32.sbatch"
for m in windows axial full; do
    sbatch --job-name=sys819-final-hr160-$m --export=ALL,SYS819_MECH=$m "$SYS819_REPO/scripts/rorqual/final160.sbatch"
done
sys819_activate   # status.py needs the venv
for q in "$SYS819_REPO"/configs/queue/phase6_*.txt; do python "$SYS819_REPO/scripts/status.py" "$q"; done
```

The five jobs run in parallel. Each 32 px job (9 runs) took 1.6–2.6 h of its 3 h limit; each
160 px job (one mechanism, 3 runs) took 3.4–4.2 h of its 8 h limit. About 15 H100 hours in total.

### 4. Collect

`collect_runs.py` turns finished run folders (`config.yaml`, `metrics.jsonl`, `summary.json`) into
two small committed CSVs; raw run folders and checkpoints stay out of the repository. A folder
without `summary.json` is skipped.

```bash
python scripts/collect_runs.py /path/to/final-runs        # or set $SYS819_RUNS and omit the path
                                                          # -> results/final_runs.csv, results/final_curves.csv
```

### 5. Aggregate

```bash
python scripts/aggregate_results.py
```

It reads only committed CSVs (`results/final_runs.csv`, `final_curves.csv`, `cost_benchmark.csv`,
`hpsearch_cifar10.csv`) and writes `results/main_table.{csv,md}`, `results/hypotheses.{csv,md}`
and the figures `cost_vs_tokens.png`, `accuracy_by_mechanism.png`, `accuracy_vs_cost.png`,
`val_curves.png` and `attention_keys.png`. A condition with fewer than 3 seeds shows as partial
and keeps H3/H4 provisional.

### Tests

`pytest` (CPU only) covers the models and attention masks, training and resume, the queue, the
search, the cost counts, the result collection and aggregation, and the whole CIFAR-10-HR build
on a synthetic ImageNet tree. It also checks the committed split files. Tests on real CIFAR-10
are skipped when `$SYS819_DATA` lacks it.
