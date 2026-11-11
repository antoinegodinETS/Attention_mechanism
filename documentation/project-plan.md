# SYS819 Attention Mechanism: Project Start Plan

Written 2026-11-06, from `Proposition SYS819.pdf` and `Rapport1 SYS819.pdf`.

## Summary

About 5 weeks remain before the S13 presentation and S14 final study, and no code exists yet, so the project is roughly one week behind its own schedule.
The GPU budget at 160 px is the deciding constraint.

- **The project.** A solo study for SYS819 *Apprentissage profond* (ÉTS, fall 2026, Marco Pedersoli). It compares three attention mechanisms inside one Vision Transformer trained from scratch: full attention (the ViT baseline), Swin-style shifted windows, and axial attention. Only the attention block changes between models.
- **Measures.** Top-1 accuracy (mean ± std over 3 seeds), FLOPs, training time and peak GPU memory.
- **Data.** CIFAR-10 at 32 px, and CIFAR-10-HR (built from ImageNet-21K) at 32 px and 160 px.
- **Deadlines.** Presentation in S13 and experimental study in S14, inferred as Nov 30–Dec 4 and Dec 7–11. Confirm against the course plan.
- **Compute.** At 32 px a run takes about 20–40 min on the RTX 2060. At 160 px with 4×4 patches, one full-attention run is estimated at 12–25 GPU-hours, and the 9-run 160 px block at 67–138 GPU-hours, which does not fit safely on that GPU. On Rorqual's H100s the same block is about 5–14 GPU-hours, so the documented 160 px experiment fits there. The main PC (CPU only) is for tests, smoke runs, FLOP counting, building the dataset and analysis.

**Main recommendations**

1. Run a cost benchmark that needs no training first; it tests H1 and H2 directly.
2. Time real training steps on the RTX 2060 early in S10, then pick the 160 px schedule with a fixed rule (see [Compute budget](#compute-budget)).
3. Build CIFAR-10-HR on the Nibi cluster from the Alliance's copy of ImageNet-21K (Winter 2021 release, `/datashare/imagenet/winter21_whole`; access confirmed Nov 9), aiming to finish by Fri Nov 13.
4. Use each technique as its original paper does: a CLS token for full attention (ViT), global average pooling for shifted windows (Swin) and axial attention. Keep a flat (isotropic) Swin-style model with no patch merging, so that otherwise only the attention block changes.

## The project as documented

The proposal (Oct 2) and Report 1 (Oct 30) define a controlled comparison: same ViT, same training, three attention blocks.
Page numbers below are the printed ones.

| Item | What the documents say | Source |
| --- | --- | --- |
| Research question | « Quel est l'effet du mécanisme d'attention sur la précision et le coût d'un ViT entraîné de zéro ? » | Proposal p.2; Report 1 p.1 |
| Mechanisms | Full attention, O(N²); shifted windows (M×M, shifted by ⌊M/2⌋ on alternating layers, cyclic shift plus mask), O(N); axial (column pass then row pass, each with its own projections), O(N√N) | Proposal p.2; Report 1 pp.5–8 |
| Isolation principle | Same base architecture, trained from scratch; only the attention block changes | Proposal p.3; Report 1 pp.1, 10 |
| Backbone | Pre-LN encoder, MSA plus 2-layer GELU MLP, residuals, learned position embeddings, CLS token with final LN and linear head | Report 1 p.5 |
| Patch and window | 4×4 patches: N = 64 at 32 px, N = 1600 at 160 px; M = 4 | Report 1 p.6, Table 2, Annex A |
| Cost formulas | Ω(MSA) = 4Nd² + 2N²d; Ω(W-MSA) = 4Nd² + 2M²Nd; Ω(axial) = 2(4Nd² + 2N√N·d) | Report 1 pp.6–8 |
| Dataset 1 | CIFAR-10: 50k train, 10k test, 32×32 | Report 1 p.2 |
| Dataset 2 | CIFAR-10-HR from ImageNet-21K: same 10 classes, 6,000 per class (5,000 train, 1,000 test), synsets mapped through WordNet; images under 160 px dropped, the rest resized and center-cropped to 160×160, plus a 32×32 copy of the same images | Report 1 pp.3–4 |
| Validation split | 10 % of the training images | Report 1 p.10 |
| HP search | Random search, same budget and space for each mechanism, on validation | Proposal p.3; Report 1 p.10 |
| Final protocol | Retrain the best configuration with 3 seeds, evaluate once on test | Report 1 p.10 |
| Hypotheses | H1: at 32 px all three cost about the same, axial the most. H2: at 160 px windows and axial cut FLOPs a lot. H3: windows beat full attention on accuracy. H4: axial accuracy close to full attention. | Report 1 p.10 |
| Literature anchor | Lee et al.: Swin 94.46 % vs ViT 93.58 % on CIFAR-10, 100 epochs, 7.1 M vs 2.8 M parameters | Report 1 p.9, Table 3 |
| Tools (proposal) | PyTorch on Google Colab | Proposal p.3 |

### Deliverables

| Deliverable | Week | Status |
| --- | --- | --- |
| Project proposal | Oct 2 | Done |
| Literature synthesis (Report 1) | S8 | Done (Oct 30) |
| Implementation plus CIFAR-10 HP search | S9–S10 | Not started |
| Training and evaluation on the ImageNet subset | S11–S12 | Not started |
| Presentation | S13 | Due |
| Experimental study (final report) | S14 | Due |

### What Report 1 changed relative to the proposal

1. The ImageNet subset is now concrete (CIFAR-10-HR), and Tiny-ImageNet, STL-10 and Imagenette are explicitly rejected.
2. There are three data conditions, not two: CIFAR-10 at 32 px, CIFAR-10-HR at 32 px and at 160 px. The HR pair isolates resolution on identical images.
3. Fixed numbers: 4×4 patches, M = 4, 10 % validation, 3 seeds, test evaluated once, peak GPU memory.
4. Hypotheses H1 to H4 are new, and the cost analysis is quantified.
5. Not yet in the documents: compute moved from Google Colab to a local RTX 2060 under WSL.

### Week calendar (inferred, verify)

The proposal (Fri Oct 2) ends S4 and Report 1 (Fri Oct 30) ends S8, so S1 is the week of Sept 7.
A break week after S8 would shift everything one week later.

| Week | Dates (Mon–Fri) |
| --- | --- |
| S10 | Nov 9–13 |
| S11 | Nov 16–20 |
| S12 | Nov 23–27 |
| S13 | Nov 30–Dec 4: presentation |
| S14 | Dec 7–11: experimental study due |

## Compute budget

The 32 px experiments are cheap on either GPU. The 160 px block at 4×4 patches is about 67–138 GPU-hours on the RTX 2060 but only about 5–14 GPU-hours on Rorqual's H100s, so the documented experiment fits if it runs on Rorqual.

Reference model: d = 192, depth 9, 3 heads, MLP ratio 2, average-pooling head (close to Lee et al.'s 2.8 M ViT; the CLS head for full attention changes the counts negligibly).
FLOP counts were checked with a CPU prototype and match the Report 1 formulas within 1 %.
GPU times are estimates (45k training images, training step about 3× forward) to be replaced by measured numbers.
They assume 4–8 effective TFLOPS with fp16 on the RTX 2060, and 40–100 effective TFLOPS on one Rorqual H100 (SXM5, 80 GB). A model this small uses only a fraction of an H100's peak, so the H100 range is deliberately conservative.

| Resolution / patch | Tokens N | Mechanism | Params | GFLOPs per image (forward) | Est. time per 100 epochs, RTX 2060 | Est. time per 100 epochs, 1 H100 (Rorqual) |
| --- | --- | --- | --- | --- | --- | --- |
| 32 px / P4 | 64 | full | 2.70 M | 0.37 | ~20–40 min | ~5–15 min |
| 32 px / P4 | 64 | windows | 2.70 M | 0.35 | ~20–40 min | ~5–15 min |
| 32 px / P4 | 64 | axial | 4.03 M | 0.52 | ~25–40 min | ~5–15 min |
| 160 px / P4 (documented) | 1600 | full | 2.99 M | 26.2 | ~12–25 h | ~1–2.5 h |
| 160 px / P4 (documented) | 1600 | windows | 2.99 M | 8.7 | ~4–8 h | ~20–50 min |
| 160 px / P4 (documented) | 1600 | axial | 4.33 M | 13.6 | ~6–13 h | ~30–80 min |
| 160 px / P8 (fallback) | 400 | full | 2.79 M | 3.26 | ~1.5–3 h | ~10–20 min |
| 160 px / P8 (fallback) | 400 | windows | 2.79 M | 2.20 | ~1–2 h | ~5–15 min |
| 160 px / P8 (fallback) | 400 | axial | 4.12 M | 3.33 | ~1.5–3 h | ~10–20 min |

At 32 px, wall time is dominated by data loading and small kernels rather than FLOPs, so the 32 px times are rough on both GPUs.

What follows from the table:

- H1 is already supported analytically: at 32 px full and windows are within 6 % of each other, axial costs about 1.4×.
- Report 1's "≈5× windows, ≈2× axial" holds for the attention sublayer only (4.96× and 2.34×). Over the whole model it is 3.0× and 1.9×.
- With 8×8 patches at 160 px, axial (3.33) costs about the same as full (3.26), which weakens the H2 contrast for axial.
- Axial has 1.5× the parameters (4.0 M vs 2.7 M), the same kind of confound Report 1 criticises in Lee et al.

### Machines

- **Main PC (CPU only, 8 cores, ~7 GB RAM in WSL).** Measured 0.9–1.3 s per training step at batch 128, 32 px: 5–8 min per epoch, 9–13 h per 100-epoch run. Use it for tests, smoke runs on small subsets, FLOP counting, building CIFAR-10-HR, aggregation and figures. Never for runs in the experiment matrix.
- **GPU PC (RTX 2060, WSL, runs started by hand).** Turing has fp16 tensor cores but no bf16: use fp16 autocast with a GradScaler. FlashAttention needs a newer GPU; the memory-efficient SDPA backend should work (check it in phase 0). Assume 6 GB VRAM until `nvidia-smi` says otherwise. Full attention at 160 px / P4 fits only with memory-efficient SDPA at batch 32, with gradient accumulation to 128. Keep data and runs on the WSL ext4 disk, not on `/mnt/c`.
- **Rorqual cluster (Digital Research Alliance of Canada / Alliance de recherche numérique du Canada).** 81 GPU nodes with 4× NVIDIA H100 SXM5 80 GB each; jobs last 1 hour to 7 days, and about half the GPU nodes are split into smaller MIG slices (1g.10gb, 2g.20gb, 3g.40gb) per the Alliance documentation. Confirm your allocation and current limits on the Alliance wiki. This changes the plan:
    - The documented 160 px / P4 block drops from ~67–138 RTX 2060 GPU-hours to ~5–14 H100 GPU-hours (see [Total budget](#total-budget)). Several runs can go in parallel as separate jobs, so the [decision rule](#the-160-px-decision-rule) is likely to land on option 1 (P4, 100 epochs, as documented).
    - Jobs go through the Slurm scheduler (`sbatch`) with a wall-time limit and a queue wait. Keep the resumable queue and checkpoint/resume from phase 2 so a job that hits its time limit restarts where it stopped. Ask for one GPU per run rather than whole nodes; the 32 px runs are small enough for a MIG slice (for example 3g.40gb), which usually waits less in the queue.
    - Report 1 requires every time and memory measurement on the same GPU type. Choose one machine for the cost benchmark and the timing/memory columns of the final table, either all on Rorqual or all on the RTX 2060, and state it in the report. Accuracy runs can use either machine.
    - Newer GPUs support bf16 and FlashAttention. Keep fp16 with a GradScaler and the same SDPA kernel on every machine, so results do not depend on where a run happened.
    - Set up early in S10: confirm the account and allocation are active, install a virtual environment with the cluster's PyTorch modules or wheels, copy CIFAR-10 and CIFAR-10-HR to project or scratch storage (scratch files can be purged), and run `check_env.py` plus a 2-epoch smoke job.
    - The ImageNet-21K extraction for CIFAR-10-HR can also run on the cluster, which avoids downloading ~1 TB to the main PC, if the cluster's storage and network rules allow it.

### Total budget

| Block | Runs | Est. GPU-hours, RTX 2060 | Est. GPU-hours, H100 (Rorqual) |
| --- | --- | --- | --- |
| Cost benchmark (no training) | — | ~1 | ~0.5 |
| HP search, CIFAR-10 32 px: 3 × 8 trials × 30 epochs | 24 | ~3–5 | ~1–2 |
| CIFAR-10 finals: 3 × 3 seeds × 100 epochs | 9 | ~3–6 | ~1–2 |
| CIFAR-10-HR 32 px finals | 9 | ~3–6 | ~1–2 |
| CIFAR-10-HR 160 px finals, P4, 100 epochs | 9 | ~67–138 | ~5–14 |
| … same, P4, 50 epochs | 9 | ~34–69 | ~3–7 |
| … same, P8, 100 epochs | 9 | ~12–24 | ~1–2.5 |
| Re-runs and slack | — | ~25 % | ~25 % |
| **Total, documented plan (P4, 100 epochs)** | 60 | **~95–195** | **~11–28** |

On Rorqual, runs can go in parallel as separate jobs, so the calendar depends more on queue waits than on GPU-hours.

### The 160 px decision rule

After timing full attention at 160 px / P4 on the GPU that will run the 160 px block (Rorqual if available, otherwise the RTX 2060), project the 9-run block:

1. If it fits in 40 GPU-hours at 100 epochs, run it as documented.
2. Otherwise, if it fits at 50 epochs, use P4 at 50 epochs for all three, and state the shorter schedule as a limitation.
3. Otherwise, switch to P8 at 100 epochs and state the weaker axial contrast.

The 40-hour cap assumes about 12 usable GPU-hours a day across S11–S12 on the RTX 2060. On Rorqual, the estimated 5–14 H100 GPU-hours land on option 1 unless your allocation or queue waits say otherwise.
The cost benchmark always runs at P4, so H1 and H2 are tested as documented whatever the training branch.

## Work phases

Nine phases take the project from an empty repository to the S14 report; phases 0–3 must land in S10.
Effort is in focused working days (d) for one person; "GPU" means a run started by hand on the RTX 2060.

### Phase 0: Decisions, environment, repository (Mon Nov 9, ~0.5–1 d)

- [ ] Check the remaining facts in [Decisions](#decisions) (S13/S14 dates, Rorqual allocation).
- [ ] Create the repository layout and a `pyproject.toml`; add `data/`, `runs/`, `checkpoints/`, `*.pt` and `*.npy` to `.gitignore`.
- [ ] Main PC: CPU virtual environment. GPU PC: clone onto the WSL ext4 disk, install the CUDA PyTorch wheel (Windows NVIDIA driver only).
- [ ] Write `scripts/check_env.py`: torch version, CUDA, device name, capability (expect 7.5), VRAM, which SDPA backends run.

**Done when:** `pytest` runs on both PCs and `check_env.py` confirms CUDA, fp16 and memory-efficient SDPA on the GPU PC.

### Phase 1: Data (S10; CIFAR-10 ~0.5 d; CIFAR-10-HR 2–4 d, target Fri Nov 13)

- [ ] **CIFAR-10:** torchvision loader, fixed seeded stratified 45k/5k split, committed index file, fast in-memory loader.
- [ ] **CIFAR-10-HR** (built on Nibi from the Alliance's ImageNet-21K copy; see `documentation/data.md`):
    1. Synset mapping, starting from CINIC-10's published list; commit it.
    2. Acquisition: read the needed synset folders in place from the Alliance's ImageNet-21K copy on Nibi (`/datashare/imagenet/winter21_whole`, opt-in after an image-net.org registration; access confirmed Nov 9), so nothing is downloaded. Record the release used (winter21).
    3. Filter (short side ≥ 160 px), resize, center-crop to 160×160, drop corrupt files.
    4. Sample 6,000 per class with a fixed seed; keep a manifest.
    5. Split 5,000 train / 1,000 test per class, then 10 % of train as validation.
    6. Deduplicate across splits with a perceptual hash.
    7. Write the 160 px array (~4.6 GB, as a memmap) and the 32 px copy (~184 MB).
    8. QA: per-class counts, a sample grid, balanced histogram.

**Done when:** both datasets load through one `get_dataset(name, res, split)` API with committed splits and QA outputs.

### Phase 2: Backbone and full-attention baseline (S10, ~1.5–2 d)

- [ ] `models/vit.py`: patch embedding, learned position embeddings, pre-LN blocks, MLP, classification head (CLS token for full attention, average pooling for windows and axial), stochastic depth, a pluggable attention slot taking `(x, H, W)`.
- [ ] `FullAttention` on `F.scaled_dot_product_attention`, plus a naive kernel switch for the memory benchmark.
- [ ] Training loop: AdamW, warmup plus cosine, label smoothing, fp16 with GradScaler, gradient accumulation, seeding, per-epoch validation, checkpoint and resume, JSONL metrics, wall time and peak memory.
- [ ] One fixed augmentation recipe for all mechanisms.
- [ ] CPU tests: shapes, overfit 256 images to ≥ 95 %, determinism.
- [ ] GPU: a ~5 min `calibrate.py` timing 50 steps per mechanism at 32/P4, 160/P4 and 160/P8 (this sets the 160 px decision), then one 100-epoch baseline sanity run (~30 min).

**Done when:** the sanity run reaches roughly 80–85 % CIFAR-10 validation accuracy and `results/calibration.csv` exists.

### Phase 3: Shifted-window and axial attention (S10–S11, ~2–3 d)

- [ ] `ShiftedWindowAttention`: window partition and reverse, cyclic shift by ⌊M/2⌋ on odd blocks, edge mask; flat variant, no patch merging, no relative position bias.
- [ ] `AxialAttention`: column pass then row pass, each with its own projections; optional parameter-matched variant for an ablation.
- All three are "reshape → the same SDPA call → reshape back", so only the token grouping differs.
- [ ] CPU tests: window attention with M = grid size equals full attention; the shift mask zeroes weights between non-adjacent tokens; cross-check against a reference window block (for example `timm`) used only as a test oracle; axial perturbation reaches exactly its row and column; gradients flow; each passes the overfit test.

**Done when:** all tests pass on CPU and each mechanism completes a 2-epoch GPU smoke run.

### Phase 4: Cost benchmark, no training (Mon Nov 16, ~1 d plus ~1 GPU-hour)

- [x] FLOPs from `FlopCounterMode` next to the Report 1 formulas.
- [x] Forward plus backward time per image (synchronised, warm-up, median of 50 steps).
- [x] Peak memory under SDPA (training setting) and naive attention (shows N² memory; out-of-memory is recorded as a result).
- [x] Resolutions 32, 64, 96, 128 and 160 px at P4, plus P8 at 160.

**Done when:** `results/cost_benchmark.csv` exists and the cost-vs-N figure is drafted; H1 and H2 can be stated with measured numbers.

**Done (Nov 9, Rorqual job 26203994, one full H100):** `results/cost_benchmark.csv` and `figures/cost_vs_tokens.png`. At 160 px / P4 (1,600 tokens), full / windows / axial: forward 26.2 / 8.7 / 13.7 GFLOPs; training 1.06 / 0.74 / 0.80 ms per image; peak memory 10.2 / 10.3 / 13.7 GB with the efficient SDPA kernel and 67.4 / 10.8 / 16.2 GB with the naive kernel.

### Phase 5: Hyperparameter search on CIFAR-10 (S11, ~3–5 GPU-hours overnight)

- [x] Random search, 8 trials per mechanism, shared space and seed list (trial i gets the same configuration for every mechanism), 30 epochs each on the 45k/5k split.
- Space: learning rate log-uniform 3e-4 to 3e-3; weight decay log-uniform 0.01 to 0.1; drop-path in {0, 0.05, 0.1}; warmup in {5, 10} epochs. Fixed: architecture, augmentation, batch 128, optimizer.

**Done when:** all 24 trials are in `results/hpsearch_cifar10.csv` and one chosen configuration per mechanism is in `configs/final/`.

**Done (Nov 9, Rorqual jobs 26204000–26204002, 1g.10gb slices):** all 24 trials are in `results/hpsearch_cifar10.csv`; `configs/final/` holds the selected configurations, unchanged for phase 6:

| Mechanism | Trial | Learning rate | Weight decay | Drop-path | Warmup | Val acc (epoch 30) |
| --- | --- | --- | --- | --- | --- | --- |
| full | 5 | 1.12e-3 | 0.084 | 0 | 10 | 77.0 % |
| windows | 3 | 9.01e-4 | 0.0953 | 0.1 | 10 | 72.6 % |
| axial | 3 | 9.01e-4 | 0.0953 | 0.1 | 10 | 75.1 % |

Stability: full attention trained in all 8 trials (72.6–77.0 %). Windows collapsed to about 28 % in 2 of 8 trials (lr 1.87e-3 and 2.13e-3). Axial collapsed to near chance (13–19 %) in 5 of 8 trials: every trial with lr ≥ 1.0e-3 except trial 5 (lr 1.12e-3, warmup 10). The collapse happens as the warmup passes roughly 1.3e-3 (axial trial 1 had 33.7 % validation accuracy at epoch 2, then fell to chance). Each selected setting lies inside its mechanism's stable range. The report should state this sensitivity to the learning rate as a finding.

The search runs' `peak_mem_mib` values rise by one model plus its gradients per trial (full 517 → 640 MiB), because the queue kept the previous run's model alive. The queue now frees each run before starting the next. Search memory is not reported; phase 6 numbers come from the fixed queue.

### Phase 6: Final training (S11–S12, the critical path)

- [x] 3 mechanisms × 3 seeds × 100 epochs on CIFAR-10.
- [x] The same on CIFAR-10-HR 32 px.
- [x] CIFAR-10-HR 160 px under the decision rule.
- Reuse the CIFAR-10 hyperparameters for both HR conditions; keep the effective batch at 128 with accumulation at 160 px.
- Retrain on the same 90 % split, evaluate once on test at the final epoch, no early stopping.
- Order: CIFAR-10 first (cheap, early results for the presentation), then HR-32, then HR-160 cheapest mechanism first; the optional axial ablation if time allows.

**Done (Nov 11):** all 27 runs finished on Rorqual (full H100s); HR-160 took 77 / 56 / 60 min per run for full / windows / axial.

**Done when:** all 27 runs have a `summary.json` with test accuracy, training time, time per epoch and peak memory.

**Launch (Rorqual, account def-chdesa, full H100 per job).** Queues in `configs/queue/phase6_*.txt` start from `configs/final/<mechanism>.yaml` and override only the seed and the dataset. Run ids: `<dataset>-<res>-p4-<mechanism>-final-s<seed>`. Batch 128 fits at 160 px (calibration peak 10–14 GB of 80), so there is no accumulation. CIFAR-10-HR must be in `$SYS819_DATA_STORE/cifar10_hr` (rsync from Nibi, see `documentation/data.md`). The jobs copy only the array they need to the node-local disk.

```bash
source ~/SYS819/scripts/rorqual/env.sh
cd "$SYS819_REPO"
sbatch --job-name=sys819-final-cifar10 --export=ALL,SYS819_COND=cifar10 scripts/rorqual/final32.sbatch   # 9 runs, ~1.5 GPU-h
sbatch --job-name=sys819-final-hr32 --export=ALL,SYS819_COND=hr32 scripts/rorqual/final32.sbatch         # 9 runs, ~1.5 GPU-h
for m in windows axial full; do                                                                          # ~3.0 / 3.3 / 4.4 GPU-h
    sbatch --job-name=sys819-final-hr160-$m --export=ALL,SYS819_MECH=$m scripts/rorqual/final160.sbatch
done
sys819_activate   # status.py needs the venv
for q in configs/queue/phase6_*.txt; do python scripts/status.py "$q"; done
```

The five jobs can run in parallel; they are submitted in the plan's order. Total is about 15 H100 GPU-hours. Each job resubmits itself if it reaches its time limit (3 h at 32 px, 8 h at 160 px), and resubmitting one by hand is always safe.

### Phase 7: Evaluation and figures (end S12 to start S13, ~1.5 d)

- [x] `aggregate_results.py` builds the main table (mean ± std per mechanism and condition) and an H1–H4 verdict table.
- [x] Figures from the CSVs only: cost vs tokens N (log-log, with analytic curves); accuracy by mechanism for the three conditions; accuracy vs GFLOPs and vs training time; validation curves per mechanism; a diagram of which keys each query sees.

**Done when:** every number and figure regenerates from committed CSVs with one command.

**Status (Nov 11, 27 of 27 runs):** `scripts/collect_runs.py` brings finished run folders into `results/final_runs.csv` and `results/final_curves.csv`; `PYTHONPATH=src python scripts/aggregate_results.py` regenerates `results/main_table.{csv,md}`, `results/hypotheses.{csv,md}` and the five figures in `figures/`. Every condition has 3/3 seeds and all H1-H4 verdicts are final.

### Phase 8: Written deliverables (S13–S14, ~3–4 d)

- [ ] **Presentation (S13):** problem, the three mechanisms, CIFAR-10-HR, protocol, cost results (H1/H2), accuracy results (H3/H4), the trade-off plot, limitations. Draft from CIFAR-10 and cost results if HR-160 is still running.
- [ ] **Experimental study (S14):** extend Report 1 with Implementation, Experimental setup (hardware, recipe, HP search, splits, every deviation from Report 1), Results, Discussion per hypothesis, Limitations, Conclusion; apply the [fixes](#fixes-for-the-final-report).

**Done when:** both are submitted.

### Calendar

All experiments must finish by Nov 27 to leave S13 and S14 for writing.

| Week | Main PC (CPU) | GPU PC (runs started by hand) | Milestone |
| --- | --- | --- | --- |
| S10 Nov 9–13 | Phases 0 → 2 → 3 code and tests; CIFAR-10-HR acquisition and build | `check_env`, calibration (by Wed Nov 11), baseline sanity run | 160 px decision; CIFAR-10-HR built on Nibi (Fri Nov 13) |
| S11 Nov 16–20 | Phase 4 script; analysis scaffolding | Cost benchmark; HP search (overnight); CIFAR-10 finals | Hyperparameters frozen by about Wed Nov 18 |
| S12 Nov 23–27 | Aggregation and figure scripts; start the report | HR-32 finals; HR-160 finals (multi-day) | All runs done by about Fri Nov 27 |
| S13 Nov 30–Dec 4 | Figures; presentation | Re-runs or the optional ablation only | Presentation |
| S14 Dec 7–11 | Experimental study | — | Final report |

## Experiment matrix and GPU-run protocol

Every GPU batch is one resumable command, so an interrupted night costs nothing.

- **One command per batch.** `python -m attnvit.queue configs/queue/<phase>.txt` reads one run configuration per line and runs them in order. It skips any run whose `runs/<run_id>/summary.json` exists and resumes from `last.pt`.
- **Deterministic run ids:** `<dataset>-<res>-p<P>-<mech>-<tag>-s<seed>`.
- **Run files:** `config.yaml`, `metrics.jsonl`, `summary.json` and checkpoints under `runs/<run_id>/`.
- **Bringing results back:** commit only `summary.json` and `metrics.jsonl` (or the aggregated CSVs); checkpoints stay on the GPU PC. The queue prints a one-line summary per run at the end.
- **Before each queue:** a 1-step smoke check of every configuration, so a typo fails in seconds, not at 3 a.m.
- **Status at a glance:** `scripts/status.py` lists done, running and pending runs for a queue file.

The matrix is 27 final runs: 3 mechanisms × 3 seeds × 3 conditions (CIFAR-10 32 px, CIFAR-10-HR 32 px, CIFAR-10-HR 160 px), plus 24 search trials and the optional axial ablation.

## Repository layout and dependencies

A small `src/attnvit` package with one pluggable attention module, committed configs and results, and large data kept out of git and out of OneDrive.

```
Attention_mechanism/
├── README.md                  # what, how to run, results pointer
├── pyproject.toml             # deps (CPU and CUDA install notes)
├── .gitignore                 # + data/ runs/ checkpoints/ *.pt *.npy
├── src/attnvit/
│   ├── models/
│   │   ├── vit.py             # backbone, pluggable attention slot, GAP head
│   │   └── attention.py       # Full, ShiftedWindow, Axial (+ naive kernel switch)
│   ├── data/
│   │   ├── cifar10.py
│   │   ├── cifar10_hr.py      # memmap loader for the built dataset
│   │   └── augment.py         # the fixed recipe shared by all mechanisms
│   ├── train.py               # loop, AMP, accumulation, resume, logging
│   ├── evaluate.py            # one-shot test evaluation
│   ├── cost.py                # analytic formulas + FlopCounterMode wrapper
│   ├── queue.py               # resumable sequential run queue
│   └── utils.py               # seeding, config loading, timers
├── configs/                   # base.yaml, search/, final/, queue/
├── scripts/                   # check_env, calibrate, benchmark_cost, build_cifar10_hr,
│                              # aggregate_results, make_figures, status
├── data_splits/               # committed split indices, synset mapping, HR manifest
├── tests/                     # CPU-only pytest suite
├── results/                   # committed small CSV/JSON summaries
├── figures/                   # generated PNG/PDF for slides and report
├── data/ runs/ checkpoints/   # git-ignored, on WSL ext4 on the GPU PC
└── documentation/             # proposal, report PDFs and this plan
```

Keep `data/` and any virtual environment out of OneDrive: syncing a 4.6 GB array or a `.venv` is slow and can conflict.
Point to a non-synced location with an environment variable.

| Purpose | Choice | Note |
| --- | --- | --- |
| Python | 3.11–3.13 | Same minor version on both PCs |
| Framework | PyTorch 2.x + torchvision | CUDA 12.x wheel on the GPU PC, CPU wheel on the main PC; uses SDPA, fp16 autocast, `torch.utils.flop_counter` |
| Reshapes | `einops` (optional) | Readable window and axial reshapes |
| Numerics and IO | numpy, pillow | Memmap datasets; Pillow for the HR build |
| Deduplication | `imagehash` | Near-duplicates across splits |
| Config | PyYAML with dataclasses | No Hydra |
| Logging | JSONL and CSV | wandb optional, off by default |
| Analysis | pandas, matplotlib | Figures from CSVs |
| Tests | pytest | CPU only |
| Test oracle | `timm` (dev only) | Cross-check the window block; never imported by model code |
| Environment | `uv` or `venv` + `pyproject.toml` | Pin versions once phase 0 works |

Not recommended: `flash-attn` (needs a newer GPU than the RTX 2060), `fvcore` for FLOPs (handles SDPA poorly), `torch.compile` until correctness is settled.

## Risks

The biggest risk is the 160 px GPU time, with a fallback decided in S10. ImageNet-21K access, the other big risk, is settled: CIFAR-10-HR is built from the Alliance's copy on Nibi (access confirmed Nov 9).

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| The CIFAR-10-HR build on Nibi is delayed (queue, missing synsets, too few usable images) | Low | Delays CIFAR-10-HR | Run the counts-only mapping step first; the build reports per-class counts and resumes after interruptions |
| 160 px runs do not fit the calendar | High at P4 / 100 epochs | Missing headline result | Calibrate in S10 and apply the decision rule; cheapest mechanism first |
| Out of memory on 6 GB at 160 px | Medium | Runs crash | Memory-efficient SDPA, batch 32 × accumulation 4, activation checkpointing last |
| Bug in the shift mask or axial reshape | Medium | Wrong conclusions | Phase 3 unit tests and the reference oracle |
| Low ViT accuracy from scratch | Medium | Weak comparison | Strong fixed augmentation, small model, sanity target in phase 2 |
| A manual run is forgotten or interrupted | Medium | Lost GPU time | Resumable queue, overnight batches, `status.py` |
| Axial has 1.5× the parameters | Certain | Reviewer objection | Report parameter counts; optional parameter-matched ablation |

## Decisions

Settled 2026-11-06. Most points are already answered by Report 1; the rest use the defaults in this plan.

| Topic | Decision | Basis |
| --- | --- | --- |
| 160 px configuration | 4×4 patches, N = 1600, 100 epochs, as documented; run on Rorqual. The decision rule stays only as a backup. | Report 1; Rorqual budget |
| Classification head | Each technique as its original paper does it: CLS token for full attention (ViT), global average pooling for shifted windows (Swin) and axial attention. One sentence in the final report's limitations. | Original papers |
| Swin variant | Flat (isotropic), no patch merging | Report 1 isolation principle |
| Position encoding | Learned position embeddings for all three | Report 1 p.5 |
| Axial design | Two passes with separate projections; report parameter counts | Report 1 |
| HP search and protocol | Random search, same budget and space per mechanism, 10 % validation, 3 seeds, test once; 8 trials × 30 epochs on CIFAR-10, reused for HR-32 and HR-160 | Report 1 plus plan defaults |
| Augmentation | One fixed recipe for all: crop and flip, RandAugment, mixup/cutmix, label smoothing 0.1, AdamW with warmup and cosine | Plan default |
| Timing and memory GPU | All time and memory measurements on Rorqual's H100 (same GPU type); RTX 2060 for development and smoke runs | Plan default; Report 1 p.4 requires one GPU type |
| Colab | Not needed | Rorqual access |
| Documentation | Proposal, Report 1 and this plan committed to the repository | Decided 2026-11-06 |

**Still to check (facts, not decisions)**

- [ ] Exact S13 and S14 dates and formats on Moodle (this plan assumes Nov 30–Dec 4 and Dec 7–11).
- [ ] Rorqual allocation active, and current job limits.

## Fixes for the final report

Five small corrections carry over from the proposal and Report 1 into the final study.

- State the hardware: the proposal names Google Colab, but the work runs on Rorqual's H100 GPUs (Digital Research Alliance of Canada), with a local RTX 2060 for development.
- Report 1 §6.1 (pp.8–9) gives "≈5× windows, ≈2× axial": that is the attention sublayer only (4.96× and 2.34×); whole-model ratios are 3.0× and 1.9×.
- Report 1 p.5 describes a CLS token for the backbone; state that windows and axial use average pooling, as in their original papers.
- "Swin" in Tables 2 and 3 is the hierarchical model from the literature; name this project's variant precisely.
- Figure numbering jumps from Fig. 2 to Fig. 4 (pp.6–7); reference [12] (STL-10) is incomplete (Coates, Ng and Lee, AISTATS 2011); the Table 3 caption reads "100 époque" and should read "100 époques".

## Next step

Start phase 0 on Monday Nov 9.
Two tasks cannot slip: the ImageNet-21K access check on day 1, and the RTX 2060 calibration run by Wednesday Nov 11.
Together they decide whether the documented 160 px experiment is feasible or which fallback applies.

## Appendix: prototype

A throwaway CPU prototype (torch 2.15.0+cpu) checked the FLOP formulas and the window/full equivalence.
It is a starting point for phases 2 and 3, not tested project code; the shift mask in particular needs the phase 3 unit tests before anyone relies on it.

Output:

```
window(M=grid) == full: True
32px P4 full  params=2.70M counted=  0.37 GFLOP analytic=  0.37
32px P4 swin  params=2.70M counted=  0.35 GFLOP analytic=  0.35
32px P4 axial params=4.03M counted=  0.52 GFLOP analytic=  0.52
160px P4 full  params=2.99M counted= 26.22 GFLOP analytic= 26.19
160px P4 swin  params=2.99M counted=  8.70 GFLOP analytic=  8.67
160px P4 axial params=4.33M counted= 13.65 GFLOP analytic= 13.62
160px P8 full  params=2.79M counted=  3.26 GFLOP analytic=  3.23
160px P8 swin  params=2.79M counted=  2.20 GFLOP analytic=  2.17
160px P8 axial params=4.12M counted=  3.33 GFLOP analytic=  3.30
CPU full: 0.90s/step(b128) -> 5.3 min/epoch (45k)
CPU swin: 0.89s/step(b128) -> 5.2 min/epoch (45k)
CPU axial: 1.30s/step(b128) -> 7.6 min/epoch (45k)
```

```python
# Throwaway prototype: verify FLOP counting of SDPA-based attention variants and CPU speed.
import time, math, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
torch.manual_seed(0)

def sdpa(q,k,v,mask=None):  # q,k,v: (B*, h, n, dh)
    return F.scaled_dot_product_attention(q,k,v,attn_mask=mask)

class MHA(nn.Module):
    """Multi-head attention over token groups: x (G, n, d) -> (G, n, d)."""
    def __init__(s,d,h): super().__init__(); s.h=h; s.qkv=nn.Linear(d,3*d); s.o=nn.Linear(d,d)
    def forward(s,x,mask=None):
        G,n,d=x.shape; q,k,v=s.qkv(x).view(G,n,3,s.h,d//s.h).permute(2,0,3,1,4)
        return s.o(sdpa(q,k,v,mask).transpose(1,2).reshape(G,n,d))

class Full(nn.Module):
    def __init__(s,d,h,**_): super().__init__(); s.a=MHA(d,h)
    def forward(s,x,H,W): return s.a(x)

class Window(nn.Module):
    def __init__(s,d,h,M=4,shift=0,**_): super().__init__(); s.a=MHA(d,h); s.M=M; s.s=shift
    def forward(s,x,H,W):
        B,N,d=x.shape; M=s.M; g=x.view(B,H,W,d)
        if s.s: g=torch.roll(g,(-s.s,-s.s),(1,2))
        w=g.view(B,H//M,M,W//M,M,d).permute(0,1,3,2,4,5).reshape(-1,M*M,d)
        mask=None
        if s.s:
            img=torch.zeros(H,W)
            cnt=0
            for hs in (slice(0,-M),slice(-M,-s.s),slice(-s.s,None)):
                for ws in (slice(0,-M),slice(-M,-s.s),slice(-s.s,None)):
                    img[hs,ws]=cnt; cnt+=1
            mw=img.view(H//M,M,W//M,M).permute(0,2,1,3).reshape(-1,M*M)
            allowed=(mw[:,:,None]==mw[:,None,:])           # (nW, M2, M2)
            mask=allowed.repeat(B,1,1)[:,None]              # (B*nW,1,M2,M2)
        y=s.a(w,mask).view(B,H//M,W//M,M,M,d).permute(0,1,3,2,4,5).reshape(B,H,W,d)
        if s.s: y=torch.roll(y,(s.s,s.s),(1,2))
        return y.reshape(B,N,d)

class Axial(nn.Module):
    def __init__(s,d,h,**_): super().__init__(); s.col=MHA(d,h); s.row=MHA(d,h)
    def forward(s,x,H,W):
        B,N,d=x.shape; g=x.view(B,H,W,d)
        c=s.col(g.permute(0,2,1,3).reshape(B*W,H,d)).view(B,W,H,d).permute(0,2,1,3)
        r=s.row(c.reshape(B*H,W,d)).view(B,H,W,d)
        return r.reshape(B,N,d)

class Block(nn.Module):
    def __init__(s,att,d,r=2): super().__init__(); s.n1=nn.LayerNorm(d); s.att=att; s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,r*d),nn.GELU(),nn.Linear(r*d,d))
    def forward(s,x,H,W): x=x+s.att(s.n1(x),H,W); return x+s.mlp(s.n2(x))

class ViT(nn.Module):
    def __init__(s,mech,img=32,P=4,d=192,L=9,h=3,M=4,ncls=10):
        super().__init__(); s.H=s.W=img//P; s.pe=nn.Conv2d(3,d,P,P); s.pos=nn.Parameter(torch.zeros(1,s.H*s.W,d))
        mk=lambda i: {"full":lambda:Full(d,h),"swin":lambda:Window(d,h,M,shift=(M//2 if i%2 else 0)),"axial":lambda:Axial(d,h)}[mech]()
        s.blocks=nn.ModuleList([Block(mk(i),d) for i in range(L)]); s.norm=nn.LayerNorm(d); s.head=nn.Linear(d,ncls)
    def forward(s,x):
        x=s.pe(x).flatten(2).transpose(1,2)+s.pos
        for b in s.blocks: x=b(x,s.H,s.W)
        return s.head(s.norm(x.mean(1)))   # GAP head

def analytic(mech,N,d=192,L=9,r=2,M=4):
    proj=4*N*d*d; mlp=2*N*d*r*d
    att={"full":2*N*N*d,"swin":2*M*M*N*d,"axial":4*N*int(math.isqrt(N))*d}[mech]
    if mech=="axial": proj*=2
    return 2*L*(proj+att+mlp)/1e9

# 1) equivalence check: window attention with M = grid, no shift == full attention (same weights)
f=Full(192,3); w=Window(192,3,M=8); w.a.load_state_dict(f.a.state_dict())
x=torch.randn(2,64,192); print("window(M=grid) == full:", torch.allclose(f(x,8,8),w(x,8,8),atol=1e-5))

for img,P in [(32,4),(160,4),(160,8)]:
    N=(img//P)**2
    for mech in ["full","swin","axial"]:
        m=ViT(mech,img,P); n=sum(p.numel() for p in m.parameters())
        with FlopCounterMode(display=False) as fc: m(torch.randn(1,3,img,img))
        print(f"{img}px P{P} {mech:5s} params={n/1e6:.2f}M counted={fc.get_total_flops()/1e9:6.2f} GFLOP analytic={analytic(mech,N):6.2f}")

# 2) CPU speed at 32px: forward+backward, batch 128
for mech in ["full","swin","axial"]:
    m=ViT(mech); opt=torch.optim.AdamW(m.parameters()); xb=torch.randn(128,3,32,32); yb=torch.randint(0,10,(128,))
    for i in range(4):
        if i==1: t=time.time()
        opt.zero_grad(); F.cross_entropy(m(xb),yb).backward(); opt.step()
    dt=(time.time()-t)/3; print(f"CPU {mech}: {dt:.2f}s/step(b128) -> {dt*45000/128/60:.1f} min/epoch (45k)")
```
