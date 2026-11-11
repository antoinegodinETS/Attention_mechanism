# SYS819 Attention Mechanism: Project Plan

Written 2026-11-06 from `Proposition SYS819.pdf` and `Rapport1 SYS819.pdf`; updated 2026-11-11
after phase 7.

## Status

Phases 0–7 are done (Nov 6–11): CIFAR-10 and CIFAR-10-HR are built, the three attention mechanisms
are implemented and tested, the cost benchmark, the hyperparameter search and all 27 final runs
ran on Rorqual H100s, and every table and figure regenerates from committed CSVs. **Phase 8, the
presentation (S13) and the experimental study (S14), is the remaining work.**

- **The project.** A solo study for SYS819 *Apprentissage profond* (ÉTS, fall 2026, Marco Pedersoli). It compares three attention mechanisms inside one Vision Transformer trained from scratch: full attention (the ViT baseline), Swin-style shifted windows, and axial attention. Only the attention block changes between models.
- **Measures.** Top-1 accuracy (mean ± std over 3 seeds), FLOPs, training time and peak GPU memory.
- **Data.** CIFAR-10 at 32 px, and CIFAR-10-HR (built from ImageNet-21K) at 32 px and 160 px.
- **Deadlines.** Presentation in S13 and experimental study in S14, inferred as Nov 30–Dec 4 and Dec 7–11. Confirm against the course plan.
- **Results.** In `results/main_table.md`, `results/hypotheses.md` and `figures/`; summarised in the [README](../README.md#results).

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
| Implementation plus CIFAR-10 HP search | S9–S10 | Done (Nov 9) |
| Training and evaluation on the ImageNet subset | S11–S12 | Done (Nov 11) |
| Presentation | S13 | To do |
| Experimental study (final report) | S14 | To do |

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

## What happened against the plan

The plan of Nov 6 expected experiments to run until about Nov 27, with the RTX 2060 as a likely
training GPU. Everything finished on Nov 11 instead, on Rorqual. The deviations the final report
must state:

1. **Compute moved to Rorqual's H100s** (Digital Research Alliance of Canada, account
   def-chdesa), not Google Colab (proposal) or the local RTX 2060. Every training run, every time
   and every memory number comes from one H100 80 GB (calibration and cost benchmark and phase 6:
   a full GPU; phase 5 search: 1g.10gb MIG slices). The RTX 2060 was not used for any reported number.
2. **CIFAR-10-HR was built on Nibi** from the Alliance's copy of ImageNet-21K **winter21**
   (`/datashare/imagenet/winter21_whole`), with CINIC-10's fall11 synset mapping; 36 of the 1,359
   mapped synsets are missing or empty there. Splits are 4,500 / 500 / 1,000 per class, and 388
   cross-split near-duplicates were replaced. Details in `documentation/data.md`.
3. **160 px setup: 4×4 patches (1,600 tokens), 100 epochs, batch 128 without accumulation**, as
   documented. The calibration projected the 9-run block at about 10 H100 hours, far under the
   40-hour cap of the [decision rule](#the-160-px-decision-rule), so no fallback was needed.
4. **Search stability finding.** Windows collapsed to about 28 % validation accuracy in 2 of 8
   trials (lr 1.87e-3 and 2.13e-3) and axial to near chance (13–19 %) in 5 of 8: every trial with
   lr ≥ 1.0e-3 except trial 5 (lr 1.12e-3, warmup 10). The collapse happens as the warmup passes
   roughly 1.3e-3 (axial trial 1 reached 33.7 % at epoch 2, then fell to chance). Full attention
   trained in all 8 trials (72.6–77.0 %). Each selected setting lies inside its mechanism's stable
   range. The report should state this sensitivity to the learning rate as a finding.
5. **Queue memory fix.** In the search, each trial's `peak_mem_mib` rose by one model plus its
   gradients (full 517 → 640 MiB over 8 trials), because the queue kept the previous run alive.
   The queue now frees each run before the next (commit `57dbe10`), before phase 6. Search memory
   is not reported; all reported memory comes from phase 6 or the cost benchmark.
6. **Hyperparameters chosen on CIFAR-10 were reused unchanged** for both HR conditions (as planned).
7. **Hypothesis thresholds** were not quantified in Report 1; `scripts/aggregate_results.py` fixes
   them (stated at the top of `results/hypotheses.md`): H1 windows within 10 % of full FLOPs; H2 at
   least 25 % fewer FLOPs; H3 windows ahead by more than the combined seed spread; H4 axial within
   1.0 point of full.

## Compute budget

Reference model: d = 192, depth 9, 3 heads, MLP ratio 2. Parameters and forward GFLOPs per image
are counted by `FlopCounterMode` (`results/cost_benchmark.csv`) and match the Report 1 formulas.
Times per 100-epoch run (45k training images, batch 128, fp16) are measured on one H100: the
P4 rows from the phase 6 runs (`results/main_table.md`), the P8 rows from the calibration
(`results/calibration.csv`; P8 was not trained).

| Resolution / patch | Tokens N | Mechanism | Params | GFLOPs per image (forward) | Time per 100 epochs, 1 H100 |
| --- | --- | --- | --- | --- | --- |
| 32 px / P4 | 64 | full | 2.70 M | 0.38 | 6.6 min |
| 32 px / P4 | 64 | windows | 2.70 M | 0.35 | 8.2 min |
| 32 px / P4 | 64 | axial | 4.03 M | 0.52 | 10.1 min |
| 160 px / P4 (used) | 1600 | full | 2.99 M | 26.2 | 77 min |
| 160 px / P4 (used) | 1600 | windows | 2.99 M | 8.7 | 56 min |
| 160 px / P4 (used) | 1600 | axial | 4.33 M | 13.7 | 60 min |
| 160 px / P8 (fallback, not trained) | 400 | full | 2.79 M | 3.27 | ~17 min (calibration) |
| 160 px / P8 (fallback, not trained) | 400 | windows | 2.79 M | 2.20 | ~19 min (calibration) |
| 160 px / P8 (fallback, not trained) | 400 | axial | 4.12 M | 3.33 | ~22 min (calibration) |

What follows from the table:

- At 32 px wall time follows kernel count more than FLOPs: windows has the fewest FLOPs but trains
  1.24× slower than full.
- Report 1's "≈5× windows, ≈2× axial" holds for the attention sublayer only (4.96× and 2.34×). Over the whole model it is 3.0× and 1.9×.
- At 160 px the FLOP savings (−67 %, −48 %) give smaller time savings (−27 %, −22 %), and peak
  memory with the efficient SDPA kernel is 9.9 / 10.1 / 13.4 GiB (cost benchmark). With the naive
  kernel that stores the attention matrix it is 65.8 / 10.6 / 15.8 GiB.
- With 8×8 patches at 160 px, axial (3.33) would cost about the same as full (3.27), which is why P4 was kept.
- Axial has 1.5× the parameters (4.0 M vs 2.7 M), the same kind of confound Report 1 criticises in Lee et al.

### Total budget (actual)

| Block | Runs | H100 hours (wall, from the run records) |
| --- | --- | --- |
| Calibration and cost benchmark (no training) | — | under 1.5 (job limits 30 min and 1 h) |
| HP search, CIFAR-10 32 px: 3 × 8 trials × 30 epochs (MIG slices) | 24 | 3.0 on 1g.10gb slices (0.9–1.1 per mechanism) |
| CIFAR-10 finals: 3 × 3 seeds × 100 epochs | 9 | 1.6 |
| CIFAR-10-HR 32 px finals | 9 | 2.6 |
| CIFAR-10-HR 160 px finals, P4, 100 epochs | 9 | 11.0 (three jobs of 3.4–4.2 h) |
| **Phase 6 total** | 27 | **15.3** |

The five phase 6 jobs ran in parallel; all 27 runs finished by Nov 11, 01:38.

### The 160 px decision rule

After timing full attention at 160 px / P4 on the GPU that will run the 160 px block, project the 9-run block:

1. If it fits in 40 GPU-hours at 100 epochs, run it as documented.
2. Otherwise, if it fits at 50 epochs, use P4 at 50 epochs for all three, and state the shorter schedule as a limitation.
3. Otherwise, switch to P8 at 100 epochs and state the weaker axial contrast.

**Outcome:** the H100 calibration (Nov 9) projected about 10 GPU-hours at 100 epochs, so option 1
applied. The cost benchmark always runs at P4, so H1 and H2 are tested as documented.

## Work phases

Effort was planned in focused working days (d) for one person.

### Phase 0: Decisions, environment, repository (Nov 6–9) ✅

- [x] Check the facts in [Decisions](#decisions) (the Rorqual allocation is active; S13/S14 dates still to confirm).
- [x] Repository layout and `pyproject.toml`; `data/`, `runs/`, `checkpoints/`, `*.pt` and `*.npy` in `.gitignore`.
- [x] CPU virtual environment on the PC; venv on Rorqual (`scripts/rorqual/setup_env.sh`).
- [x] `scripts/check_env.py`: torch version, CUDA, device name, capability, VRAM, which SDPA backends run.

**Done when:** `pytest` runs on the PC and `check_env.py` confirms CUDA, fp16 and memory-efficient SDPA on the training GPU.

### Phase 1: Data (Nov 9) ✅

- [x] **CIFAR-10:** torchvision loader, fixed seeded stratified 45k/5k split, committed index file, in-memory loader.
- [x] **CIFAR-10-HR**, built on Nibi from ImageNet-21K winter21 (`documentation/data.md`): CINIC-10 synset mapping, filter (short side ≥ 160 px), resize and center-crop, 6,000 per class with a fixed seed, 5,000 / 1,000 split then 10 % validation, phash dedupe across splits, 160 px and 32 px arrays, QA.

**Done when:** both datasets load through one `get_dataset(name, res, split)` API with committed splits and QA outputs.

### Phase 2: Backbone and full-attention baseline (Nov 9) ✅

- [x] `models/vit.py`: patch embedding, learned position embeddings, pre-LN blocks, MLP, CLS head for full attention and average pooling for windows and axial, stochastic depth, a pluggable attention slot taking `(x, H, W)`.
- [x] `FullAttention` on `F.scaled_dot_product_attention`, plus a naive kernel switch for the memory benchmark.
- [x] Training loop: AdamW, warmup plus cosine, label smoothing, fp16 with GradScaler, gradient accumulation, seeding, per-epoch validation, checkpoint and resume, JSONL metrics, wall time and peak memory.
- [x] One fixed augmentation recipe for all mechanisms.
- [x] CPU tests: shapes, overfitting a small set, determinism.
- [x] `calibrate.py` on a full H100 (`results/calibration.csv`, the 160 px decision); `scripts/rorqual/sanity.sbatch` for the 100-epoch baseline.

**Done when:** the sanity run reaches roughly 80–85 % CIFAR-10 validation accuracy and `results/calibration.csv` exists.

### Phase 3: Shifted-window and axial attention (Nov 9) ✅

- [x] `ShiftedWindowAttention`: window partition and reverse, cyclic shift by ⌊M/2⌋ on odd blocks, edge mask; flat, no patch merging, no relative position bias.
- [x] `AxialAttention`: column pass then row pass, each with its own projections; `axial_shared` as the optional parameter-matched variant.
- All three are "reshape → the same SDPA call → reshape back", so only the token grouping differs.
- [x] CPU tests: windows with M = grid size equals full attention; the shift mask against an independent neighbour oracle; a `timm` Swin block as test-only oracle; axial perturbation reaches exactly its row and column; gradients flow; FLOP counts match Report 1.

**Done when:** all tests pass on CPU and each mechanism completes a 2-epoch GPU smoke run (`scripts/rorqual/smoke.sbatch`).

### Phase 4: Cost benchmark, no training (Nov 9) ✅

- [x] FLOPs from `FlopCounterMode` next to the Report 1 formulas.
- [x] Forward plus backward time per image (synchronised, warm-up, median of 50 steps).
- [x] Peak memory under SDPA (training setting) and naive attention (shows N² memory; out-of-memory would be recorded as a result).
- [x] Resolutions 32, 64, 96, 128 and 160 px at P4, plus P8 at 160.

**Done (Rorqual job 26203994, one full H100):** `results/cost_benchmark.csv` and `figures/cost_vs_tokens.png`. At 160 px / P4, full / windows / axial: forward 26.2 / 8.7 / 13.7 GFLOPs; training 1.06 / 0.74 / 0.80 ms per image; peak memory 9.9 / 10.1 / 13.4 GiB with the efficient SDPA kernel and 65.8 / 10.6 / 15.8 GiB with the naive kernel. No setting ran out of memory.

### Phase 5: Hyperparameter search on CIFAR-10 (Nov 9) ✅

- [x] Random search, 8 trials per mechanism, shared space and seed list (trial i gets the same configuration for every mechanism), 30 epochs each on the 45k/5k split.
- Space: learning rate log-uniform 3e-4 to 3e-3; weight decay log-uniform 0.01 to 0.1; drop-path in {0, 0.05, 0.1}; warmup in {5, 10} epochs. Fixed: architecture, augmentation, batch 128, optimizer.

**Done (Rorqual jobs 26204000–26204002, 1g.10gb slices):** all 24 trials are in `results/hpsearch_cifar10.csv`; `configs/final/` holds the selected configurations, unchanged for phase 6:

| Mechanism | Trial | Learning rate | Weight decay | Drop-path | Warmup | Val acc (epoch 30) |
| --- | --- | --- | --- | --- | --- | --- |
| full | 5 | 1.12e-3 | 0.084 | 0 | 10 | 77.0 % |
| windows | 3 | 9.01e-4 | 0.0953 | 0.1 | 10 | 72.6 % |
| axial | 3 | 9.01e-4 | 0.0953 | 0.1 | 10 | 75.1 % |

The learning-rate collapses of windows and axial are described [above](#what-happened-against-the-plan) (item 4).

### Phase 6: Final training (Nov 9–11) ✅

- [x] 3 mechanisms × 3 seeds × 100 epochs on CIFAR-10.
- [x] The same on CIFAR-10-HR 32 px.
- [x] The same on CIFAR-10-HR 160 px, P4, 100 epochs (decision rule option 1).
- CIFAR-10 hyperparameters reused for both HR conditions; batch 128 fits at 160 px without accumulation.
- Trained on the 45k split, test evaluated once at the final epoch, no early stopping.

**Done (Nov 11, Rorqual jobs 26214580, 26214582 and 26214588–26214590, full H100 each):** all 27
runs have a `summary.json` with test accuracy, training time, time per epoch and peak memory. HR-160
took 77 / 56 / 60 min per run for full / windows / axial. Submission commands are in the README.

### Phase 7: Evaluation and figures (Nov 10–11) ✅

- [x] `aggregate_results.py` builds the main table (mean ± std per mechanism and condition) and an H1–H4 verdict table.
- [x] Figures from the CSVs only: cost vs tokens N (log-log, with analytic curves); accuracy by mechanism for the three conditions; accuracy vs GFLOPs and vs training time; validation curves per mechanism; a diagram of which keys each query sees.

**Done:** `scripts/collect_runs.py` brings finished run folders into `results/final_runs.csv` and `results/final_curves.csv`; `python scripts/aggregate_results.py` regenerates `results/main_table.{csv,md}`, `results/hypotheses.{csv,md}` and the five figures. Every condition has 3/3 seeds and all H1–H4 verdicts are final: H1 supported; H2 supported (FLOPs −67 % / −48 %, training only 27 % / 22 % faster, no memory saving); H3 not supported at 32 px, inconclusive at 160 px; H4 not supported on CIFAR-10, supported on HR-32 (borderline) and HR-160.

### Phase 8: Written deliverables (S13–S14, ~3–4 d) — remaining

- [ ] **Presentation (S13, Nov 30–Dec 4):** problem, the three mechanisms, CIFAR-10-HR, protocol, cost results (H1/H2), accuracy results (H3/H4), the trade-off plot, the search stability finding, limitations.
- [ ] **Experimental study (S14, Dec 7–11):** extend Report 1 with Implementation, Experimental setup (hardware, recipe, HP search, splits, every [deviation](#what-happened-against-the-plan)), Results, Discussion per hypothesis, Limitations, Conclusion; apply the [fixes](#fixes-for-the-final-report).
- [ ] Optional, if time allows: the parameter-matched `axial_shared` ablation (not run).

**Done when:** both are submitted.

## Experiment matrix and GPU-run protocol

The matrix is 27 final runs: 3 mechanisms × 3 seeds × 3 conditions (CIFAR-10 32 px, CIFAR-10-HR 32 px, CIFAR-10-HR 160 px), plus 24 search trials.

- **One command per batch.** `python -m attnvit.queue configs/queue/<phase>.txt` runs one configuration per line in order, skips any run whose `$SYS819_RUNS/<run_id>/summary.json` exists and resumes from `last.pt`. On Rorqual each job resubmits itself at its time limit.
- **Deterministic run ids:** `<dataset>-<res>-p<P>-<mech>-<tag>-s<seed>`.
- **Before each queue:** a 1-step smoke check of every configuration.
- **Bringing results back:** `scripts/collect_runs.py` turns run folders into `results/final_runs.csv` and `results/final_curves.csv`; run folders and checkpoints stay in Rorqual project space.
- **Status:** `scripts/status.py` lists done, partial and pending runs for a queue file.

The repository layout and the commands are in the [README](../README.md).

## Risks (outcome)

| Risk | Outcome |
| --- | --- |
| The CIFAR-10-HR build on Nibi is delayed | Built on Nov 9; every class had enough usable images |
| 160 px runs do not fit the calendar | Ran on H100s in about 11 hours; documented setup kept |
| Out of memory at 160 px | Peak 13.5 GiB of 80 GB; no accumulation or checkpointing needed |
| Bug in the shift mask or axial reshape | Covered by the phase 3 tests and the `timm` oracle |
| Low ViT accuracy from scratch | 88.8 % for full attention on CIFAR-10 |
| A run is forgotten or interrupted | Resumable queue and self-resubmitting jobs; all 27 runs finished |
| Axial has 1.5× the parameters | Remains; report the parameter counts (the `axial_shared` ablation was not run) |
| Learning-rate instability (not foreseen) | Windows and axial collapse at lr ≥ ~1e-3; selected settings are stable; report it |

## Decisions

Settled 2026-11-06. Most points were answered by Report 1; the rest use the defaults in this plan.

| Topic | Decision | Basis |
| --- | --- | --- |
| 160 px configuration | 4×4 patches, N = 1600, 100 epochs, as documented; run on Rorqual (decision rule option 1) | Report 1; calibration |
| Classification head | Each technique as its original paper does it: CLS token for full attention (ViT), global average pooling for shifted windows (Swin) and axial attention. One sentence in the final report's limitations. | Original papers |
| Swin variant | Flat (isotropic), no patch merging | Report 1 isolation principle |
| Position encoding | Learned position embeddings for all three | Report 1 p.5 |
| Axial design | Two passes with separate projections; report parameter counts | Report 1 |
| HP search and protocol | Random search, same budget and space per mechanism, 10 % validation, 3 seeds, test once; 8 trials × 30 epochs on CIFAR-10, reused for HR-32 and HR-160 | Report 1 plus plan defaults |
| Augmentation | One fixed recipe for all: crop and flip, RandAugment, mixup/cutmix, label smoothing 0.1, AdamW with warmup and cosine | Plan default |
| Timing and memory GPU | All time and memory measurements on Rorqual H100s (same GPU type) | Report 1 p.4 requires one GPU type |
| Colab | Not used | Rorqual access |
| Documentation | Proposal, Report 1 and this plan committed to the repository | Decided 2026-11-06 |

**Still to check:** the exact S13 and S14 dates and formats on Moodle (this plan assumes Nov 30–Dec 4 and Dec 7–11).

## Fixes for the final report

Corrections to carry from the proposal and Report 1 into the final study, plus what it must state.

- State the hardware: the proposal names Google Colab, but every reported number comes from Rorqual H100s (Digital Research Alliance of Canada); CIFAR-10-HR was built on Nibi.
- State the ImageNet-21K release (winter21, not the fall11 release CINIC-10 mapped), the 36 missing synsets, the 4,500 / 500 / 1,000 split per class and the 388 dedupe replacements.
- State the hypothesis thresholds (Report 1 gives none; see `results/hypotheses.md`) and that the search ran on CIFAR-10 only.
- State the learning-rate sensitivity of windows and axial found by the search.
- Report 1 §6.1 (pp.8–9) gives "≈5× windows, ≈2× axial": that is the attention sublayer only (4.96× and 2.34×); whole-model ratios are 3.0× and 1.9×.
- Report 1 p.5 describes a CLS token for the backbone; state that windows and axial use average pooling, as in their original papers.
- "Swin" in Tables 2 and 3 is the hierarchical model from the literature; name this project's variant precisely.
- Figure numbering jumps from Fig. 2 to Fig. 4 (pp.6–7); reference [12] (STL-10) is incomplete (Coates, Ng and Lee, AISTATS 2011); the Table 3 caption reads "100 époque" and should read "100 époques".

## Next step

Phase 8: draft the presentation from `results/` and `figures/` for S13, then write the experimental
study for S14.
