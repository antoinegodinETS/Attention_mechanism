# Data (phase 1)

How the two datasets were obtained, built and loaded. Both were built on 2026-11-09: CIFAR-10 on a
PC, CIFAR-10-HR on Nibi from ImageNet-21K winter21.

## The datasets at a glance

Both load through `attnvit.data.get_dataset(name, res, split)`; items are
`(uint8 tensor (3, res, res), int label)`. Data lives under `$SYS819_DATA` (default
`~/data/sys819`), never in the repository.

| `name` | `res` | train / val / test (trainval = train + val) | Committed split record | Built |
| --- | --- | --- | --- | --- |
| `cifar10` | 32 | 45,000 / 5,000 / 10,000 | `data_splits/cifar10_val_idx.json` | on a PC |
| `cifar10_hr` | 32, 160 | 45,000 / 5,000 / 10,000 | `data_splits/cifar10_hr_manifest.csv.gz` | on Nibi (winter21) |

- **CIFAR-10**: torchvision's pickled batches, held in memory. Validation is 500 seeded random
  images per class of the official training set (`numpy.random.default_rng([0, class])`).
- **CIFAR-10-HR**: the ten CIFAR-10 classes from ImageNet-21K, 6,000 images per class (4,500
  train, 500 val, 1,000 test), 160×160 plus a 32×32 copy, built by `scripts/build_cifar10_hr.py`.

Per-channel mean and std (train split) are stored in CIFAR-10-HR's `build_info.json` (CIFAR-10
uses the usual published values) and exposed as `ds.mean`, `ds.std` and `ds.normalizer()`.

## CIFAR-10-HR build pipeline

`scripts/build_cifar10_hr.py --source <ImageNet-21K root> --release <name>` runs seven resumable
steps. Each finished step leaves a marker in `<out>/work/done/`, so a killed job restarts where it
stopped; `--redo <step>` reruns a step and all later ones. The work directory refuses to resume
with different parameters.

1. **mapping**: lists the image files of the 1,359 synsets in `data_splits/cifar10_hr_synsets.csv`
   (CINIC-10's list, see `data_splits/README.md`), writes `work/synsets_report.csv` (found, missing
   or empty, file count per synset) and prints the candidate count per class. It stops if a class
   has fewer than 6,000 candidate files.
2. **filter**: for each class, walks the candidates in a seeded random order. It drops files that
   Pillow cannot decode and images whose short side is under 160 px (read from the header). It
   resizes the rest so the short side is 160 (Lanczos) and center-crops to 160×160. It stops at
   6,600 accepted images (6,000 plus a 600 reserve). The first *k* usable images of a random
   permutation are a uniform random sample of the usable images, so this equals "filter, then
   sample" without decoding every candidate. It fails, naming the classes, if a class has fewer
   than 6,000 usable images.
3. **sample**: the first 6,000 accepted images of each class; the rest is the reserve.
4. **split**: per class, a seeded permutation gives 1,000 test, then 10 % of the remaining 5,000
   as val (500) and 4,500 train.
5. **dedupe**: computes the 64-bit `imagehash.phash` of every accepted image. Any pair in
   different splits within Hamming distance 4 (`--hash-threshold`), within or across classes, is a
   leak. The val or test member (the test one for a val/test pair) is replaced by the next reserve
   image of its class, and the check repeats until no cross-split pair remains. Replacements are
   logged in `work/dedupe_log.csv`.
6. **write**: `images_160.npy` (60,000×160×160×3 uint8, 4.6 GB), `images_32.npy` (box filter,
   184 MB), `labels.npy`, `manifest.csv` and `build_info.json`. Rows are ordered train, val, test,
   so each split is a contiguous slice of the memmap. The manifest is also written gzipped to
   `data_splits/cifar10_hr_manifest.csv.gz`; commit that file. The loader refuses arrays whose
   manifest checksum differs from the committed one.
7. **qa**: per-class counts, a sample grid, and class and intensity histograms in
   `results/data_qa/` and `figures/data_qa/`, with a check that every split is exactly balanced.

The CPU test suite runs the whole pipeline on a synthetic source tree
(`tests/test_cifar10_hr_build.py`). The tree has real synset ids, images of varied sizes
(some under 160 px), one corrupt file, planted duplicates and one synset stored as a tar.

## The build on Nibi

### Source

The Alliance keeps a copy of ImageNet on **Nibi only**, in `/datashare/imagenet/` (an NFS3 mount;
the ImageNet page of the Alliance wiki says it is "available only on Nibi"). It includes the full
ImageNet-21K **Winter 2021** release, `winter21_whole`: 13,153,500 images in 19,167 classes, one
folder of flat files per synset (for example `n02691156/n02691156_10000.JPEG`). Source: the
Alliance wiki page "ImageNet", read from the Wayback Machine snapshot of 2025-11-21
(<https://web.archive.org/web/20251121224538/https://docs.alliancecan.ca/wiki/ImageNet>), and
checked on Nibi on 2026-11-09 (`ls /datashare/imagenet/winter21_whole | wc -l` gives 19167).
Nibi nodes have internet access; Rorqual's compute nodes do not, so the arrays were built on Nibi
and copied to Rorqual.

Access needs three things, all granted on 2026-11-09: an approved image-net.org account (terms of
access for non-commercial research and education), the ImageNet opt-in on CCDB
(<https://ccdb.computecanada.ca/services/opt_in>), and Nibi access in CCDB (*Resources → Access
Systems*). The wiki warns that a user in more than 16 Unix groups may be refused the NFS3 mount.

### Steps

1. **Environment** (Nibi login node, once). `scripts/rorqual/setup_env.sh` works on Nibi too. The
   wheelhouse ships `imagehash` 4.1.0, which calls `PIL.Image.ANTIALIAS` (removed in Pillow 10);
   the script upgrades `imagehash` alone to >= 4.3 from PyPI, and the build job checks it.

   ```bash
   git clone <repo-url> ~/src/Attention_mechanism && cd ~/src/Attention_mechanism
   bash scripts/rorqual/setup_env.sh                  # or PYTHON_MODULE=python/3.X bash ...
   ```

2. **Counts only** (the mapping step; read `sys819-build-hr-<jobid>.out` and
   `$SCRATCH/sys819-data/cifar10_hr/work/synsets_report.csv`):

   ```bash
   sbatch --account=def-<sponsor> --export=ALL,STEPS=mapping scripts/nibi/build_cifar10_hr.sbatch
   ```

3. **Full build** (resumes after the mapping step; 16 cores, 32 GB, 4 h requested):

   ```bash
   sbatch --account=def-<sponsor> scripts/nibi/build_cifar10_hr.sbatch
   ```

   Equivalent direct call: `python scripts/build_cifar10_hr.py --source
   /datashare/imagenet/winter21_whole --release winter21 --workers 16`. The data root is
   `$SYS819_DATA`, by default `$SCRATCH/sys819-data` (falling back to `/scratch/$USER`).
4. **Commit the split record** from the Nibi clone: `data_splits/cifar10_hr_manifest.csv.gz`,
   `results/data_qa/cifar10_hr_*` and `figures/data_qa/cifar10_hr_*`. A rebuild must reproduce the
   committed manifest; the loader refuses arrays whose manifest checksum differs.
5. **Copy the arrays** (run on the destination, which pulls through the Nibi login node;
   `--exclude work` skips the intermediate files):

   ```bash
   # Rorqual login node, into project space:
   rsync -av --progress --exclude work <user>@nibi.alliancecan.ca:<nibi-data-root>/cifar10_hr/ \
       ~/links/projects/<account>/$USER/sys819-data/cifar10_hr/
   # a PC (WSL):
   rsync -av --progress --exclude work <user>@nibi.alliancecan.ca:<nibi-data-root>/cifar10_hr/ ~/data/sys819/cifar10_hr/
   ```

   `<nibi-data-root>` is the `SYS819_DATA` printed at the top of the job log. Logins from one
   cluster to another may ask for multifactor authentication; the Globus collections
   `alliancecan#nibi` and `alliancecan#rorqual` are the alternative.

### What the build produced

From `results/data_qa/cifar10_hr_qa.json` (release `winter21`, manifest SHA-256 `664e6c8c…`):

- **Synsets**: 36 of the 1,359 mapped synsets are missing or empty in winter21. Candidate files per
  class range from 21,939 (truck) to 812,069 (bird).
- **Filter**: about 7,200–7,950 candidates were examined per class to accept 6,600; 8.5–16.9 %
  were dropped for a short side under 160 px, none for being corrupt.
- **Dedupe**: 388 cross-split near-duplicates (phash, Hamming distance ≤ 4) were replaced, 125 in
  val and 263 in test, using 8–162 reserve images per class.
- **Result**: 60,000 images, exactly 4,500 / 500 / 1,000 per class for train / val / test.
  Train-split mean (0.4754, 0.4666, 0.4223); std (0.2593, 0.2552, 0.2730) at 160 px and
  (0.2402, 0.2366, 0.2561) at 32 px.

CINIC-10 mapped synsets of the **fall11** release; winter21 removed 2,674 synsets in the 2019
cleanup, mostly in the *person* subtree (<https://www.image-net.org/update-sep-17-2019.php>),
which none of the ten classes is under.

### Resources

- **Disk** under `$SYS819_DATA/cifar10_hr`: 4.6 GB + 184 MB of arrays, plus about 5 GB in
  `work/` (6,600 crops per class in `work/pool` and the candidate list). `--drop-pool` deletes the
  pool after a successful write.
- **Reads**: only the examined images (about 75,000 JPEGs) from the NFS mount, instead of the
  1.1 TB release.
- **Memory**: bounded by design (chunked statistics and hashing); 32 GB was requested.

## Alternative: manual download (not used)

Without Nibi, an approved image-net.org account gives the full-release archive
(`winter21_whole.tar.gz`, about 1.1 TB), commonly described as a tar of per-synset tars. Extract
only the 1,359 synset tars listed in `data_splits/cifar10_hr_synsets.csv` and point `--source` at
them; the build reads per-synset `<wnid>.tar` files in place. This route was never tried.
