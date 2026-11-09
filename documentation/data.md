# Data (phase 1)

Written 2026-11-09. How the three datasets are obtained, built and loaded, and what was checked
about ImageNet-21K access for CIFAR-10-HR.

## The datasets at a glance

All three load through `attnvit.data.get_dataset(name, res, split)`; items are
`(uint8 tensor (3, res, res), int label)`. Data lives under `$SYS819_DATA` (default
`~/data/sys819`), never in the repository.

| `name` | `res` | train / val / test (trainval = train + val) | Committed split record | Status |
| --- | --- | --- | --- | --- |
| `cifar10` | 32 | 45,000 / 5,000 / 10,000 | `data_splits/cifar10_val_idx.json` | built on the main PC |
| `imagenette` | 32, 160 | 8,522 / 947 / 3,925 | `data_splits/imagenette_val.json` | built on the main PC |
| `cifar10_hr` | 32, 160 | 45,000 / 5,000 / 10,000 | `data_splits/cifar10_hr_manifest.csv.gz` | pipeline ready; build on Nibi |

- **CIFAR-10**: torchvision's pickled batches, held in memory. Validation is 500 seeded random
  images per class of the official training set (`numpy.random.default_rng([0, class])`).
- **Imagenette-160** (the fallback): the fastai `imagenette2-160.tgz` archive (SHA-256
  `64d0c485…db02fe5`). Every image already has a 160 px short side; it is center-cropped to
  160×160 and box-filtered to 32×32. Our test split is the official `val` folder; our val split is
  a seeded stratified 10 % of the official `train` folder. Classes are not exactly balanced
  (772–894 train images per class).
- **CIFAR-10-HR**: the ten CIFAR-10 classes from ImageNet-21K, 6,000 images per class (4,500
  train, 500 val, 1,000 test), 160×160 plus a 32×32 copy, built by `scripts/build_cifar10_hr.py`.

Per-channel mean and std (train split) are stored in each built dataset's `build_info.json` and
exposed as `ds.mean`, `ds.std` and `ds.normalizer()`.

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

## Getting ImageNet-21K: the Nibi route (primary)

The Alliance keeps a copy of ImageNet on **Nibi only**, in `/datashare/imagenet/`, an NFS3
mount. That copy includes the full ImageNet-21K **Winter 2021** release (`winter21_whole`:
13,153,500 images in 19,167 classes, one folder per synset). Source: the Alliance wiki page
"ImageNet", marked as a draft. `docs.alliancecan.ca` blocks automated access, so it was read
from the Wayback Machine snapshot of 2025-11-21:
<https://web.archive.org/web/20251121224538/https://docs.alliancecan.ca/wiki/ImageNet>.
The captain checked it on Nibi on 2026-11-09:

- `ls /datashare/imagenet/` lists `DownSampled  ILSVRC2012  tiny-imagenet-200  winter21_whole`.
- `ls /datashare/imagenet/winter21_whole | wc -l` gives 19167.
- Synset folders hold flat files such as `n02691156/n02691156_10000.JPEG`; `n02691156`
  (airplane) holds 1,434 files.
- The wiki tree also shows a `tars/` entry; the count above suggests there is none, and the build
  does not use it.

The archived Nibi wiki page (snapshot 2026-10-29,
<https://web.archive.org/web/20261029135549/https://docs.alliancecan.ca/wiki/Nibi>) gives:

- login node `nibi.alliancecan.ca`, which is also the node for rsync and scp;
- **all Nibi nodes have internet access**;
- `/scratch` has a 1 TB soft quota per user with a 60-day grace period;
- CPU nodes have 192 cores and about 750 GB of memory.

**Rorqual**: its archived page (snapshot 2026-11-02) does not mention ImageNet or `/datashare`,
and the ImageNet page says the dataset is "available only on Nibi". Rorqual's compute nodes have
no internet access, so the arrays are built on Nibi and copied over.

### Steps for the captain

1. **image-net.org account**: sign up at <https://image-net.org/download.php>, log in and accept
   the terms of access (non-commercial research and education). The full-dataset download
   requires an approved account. **Done** (access confirmed 2026-11-09).
2. **Alliance opt-in**: on CCDB, open <https://ccdb.computecanada.ca/services/opt_in> and opt in
   to the ImageNet dataset. This confirms the image-net.org registration. **Done** (access to
   `/datashare/imagenet` confirmed).
3. **Nibi access**: request access to Nibi in CCDB (*Resources → Access Systems*) if it is not
   active yet. **Done** (login on `l3.nibi` as `godinets` confirmed). The wiki warns that a user
   in more than 16 Unix groups may be refused access to the NFS3 mount.
4. **Environment on Nibi** (login node, once): clone the repository and create the venv.
   `scripts/rorqual/setup_env.sh` follows the standard Alliance software stack and is expected to
   work on Nibi too (not verified). If the wheelhouse has no `imagehash`, run
   `pip install imagehash` in the venv; Nibi has internet access.

   ```bash
   git clone <repo-url> ~/src/Attention_mechanism && cd ~/src/Attention_mechanism
   bash scripts/rorqual/setup_env.sh                  # or PYTHON_MODULE=python/3.X bash ...
   source ~/venvs/sys819-attention/bin/activate && python -c "import imagehash" || pip install imagehash
   ```

5. **Quick check** (lists the candidates and prints per-class counts and missing synsets, then
   stops; read `sys819-build-hr-<jobid>.out` and
   `$SCRATCH/sys819-data/cifar10_hr/work/synsets_report.csv`):

   ```bash
   sbatch --account=def-<sponsor> --export=ALL,STEPS=mapping scripts/nibi/build_cifar10_hr.sbatch
   ```

6. **Full build** (resumes after the mapping step; 16 cores, 32 GB, 4 h requested):

   ```bash
   sbatch --account=def-<sponsor> scripts/nibi/build_cifar10_hr.sbatch
   ```

   Equivalent direct call: `python scripts/build_cifar10_hr.py --source
   /datashare/imagenet/winter21_whole --release winter21 --workers 16`.
7. **Commit the split record** from the Nibi clone (or copy the files to your PC clone):
   `data_splits/cifar10_hr_manifest.csv.gz`, `results/data_qa/cifar10_hr_*` and
   `figures/data_qa/cifar10_hr_*`.
8. **Copy the arrays** (run on the destination, which pulls through the Nibi login node;
   `--exclude work` skips the 5 GB of intermediate files):

   ```bash
   # main PC or GPU PC (WSL):
   rsync -av --progress --exclude work <user>@nibi.alliancecan.ca:<nibi-data-root>/cifar10_hr/ ~/data/sys819/cifar10_hr/
   # Rorqual login node (rorqual.alliancecan.ca), into project space:
   rsync -av --progress --exclude work <user>@nibi.alliancecan.ca:<nibi-data-root>/cifar10_hr/ \
       ~/links/projects/<account>/$USER/sys819-data/cifar10_hr/
   ```

   `<nibi-data-root>` is the `SYS819_DATA` printed at the top of the job log (default
   `$SCRATCH/sys819-data`). Logins from one cluster to another may ask for multifactor
   authentication; the Globus collections `alliancecan#nibi` and `alliancecan#rorqual` are the
   alternative.

**Not verified on Nibi** (no cluster access from this machine): the Python module version, that
`$SCRATCH` is set (the job falls back to `/scratch/$USER`), the runtime and memory figures, and
whether `imagehash` is in the wheelhouse.

### Release and synset coverage

CINIC-10 mapped synsets of the **fall11** release (21,841 synsets, 14,197,122 images, the counts
image-net.org still shows). **winter21** removed 2,674 synsets in the 2019 cleanup, mostly in the
*person* subtree (<https://www.image-net.org/update-sep-17-2019.php>, read through the Wayback
Machine). None of the ten classes is under *person*, so few mapped synsets should be missing.
The exact winter21 synset list is not public: the copy on Hugging Face (`timm/imagenet-w21-wds`)
is gated. The mapping step therefore reports missing and empty synsets on the real tree.

Offline evidence that 6,000 usable images per class exist: `timm`'s
`imagenet21k_miil_w21_synsets.txt` lists the 10,450 winter21 classes that Ridnik et al. (2021,
*ImageNet-21K Pretraining for the Masses*) kept after dropping classes with fewer than 500
images. Of the mapped synsets it contains, per class: airplane 21, automobile 29, bird 679, cat 42,
deer 25, dog 72, frog 29, horse 59, ship 42, truck 19. So even the smallest class (truck) has at
least 19 × 500 = 9,500 images before the 160 px filter. CINIC-10 also reports at least 21,939
fall11 images per class. ImageNet photos are mostly larger than 160 px on the short side, so the
filter should keep most of them; the filter step prints the real rejection counts.

The build records `--release winter21` in `build_info.json` and in the QA JSON.

### Resources

- **Disk** under `$SYS819_DATA/cifar10_hr`: 4.6 GB + 184 MB of arrays, plus about 5.1 GB in
  `work/pool` (6,600 crops per class) and the candidate list. `--drop-pool` deletes the pool after
  a successful write.
- **Reads**: only the visited images, roughly 70,000–80,000 JPEGs (about 10 GB) from the NFS
  mount, instead of the 1.1 TB release.
- **Memory**: bounded by design (chunked statistics and hashing); 32 GB is a safe request.
- **Time**: on the main PC, decoding and cropping a 500×375 JPEG takes about 3 ms per core and
  the phash about 2 ms (measured 2026-11-09). Filter and dedupe should therefore take minutes on
  16 cores, plus NFS read time; listing about 1,359 folders over NFS may take longer. 4 h is
  requested, with generous margin.

## Alternative: manual download from image-net.org (short)

Without Nibi, an approved image-net.org account gives the full-release archives from the
download page after login: `fall11_whole.tar` (about 1.3 TB) or the newer `winter21_whole.tar.gz`
(about 1.1 TB). They are commonly described as a tar of per-synset tars (`n02691156.tar`, ...).
**Not verified**: image-net.org refused connections from this machine, and the file list is only
shown after login.

To keep only the needed synsets, stream the archive once and extract the 1,359 member tars
(`data_splits/cifar10_hr_synsets.csv`). The whole archive is read but not stored:

```bash
cut -d, -f1 data_splits/cifar10_hr_synsets.csv | tail -n +2 | sed 's/$/.tar/' > /tmp/hr_members.txt
# Member names may carry a leading directory; check with `tar -tzf <archive> | head` first.
curl -u <user> -L <archive-url> | tar -xzf - -C /path/to/in21k_subset -T /tmp/hr_members.txt
python scripts/build_cifar10_hr.py --source /path/to/in21k_subset --release winter21
```

The build reads per-synset `<wnid>.tar` files in place, so they need not be unpacked. The
per-synset download API that CINIC-10 used (through ImageNet_Utils) may no longer exist.
