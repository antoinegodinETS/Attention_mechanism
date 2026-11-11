# data_splits

Committed split indices, the CIFAR-10-HR synset mapping and its manifest. Scripts
write these files; they are verified, never silently regenerated.

| File | What | Made by |
| --- | --- | --- |
| `cifar10_val_idx.json` | 5,000 validation indices (500 per class) into torchvision's CIFAR-10 training set; train is the other 45,000. Holds the seed, method and a SHA-256 of the training labels. | `scripts/make_cifar10_split.py` |
| `cinic10_synsets-to-cifar-10-classes.txt` | Verbatim copy of CINIC-10's synset-to-CIFAR-10-class list (see below). | downloaded |
| `cifar10_hr_synsets.csv` | The same mapping as one row per synset: `synset, label, class, words, depth, cinic_images`. | `scripts/make_hr_synsets.py` |
| `cifar10_hr_manifest.csv.gz` | One row per CIFAR-10-HR image (array row, split, label, synset, source file, phash), from the winter21 build on Nibi (2026-11-09). The loader checks the arrays against it. | `scripts/build_cifar10_hr.py` |

## Source of the CINIC-10 synset list

- Repository: https://github.com/BayesWatch/cinic-10 (Darlow et al., 2018, *CINIC-10 is not
  ImageNet or CIFAR-10*, arXiv:1810.03505), MIT licence.
- File: `synsets-to-cifar-10-classes.txt` on branch `master`, last changed in commit
  `1854aabe2dee1f3cff0d4728fb5b58259a505f08` (2018-09-21, "Added all files").
- Downloaded 2026-11-09 from
  `https://raw.githubusercontent.com/BayesWatch/cinic-10/master/synsets-to-cifar-10-classes.txt`.
- Verified: git blob `0efc9cc4fb2bb55251d15ec55fd29c6221053efc` (equal to the blob GitHub lists for
  that path), SHA-256 `32e4998cdc44bb4bb42c235154cd14110778f1649ed9123b1ff928567b491b57`,
  1,372 lines, 1,362 synset lines.

CINIC-10 built this list from the WordNet hierarchy of the **fall11** ImageNet release. It covers
1,359 distinct synsets (airplane 33, automobile 41, bird 872, cat 53, deer 37, dog 95, frog 54,
horse 87, ship 64, truck 23). Three synsets appear twice: two are repeated inside one class, and
`n03770679` "minivan" is listed under both automobile and truck. CINIC-10's
`imagenet-contributors.csv` (same commit, 6 MB, not committed here) uses minivan for automobile
only, so `cifar10_hr_synsets.csv` keeps it there. The `cinic_images` column counts the images
CINIC-10 took from each synset (21,000 per class in total). 81 listed synsets contributed nothing
to CINIC-10, and 36 are not in the fall11 synset list at all (WordNet nodes without images).

Regenerate the CSV with:

```bash
python scripts/make_hr_synsets.py --contributors \
    https://raw.githubusercontent.com/BayesWatch/cinic-10/master/imagenet-contributors.csv
```
