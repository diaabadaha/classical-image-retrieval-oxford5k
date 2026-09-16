# Classical Image-to-Image Retrieval on Oxford5k

An end-to-end **instance-level image retrieval** pipeline built from classical
computer-vision primitives — SIFT + RootSIFT features, a K-means **Bag-of-Visual-Words**
vocabulary, TF-IDF weighting, and optional **PCA** — evaluated on the Oxford5k Buildings
benchmark with mAP and Precision@K. PCA lifts mAP from **0.284 → 0.351** while *reducing*
query time.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/vision-OpenCV%20SIFT-5c3ee8.svg)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-f7931e.svg)](https://scikit-learn.org/)
[![mAP](https://img.shields.io/badge/Oxford5k%20mAP-0.351-brightgreen.svg)](#results)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)

> Birzeit University — ENCS5343 Computer Vision — Project 1, First Semester 2025/2026.

---

## Table of contents

- [Overview](#overview)
- [Results](#results)
- [Pipeline](#pipeline)
- [Dataset & evaluation](#dataset--evaluation)
- [Quickstart](#quickstart)
- [Repository structure](#repository-structure)
- [Design notes](#design-notes)
- [Authors](#authors)
- [License](#license)

---

## Overview

Unlike classification, instance retrieval must stay robust to viewpoint, illumination,
scale, and background clutter — the *same* building photographed from different angles
should still rank first. This project implements the classic **Bag-of-Visual-Words**
retrieval recipe, enhanced with RootSIFT, TF-IDF, and PCA, and measures it honestly
against the official Oxford5k ground truth.

Classical pipelines like this remain valuable: they're interpretable, need no training
labels, and form a strong, well-understood baseline against which deep methods are
compared.

## Results

Evaluated over the **55 official Oxford5k queries**:

| Configuration | **mAP** | P@5 | R@5 | P@10 | R@10 | Time / query |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| BoVW + TF-IDF | 0.2837 | 0.5491 | 0.1283 | 0.4545 | 0.1706 | 96.73 ms |
| **BoVW + TF-IDF + PCA (256-D)** | **0.3508** | **0.6255** | **0.1442** | **0.5182** | **0.1981** | **91.94 ms** |

**PCA improves every metric *and* runs faster** — removing redundant TF-IDF dimensions
yields a more compact, more discriminative representation.

Single-query example (`all_souls_000013.jpg`):

| | Without PCA | With PCA (256-D) |
|:--|:--:|:--:|
| Average Precision | 0.177 | **0.349** |
| Precision@5 | 0.60 | **1.00** |
| Precision@10 | 0.50 | **0.90** |

### How it stacks up against the literature

| Method | Type | Oxford5k mAP |
|:--|:--:|:--:|
| BoW (large vocabulary) | Classical | ~0.36 |
| VLAD + SSR | Classical | ~0.38 |
| Improved Fisher Vector | Classical | ~0.42 |
| R-MAC (CNN) | Deep | ~0.67 |
| **This work — BoVW + TF-IDF** | Classical | 0.2837 |
| **This work — BoVW + TF-IDF + PCA** | Classical | **0.3508** |

*(Reference numbers differ in evaluation details; provided as approximate context.)*

## Pipeline

```
Query / DB images
      │
      ▼
Preprocessing        resize (max side 1024) → grayscale → CLAHE
      │
      ▼
Local features       SIFT keypoints → 128-D descriptors → RootSIFT (L1-norm + √)
      │
      ▼
Visual vocabulary    MiniBatch K-means, K = 2000 visual words
      │
      ▼
BoVW histogram       nearest-word assignment → TF-IDF weighting → L2-normalise
      │
      ▼
(optional) PCA       project to 256-D → L2-normalise again
      │
      ▼
Ranking              cosine similarity (dot product) → top-K
```

**Design highlights**

- **RootSIFT** makes the descriptor space friendlier to Euclidean clustering and matching.
- **CLAHE** stabilises local features across lighting changes without amplifying noise.
- **TF-IDF** down-weights ubiquitous visual words so distinctive structures dominate the match.
- **PCA (256-D)** is fit on the database only, then applied consistently to queries.

## Dataset & evaluation

- **Dataset:** [Oxford5k Buildings](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/)
  — Oxford landmarks under varied viewpoints/conditions.
- **Ground truth:** official `good`/`ok` treated as relevant, `junk` ignored, the query
  itself excluded from its own results.
- **Metrics:** Precision@K, Recall@K, mean Average Precision (mAP, primary), and average
  retrieval time per query.

## Quickstart

```bash
pip install numpy opencv-contrib-python scikit-learn matplotlib tqdm

# Place the Oxford5k images and ground truth as:
#   dataset/images/         (all .jpg images)
#   dataset/groundtruth.json

cd modules

# Run the pipeline stages in order — each writes artifacts the next stage reads:
python preprocessing.py     # resize + grayscale + CLAHE  → dataset/preprocessed
python descriptors.py       # SIFT + RootSIFT             → dataset/descriptors
python codebook.py          # K-means (K=2000)            → dataset/codeBook
python VBoW.py              # BoVW histograms + TF-IDF    → dataset/vbow
python buildPca.py          # fit PCA (256-D)             → dataset/pca
python retrieveEvaluate.py  # full evaluation over 55 queries (mAP, P@K, R@K, time)
python singleQuery.py       # qualitative top-10 for one query, with GT colour borders
```

> `retrieveEvaluate.py` and `singleQuery.py` default to `usePca=True, pcaDim=256`.
> Set `usePca=False` to reproduce the baseline row.

## Repository structure

```
.
├── modules/
│   ├── preprocessing.py      # resize (keep aspect) + grayscale + CLAHE
│   ├── descriptors.py        # SIFT extraction + RootSIFT normalisation
│   ├── codebook.py           # MiniBatch K-means visual vocabulary (K=2000)
│   ├── VBoW.py               # BoVW histograms + TF-IDF + L2 norm
│   ├── buildPca.py           # fit + save PCA projection (256-D)
│   ├── encodeQuery.py        # encode a query into the retrieval space
│   ├── retrieveEvaluate.py   # batch evaluation: mAP, P@K, R@K, timing
│   ├── singleQuery.py        # qualitative single-query visualisation
│   └── helpers/
│       ├── ioStore.py        # centralised load/save of all artifacts
│       ├── pcaProject.py     # PCA fit / transform
│       └── checkNPYFiles.py  # quick artifact sanity check
├── The Report.pdf            # full write-up with figures
└── README.md
```

## Design notes

- **Vocabulary size (K=2000)** trades discriminative power against memory and query
  time; larger vocabularies sharpen distinctions but slow retrieval and raise memory.
- **Modular, resumable stages.** Each script persists `.npy`/`.json` artifacts through a
  single `ioStore` layer, so any stage can be re-run without redoing the earlier ones.
- **Fair evaluation.** Retrieval time includes both query encoding and ranking, not just
  the similarity search.

## Author

- **Diaa Badaha** — 1210478

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
