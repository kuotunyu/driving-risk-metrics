# Audit of the source material

This repository grew out of a set of CamVid segmentation notebooks. Before
building anything, the existing results were re-derived from the notebook files
and the annotation masks. Three findings came out of that audit, and they are
what the package is designed around.

Everything here is reproducible:

```bash
python scripts/audit_notebooks.py /path/to/notebooks
python scripts/analyse_dataset.py --root /path/to/CamVid
```

No notebook is executed by the audit — they are parsed as JSON.

---

## 1. The reported results were not comparable

Four notebooks shared one `iou()` implementation and one `num_class = 11`
declaration. Their mIoU values were treated as a ranking of the four
architectures:

| Notebook | classes in stored output | declared | final mIoU | markdown claimed |
|---|---:|---:|---:|---:|
| `FCN_PyTorch_corrected` | **3** | 11 | 0.8157 | 0.8688 |
| `FCN_PyTorch_wandb_sweep` | 11 | 11 | 0.6780 | 0.4195 |
| `SETR_PyTorch_CamVid` | 11 | 11 | 0.5657 | — |
| `Semantic_Segmentation_DeepLabv3` | 11 | 11 | 0.3808 | — |

Two separate problems.

**A three-class run reported as an eleven-class one.** The stored output of
`FCN_PyTorch_corrected` is:

```
epoch19, pix_acc: 0.9419084303449876, meanIoU: 0.8157271540831088,
IoUs: [0.87150295 0.93709133 0.63858718]
```

Three per-class values. Whatever produced that output was not solving the
eleven-class problem the source declares, so "FCN 0.82 beats SETR 0.57" compares
a three-class task against an eleven-class one. A mean over three classes and a
mean over eleven are different quantities and there is nothing in either number
that says so.

**Prose that disagrees with the outputs.** Every markdown claim in these
notebooks differs from that notebook's own stored cell output. For the FCN
notebooks alone there are four distinct candidate answers in circulation —
0.8688, 0.8157, 0.4195, 0.6862 — depending on whether you read the prose, the
output, the sweep's baseline note, or the sweep's best run.

This is not a criticism of the material. It is the ordinary failure mode of
notebook-based research: outputs outlive the code that produced them, and prose
outlives both. It is also precisely why this package accumulates one confusion
matrix per protocol and carries the class count inside the result object.

### What the package does about it

- `IoUResult.n_classes_counted` travels with every score, so a three-class mean
  can never again be mistaken for an eleven-class one.
- `dataset_iou` and `per_image_nanmean_iou` implement the two aggregations
  side by side, so the gap between them is measured rather than argued about.
  The notebooks used the per-image variant, which flatters rare classes: when a
  class appears in few images, one lucky image moves the number a long way.

---

## 2. The classes that matter are a rounding error in the pixel budget

Measured from the 367 training annotation masks (60,913,605 labelled pixels,
void excluded):

| Class | pixels | share | in images |
|---|---:|---:|---:|
| Road | 20,076,880 | 32.96% | 100.0% |
| Building | 14,750,079 | 24.22% | 99.5% |
| Sky | 10,682,767 | 17.54% | 99.7% |
| Tree | 6,166,762 | 10.12% | 86.9% |
| Car | 3,719,877 | 6.11% | 98.1% |
| Pavement | 2,845,085 | 4.67% | 95.1% |
| SignSymbol | 743,859 | 1.22% | 95.6% |
| Fence | 714,595 | 1.17% | 47.1% |
| Pole | 623,349 | 1.02% | 99.7% |
| Pedestrian | 405,385 | 0.67% | 86.4% |
| Bicyclist | 184,967 | 0.30% | 54.8% |

Road, Building and Sky hold **74.71%** of labelled pixels. Pedestrian,
Bicyclist and Pole together hold **1.99%** — outweighed 37 to 1.

A loss averaged over pixels is not malfunctioning when it neglects them; it is
optimising what it was given. The consequence is visible in the notebooks'
own results. `Semantic_Segmentation_DeepLabv3` at epoch 19:

```
pixel accuracy 0.8121      mIoU 0.3808

Road      .8959     Fence       .1094
Sky       .8462     Bicyclist   .0402
Tree      .7871     Pedestrian  .0003
Building  .6608     SignSymbol  .0000
Pavement  .5762     Pole        .0000
Car       .2726
```

An 81%-pixel-accurate model that never correctly labelled a pedestrian, a sign
or a pole. `SETR_PyTorch_CamVid` has the same shape at higher overall numbers:
pixel accuracy 0.8907, mIoU 0.5702, Pole 0.0007, Pedestrian 0.1939.

The headline metric is not wrong. It is answering a question nobody driving a
car would ask.

---

## 3. CamVid's validation split is not a sample of its training split

| Class | train % | val % | val/train |
|---|---:|---:|---:|
| Bicyclist | 0.304 | 2.256 | **7.43×** |
| Fence | 1.173 | 3.131 | 2.67× |
| Pavement | 4.671 | 8.870 | 1.90× |
| Tree | 10.124 | 16.619 | 1.64× |
| Building | 24.215 | 26.412 | 1.09× |
| Pedestrian | 0.666 | 0.662 | 0.99× |
| Road | 32.960 | 29.455 | 0.89× |
| SignSymbol | 1.221 | 0.908 | 0.74× |
| Pole | 1.023 | 0.574 | 0.56× |
| Sky | 17.538 | 9.336 | 0.53× |
| Car | 6.107 | 1.776 | **0.29×** |

All 101 validation frames contain all 11 classes, against 54.8% of training
frames for Bicyclist. The validation split is one contiguous sequence with very
low scene diversity.

This has a direct consequence for the rare-class story. Bicyclist is genuinely
rare in training (0.30%) but *seven times less rare* in validation (2.26%), so a
respectable Bicyclist IoU on this split — SETR reaches 0.526 — is not evidence
of good rare-class performance. It is partly evidence that the class stopped
being rare.

Any per-class number on CamVid val is measured under a different class prior
than the one the model trained on. `scripts/analyse_dataset.py` reports both
priors for exactly this reason, and no conclusion in this repository should be
stated about "rare classes" without saying which split's notion of rare is meant.

---

## What was checked and found clean

- **No credentials.** All source cells across 22 notebooks were scanned for
  Google, OpenAI, HuggingFace and GitHub token patterns. The one apparent hit —
  an `AIza…` string in `stable_diffusion_basic.ipynb` — is a false positive: it
  is a substring inside a base64-encoded image blob, surrounded by unrelated
  base64. The only `token=` assignment is the Textual Inversion placeholder
  `<cat-toy>`. Nothing needs revoking. The CI `guard` job re-runs this check
  over the tracked tree on every push.
- **No cross-split leakage.** `build_manifest` hashes every file and refuses to
  build a manifest if identical contents appear in two splits. CamVid's train
  and val splits are disjoint by content: 468 samples, no duplicates.
