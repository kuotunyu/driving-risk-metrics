# driving-risk-metrics

**Segmentation metrics that know which mistakes kill people.**

A model can score 81% pixel accuracy on CamVid while never once correctly
labelling a pedestrian. That is not a hypothetical — it is the measured result
of a DeepLabV3 run reproduced in this repository's motivating audit
([`docs/AUDIT.md`](docs/AUDIT.md)). The model is excellent at road, sky and
buildings, and blind to people, poles and signs. Its headline metric does not
mention this.

This package provides the evaluation that does.

```
                 pixel share of CamVid train         final per-class IoU
                                                     (DeepLabV3, epoch 19)
  Road           ████████████████████████ 32.96%     0.8959
  Building       ██████████████████ 24.22%           0.6608
  Sky            █████████████ 17.54%                0.8462
  ...
  Pole           ▏ 1.02%                             0.0000   ← never found one
  Pedestrian     ▏ 0.67%                             0.0003   ← never found one
  Bicyclist      ▏ 0.30%                             0.0402
```

Road, Building and Sky are **74.7%** of labelled training pixels. Pedestrian,
Bicyclist and Pole together are **1.99%** — outweighed 37 to 1. A loss averaged
over pixels is not malfunctioning when it ignores them. It is doing exactly what
it was asked to do. The problem is what it was asked.

---

## Status

Honest accounting of what exists today:

| Component | State |
|---|---|
| Evaluation core (confusion, both IoU protocols, risk, blind-spot, distance bands) | **Done** |
| Dataset analysis + frozen split manifest with SHA-256 | **Done**, runs on real CamVid |
| Notebook audit (the four-incomparable-numbers finding) | **Done** |
| Evaluation runner, harm-model sensitivity sweep, HTML report | **Done** |
| Synthetic fixtures that exercise the whole chain without a model | **Done** |
| Model definitions (FCN8s, DeepLabV3, SETR-PUP, SegFormer) + training script | **Done**, shapes verified |
| **The actual four-model comparison** | **Not run yet** |

**No model comparison numbers appear anywhere in this repository**, because none
have been produced under this protocol yet. The per-class figures quoted above
are read out of the original notebooks and are labelled as such throughout.

95 tests run on CPU with no dataset, no checkpoint and no GPU.

## The three measurements

### 1. Risk-weighted error — what the mistakes would cost

Instead of averaging IoU uniformly across classes, score each confusion by the
danger it represents. The cost of predicting `p` where the truth is `t`:

```
C[t][p] = max(0, miss(t) - miss(p))               # severity you failed to report
        + phantom(p)  if miss(p) > miss(t)        # nuisance from over-reporting
```

Read plainly: **you are charged for the danger you failed to signal, plus a
smaller charge for crying wolf.** Three consequences fall out of the subtraction,
and all three are the reason it is written this way rather than as the more
obvious `miss(t) + phantom(p)`:

- Pedestrian → Road costs **97**. A person became empty tarmac.
- Pedestrian → Bicyclist costs **0**. A planner brakes for either; the
  substitution is harmless. Under the naive additive rule this would have cost
  *more* than calling the person "road", which is backwards.
- Road → Pedestrian costs **5**. An unnecessary brake is a real cost, but it is
  not in the same category as the first one.

This makes it a **safety-consequence metric, not a classification metric**: a
model that labels every cyclist a pedestrian scores perfectly here. That is
intended, and it is why mIoU is always reported alongside — only mIoU catches
that particular confusion.

The harm model is the subjective part, so it lives in one inspectable place
([`taxonomy.py`](src/drivemetrics/taxonomy.py)), never inside a metric, and
every headline number is reported with a sensitivity sweep over it. A conclusion
that does not survive the sweep is reported as not surviving it.

### 2. Blind-spot rate — how often a present hazard is missed entirely

Averages let a model trade a total failure on one image against a good result on
another. For driving that trade is not admissible. This metric asks a per-image
yes/no question instead — *this pedestrian was there; was anything recovered at
all?* — and reports the failure fraction.

It is deliberately crude, because crude is harder to game. Ninety-nine perfectly
segmented pedestrian-free images cannot dilute one total failure, since they
never contained a pedestrian. Because the "recovered at all" bar is a threshold,
and thresholds invite cherry-picking, the API returns the whole curve over
thresholds and the report renders it rather than publishing one flattering point.

### 3. Distance-stratified evaluation — near misses and far misses are different

"0.17 IoU on pedestrians" is much less actionable than "0.44 within 15 m, 0.02
beyond 40 m". Distance is what turns a perception error into a stopping-distance
problem. Image rows are projected to ground distance with a flat-ground pinhole
model and bucketed.

**CamVid publishes no camera calibration.** The model therefore runs on three
explicitly declared assumptions — horizon row, focal length, camera height — and
every result carries them in its output so a number can never be quoted without
them. They are parameterised by *horizon row* rather than pitch angle on purpose:
the horizon is visible in the image and checkable by eye, pitch is not, and an
error in pitch is silent.

---

## Findings so far

Two results from the audit, both reproducible with the scripts in this repo.

### The source notebooks' results were never comparable

Four CamVid notebooks shared one `iou()` function and one `num_class = 11`
declaration, and their mIoU values were routinely compared:

| Notebook | classes in stored output | final mIoU | markdown claimed |
|---|---:|---:|---:|
| `FCN_PyTorch_corrected` | **3** | 0.8157 | 0.8688 |
| `FCN_PyTorch_wandb_sweep` | 11 | 0.6780 | 0.4195 |
| `SETR_PyTorch_CamVid` | 11 | 0.5657 | — |
| `Semantic_Segmentation_DeepLabv3` | 11 | 0.3808 | — |

The first run's stored output is `IoUs: [0.87150295 0.93709133 0.63858718]` —
three values. It was an 11-class number in name only, so "FCN 0.82 beats SETR
0.57" compared a 3-class problem against an 11-class one. Every notebook's prose
also disagrees with its own cell output.

Hence `IoUResult.n_classes_counted`, which makes the class count travel with the
score, and `dataset_iou` vs `per_image_nanmean_iou`, which measures the
aggregation gap instead of arguing about it.

### CamVid's standard val split is not a sample of its train split

Measured with `scripts/analyse_dataset.py`:

| Class | train % | val % | ratio |
|---|---:|---:|---:|
| Bicyclist | 0.304 | 2.256 | **7.4×** |
| Fence | 1.173 | 3.131 | 2.7× |
| Car | 6.107 | 1.776 | **0.29×** |
| Sky | 17.538 | 9.336 | 0.53× |

All 101 validation frames contain all 11 classes; it is one contiguous sequence
with very low scene diversity. This matters directly for rare-class claims: a
"rare" class in training is not rare in validation, so a good Bicyclist IoU on
this split is not evidence of good rare-class performance. Any per-class number
on CamVid val is measuring a different distribution from the one the model
trained on, and this repo reports both priors rather than assuming they match.

---

## Install

```bash
pip install -e ".[dev]"
```

The evaluation core depends on **numpy only** — deliberately, so that every
metric is testable on a CPU-only machine with no deep-learning stack, dataset,
or checkpoint. Torch and Pillow are extras needed only to produce predictions
(`[train]`) or render figures (`[report]`).

## Quickstart

```python
import numpy as np
from drivemetrics import (
    ConfusionMatrix, DEFAULT_HARM, dataset_iou, expected_risk,
    risk_contribution_by_confusion,
)

cm = ConfusionMatrix()
for target, pred in predictions:        # any (H, W) int arrays of class indices
    cm.update(target, pred)

iou = dataset_iou(cm)
print(f"mIoU {iou.mean:.4f} over {iou.n_classes_counted} classes")

risk = expected_risk(cm, DEFAULT_HARM)
print(f"risk {risk.expected_risk:.3f}  skill {risk.risk_skill:.3f}")

for row in risk_contribution_by_confusion(cm, DEFAULT_HARM, top_k=5):
    print(f"  {row['true']:>10} -> {row['pred']:<10} {row['risk_share']:.1%}")
```

Reproduce the dataset evidence:

```bash
python scripts/analyse_dataset.py --root /path/to/CamVid
```

Audit notebooks for incomparable results:

```bash
python scripts/audit_notebooks.py /path/to/notebooks
```

Exercise the whole pipeline with no model at all — synthetic fixtures, banner-marked
as such, that produce a full report:

```bash
python scripts/evaluate.py --root /path/to/CamVid --split val --synthetic
```

Then render the report:

```bash
python scripts/report.py
```

Train a model and score it:

```bash
python scripts/train.py --root /path/to/CamVid --model fcn8s --epochs 40
```

`notebooks/train_colab.ipynb` runs the whole comparison on a Colab GPU.

Run the tests:

```bash
pytest
```

---

## Validating the pipeline without a model

`drivemetrics.synthetic` generates deterministic predictions by degrading ground
truth under declared profiles. They exist so the evaluation, sweep and report can
be debugged before GPU time is spent, and so CI can exercise the report without
shipping a checkpoint. Everything derived from them is stamped `synthetic: true`
and the report carries a banner.

One profile is deliberately built to reproduce the failure this repository is
about — `synth-background-biased` never recovers a pole, a sign or a pedestrian,
imitating the observed DeepLabV3 run. Another, `synth-hazard-priority`, recovers
hazards almost perfectly while being sloppy on background boundaries.

On the CamVid validation split those two land as:

| profile | mIoU | pixel acc | VRU recall | Pedestrian blind-spot |
|---|---:|---:|---:|---:|
| `synth-background-biased` | 0.6531 | 0.9572 | 0.478 | **101/101 images** |
| `synth-hazard-priority` | 0.6494 | 0.7209 | 0.982 | 0/101 |

mIoU ranks the first one **higher**. It also never registers a single pedestrian.
Across harm models spanning a 100× range, the risk ranking disagrees with the
mIoU ranking at 86% of settings. These are fixtures, not models — but the
codepath that catches this is exercised on every CI run, so it will be working
when the real numbers arrive.

## Limitations

- **The harm model is ordinal, not calibrated.** Its tiers (100:30:10:3:1) encode
  a defensible *ordering*, not injury statistics. Magnitudes are not meaningful;
  only the sensitivity sweep is.
- **Distances are approximate.** Flat-ground, assumed intrinsics, no calibration
  data. Good enough to separate near from far, not to certify a range.
- **The risk metric is not a classification metric.** By design it does not
  penalise confusing two equally dangerous classes. Read it with mIoU, never
  instead of it.
- **CamVid is small and its val split is one sequence.** 367 train / 101 val
  frames, with the distribution shift documented above. Conclusions drawn here
  are about this dataset and should not be generalised to road perception at
  large.
- **This is a research and teaching artefact.** It is not a safety case, and
  nothing here constitutes validation of a real vehicle.

## Licence

MIT. CamVid is not redistributed; obtain it separately. The split manifest in
`reports/` records the exact bytes used for any reported result.
