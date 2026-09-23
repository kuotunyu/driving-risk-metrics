# Post-release analysis: inference resolution and small-person critical misses

**Pre-registered analysis plan.** Dated 2026-09-23. This document was committed
to the repository before any result of this analysis existed: no checkpoint had
been run at a non-formal input geometry, and no statistic below had been
computed, when it was written. Its commit is the time stamp of the
pre-registration. Anything reported later that departs from this plan will be
labelled exploratory.

This is a post-release analysis of the v1.0.x release. It does not change any
published number, claim, figure, tag or release, and it does not use the locked
evaluation cohort.

## 1. Research question

Critical misses of small people (instances in the frozen `small` area tertile
whose semantic `correct_fraction` is below 0.5) are the headline weakness of the
released models. Are they caused mainly by the **models**, or mainly by the
**inference input being downscaled** to about 0.711 of source resolution
(720x1280 source images resized to 512x910 and padded to 512x1024)?

## 2. Hypotheses

- **H_res (resolution-limited).** Without retraining, raising the inference input
  to the native 720x1280 lowers the small-person critical-miss rate clearly, and
  monotonically across scales 0.711, 0.85 and 1.0. The effect is concentrated in
  the small tertile; the large tertile is nearly unchanged.
- **H_model (model-limited).** Raising the input resolution leaves the
  small-person miss rate essentially unchanged (inside the equivalence bounds of
  §4.3).
- **Confound: train/test scale mismatch (the FixRes effect).** Every model was
  trained only at scale 0.711. At native resolution objects are 1.41 times larger
  in pixels than anything seen in training and may fall outside the training
  distribution. A native arm that does not improve therefore **cannot** be
  attributed to the model on its own; the diagnostic in §4.4 decides whether it
  can.

What this experiment can answer is: *on these nine checkpoints, how many small
people are recovered by raising the inference resolution alone*. It does not
answer how many would be recovered by training at native resolution; that would
require retraining and is out of scope.

## 3. Design

- **Checkpoints.** The nine formal checkpoints (`segformer_b2`,
  `upernet_convnextv2_tiny`, `upernet_dinov2_small`, each at seeds 17, 42 and 73).
  Each checkpoint's SHA-256 must equal the value in
  [`docs/experiment-card.md`](../../experiment-card.md), and the protocol hash in
  its metadata must equal the formal protocol hash
  (`b33c842250f6afcc7bd7c1108b29bf84f342dda5bb5420e64d6be48773c4369f`).
- **Cohort.** The calibration split only: 700 images, all eligible, manifest
  SHA-256 `9f7288d24227678174eb596ed59fdc6c6879b8e01ae554191eef72e1a375b097`.
- **Arms (pre-registered).**

  | Arm | Model input | Scale | Notes |
  | --- | --- | --- | --- |
  | `formal` | 512x910, padded to 512x1024 | 0.711 | Calls the frozen released preprocessing and restoration functions unchanged |
  | `s085` | 612x1088 | 0.850 | Intermediate scale, no padding |
  | `native` | 720x1280 | 1.000 | No resize, no padding |

  The tool also implements two optional controls, `s060` (432x768, a downward
  positive control) and `formal_unpadded` (512x910 without padding, isolating the
  padding effect). They are off by default and are **not** part of this
  pre-registration; any result from them would be exploratory. If compute forces
  a reduction, `formal` and `native` are the minimal pre-registered pair and the
  `s085` supplement of §4.4 is then reported as unavailable.
- **Inference.** Batch size 1, float32, evaluation mode, no gradients, and the
  same bilinear upsampling of logits to the model-input size that the released
  evaluation uses; argmax; the predicted labels are restored to 1280x720 with
  nearest-neighbour resampling and scored at source geometry. Whole-image
  inference, no sliding window (memory is sufficient, and a sliding window would
  add context and stitching as new variables).
- **Scoring.** The repository's existing rules, with no definition changed:
  corroborated instances (the footprint is the pixels on which the semantic mask
  and the instance annotation agree), `instance_coverages`, the frozen area
  tertiles (`docs/evidence/bdd100k_semseg_v1/area_tertiles.json`, SHA-256
  `5f9365d5b9189b49649e34fc8403f16f4934d630ceb5cf903429007a52997206`, assigned by
  area in native pixels and therefore independent of the inference resolution),
  and critical miss = `correct_fraction < 0.5`. A 19x19 confusion matrix is also
  stored per arm, run and image for the mIoU diagnostic.
- **Parity gate.** Before the sweep, the `formal` arm must reproduce the released
  evaluation path (the released per-sample scorer at temperature 1) pixel for
  pixel on 20 calibration images for each of the nine checkpoints. On an A100
  the criterion is exact: zero mismatched pixels. If only an L4 is available, the
  relaxed criterion is zero change in any instance's critical-miss decision;
  the pixel mismatch count is still recorded. The criterion used and the GPU are
  recorded with the results.

## 4. Pre-registered analysis

### 4.1 Primary quantity

For each model *m*, seed *s* and arm *a*: *M(m, s, a)*, the number of
small-tertile person instances in the calibration split that are critical
misses. The denominator is *N* = 355 corroborated small-tertile person instances
(in 127 images), identical for every arm because ground truth and tertiles do not
depend on the arm.

### 4.2 Primary contrast and statistic

One primary contrast per model (three in total):

Δ_m = Σ_s [ M(m, s, formal) − M(m, s, native) ] / (3 × N),

the excess critical-miss rate of `formal` over `native`, averaged over the three
seeds.

- **Interval.** Paired cluster bootstrap with the **image** as the resampling
  unit. All 700 images take part; an image without a small person contributes
  zero to both numerator and denominator. The statistic is a ratio of sums. 5,000
  resamples, generator seed 20260831 (the repository's bootstrap seed),
  percentile interval. Seeds are held fixed and averaged, not resampled; the
  per-seed Δ is reported beside the seed mean.
- **Multiplicity.** Bonferroni over the three models: the primary interval is
  the 98.33% interval (1 − 0.05/3). The 95% interval is reported as a
  description.

### 4.3 Decision rules

Each model is classified on its own. An overall conclusion requires at least
two of the three models in the same category.

- **A — supports H_res:** the lower bound of the 98.33% interval is above 0 and
  the point estimate Δ is at least 0.10 (ten percentage points, about 36 small
  people per seed).
- **B — partial support:** the lower bound is above 0 but the point estimate is
  below 0.10.
- **C — supports H_model (resolution negligible):** the entire 98.33% interval
  lies inside (−0.05, +0.05) (equivalence bound of five percentage points).
- **D — inconclusive:** anything else.

If the three models fall into different categories, the overall conclusion is
recorded as "model-dependent" and each model is reported separately.

### 4.4 Scale-mismatch diagnostic (decides whether C can stand)

For a model, the `native` arm is treated as confounded by the train/test scale
shift if either holds:

- (a) the seed-mean mIoU of `native` is more than 0.02 below that of `formal`;
- (b) the 95% interval of the large-tertile person critical-miss rate difference
  `formal − native` (same bootstrap as §4.2) lies entirely below 0, that is,
  `native` makes large people worse.

When confounded, a category C is downgraded to D (the result cannot be
attributed to the model), and the same statistic computed for `formal − s085` is
reported as a descriptive supplement. It does not re-classify the model.

### 4.5 Secondary analyses (all descriptive; none changes §4.3)

1. For each (model, seed), an exact McNemar test of `formal` against `native` on
   the discordant small-person pairs *b* and *c*; Holm correction over the nine
   tests.
2. Dose response: the seed-mean small-person miss rate at `formal`, `s085` and
   `native` (and `s060` if run), and whether it is monotone in scale.
3. The same statistics as §4.2 for small riders (*n* = 14) and for small person
   and rider pooled. Riders are described only; no conclusion is drawn.
4. Small motorcycles and bicycles (*n* = 9 and 20): counts only.
5. The "pixel-count" hypothesis: person instances binned by model-input area
   (native area × scale²) in power-of-two bins, with the miss rate of every arm
   per bin. If the arms' curves overlap, misses are governed mainly by the number
   of input pixels.
6. The fraction of small people still missed at `native` (a residual floor due
   to the model or the annotation).
7. mIoU and per-class IoU, including the vulnerable-road-user classes person,
   rider, motorcycle and bicycle, for every arm, from the stored confusions.

### 4.6 What will not be done

No threshold will be tuned, no arm will be selected after the fact, the locked
cohort will not be looked at, the tertiles will not be changed, and no number
from this analysis will be compared directly with a locked-cohort number
(different cohorts). Any analysis that departs from this plan will be labelled
exploratory in the report.

## 5. Data and cohort boundary

- The new command accepts only a manifest whose split is `calibration`; any other
  split is refused in code.
- Only the 700 calibration images, their semantic masks and their instance
  bitmasks are extracted on the compute runtime; no validation or locked-cohort
  image is present there.
- The calibration split was previously used only to fit the temperature (which
  does not change the argmax). It took no part in training or model selection.
- Unchanged by this analysis: the v1.0.x tags and releases, `docs/evidence/**`,
  `docs/claims.yaml`, `configs/**`, `data/transforms.py`, `evaluation/engine.py`,
  `analysis/extended.py`, and the locked cohort.
- Publication, decided separately later: a separate document clearly labelled
  "post-release analysis, calibration split, not the locked cohort", with its own
  claims file. Only aggregate numbers are published; no images, crops, bitmasks or
  per-instance tables.

## 6. Risks

- The train/test scale mismatch can make the `native` result hard to attribute;
  §4.4 addresses this, with `s085` as the intermediate arm.
- The DINOv2 adapter's position embeddings started from a randomly initialised
  224-pixel table and are interpolated; higher resolution interpolates them
  further. The DINOv2 result is the most likely to be affected by the mismatch
  and is interpreted separately.
- The parity gate can fail on a handful of argmax pixels because of the GPU model
  or the kernel algorithms Torch selects. On an A100 a failure stops the run for
  investigation; the criterion is not relaxed. The relaxed criterion of §3
  applies only when the run is on an L4.
- Statistical power: 355 small-person instances in 127 images; only 14 small
  riders, which cannot support a conclusion.
- The gap between the corroborated footprint and the whole-instance area over
  which the tertiles were learned (already disclosed for the released instance
  metrics) applies here as well. All arms share the same ground truth, so it does
  not affect the paired contrasts.
- The instance annotations come from the third-party mirror disclosed in the
  dataset card; that disclosure carries over to this analysis.
