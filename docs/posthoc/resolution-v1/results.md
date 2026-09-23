# Post-release analysis: inference resolution and small-person critical misses — results

**This is a post-release analysis on the calibration split, not the locked cohort.** It follows
the [analysis plan](analysis-plan.md), which was committed before any result existed. It changes
no released number, claim, figure, tag or release. Its images are not the locked cohort's, so
its numbers must not be compared with the numbers in the README.

## Answer

Without retraining, raising the inference input from the formal scale (512x910, about 0.711 of
the source) to the native 720x1280 recovers some small people for two of the three models, but
less than the pre-registered threshold for a clear effect (ten percentage points), and most
small people are still missed at native resolution. The native input is also outside the scale
the models were trained at, and for all three models it makes large people worse.

The pre-registered overall conclusion is **B, partial support** for the resolution hypothesis.
Downscaling at inference explains a small part of the small-person misses; most of them come
from somewhere else. This analysis cannot say what training at native resolution would do.

## Primary result

The table gives the small-person critical-miss rate: the share of small-tertile people with
less than half of their pixels classified correctly, averaged over seeds 17, 42 and 73. The
pre-registered contrast is formal minus native, with its Bonferroni-adjusted (98.33%) paired
bootstrap interval over images.

| Model | formal | s085 | native | formal − native | interval | category |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| SegFormer-B2 | 0.748 | 0.633 | 0.658 | 0.090 | 0.035 to 0.146 | B, partial support <!-- claim: p1.posthoc.resolution.segformer; rounded: 3; fields: formal_miss_rate,s085_miss_rate,native_miss_rate,delta,delta_low,delta_high --> |
| UperNet-ConvNeXtV2-Tiny | 0.715 | 0.690 | 0.691 | 0.024 | -0.037 to 0.088 | D, inconclusive <!-- claim: p1.posthoc.resolution.convnextv2; rounded: 3; fields: formal_miss_rate,s085_miss_rate,native_miss_rate,delta,delta_low,delta_high --> |
| UperNet-DINOv2-Small | 0.820 | 0.828 | 0.772 | 0.048 | 0.010 to 0.085 | B, partial support <!-- claim: p1.posthoc.resolution.dinov2; rounded: 3; fields: formal_miss_rate,s085_miss_rate,native_miss_rate,delta,delta_low,delta_high --> |

Every arm is scored on the same 355 small people, found in 127 of the 700 images. <!-- claim: p1.posthoc.resolution.counts -->

The plan also reports each seed's formal − native difference beside the mean:

| Model | seed 17 | seed 42 | seed 73 |
| --- | ---: | ---: | ---: |
| SegFormer-B2 | 0.090 | 0.096 | 0.085 <!-- claim: p1.posthoc.resolution.segformer; rounded: 3; fields: delta_seed_17,delta_seed_42,delta_seed_73 --> |
| UperNet-ConvNeXtV2-Tiny | 0.034 | 0.003 | 0.037 <!-- claim: p1.posthoc.resolution.convnextv2; rounded: 3; fields: delta_seed_17,delta_seed_42,delta_seed_73 --> |
| UperNet-DINOv2-Small | 0.068 | 0.054 | 0.023 <!-- claim: p1.posthoc.resolution.dinov2; rounded: 3; fields: delta_seed_17,delta_seed_42,delta_seed_73 --> |

The decision rules, fixed in advance: **A** when the interval's lower bound is above zero and the
estimate is at least 0.10; **B** when the lower bound is above zero but the estimate is below
0.10; **C** when the whole interval lies within plus or minus 0.05; **D** otherwise. An overall
conclusion needs two of the three models in the same category.

## Scale-mismatch diagnostic

Every model was trained only at the formal scale, so the native arm may be out of distribution.
The plan treats a model's native arm as confounded when native lowers its seed-mean mIoU by
more than the limit set in the plan, or when native makes large people reliably worse. All three
models are confounded:

| Model | formal mIoU | native mIoU | drop | large people, formal − native | descriptive interval | confounded |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| SegFormer-B2 | 0.555 | 0.528 | 0.027 | -0.044 | -0.079 to -0.013 | yes <!-- claim: p1.posthoc.resolution.segformer; rounded: 3; fields: formal_miou,native_miou,miou_drop,large_person_delta,large_person_delta_low_95,large_person_delta_high_95 --> |
| UperNet-ConvNeXtV2-Tiny | 0.592 | 0.551 | 0.041 | -0.070 | -0.110 to -0.035 | yes <!-- claim: p1.posthoc.resolution.convnextv2; rounded: 3; fields: formal_miou,native_miou,miou_drop,large_person_delta,large_person_delta_low_95,large_person_delta_high_95 --> |
| UperNet-DINOv2-Small | 0.437 | 0.419 | 0.018 | -0.044 | -0.073 to -0.016 | yes, by large people only <!-- claim: p1.posthoc.resolution.dinov2; rounded: 3; fields: formal_miou,native_miou,miou_drop,large_person_delta,large_person_delta_low_95,large_person_delta_high_95 --> |

A negative large-person difference means native misses more large people than formal. Being
confounded can only turn a C into a D; it does not change a B. It does mean the native arm
mixes two effects: more pixels on small people, and inputs unlike anything seen in training.

Because every model is confounded, the plan reports the intermediate scale as a descriptive
supplement. It does not re-classify any model:

| Model | formal − s085 | interval |
| --- | ---: | --- |
| SegFormer-B2 | 0.115 | 0.047 to 0.183 <!-- claim: p1.posthoc.resolution.segformer; rounded: 3; fields: formal_minus_s085,formal_minus_s085_low,formal_minus_s085_high --> |
| UperNet-ConvNeXtV2-Tiny | 0.025 | -0.022 to 0.074 <!-- claim: p1.posthoc.resolution.convnextv2; rounded: 3; fields: formal_minus_s085,formal_minus_s085_low,formal_minus_s085_high --> |
| UperNet-DINOv2-Small | -0.008 | -0.045 to 0.030 <!-- claim: p1.posthoc.resolution.dinov2; rounded: 3; fields: formal_minus_s085,formal_minus_s085_low,formal_minus_s085_high --> |

SegFormer-B2 misses fewer small people at the intermediate scale than at native, so its miss
rate does not fall monotonically with scale; none of the three models is monotone.

## Secondary analyses

All descriptive, and none changes the classification above. They are in
[`evidence/summary.json`](evidence/summary.json) under `secondary`: the exact McNemar tests of
formal against native for each model and seed with Holm correction, the miss rate at every
scale, small riders and small people and riders pooled, small motorcycles and bicycles, the miss
rate by model-input area, the share of small people still missed at native, and per-class IoU
for every arm. Small riders are too few to support any conclusion.

## What this does not show

- What training at native resolution, or at several scales, would recover. That needs
  retraining and is out of scope.
- Anything about the locked cohort. The calibration split was used before only to fit the
  temperature, which does not change the predicted class.
- Anything about DINOv2 in general. This adapter's position embeddings were not loaded from
  the pretrained checkpoint (see the
  [experiment card](../../experiment-card.md#threats-to-validity)), and higher resolution
  interpolates them further, so its result is the most exposed to the scale mismatch.

## How the numbers were produced and checked

- Run on Colab with one NVIDIA A100-SXM4-40GB, at commit
  `fb376783dc6f8ba026849c5973e1167c07450e9d`, with the committed `uv.lock`, on the nine released
  checkpoints, whose SHA-256 values match the [experiment card](../../experiment-card.md).
- Before the sweep, the formal arm reproduced the released evaluation path pixel for pixel on
  twenty calibration images for each of the nine checkpoints (the exact criterion of the plan).
- [`evidence/summary.json`](evidence/summary.json) is the unedited output of
  `driving-risk resolution-sweep analyse`. Re-running the analysis on a second machine from the
  per-image records gave the same bytes. The per-image records name images and instances and are
  not published.
- [`evidence/resolution-evidence.json`](evidence/resolution-evidence.json) is derived from the
  summary by `driving-risk resolution-sweep evidence`, which copies numbers and recomputes none.
  Every number on this page is traced to [`claims.yaml`](claims.yaml) by the claims validator;
  `tests/contract/test_posthoc_resolution.py` re-derives the evidence and runs the validator on
  every pull request.
- No part of the analysis departed from the plan. One logistics detail did: the Colab notebook
  fetched the code from this public repository at the pinned commit instead of from a git bundle.

```bash
uv run driving-risk resolution-sweep evidence \
  --summary docs/posthoc/resolution-v1/evidence/summary.json \
  --output docs/posthoc/resolution-v1/evidence/resolution-evidence.json
python .agents/skills/auditing-driving-risk-claims/scripts/validate_claims.py \
  --claims docs/posthoc/resolution-v1/claims.yaml --document docs/posthoc/resolution-v1/results.md
```
