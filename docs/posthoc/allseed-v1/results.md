# Post-release analysis: every seed's instance coverage, and intervals for the point-estimate metrics — results

**This is a post-release analysis of the released locked-cohort predictions.** It follows the
[analysis plan](analysis-plan.md), which was committed before any number below existed. It
re-aggregates the per-image artifacts the released evaluation wrote; no model was run and no
released number, claim, figure, tag or release was changed. The cohort is the README's locked
cohort, so these numbers describe the same images and predictions as the README.

## Answer

- **The seed-17 instance numbers in the README are representative.** For every model, seed 17's
  small-person critical-miss rate is within half a percentage point of the mean over the three
  seeds.
- **Every model critically misses most small people.** The lower bound of each model's
  Bonferroni-adjusted interval is above one half, so all three are classified "majority missed".
- **The three models are separable on small people.** UperNet-ConvNeXtV2-Tiny misses the fewest,
  SegFormer-B2 more and UperNet-DINOv2-Small the most, and every pairwise interval excludes zero.
  That order is the same at every critical-miss threshold from 0.1 to 0.9.
- **On the confidence-based metrics SegFormer-B2 leads,** most clearly on selective risk (AURC).
  Temperature scaling closes its Brier-score lead over UperNet-ConvNeXtV2-Tiny and turns its
  calibration-error lead into a small deficit.

## Primary result: the small-person critical-miss rate

The rate is the share of small-tertile people with less than half of their pixels classified
correctly. The interval is the Bonferroni-adjusted (98.33%) two-stage paired bootstrap interval
over images and seeds.

| Model | three-seed mean | interval | seed 17 | seed 42 | seed 73 | category |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| SegFormer-B2 | 0.758 | 0.688 to 0.822 | 0.762 | 0.742 | 0.771 | majority missed <!-- claim: p1.posthoc.allseed.rate.segformer; rounded: 3; fields: estimate,low,high,seed_17,seed_42,seed_73 --> |
| UperNet-ConvNeXtV2-Tiny | 0.662 | 0.593 to 0.730 | 0.662 | 0.652 | 0.671 | majority missed <!-- claim: p1.posthoc.allseed.rate.convnextv2; rounded: 3; fields: estimate,low,high,seed_17,seed_42,seed_73 --> |
| UperNet-DINOv2-Small | 0.846 | 0.798 to 0.891 | 0.851 | 0.848 | 0.840 | majority missed <!-- claim: p1.posthoc.allseed.rate.dinov2; rounded: 3; fields: estimate,low,high,seed_17,seed_42,seed_73 --> |

Every model and seed is scored on the same 462 small people, found in 214 of the 998 images. <!-- claim: p1.posthoc.allseed.counts -->

The decision rule, fixed in advance: "majority missed" when the interval's lower bound is above
0.5, "minority missed" when its upper bound is below 0.5, and "undetermined" otherwise.

## Paired differences between models

| Pair | difference | interval | separable |
| --- | ---: | --- | --- |
| SegFormer-B2 minus UperNet-ConvNeXtV2-Tiny | 0.097 | 0.035 to 0.154 | yes <!-- claim: p1.posthoc.allseed.pair.segformer-minus-convnextv2; rounded: 3; fields: estimate,low,high --> |
| SegFormer-B2 minus UperNet-DINOv2-Small | -0.088 | -0.147 to -0.032 | yes <!-- claim: p1.posthoc.allseed.pair.segformer-minus-dinov2; rounded: 3; fields: estimate,low,high --> |
| UperNet-ConvNeXtV2-Tiny minus UperNet-DINOv2-Small | -0.185 | -0.242 to -0.130 | yes <!-- claim: p1.posthoc.allseed.pair.convnextv2-minus-dinov2; rounded: 3; fields: estimate,low,high --> |

A positive difference means the first model misses more small people. The intervals are
Bonferroni-adjusted (98.33%), like those of the primary result.

## Robustness to the critical-miss threshold

The released definition of a critical miss, less than half of an instance's pixels correct, is
unchanged. As a description, the seed-mean small-person rate at three of the nine thresholds:

| Threshold | SegFormer-B2 | UperNet-ConvNeXtV2-Tiny | UperNet-DINOv2-Small |
| --- | ---: | ---: | ---: |
| one tenth | 0.638 | 0.534 | 0.740 <!-- claim: p1.posthoc.allseed.threshold.01; rounded: 3; fields: segformer_b2,upernet_convnextv2_tiny,upernet_dinov2_small --> |
| one half | 0.758 | 0.662 | 0.846 <!-- claim: p1.posthoc.allseed.threshold.05; rounded: 3; fields: segformer_b2,upernet_convnextv2_tiny,upernet_dinov2_small --> |
| nine tenths | 0.931 | 0.864 | 0.944 <!-- claim: p1.posthoc.allseed.threshold.09; rounded: 3; fields: segformer_b2,upernet_convnextv2_tiny,upernet_dinov2_small --> |

At all nine thresholds the order of the three models is the one it is at one half.

## The small vulnerable road users the plan describes but does not rate

Rider, motorcycle and bicycle have too few small instances for a rate to be interpreted. Their
counts, averaged over the three seeds:

| Model | small riders missed | small motorcycles missed | small bicycles missed |
| --- | ---: | ---: | ---: |
| SegFormer-B2 | 17.0 | 14.0 | 26.0 <!-- claim: p1.posthoc.allseed.small.segformer; rounded: 1; fields: rider_misses,motorcycle_misses,bicycle_misses --> |
| UperNet-ConvNeXtV2-Tiny | 16.3 | 13.3 | 24.3 <!-- claim: p1.posthoc.allseed.small.convnextv2; rounded: 1; fields: rider_misses,motorcycle_misses,bicycle_misses --> |
| UperNet-DINOv2-Small | 17.0 | 14.0 | 29.3 <!-- claim: p1.posthoc.allseed.small.dinov2; rounded: 1; fields: rider_misses,motorcycle_misses,bicycle_misses --> |

The cohort holds 17 small riders, 14 small motorcycles and 32 small bicycles. <!-- claim: p1.posthoc.allseed.small.segformer; rounded: 0; fields: rider_count,motorcycle_count,bicycle_count -->

## Intervals for the point-estimate metrics

All descriptive (95%, no multiplicity adjustment). The table gives the two best models on mean
IoU, SegFormer-B2 minus UperNet-ConvNeXtV2-Tiny. For selective risk, Brier score, calibration
error and cost lower is better, so a negative difference favours SegFormer-B2; for IoU higher is
better, so a negative difference favours UperNet-ConvNeXtV2-Tiny.

| Quantity | difference | interval | excludes zero |
| --- | ---: | --- | --- |
| AURC, uncalibrated | -0.0074 | -0.0084 to -0.0063 | yes <!-- claim: p1.posthoc.allseed.secondary.aurc-uncalibrated; rounded: 4; fields: estimate,low,high --> |
| AURC, calibrated | -0.0073 | -0.0084 to -0.0062 | yes <!-- claim: p1.posthoc.allseed.secondary.aurc-calibrated; rounded: 4; fields: estimate,low,high --> |
| Brier score, uncalibrated | -0.0075 | -0.0103 to -0.0049 | yes <!-- claim: p1.posthoc.allseed.secondary.brier-uncalibrated; rounded: 4; fields: estimate,low,high --> |
| Brier score, calibrated | -0.0001 | -0.0024 to 0.0021 | no <!-- claim: p1.posthoc.allseed.secondary.brier-calibrated; rounded: 4; fields: estimate,low,high --> |
| Classwise ECE, uncalibrated | -0.0017 | -0.0019 to -0.0016 | yes <!-- claim: p1.posthoc.allseed.secondary.classwise-ece-uncalibrated; rounded: 4; fields: estimate,low,high --> |
| Classwise ECE, calibrated | 0.0002 | 0.0001 to 0.0004 | yes <!-- claim: p1.posthoc.allseed.secondary.classwise-ece-calibrated; rounded: 4; fields: estimate,low,high --> |
| Top-label ECE, uncalibrated | -0.0164 | -0.0179 to -0.0150 | yes <!-- claim: p1.posthoc.allseed.secondary.toplabel-ece-uncalibrated; rounded: 4; fields: estimate,low,high --> |
| Top-label ECE, calibrated | 0.0058 | 0.0040 to 0.0076 | yes <!-- claim: p1.posthoc.allseed.secondary.toplabel-ece-calibrated; rounded: 4; fields: estimate,low,high --> |
| IoU, person | -0.0368 | -0.0550 to -0.0209 | yes <!-- claim: p1.posthoc.allseed.secondary.iou-person; rounded: 4; fields: estimate,low,high --> |
| IoU, rider | -0.0759 | -0.1923 to 0.0405 | no <!-- claim: p1.posthoc.allseed.secondary.iou-rider; rounded: 4; fields: estimate,low,high --> |
| IoU, motorcycle | -0.0263 | -0.1014 to 0.0293 | no <!-- claim: p1.posthoc.allseed.secondary.iou-motorcycle; rounded: 4; fields: estimate,low,high --> |
| IoU, bicycle | -0.0683 | -0.1230 to 0.0015 | no <!-- claim: p1.posthoc.allseed.secondary.iou-bicycle; rounded: 4; fields: estimate,low,high --> |
| Cost, vru_priority profile | 0.0002 | -0.0011 to 0.0014 | no <!-- claim: p1.posthoc.allseed.secondary.cost-vru-priority; rounded: 4; fields: estimate,low,high --> |
| Cost, drivable_boundary profile | 0.0001 | -0.0014 to 0.0017 | no <!-- claim: p1.posthoc.allseed.secondary.cost-drivable-boundary; rounded: 4; fields: estimate,low,high --> |

Top-label ECE is not a published metric; the plan added it because it is the form most readers
know. Every level and every other pair, including UperNet-DINOv2-Small, is in
[`evidence/summary.json`](evidence/summary.json) under `secondary.intervals`. For each of the four
vulnerable-road-user IoUs and both cost profiles, UperNet-DINOv2-Small is worse than each of the
other two models and the paired interval excludes zero.

## What this does not show

- Anything beyond three seeds per model: the seed stage of every interval rests on three values.
- A family of tests. The secondary intervals are many and unadjusted; they describe uncertainty.
- Pixel accuracy by image band, which still has no interval.
- Verified instance labels. They come from the third-party mirror disclosed in the dataset card;
  the set that was read is identified by its digest.

## How the numbers were produced and checked

- Run on a Colab CPU runtime at commit `4c6e8afb47016ad9093b113db63aa996d4410f3c`, with the
  committed `uv.lock`, over the 17,964 released artifacts of the nine runs.
- Integrity gate, passed: every artifact is the payload its evaluation recorded, belongs to this
  study, and keeps its predicted classes after temperature scaling; the instance bitmasks are the
  released set.
- Reproduction gate, passed: from the extraction alone, the new code recomputed the nine released
  paired intervals, the instance blocks of the first approved seed, the seed-mean AURC and the
  seed-mean ECE and Brier score, and every one equalled the released evidence exactly
  ([`evidence/reproduction.json`](evidence/reproduction.json)).
- [`evidence/summary.json`](evidence/summary.json) is the unedited output of
  `driving-risk allseed analyse`. On a second machine, everything in it except AURC was
  recomputed from the downloaded extraction and came out equal; AURC needs the per-image
  confidence histograms, which stayed on the Colab runtime.
- [`evidence/allseed-evidence.json`](evidence/allseed-evidence.json) is derived from the summary by
  `driving-risk allseed evidence`, which copies numbers and recomputes none. Every number on this
  page is traced to [`claims.yaml`](claims.yaml) by the claims validator;
  `tests/contract/test_posthoc_allseed.py` re-derives the evidence and runs the validator on every
  pull request.
- No part of the analysis departed from the plan.

```bash
uv run driving-risk allseed evidence \
  --summary docs/posthoc/allseed-v1/evidence/summary.json \
  --output docs/posthoc/allseed-v1/evidence/allseed-evidence.json
python .agents/skills/auditing-driving-risk-claims/scripts/validate_claims.py \
  --claims docs/posthoc/allseed-v1/claims.yaml --document docs/posthoc/allseed-v1/results.md
```
