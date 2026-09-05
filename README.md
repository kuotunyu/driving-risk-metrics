# driving-risk-metrics

**Does a higher mIoU mean a safer segmentation model? On this cohort it does not.**

[繁體中文](README.zh-TW.md)

Three contemporary semantic segmentation models were trained on BDD100K under one
frozen protocol, three seeds each, and evaluated once on a locked 998-image cohort
that was never used for training, checkpoint selection, temperature fitting or
sample selection. The headline metric and the safety metric disagree about whether
the top two models differ, and the disagreement is the result this repository exists
to report.

## The finding

The paired bootstrap interval for the two best models **includes zero on mean IoU**:

> The paired difference in mean IoU between SegFormer-B2 and UperNet-ConvNeXtV2-Tiny is -0.010027276977824351, and its bootstrap interval from -0.02224922437284147 to 0.0023369779504553204 includes zero. <!-- claim: p1.interval.miou.segformer-minus-convnextv2 -->

The same comparison on recall over the vulnerable-road-user classes **excludes zero**:

> The paired difference in critical-class recall between SegFormer-B2 and UperNet-ConvNeXtV2-Tiny is -0.023304517439508565, and its bootstrap interval from -0.03998178645924999 to -0.008046430169669789 excludes zero. <!-- claim: p1.interval.critical-recall.segformer-minus-convnextv2 -->

A reader who ranks by mean IoU alone would conclude the two models are
interchangeable. On the classes that a braking decision depends on, they are not.

Ranking is not the issue here. The order of the three models is the same under all
three metrics, and that is reported as plainly as a reversal would have been:

> Ranking the three models by critical_recall produces the same order as ranking them by miou: no reversal is observed. <!-- claim: p1.ranking.critical-recall.no-reversal -->
> Ranking the three models by pixel_accuracy produces the same order as ranking them by miou: no reversal is observed. <!-- claim: p1.ranking.pixel-accuracy.no-reversal -->

What changes between metrics is not the order but whether the top two can be
separated at all.

## Headline results

Averaged over three seeds on the locked cohort. Numbers appear at full precision
throughout this page; a rounded copy would be a second number for one quantity, and
this project refuses to publish one.

| Model | mean IoU | critical-class recall | pixel accuracy |
| --- | --- | --- | --- |
| UperNet-ConvNeXtV2-Tiny <!-- claim: p1.metrics.convnextv2 --> | 0.6320100232208011 | 0.8105162716623479 | 0.9387763736063249 |
| SegFormer-B2 <!-- claim: p1.metrics.segformer --> | 0.6219827462429768 | 0.7872117542228393 | 0.9385890837416806 |
| UperNet-DINOv2-Small <!-- claim: p1.metrics.dinov2 --> | 0.47424706184502113 | 0.520379600009604 | 0.9141344038041649 |

> Every paired interval is a two-stage paired bootstrap over summed confusions at 0.95 confidence, using 5000 resamples from seed 20260831. <!-- claim: p1.interval.method -->

## Where the pixel metrics hide the failure

Pixel-weighted metrics let one bus outvote fifty pedestrians. This repository also
scores every annotated instance with equal weight, and calls an instance a
**critical miss** when less than half of it is classified correctly. Instances are
grouped into size tertiles learned from the training split alone.

Read that way, the best model on this cohort fails almost completely on the small
end of exactly the classes the study protects:

> UperNet-ConvNeXtV2-Tiny recovers less than half the pixels of 306 of the 462 smallest-tertile person instances. <!-- claim: p1.instances.convnextv2.person-small -->
> UperNet-ConvNeXtV2-Tiny recovers less than half the pixels of every one of the 17 smallest-tertile rider instances. <!-- claim: p1.instances.convnextv2.rider-small -->
> UperNet-ConvNeXtV2-Tiny recovers less than half the pixels of every one of the 14 smallest-tertile motorcycle instances. <!-- claim: p1.instances.convnextv2.motorcycle-small -->

Against cars, the same model on the same instances:

> UperNet-ConvNeXtV2-Tiny recovers less than half the pixels of 822 of the 3749 smallest-tertile car instances. <!-- claim: p1.instances.convnextv2.car-small -->
> Across all classes UperNet-ConvNeXtV2-Tiny misses more than half the pixels of 1399 of the 4514 smallest-tertile instances. <!-- claim: p1.instances.convnextv2.small-overall -->

A model that recovers roughly four out of five small cars and roughly one out of
three small pedestrians has a mean IoU that says none of this.

Instance coverage is measured over the footprint that the semantic and instance
annotations corroborate, not over the raw bitmask, because the two annotations
disagree at object boundaries and scoring over pixels only one of them claims would
attribute an annotation artefact to the model:

> Instance coverage scores 12860 instances over the footprint that the semantic and instance annotations corroborate, a mean corroborated fraction of 0.94643690780947; 115 instances had no corroborated pixel and were excluded. <!-- claim: p1.instances.corroboration -->
> Every one of the 998 locked-cohort ground-truth masks was verified against the frozen manifest before it was scored. <!-- claim: p1.ground-truth.masks -->

## Calibration does not always help

Temperature scaling is fitted on a held-out calibration split and applied to the
locked cohort. It lowered the calibration error of two models and raised it for the
third:

> Temperature scaling lowered the expected calibration error of UperNet-ConvNeXtV2-Tiny on the locked cohort, from 0.004609387187919981 to 0.0032855195799122. <!-- claim: p1.calibration.convnextv2.ece -->
> Temperature scaling lowered the expected calibration error of UperNet-DINOv2-Small on the locked cohort, from 0.005448051902032049 to 0.003985369701553616. <!-- claim: p1.calibration.dinov2.ece -->
> Temperature scaling raised the expected calibration error of SegFormer-B2 on the locked cohort, from 0.0028840449773854925 to 0.0035196866015977167. <!-- claim: p1.calibration.segformer.ece -->

That is not one unlucky seed. Every seed moved the same way, which is why the
per-seed values are published rather than the mean alone:

> Every SegFormer-B2 seed moved the same way after temperature scaling: calibrated 0.003590665043061051, 0.0034857099631801494, 0.0034826847985519496 against uncalibrated 0.0028459279224686924, 0.002904871031420072, 0.0029013359782677135. <!-- claim: p1.calibration.segformer.ece-per-seed -->

A model that was already close to calibrated can be made worse by a correction
fitted elsewhere, and a study that published only the seed mean could not show it.

## Thin classes are labelled, not hidden

> All three models score an IoU of 0.0 on the class train, which carries 109005 labelled pixels in 7 of the cohort's images. <!-- claim: p1.per-class.train -->

Read alone, a zero looks like a model failure. Read beside its support it is a
statement about the cohort. Every per-class row in the generated report carries its
pixel count and its image count, and a class appearing in fewer than 50 images is
marked thin.

Accuracy also varies by where in the frame a road user appears:

> Pixel accuracy in the middle third of the image, where distant road users appear, is 0.9063993962745598 for UperNet-ConvNeXtV2-Tiny, 0.9039825378430191 for SegFormer-B2, 0.8681091984773754 for UperNet-DINOv2-Small. <!-- claim: p1.bands.middle -->

These bands are normalized image rows. They are not depth and not metric distance.

## How every number on this page is checked

Each result sentence above carries a `<!-- claim: ... -->` marker. The claim names an
artifact under [`docs/evidence/bdd100k_semseg_v1/`](docs/evidence/bdd100k_semseg_v1),
a JSON pointer inside it, and the protocol and dataset manifest hashes the artifact
must carry. Two independent checks enforce it:

```bash
uv run --frozen driving-risk audit-claims --claims docs/claims.yaml
uv run --frozen python .agents/skills/auditing-driving-risk-claims/scripts/validate_claims.py \
  --claims docs/claims.yaml --repo-root . --document README.md --document README.zh-TW.md
```

The first proves every registry claim reproduces from its own artifact. The second
reads these documents, traces every marked sentence, and **reports any line that
states a metric and a number without a marker**. A number nobody can trace is the
failure this project exists to prevent, so an untraceable one fails the build rather
than shipping.

The evidence is also self-checking inside the ordinary test run: a change to any
published number, in any tracked artifact, fails the test suite.

Supporting records:

- [`docs/protocol.md`](docs/protocol.md) — the frozen protocol and its revisions.
- [`docs/experiment-card.md`](docs/experiment-card.md) — the nine runs, their hashes and the method history.
- [`docs/model-card.md`](docs/model-card.md) — the three architectures and their permitted use.
- [`docs/dataset-card.md`](docs/dataset-card.md) — BDD100K provenance, licence and the frozen splits.
- [`docs/verification/analysis-reproduction.md`](docs/verification/analysis-reproduction.md) — three independent executions of the analysis, and what agreed.
- [`docs/verification/mutation-audit.md`](docs/verification/mutation-audit.md) — the mutation score of the pure core and every surviving mutant's disposition.

## Reproducing this

```bash
uv sync --frozen --all-groups --extra train
uv run --frozen python -m drivemetrics.dev verify
uv run --frozen driving-risk --help
```

`verify` runs eight stages in a fixed order and stops at the first failure: the
private-file guard, format check, lint, type check, the full test suite, 100 percent
statement and branch coverage on first-party code, schema contracts, and
documentation links. There are no coverage exemptions, no `pragma: no cover`
comments and no omitted first-party paths.

The formal pipeline, in order:

```bash
driving-risk data preflight --config configs/protocols/bdd100k_semseg_v1.yaml --data-root PATH --output PATH
driving-risk data tertiles --manifest PATH --labels-root PATH --instance-root PATH --output PATH
driving-risk train --config configs/run_segformer_b2.yaml --manifest PATH --data-root PATH --seed 17 --output-dir PATH --device cuda
driving-risk calibrate --config configs/protocols/bdd100k_semseg_v1.yaml --manifest PATH --checkpoint PATH --data-root PATH --output-dir PATH --device cuda
driving-risk evaluate --config configs/protocols/bdd100k_semseg_v1.yaml --manifest PATH --checkpoint PATH --data-root PATH --output-dir PATH --device cuda --temperature PATH
driving-risk index --runs-root PATH --config configs/protocols/bdd100k_semseg_v1.yaml --manifest PATH --risk-profile configs/risk_profiles/vru_priority.yaml --output PATH
driving-risk aggregate --index PATH --output-dir PATH
driving-risk gallery --index PATH --output PATH
driving-risk extended-metrics --index PATH --output PATH
driving-risk report --claims docs/claims.yaml --artifacts-dir docs/evidence/bdd100k_semseg_v1 --output-dir site
```

The nine training runs took about 12 hours each on an A100. The analysis runs on
CPU from the stored prediction artifacts, and has been executed three times on
different days and runtimes; what agreed between executions is recorded in
[`docs/verification/analysis-reproduction.md`](docs/verification/analysis-reproduction.md).

BDD100K is not redistributed here, and neither are the checkpoints or the roughly
54 GiB of per-image prediction artifacts. The committed evidence is the analysis
output the claims cite.

## What this does not support

- **Other regions and other cameras.** BDD100K is dashcam footage from a specific
  collection. Nothing here transfers to another sensor, mounting or region without
  new evidence.
- **Depth or distance.** The image bands are normalized image rows. No depth is
  estimated anywhere in this repository.
- **Real-time inference.** No latency, throughput or embedded-deployment claim is
  made or measured.
- **Models outside the approved list.** Only `segformer_b2`,
  `upernet_convnextv2_tiny` and `upernet_dinov2_small` were trained under this
  protocol. Results are about these models trained this way, not about these
  architectures in general.
- **A production safety case.** Instance coverage and risk-weighted cost are
  evaluation tools. They are not a safety argument and not a substitute for one.

## Licence

MIT, see [LICENSE](LICENSE). BDD100K is distributed by its authors under its own
terms and is not redistributed here; see
[`docs/dataset-card.md`](docs/dataset-card.md).
