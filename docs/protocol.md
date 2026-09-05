# Experiment protocol

The frozen protocol every formal run must satisfy. The machine-readable source
of truth is [`../configs/protocols/bdd100k_semseg_v1.yaml`](../configs/protocols/bdd100k_semseg_v1.yaml);
this page explains what each constant is for and which choices are not
negotiable once a run has started.

- Schema: `bdd100k-semseg-protocol/v1`
- Protocol SHA-256: `b33c842250f6afcc7bd7c1108b29bf84f342dda5bb5420e64d6be48773c4369f`

The hash is over the validated semantic content, not the file's formatting, so
reindenting the YAML does not change it. Every run record carries this hash, and
a checkpoint whose metadata declares a different one is refused at evaluation.

## Why the protocol is hashed at all

The question this project asks is whether the mIoU ranking of three
architectures survives being re-scored under risk-weighted, calibration, and
spatial-exposure metrics. That question is only answerable if every model met
identical conditions. Freezing the conditions in a hashed file makes "identical"
checkable rather than asserted.

Changing any constant below is a new protocol version with a new hash. Results
computed under different hashes are not comparable and must never be pooled.

## Input geometry

| Constant | Value |
| --- | --- |
| Resize | 512 × 910 |
| Pad to | 512 × 1024 |
| Image pad value, after normalization | 0.0 |
| Mask pad value | 255 (ignore) |

Training runs on the padded canvas. **Metrics are computed at source geometry**,
by mapping predictions back before scoring. Scoring on the padded canvas would
let padding and resampling change per-class results, and would shrink small
instances before the metric that exists to protect them ever sees them.

## Eligibility

A pair enters a cohort by assignment and is used only if it is eligible: its
image and its label must share one pixel geometry. Ineligibility is decided from
file headers, never from labels, metrics or models; it is applied after the
deterministic assignment so that no other sample moves; and every ineligible ID
is recorded with its reason inside the frozen manifest and covered by its hash.
Under `bdd100k_semseg_v1` six official pairs are ineligible, four assigned to
`train` and two to `locked_validation`; the dataset card names them. Eligible
counts are therefore 6,296 / 700 / 998, and the locked denominator is 998.

## Training

| Constant | Value |
| --- | --- |
| Steps | 30,000 |
| Warmup steps | 1,000 |
| Effective batch size | 16 |
| Horizontal flip probability | 0.5 |
| Learning-rate decay | Polynomial, power 0.9 |
| Checkpoint selection | Final step only |
| Seeds | 17, 42, 73 |

**Final-step-only checkpointing is a selection rule, not a storage decision.**
Choosing among checkpoints requires a criterion, any criterion is computed on
some cohort, and the only untouched cohort is the locked one. Removing the
choice removes the leak.

| Model | Optimizer | Learning rate | Other |
| --- | --- | --- | --- |
| `segformer_b2` | AdamW | 6e-5 | weight decay 0.01 |
| `upernet_convnextv2_tiny` | AdamW | 1e-4 | weight decay 0.05 |
| `upernet_dinov2_small` | AdamW | 1e-4 | weight decay 0.05 |

Nine formal runs: three architectures times three seeds. Every reported number
is a mean over seeds with an interval, never a single seed.

The three are contemporary and each carries a different pretraining paradigm,
because pretraining strongly shapes how confident a model is and half these
metrics measure confidence. `segformer_b2` is supervised.
`upernet_convnextv2_tiny` is pretrained by fully convolutional masked
autoencoding. `upernet_dinov2_small` uses a self-supervised foundation-model
backbone. The two UPerNet variants deliberately share a decoder, which holds
the decoder fixed and isolates the backbone, while SegFormer varies both.

Every backbone starts from classification or self-supervised weights only,
never from a checkpoint already trained for segmentation. One model starting
from segmentation supervision would carry a head start the others never had,
and every ranking below would be measuring that instead.

A mask-classification model such as Mask2Former is deliberately excluded. It
can be post-processed into per-pixel probabilities, but this protocol
calibrates on logits, and probabilities entering that stage would be silently
miscalibrated rather than refused.

## Calibration

Scalar temperature, fitted by multiclass negative log likelihood on the frozen
700-image calibration cohort, after training and before evaluation. Temperature
is never fitted on `train` and never on `locked_validation`.

## Statistics

| Constant | Value |
| --- | --- |
| Method | Two-stage paired bootstrap |
| Resamples | 5,000 |
| Seed | 20260831 |
| Confidence | 0.95 |

Two-stage because two things vary independently: which images are in the cohort,
and which seed a model happened to draw. The common image axis is resampled once
and shared across every run, so paired differences stay paired; seeds are then
resampled within each model.

The paired estimand is the DIFFERENCE of model means, oriented left minus right
as the interval key names it. The bootstrap signs the right model's runs
negative and sums the two group means; averaging them would report half the
difference, which is what this project published until P1-17 and is recorded in
`verification/mutation-audit.md`. The image axis is sorted and the runs are
ordered by the approved model and seed lists, so a published interval cannot
depend on the order an index happened to list them in.

Per-class values are published with the class names, the pixel support and the
number of images each class appears in, all read from the same summed confusions
that produce the scores; the nine runs must agree on that support exactly, because
they scored one ground truth, and a run that does not is another study. A class
whose denominator is zero is published as `null`, never as `0.0`. Calibration is
published per seed beside its mean. The ranking document carries, for every pair
and every metric, whether the paired interval excludes zero — separability is a
different question from order, and on this cohort the two answers differ.

## Ground-truth metrics

Per-band pixel accuracy and instance coverage are the only published numbers that
open the label files again after evaluation, so they carry rules the
confusion-derived metrics never need.

**Predictions are scattered, never reshaped.** An artifact holds one prediction
per NON-IGNORED pixel, in row-major order of the source mask, because that is
exactly the set of pixels the evaluation scored. Reshaping that flat array onto
the image grid would misalign every pixel after the first ignored one and would
still produce a plausible-looking number.

**Ignored pixels appear in no denominator.** No model was asked about them, and
counting them would publish a model as wrong about pixels it was never shown.

**The masks are located through the frozen manifest**, by the relative paths the
runs themselves read, and each file is checked against the digest the manifest
recorded. A directory that merely resembles a BDD100K tree cannot stand in for
the cohort, and a mask edited after the runs read it cannot pass unnoticed.

**Instance categories are translated before anything is compared.** BDD100K
numbers its instance categories 1-8 while the semantic masks carry the nineteen
Cityscapes train IDs. The correspondence is written as names and the IDs are
derived from the semantic class list, so a change to that list breaks the
translation loudly instead of silently remapping every published instance.

**An instance is scored over its corroborated footprint**: the pixels where the
instance bitmask and the semantic mask both say it is that class. The two are
separate rasterizations of the same image and do not agree at the boundary.
Measured over sixty locked-validation images and 664 instances, they agree on a
median 98.65% of an instance's pixels — and on ALL of an instance's pixels for
only 8.9% of instances. A rule demanding total agreement therefore discards nine
instances in ten and keeps a sample selected by which objects the two rasterizers
happened to round the same way, which is a property of the annotations and not of
any model. Boundary disagreement narrows the footprint instead; only an instance
with no corroborated pixel at all is excluded, and those are counted.

The footprint is smaller than the whole instance — a measured median of 98.7% and
mean of 94.8% of its non-ignored pixels — while the frozen tertile edges were
learned over whole instances. That gap is far below the spacing of the edges, but
it is a real systematic difference, so `mean_corroborated_fraction` is published
beside the counts rather than left for a reader to discover.

These rules were written after the first implementation of these blocks met real
data. Its test fixture had no ignored pixels, a four-wide class space and
single-byte annotation IDs, so it agreed with a wrong implementation on every one
of them; and the corroboration rule that replaced the first round of fixes was
itself replaced once measurement showed what it was really counting.

## Rules that outlive the constants

1. **The locked cohort is spent the moment it influences a decision.** Not only
   training — choosing a checkpoint, a threshold, which seed to report, whether
   to rerun, or when to stop. If a locked number touched any of those, the study
   needs a new protocol version and a new cohort.
2. **Provenance is measured, never typed.** Commit and lock hash are computed;
   hardware is queried on the machine that runs the job. A plausible GPU name
   recalled from memory is a synthetic value in a field the whole report rests on.
3. **A ranking reversal is an observation, not a success criterion.** If the
   ranking is stable, that is reported just as plainly. The report must state
   which occurred.
4. **Every published number carries its evidence type** and traces to an artifact
   hash that can be recomputed.
