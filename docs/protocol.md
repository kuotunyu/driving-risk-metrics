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
