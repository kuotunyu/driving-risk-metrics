# P1 smoke rehearsal

Evidence that the end-to-end chain runs on the real frameworks and is
deterministic, plus the measured numbers that size the formal jobs.

Everything below is **synthetic**. No dataset was read, no pretrained weights
were downloaded, and no model was trained to convergence. Nothing here is a
result about any architecture, and nothing here may be cited as one.

- Run ID: `20260901T113432Z-995b6a7`
- Executed: `2026-09-01T11:34:32Z` to `2026-09-01T11:40:37Z`
- Evidence type: `synthetic`

## What was exercised

The real `TorchTrainingBackend` and `TorchEvaluationBackend`, not the fakes the
unit and integration suites inject. That means real model construction, real
optimizer steps, a real checkpoint written and read back, real inference, real
metric-sufficient artifact writing, and real content hashing.

| Stage | Real or faked |
| --- | --- |
| Model construction (`fcn_resnet50`, 19 classes) | Real, `pretrained=False` |
| Optimizer steps at effective batch 16 | Real, SGD with the pinned settings |
| Checkpoint save and restore | Real `torch.save` and `torch.load` |
| Evaluation and per-image artifacts | Real, through `driving-risk evaluate` |
| Content hashing and run record | Real |

`pretrained=False` throughout, so the rehearsal never reaches the network. That
also means the weights are randomly initialized: the predictions are meaningless
by construction, which is the point. A smoke test that produced plausible-looking
scores would invite exactly the misreading this project exists to prevent.

## Cohort

Four synthetic samples at 128 × 256 source geometry, generated from a fixed
seed, laid out in the real protocol directory structure and frozen into a real
manifest. The full protocol is used unchanged, including the 512 × 910 resize
onto the 512 × 1024 canvas, so the geometry path under test is the formal one.

## Determinism

The whole rehearsal was run twice within each of two separate invocations, four
independent runs in total, each in a fresh temporary directory.

- Checkpoint SHA-256 `dbad533fa4922594e17e2ab8cdb9cde40741fc49077b99934c0b7f1aac8534a5`,
  identical in all four runs
- All eight prediction artifact hashes identical between paired runs
- Four of four samples evaluated every time

Identical checkpoint bytes across separate process invocations is the strong
form of this claim: it covers seeding, model initialization, data ordering, the
flip draws, gradient accumulation, and the optimizer update.

## Measured numbers

These replace the previous estimates. They are measured on this desktop
(Intel Core i7-13700, 24 logical processors), CPU only.

| Quantity | Measured |
| --- | --- |
| CPU training throughput | 4.81 s per image at 512 × 1024, forward and backward |
| CPU inference | roughly 1.4 s per image |
| Peak process memory | 10,409 MiB |
| Artifact size | 113 KiB per 128 × 256 image |

The 10.2 GiB peak is worth noting on its own: it is CPU training at micro batch
4 on a 32 GiB machine, so the desktop cannot host two such jobs at once.

## Storage projection for the formal runs

Prediction artifacts are per-pixel at source geometry, so the synthetic figure
must be scaled by the pixel ratio before it says anything about the real cohort.
BDD100K source geometry is 1280 × 720, which is 28× the synthetic area.

| Quantity | Projection |
| --- | --- |
| Per real image | 3.1 MiB |
| Per run, 1,000 locked-validation images | 3.0 GiB |
| Nine formal runs | 27 GiB |

Storing calibrated artifacts alongside uncalibrated ones roughly doubles this to
about 54 GiB. Both fit comfortably in the free space recorded for the artifact
volume, but the number belongs in the plan before nine jobs are launched rather
than after.

## What was not done

- **No local GPU was used.** The shared RTX 4090 was never touched. The GPU
  rehearsal was run on a Colab A100 instead, because the formal runs will be on
  A100 and that is the only number worth having.
- **No CamVid smoke.** The plan permits it only after its public licence and
  source are documented. The only local copy sits under a reference directory
  whose provenance cannot be established, so it is deliberately skipped.
  Synthetic data raises no licensing question at all and covers the same chain.
- **No convergence, no accuracy, no comparison.** Two optimizer steps on random
  weights over four synthetic images. Any number derived from this rehearsal
  describes the plumbing, never a model.

## GPU rehearsal, measured

Run on a Colab A100 at `2026-09-01T14:39Z`, on commit `60741c3`, against the
uploaded copy of the cohort whose four manifest hashes were verified to match
the frozen originals before the measurement started.

| Quantity | Measured |
| --- | --- |
| Hardware | NVIDIA A100-SXM4-40GB, 40,960 MiB |
| Stack | torch 2.13.0+cu130, CUDA 13.0 |
| Per step | **1.309 s** at effective batch 16 |
| Peak GPU memory | **8.7 GiB** allocated, of 40 GiB available |
| One formal run | **10.9 hours** at 30,000 steps |
| Nine formal runs | **98.1 hours** |

One untimed warm-up step preceded the twenty timed steps, so cuDNN autotuning
and allocator warm-up are excluded. No checkpoint and no artifact were written.

**This is three to four times the 2.5-to-3.5-hour figure the plan had carried
since the start, which was never measured.** Two consequences follow directly
and are recorded in the private handoff: the no-resume-checkpointing decision
was explicitly conditional on a run staying under roughly 4.5 hours and must now
be revisited, and 98 hours is a scale that has to be authorized on the measured
number rather than the estimate.

Two observations for whoever tunes this. Peak memory is 8.7 GiB of 40, so the
micro batch of 4 leaves most of the card idle; and the backend sets
`cudnn.deterministic = True` with `benchmark = False`, which buys exact
reproducibility at a real and deliberate cost in speed. Neither is a defect.

## Reproduction

The rehearsal builds its own cohort and configs in a temporary directory and
leaves nothing behind. Its outputs are recorded under the ignored `artifacts/`
boundary at `artifacts/smoke/20260901T113432Z-995b6a7/`, and are not committed.
