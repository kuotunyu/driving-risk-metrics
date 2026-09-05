# Experiment card: BDD100K semantic segmentation under `bdd100k_semseg_v1`

One protocol, three models, three seeds, nine runs, one locked evaluation. This card
records what was fixed before any run started, what each run produced, and every
method change made after the runs began.

## Protocol and cohort identity

| Field | Value |
| --- | --- |
| Protocol | `bdd100k_semseg_v1` |
| Protocol SHA-256 | `b33c842250f6afcc7bd7c1108b29bf84f342dda5bb5420e64d6be48773c4369f` |
| Classes | 19 BDD100K train IDs |
| Critical class IDs | 11 `person`, 12 `rider`, 17 `motorcycle`, 18 `bicycle` |
| Training steps | 30000 |
| Seeds | 17, 42, 73 |
| Environment lock SHA-256 | `34a90c4086d57333ad7dcfcb82783c4a42c08e6a674353d25058dce600a8a632` |

Frozen split manifests, all four hashed before the first run and unchanged since:

| Split | Purpose | Manifest SHA-256 |
| --- | --- | --- |
| `source_train` | the pool the training split was drawn from | `5a2f2ddf4a9076398855c1c79b3b7e90cfff79488afdea5eb11b3785ffa0672b` |
| `train` | training only | `1d621fd5a86d6e7cb35d090fee0e60ae8580904a4ea2ccbb5f98b94bff84ef68` |
| `calibration` | temperature fitting only | `9f7288d24227678174eb596ed59fdc6c6879b8e01ae554191eef72e1a375b097` |
| `locked_validation` | evaluated once, 998 images | `d43bda8f556747f8500bc14269957f2218c52c02917358e5456ec75faa12944f` |

The locked split was never used to select a hyperparameter, a checkpoint, a
temperature, a threshold or a gallery sample. The area tertiles that group instances
by size were learned from `train` alone and frozen to
[`evidence/bdd100k_semseg_v1/area_tertiles.json`](evidence/bdd100k_semseg_v1/area_tertiles.json),
SHA-256 `5f9365d5b9189b49649e34fc8403f16f4934d630ceb5cf903429007a52997206`; the
reproduction of that file from the licensed training data is recorded in
[`verification/tertile-reproduction.md`](verification/tertile-reproduction.md).

## The nine runs

Every run trained to step 30000 on a single A100-SXM4-40GB in Google Colab and
reported `succeeded` for training, uncalibrated evaluation and calibrated
evaluation, with 998 of 998 evaluation artifacts on both passes.

| run_id | model | seed | status | final step | temperature | checkpoint SHA-256 |
| --- | --- | ---: | --- | ---: | --- | --- |
| `segformer_b2-seed-17` | segformer_b2 | 17 | succeeded | 30000 | 1.8739325125709765 | `94904bef3125613b8785b9e20cf65cc80e65704fa8a54db8ebfa9c5e3fc1b698` |
| `segformer_b2-seed-42` | segformer_b2 | 42 | succeeded | 30000 | 1.887354851996863 | `7bc6d3a9a2bdc06ea0ee38660336fae87c068f2be6fe8e4874e10029860dc438` |
| `segformer_b2-seed-73` | segformer_b2 | 73 | succeeded | 30000 | 1.8750506588178484 | `f75181d5c17a36d18926cc06f3fb66a1929de4f540c051603d8d1e32975f7676` |
| `upernet_convnextv2_tiny-seed-17` | upernet_convnextv2_tiny | 17 | succeeded | 30000 | 2.518406969695241 | `8ea366d4648da5bb1cb09585d5c51186ace58169a71c44a16f0517bd9320a027` |
| `upernet_convnextv2_tiny-seed-42` | upernet_convnextv2_tiny | 42 | succeeded | 30000 | 2.5048376587726593 | `f5754121e1cabc7eb00c3d30d08e9ccc24ffd5b1da69638feaab02a5881d5461` |
| `upernet_convnextv2_tiny-seed-73` | upernet_convnextv2_tiny | 73 | succeeded | 30000 | 2.519636098701938 | `7d6730bc303b50e72592b876a3f6d609fe5703ee14db72a655ab34b87f93a7c0` |
| `upernet_dinov2_small-seed-17` | upernet_dinov2_small | 17 | succeeded | 30000 | 2.1378234290930633 | `194b9a388939671eabf12bca9409c95bdd96513d15737dd410acd2d836a6d534` |
| `upernet_dinov2_small-seed-42` | upernet_dinov2_small | 42 | succeeded | 30000 | 2.1442234879672153 | `def1fec93265fbdf018a3c3b6fa81d8a10bf3f5a54147d7eea08e30a962d5c29` |
| `upernet_dinov2_small-seed-73` | upernet_dinov2_small | 73 | succeeded | 30000 | 2.168452384729197 | `1f6ac669714f39d5798037775c95d5469ce5ab338ee32e74b8cd0e3e93ca4585` |

Two readings across runs matter more than any single row. The nine checkpoint
digests are all distinct, which is what three seeds are for and what a cached or
copied checkpoint would have broken. Within each model the three temperatures agree
closely, so the over-confidence each model needs corrected is a property of the
architecture and the protocol rather than an accident of one run.

**Two training commits appear across the nine runs, and this is stated rather than
smoothed over.** Runs 01 to 03 were produced at commit `67cbeee`; runs 04 to 09 at
`a051943`, which added opt-in resume checkpointing and made written artifact bytes
independent of the writing platform. Neither change touches the training path taken
when `--resume-dir` is absent, the protocol hash is identical across all nine, and
the formal index gates on the protocol hash. The set remains comparable; a reader
should nonetheless know that two commits are involved.

## Analysis

The analysis reads the stored per-image prediction artifacts, never the images, and
runs on CPU. It has been executed three times, on different days and different
runtimes:

| commit | date | what changed since the previous execution |
| --- | --- | --- |
| `cc139a0` | 2026-09-04 | first full analysis of the nine runs |
| `ab2954d` | 2026-09-05 | one corroboration rule in `extended.py` |
| `bd47e64` | 2026-09-05 | document shapes: support, per-seed calibration, `by_class`, schemas |

Five of six documents were byte-identical between the first two executions. Between
the second and third, no value changed anywhere; only the shapes did. Both
comparisons, with their hash tables, are in
[`verification/analysis-reproduction.md`](verification/analysis-reproduction.md).

Interval method: a two-stage paired bootstrap over summed confusions, 0.95
confidence, 5000 resamples, seed 20260831. Pairing is over the images of the locked
cohort, which is why a difference between two models can have a tighter interval
than either model's own value.

## Method history: what contact with real data changed

Nothing in this section was planned. Each entry is a defect or a rule that only
appeared when the code met the full cohort, and each is recorded because a method
that changed silently is not a method.

- **Six defects in `extended.py`**, found at `cc139a0` on first contact with the
  real prediction artifacts, and one further rule replaced at `ab2954d`. All are
  described in [`protocol.md`](protocol.md).
- **The corroboration rule was replaced after measuring, not after arguing.** The
  first rule required total agreement between the semantic and instance annotations
  over an instance's pixels. Measured over 60 images and 664 instances, median
  agreement was 0.9865 and the total-agreement rule admitted only 8.9 percent of
  instances, while a footprint rule admitted 98.6 percent. Scoring over the
  corroborated footprint is the published rule, and the fraction it covers is
  published beside every instance block.
- **Calibrated artifacts were being ignored.** `calibrated_artifacts_dir` was
  present in the run index and read by nothing; the metric table only ever used the
  uncalibrated directory. Harmless for mean IoU, whose argmax is
  temperature-invariant, and exactly wrong for calibration error and Brier score.
- **A degenerate selective-risk curve now reports null rather than a number.** A
  model whose pixels all carry one confidence produces a one-point curve, which has
  no span and therefore no area.
- **Written artifact bytes were platform-dependent.** Nine first-party writers
  inherited the platform line ending, so the same document written on Windows and in
  CI differed by hash. Fixed at `a051943` and pinned by a contract test.
- **The paired interval reported half the difference its key named**, and the cohort
  order was taken from the first run while the validator proved only set equality.
  Both were found by the mutation audit and fixed before any claim was written.

## Verification at release

- Eight-stage `verify` at exit 0: private guard, format, lint, types, tests, 100
  percent statement and branch coverage on first-party code, schema contracts,
  documentation links.
- Mutation testing over the pure core, with every surviving mutant either killed or
  documented as equivalent with its measurement or its reading, in
  [`verification/mutation-audit.md`](verification/mutation-audit.md).
- Every published number traced to an artifact by
  [`claims.yaml`](claims.yaml) and enforced by `driving-risk audit-claims`,
  by the `auditing-driving-risk-claims` skill validator over both READMEs, and by
  the ordinary test suite.

## Known limitations of this experiment

- One dataset, one region, one camera configuration.
- Three models under one training recipe. No architecture search, no recipe search.
- One locked evaluation. Every number is a measurement on that cohort, not an
  estimate of population performance.
- Class `train` appears in 7 images of the locked cohort. Its per-class values are
  published with that count attached and should not be read as a model property.
