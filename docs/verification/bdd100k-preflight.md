# BDD100K preflight and frozen cohorts

Evidence that the four formal cohorts were frozen deterministically from a
licensed local copy of BDD100K, and that the freeze reproduces exactly.

Nothing here contains an absolute path, a sample listing, or any dataset
content. The manifests themselves are generated locally and are never committed.

- Preflight run: `2026-09-01T10:58:40Z` to `2026-09-01T11:04:02Z`
- Reproducibility run: `2026-09-01T11:04Z` to `2026-09-01T11:09Z`
- Amendment A1 (ineligible pairs) freeze: `2026-09-02T02:15Z` to `2026-09-02T02:19Z`,
  reproduced byte-identically in a second directory the same hour; see the
  amendment section below
- Protocol: `configs/protocols/bdd100k_semseg_v1.yaml`
- Protocol SHA-256: `b33c842250f6afcc7bd7c1108b29bf84f342dda5bb5420e64d6be48773c4369f`
- Superseded protocol SHA-256: `5a386b2b531481ca23f20b12f780d092a33a1d0105d5d34ace88ac2089a019cf`, which named three older
  architectures. The cohorts below are unchanged by that revision: they are
  frozen from the dataset and do not depend on which models are trained.

## Dataset and licence

| Field | Value |
| --- | --- |
| Dataset | BDD100K, 10K semantic segmentation subset |
| Version identifier | `10k-semantic-v1` |
| Source | Official BDD100K distribution, `https://bdd-data.berkeley.edu/` |
| Images archive SHA-256 | `fd95e3ba04afeb89f724e080ea738185decaefe0250471a3a340c19f1f79a118` |
| Semantic maps archive SHA-256 | `d6642e9efeeb30b4eac351a06f83753b87a6f8bd4def1baf940908159c322efe` |
| Licence | The BDD100K licence shipped with the distribution governs use |

The licence was accepted by a human before any download. No licence acceptance,
account creation, or credential handling is automated anywhere in this
repository, and no credential is stored in it.

Instance segmentation annotations for the same 10K image cohort are no longer
served by the official endpoints. They were obtained from a third-party mirror
with explicit human authorization, and the mirror hash is recorded as a local
mirror hash only. It is not claimed to be bit-identical to the unavailable
official archive, and the third-party page's licence labelling confers no
additional permission beyond the shipped BDD100K licence.

## Cohort counts

Every count is asserted by the preflight, which refuses to write anything unless
all of them hold.

| Cohort | Assigned | Eligible | Derivation |
| --- | --- | --- | --- |
| `source_train` | 7,000 | 6,996 | Official train split, paired image and train-ID mask |
| `train` | 6,300 | 6,296 | Deterministic SHA-256 filename order over all 7,000 source IDs |
| `calibration` | 700 | 700 | The remainder of that same ordering |
| `locked_validation` | 1,000 | 998 | Official validation split, scored once and never fitted on |

Assigned counts are the official split sizes and the sizes the deterministic
partition is computed over. Eligible counts subtract the pairs recorded as
ineligible by amendment A1 below; consumers read only the eligible list.

Unlabeled official test metadata is 2,000 images and is not part of any cohort.

Contamination checks, all verified after the freeze: `train` and `calibration`
are disjoint, their union is exactly the 7,000 source IDs, and no
`locked_validation` ID appears in any other cohort.

## Two kinds of hash

A cohort has two distinct fingerprints, and conflating them reads as dataset
drift when nothing has drifted. Both are recorded so that either can be checked.

- **Manifest hash** binds membership, relative paths and the interleaved image
  and label file hashes. It changes if any file byte changes.
- **Membership hash** binds only the split name and the ordered sample IDs. It is
  the narrower, longer-lived fact, computed by
  `drivemetrics.data.splits.cohort_membership_sha256`.

| Cohort | Manifest SHA-256 | Membership SHA-256 |
| --- | --- | --- |
| `source_train` | `5a2f2ddf4a9076398855c1c79b3b7e90cfff79488afdea5eb11b3785ffa0672b` | `6f5cfb413a4a0c55fa744f122db19eb5ea572e0b8c371685892adec0544f4ace` |
| `train` | `1d621fd5a86d6e7cb35d090fee0e60ae8580904a4ea2ccbb5f98b94bff84ef68` | `ec099f7ada79d511f457a4d12bf66a259c11056b109267a4bdb25f29d66e642f` |
| `calibration` | `9f7288d24227678174eb596ed59fdc6c6879b8e01ae554191eef72e1a375b097` | `5e6615a353216f7d7ae1efff89824841ec72594db404bf60d49e3c50f9e4d2fc` |
| `locked_validation` | `d43bda8f556747f8500bc14269957f2218c52c02917358e5456ec75faa12944f` | `b7329e63da35d85168cd777b2e6cd5730fe584a2111ebf02594a35a9253e6ecf` |

These are the amendment A1 values, current since 2026-09-02. The `calibration`
membership hash is unchanged from the original freeze because none of its
members was ineligible; every manifest hash changed because the manifest
document gained the two ineligibility fields. The superseded values are kept
in the amendment section so that either freeze can be recognised.

## Reproducibility

The preflight was run twice into different output directories. All four manifest
files are byte-identical between the two runs, and all four manifest hashes
agree. The command refuses to overwrite an existing frozen manifest, which is
why a second directory is required rather than a rerun in place.

```bash
export BDD100K_ROOT=/path/to/your/licensed/bdd100k/raw

uv run driving-risk data preflight \
  --config configs/protocols/bdd100k_semseg_v1.yaml \
  --data-root "$BDD100K_ROOT" \
  --output artifacts/manifests/bdd100k_semseg_v1
```

`$BDD100K_ROOT` must contain `images/10k/{train,val}` and
`labels/sem_seg/masks/{train,val}`.

## Fail-closed rehearsal

Before the real freeze, the preflight was pointed at a deliberately broken copy
of `tests/fixtures/bdd100k_tiny_manifest_input.json` with one validation label
removed. It exited 1 with `missing label for sample IDs: ['sample-a']`, naming
the missing pair, and wrote no manifest.

The assertion is on the message, not merely on a non-zero exit. A rehearsal
pointed at a path that does not exist also exits non-zero, so an exit-code-only
check would pass while proving nothing about pair detection.

## Amendment A1: ineligible pairs (2026-09-02)

Discovered by the first formal training run, on its first micro batch, through
the loader guard `image and mask spatial shapes must match`. No formal artifact
existed yet, which is the only moment a frozen cohort may be amended.

**Observed.** A header scan of all 8,000 official pairs found six whose JPEG is
stored as a 720x1280 portrait beside a 1280x720 label. None carries an EXIF
orientation tag, so `exif_transpose` is a no-op. To test whether any rigid
transform aligns them, an edge-alignment score (mean image gradient on label
boundaries divided by mean gradient elsewhere) was computed for eight candidate
transforms of each of the four training images: both 90-degree rotations,
transpose, transverse, and a non-uniform resize with its three flips. Over 30
ordinary pairs the score has mean 3.38 and median 2.70; the same pairs with a
deliberately mirrored label score mean 1.43. The best candidate for each
defective pair scored between 1.48 and 1.69, and the worst 0.40, that is, at
the level of the misaligned controls. The labels belong to a different view.

**Rule.** A pair is ineligible when its image and label pixel geometry differ.
Decided from file headers only. Applied by the preflight after the deterministic
SHA-256 assignment over all 7,000 source IDs, so no other membership moves.
Recorded per cohort inside the manifest as `ineligible_sample_ids` and
`ineligibility_reasons`, both covered by the manifest hash.

| Cohort | Ineligible ID | Reason |
| --- | --- | --- |
| `train` | `3d581db5-2564fb7e` | image 720x1280 but label 1280x720 |
| `train` | `52e3fd10-c205dec2` | image 720x1280 but label 1280x720 |
| `train` | `781756b0-61e0a182` | image 720x1280 but label 1280x720 |
| `train` | `78ac84ba-07bd30c2` | image 720x1280 but label 1280x720 |
| `locked_validation` | `80a9e37d-e4548ac1` | image 720x1280 but label 1280x720 |
| `locked_validation` | `9342e334-33d167eb` | image 720x1280 but label 1280x720 |

These six are exactly the six IDs the instance-annotation mirror lacks (see the
intersection below), so the intersection, its counts and the area tertiles are
unchanged by this amendment.

**Superseded freeze (2026-09-01), kept for recognition only.** Manifest hashes
`87bb7314…`, `e0109c1b…`, `79b391b1…`, `20ca13a1…` and membership hashes
`2d198a2d…`, `7c812b14…`, `5e6615a3…`, `dfe79904…` for `source_train`, `train`,
`calibration`, `locked_validation`. Any run record or index that names one of
those hashes predates the amendment and is not a formal result.

## Semantic and instance intersection

Instance annotations come from a mirror whose sample set differs slightly from
the current official image and semantic cohort, so the intersection is audited
rather than assumed. Every exclusion is recorded with its reason.

| Split | Semantic | Instance | Retained | Dropped |
| --- | --- | --- | --- | --- |
| train | 7,000 | 7,000 | 6,996 | 8 |
| val | 1,000 | 1,000 | 998 | 4 |

Excluded sample IDs, by reason:

- train, missing instance annotation: `3d581db5-2564fb7e`, `52e3fd10-c205dec2`,
  `781756b0-61e0a182`, `78ac84ba-07bd30c2`
- train, missing semantic annotation: `fee92217-63b3f87f`, `ff1e4d6d-f4d85cfd`,
  `ff3d3536-04986e25`, `ff3da814-c3463a43`
- val, missing instance annotation: `80a9e37d-e4548ac1`, `9342e334-33d167eb`
- val, missing semantic annotation: `ff55861e-a06b953c`, `ff7b98c7-3cb964ac`

The intersection affects only instance-conditioned metrics. Semantic metrics use
the full frozen cohorts above.

## Area tertiles

Instance area tertiles are learned from the training intersection only. Four of
the 6,300 frozen `train` IDs lack instance annotations, leaving 6,296 eligible
images and 80,249 instances. The calibration and locked-validation cohorts are
never read: a tertile edge learned from either would leak held-out structure
into how every instance in the study is bucketed. Both exclusions are asserted
in the freezing step rather than assumed.

Areas are counted at source geometry, because that is where every metric in this
project is computed and a resized area would rank instances differently.

| Instance category | Instances | Small ≤ | Medium ≤ |
| --- | --- | --- | --- |
| 1 | 8,946 | 349 | 987 |
| 2 | 419 | 409 | 1,536 |
| 3 | 64,751 | 390 | 2,110 |
| 4 | 3,469 | 1,024 | 5,821 |
| 5 | 1,494 | 886 | 6,017 |
| 6 | 59 | 2,079 | 8,100 |
| 7 | 368 | 557 | 2,062 |
| 8 | 743 | 400 | 1,437 |

Edges are in source pixels. Category 6 carries only 59 instances across the
whole training intersection, so any per-tertile statistic conditioned on it will
be thin; that is a property of the dataset and must be reported alongside any
such number rather than smoothed over.

## What this evidence does not establish

The cohorts are frozen and reproducible. No model has been trained, no
checkpoint exists, and no score has been computed. `docs/claims.yaml` is still
empty. Any number presented as a result of this project before those exist would
be fabricated.
