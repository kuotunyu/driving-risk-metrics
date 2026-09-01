# BDD100K preflight and frozen cohorts

Evidence that the four formal cohorts were frozen deterministically from a
licensed local copy of BDD100K, and that the freeze reproduces exactly.

Nothing here contains an absolute path, a sample listing, or any dataset
content. The manifests themselves are generated locally and are never committed.

- Preflight run: `2026-09-01T10:58:40Z` to `2026-09-01T11:04:02Z`
- Reproducibility run: `2026-09-01T11:04Z` to `2026-09-01T11:09Z`
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

| Cohort | Count | Derivation |
| --- | --- | --- |
| `source_train` | 7,000 | Official train split, paired image and train-ID mask |
| `train` | 6,300 | Deterministic SHA-256 filename order over `source_train` |
| `calibration` | 700 | The remainder of that same ordering |
| `locked_validation` | 1,000 | Official validation split, scored once and never fitted on |

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
| `source_train` | `87bb73143efaf157a6706b9b7c7dea7ced7464c6f67e2940cbd13a2e15dcf1c5` | `2d198a2dcf9742e6105744a887cddb034274c3e0a17a2fe9a6eeaca15b92164a` |
| `train` | `e0109c1ba00dcf8997b1ec039e7faee31550dac480731dc25dc9652be744fb95` | `7c812b14f9ea576a8087bb6208816d47d545f35301f6ae5c66117d11f41ac746` |
| `calibration` | `79b391b1944a55e29e48e8d3808b3c4353c8c618e53ea69e7fb0cb5114304a81` | `5e6615a353216f7d7ae1efff89824841ec72594db404bf60d49e3c50f9e4d2fc` |
| `locked_validation` | `20ca13a1bacb7103fdc903cd52ed3f12c433e5cfae8254398522429802858e60` | `dfe79904d5853189dff15fb7ddca0c06c0bf342b52ae71366f187edd913546ee` |

The `source_train` and `locked_validation` manifest hashes, and the `train` and
`calibration` membership hashes, all reproduce the values recorded during the
earlier read-only verification of the same local copy. The dataset has not
drifted and the deterministic split has not moved.

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
