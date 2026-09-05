# Dataset card: BDD100K 10K semantic segmentation

What this project evaluates on, where it came from, and what it can and cannot
support. Frozen cohort hashes and the reproduction command live in
[`verification/bdd100k-preflight.md`](verification/bdd100k-preflight.md).

## Identity

| Field | Value |
| --- | --- |
| Name | BDD100K, 10K semantic segmentation subset |
| Version identifier used here | `10k-semantic-v1` |
| Publisher | Berkeley DeepDrive |
| Task | Dense semantic segmentation, 19 Cityscapes-compatible train IDs |
| Ignore index | 255 |
| Source geometry | 1280 × 720 |
| Licence | The BDD100K licence shipped with the distribution |

The dataset is not redistributed by this repository, and no image, mask, or
manifest is committed. Users supply their own licensed copy and point
`BDD100K_ROOT` at it.

## Composition

| Cohort | Assigned | Eligible | Role |
| --- | --- | --- | --- |
| `train` | 6,300 | 6,296 | Gradient updates only |
| `calibration` | 700 | 700 | Scalar temperature fitting only |
| `locked_validation` | 1,000 | 998 | Scored once per checkpoint, never fitted on |

`train` and `calibration` partition the official 7,000-image train split by
deterministic SHA-256 filename order. `locked_validation` is the official
validation split. The official 2,000-image test split is unlabeled and unused.

**Eligibility (amendment A1, 2026-09-02, before any formal training).** A pair
is eligible only if its image and its label share one pixel geometry. Six
official pairs fail this: the JPEG is stored as a 720x1280 portrait beside a
1280x720 label, with no EXIF orientation tag, and no rotation, transpose or
resize of the image aligns it with the label (edge alignment at the level of
deliberately misaligned controls; see the preflight evidence). Such a pair has
no usable supervision and no scorable prediction. The rule is decided from file
headers alone, applied after cohort assignment so that no other membership
moves, and every excluded ID is listed with its reason inside the frozen
manifest and covered by its hash.

- `train`: `3d581db5-2564fb7e`, `52e3fd10-c205dec2`, `781756b0-61e0a182`, `78ac84ba-07bd30c2`
- `locked_validation`: `80a9e37d-e4548ac1`, `9342e334-33d167eb`

Instance annotations cover 6,996 train and 998 validation images; the six
geometry-ineligible IDs are exactly the six that the instance mirror also
lacks. The twelve intersection exclusions and their reasons are listed in the
preflight evidence. Instance area tertiles are learned from the training
intersection only.

## What the labels mean

Nineteen train IDs following the Cityscapes convention, with 255 as ignore.
Metrics are computed at source geometry rather than on the padded 512 × 1024
training canvas, so small instances are not erased by resizing before they are
scored.

Two naming cautions that this project treats as load-bearing:

- `normalized_image_band` means the top, middle, and bottom thirds of the frame.
  It is an image region, **not** distance or depth. Nothing in BDD100K's 2D
  semantic labels supports a distance claim.
- The `drivable_boundary` risk profile is a road and sidewalk false-negative
  proxy, not an exact boundary-confusion metric.

## Known limitations

- **Single dataset, single geography.** BDD100K is US-centric dashcam footage.
  Nothing here supports a claim about model behaviour in other regions.
- **2D only.** No depth, no LiDAR, no 3D geometry. Any spatial statement is about
  image regions.
- **Class imbalance is severe.** In the training intersection, one instance
  category carries 64,751 instances and another carries 59. Per-class and
  per-tertile statistics conditioned on rare categories are thin and must be
  reported with their support.
- **Instance annotations come from a mirror.** The official endpoints no longer
  serve them. The mirror hash is a local hash and is not claimed to match the
  unavailable official archive.
- **Not a safety argument.** Risk-weighted metrics here are a controlled study of
  how metric choice changes model ranking. They are not evidence that any model
  is safe to deploy.

## Intended use

Reproducible comparison of three segmentation architectures under standard and
risk-weighted metrics on one frozen cohort. Not intended for model selection for
deployment, for benchmarking against published leaderboard numbers, or for any
commercial use the BDD100K licence does not grant.

## Regenerating the frozen area tertiles

The classwise area tertiles in `docs/evidence/bdd100k_semseg_v1/area_tertiles.json`
were learned at P1-14 from the training cohort's instance bitmasks, over
whole-instance areas as `instance_areas` counts them, with no semantic-mask
filter. They are regenerated, never edited, with

```
driving-risk data tertiles --manifest <train manifest> --instance-root <bitmasks/train> --output <path>
```

and the regenerated file must be byte-identical to the tracked one. The file P1-14
wrote on Windows carried CRLF line endings (630 bytes, SHA-256
`27330b08bf46929ad060adae02618ab6f612173de8e674fdb476658a41e33799`); the repository
normalises text to LF, so the tracked copy is the same content at 581 bytes,
SHA-256 `5f9365d5b9189b49649e34fc8403f16f4934d630ceb5cf903429007a52997206`, and that
is the hash the extended metrics publish as `tertile_edges_sha256`. Areas are
whole instances while coverage is measured over corroborated footprints; the
extended metrics publish `mean_corroborated_fraction` so that gap is visible.
