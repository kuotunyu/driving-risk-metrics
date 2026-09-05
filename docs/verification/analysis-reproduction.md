# Reproducing the analysis of the nine formal runs

The nine training runs are expensive and were executed once. Everything published
about them is produced by re-running the analysis over their stored prediction
artifacts, so the analysis itself has to be shown to be deterministic: the same
inputs, executed on a different day on a different machine, must give the same
document. This records the third such execution and what it establishes.

## The three executions

| commit | date | runtime |
| --- | --- | --- |
| `cc139a0` | 2026-09-04 | Colab CPU |
| `ab2954d` | 2026-09-05 | Colab CPU, new runtime |
| `bd47e64` | 2026-09-05 | Colab CPU, new runtime |

Between `cc139a0` and `ab2954d` only `extended.py` changed, and five of the six
documents were byte-identical. Between `ab2954d` and `bd47e64` the document shapes
changed on purpose: per-class support and names, per-seed calibration, instance
`by_class`, a tertile hash in place of a Colab path, `schema_version` on every
document and a `separability` block. So byte-identity is expected only where a
producer did not change, and the stronger test is that **no value moved**.

## The two hash tables

| document | bytes at `ab2954d` | SHA-256 at `ab2954d` |
| --- | ---: | --- |
| `formal_run_index.json` | 527,851 | `a95b33e2761a5d3fb07a546b07f38df4857297c1e874badf11c8c2a3a1b9ffb1` |
| `gallery-manifest.json` | 13,522 | `b873217f2af9e49beb2b74e787e5b6a09878a369912070388a54ce8099ff3392` |
| `intervals.json` | 2,849 | `c42006b8240c38a7e2704544743c8e732008244de8b86e6cbcc414da77ba3604` |
| `rankings.json` | 912 | `0a3224e2ce3726ea085f9b26c95be55106df06fa41442a321061ab714f696e2f` |
| `metrics.json` | 5,905 | `20f6ad9e4cf065a1301f5effc4770922243dcb819d9b19b5221e154370ae6181` |
| `extended-metrics.json` | 5,002 | `ceac6a9af88799a6ed2ae251f0475b97501c1a51ce78c831d08b79c44179939c` |

| document | bytes at `bd47e64` | SHA-256 at `bd47e64` |
| --- | ---: | --- |
| `formal_run_index.json` | 527,851 | `a95b33e2761a5d3fb07a546b07f38df4857297c1e874badf11c8c2a3a1b9ffb1` |
| `gallery-manifest.json` | 13,522 | `b873217f2af9e49beb2b74e787e5b6a09878a369912070388a54ce8099ff3392` |
| `intervals.json` | 2,898 | `8beeb35dbf61549456d44c60f4bc6f20ddc530876ab597620f0d1ffc142b0c6d` |
| `rankings.json` | 3,224 | `b42d21e79eb8eed5c9172ded83c16b00aee5e2862be0cf1a71e9b6f269d640ff` |
| `metrics.json` | 9,373 | `4f47011d8df98b8435ccf1e954d02e8cfa5aa7f0e03afd8676f47160c072ddb3` |
| `extended-metrics.json` | 21,836 | `ecf4e46ff6b31fd7c96926fdde9bcd935d106d580ee3764ecc3751fcb069a849` |

Each `bd47e64` hash was checked against the table the notebook printed for itself
before any of these files was read, and again after they were copied into this
repository.

## The comparison, and its result

Both sets were flattened to `{JSON pointer: leaf value}` and compared. The result
across all six documents:

| document | bytes | changed values | added leaves | leaves that moved |
| --- | --- | ---: | ---: | ---: |
| `formal_run_index.json` | identical | 0 | 0 | 0 |
| `gallery-manifest.json` | identical | 0 | 0 | 0 |
| `intervals.json` | differ | 0 | 1 | 0 |
| `rankings.json` | differ | 0 | 55 | 0 |
| `metrics.json` | differ | 0 | 208 | 114 |
| `extended-metrics.json` | differ | 0 | 311 | 18 |

**No value changed anywhere.** The only leaf removed and not moved is
`instances[*].tertile_edges_from`, which held an absolute Colab path and is
replaced by `tertile_edges_sha256`; a path cannot be verified and a hash can.

The leaves counted as moved are `per_class/<model>/{iou,recall}`, now under
`per_class/by_model/<model>`, and `normalized_image_bands/<model>`, now under
`normalized_image_bands/by_model/<model>`. Each was compared at its new pointer
against its old one: 6 lists of 19 per-class values and 3 band blocks, all
identical value for value. The pooled instance figures — `instance_count`,
`excluded_without_semantic_pixels`, `mean_corroborated_fraction` and `by_tertile`
— are unchanged for all three models after `by_class` was added beside them.

## One thing this does not establish

`formal_run_index.json` is byte-identical for a weaker reason than the others: the
notebook creates the index only when it does not already exist on Drive, and it
did exist from the previous execution. It is the same file, not an independent
recomputation, and it is recorded that way here rather than counted as evidence.
The byte-identity claim rests on `gallery-manifest.json`, which was recomputed,
and on the unchanged values everywhere else.

## What follows from this

- The six documents under `docs/evidence/bdd100k_semseg_v1/` are the `bd47e64`
  set, and every claim in `docs/claims.yaml` cites one of the five that carry a
  `protocol_hash` and a `dataset_manifest_hash`. The run index carries neither, so
  no claim may cite it; `audit-claims` would report a mismatch if one did.
- The ground truth did not change between executions and the figures that depend
  only on it did not either: 998 masks verified, 12,860 instances scored, 115
  excluded, a mean corroborated fraction of 0.94643690780947, and tertile edges at
  `5f9365d5…`.
