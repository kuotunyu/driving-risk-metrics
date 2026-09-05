# Reproducing the frozen area tertiles

The extended metrics place every instance in a size tertile using edges that were
learned from the training cohort at P1-14 and frozen to `area_tertiles.json`. That
file was cited by the published evidence but could not be regenerated: the script
that produced it was not kept, and `learn_area_tertiles` was called from nowhere
but tests. A cited artifact that cannot be reproduced is a number without a
method, so `driving-risk data tertiles` was added and required to reproduce the
frozen file exactly.

## The question the reproduction had to settle

Two definitions of an instance's area were possible and the frozen file did not
say which it used: the instance's whole bitmask area, or only its pixels that the
semantic mask does not ignore. The two give different edges. The reproduction was
run under one definition and compared; a mismatch would have meant trying the
other, and a second mismatch would have stopped the work with both outputs
recorded rather than a third guess.

## What was run, 2026-09-05, at `b2c38ed`

```
driving-risk data tertiles \
  --manifest artifacts/manifests/bdd100k_semseg_v1/train.json \
  --instance-root ../data/BDD100K/raw/labels/ins_seg/bitmasks/train \
  --output <scratch>/area_tertiles.regenerated.json
```

```
{"command": "data tertiles", "eligible_images": 6296, "missing_bitmasks": [],
 "total_instances": 80249}
```

The producer counts each instance's area as `instance_areas` does: every pixel
carrying the instance's annotation ID, with no semantic-mask filter — the
whole-instance definition.

## The comparison

| file | bytes | SHA-256 |
| --- | --- | --- |
| regenerated | 581 | `5f9365d5b9189b49649e34fc8403f16f4934d630ceb5cf903429007a52997206` |
| frozen at P1-14, as written (CRLF) | 630 | `27330b08bf46929ad060adae02618ab6f612173de8e674fdb476658a41e33799` |
| frozen at P1-14, as the repository tracks it (LF) | 581 | `5f9365d5b9189b49649e34fc8403f16f4934d630ceb5cf903429007a52997206` |

- Content identical (parsed JSON equal): **yes**.
- Bytes identical to the frozen file as written: no — line endings only.
- Bytes identical to the frozen file as tracked: **yes**.

The P1-14 file was written on Windows without a newline pin and carries CRLF
endings; `.gitattributes` (`* text=auto eol=lf`) normalises every tracked text
file to LF. The regenerated file matches the tracked bytes exactly on the first
attempt, under the whole-instance definition. The second definition was therefore
never needed, and the frozen edges are confirmed to have been learned over whole
instances: 6,296 eligible images, 80,249 instances, per-category counts and edges
as the file states them.

## What follows from this

- `docs/evidence/bdd100k_semseg_v1/area_tertiles.json` is the regenerated file.
  Its SHA-256, `5f9365d5…`, is the value the extended metrics publish as
  `tertile_edges_sha256`.
- Coverage is measured over each instance's corroborated footprint — the pixels
  both annotations agree carry its class — while the edges were learned over whole
  instances. Measured over the locked cohort the footprint is on average 94.6% of
  the instance (`mean_corroborated_fraction`); the gap is far below the spacing of
  the edges, and it is published beside every instance block rather than left for
  a reader to discover.
- The command is part of the verified CLI (`tests/unit/data/test_tertiles.py`,
  `tests/unit/cli/test_app.py`) and refuses to overwrite an existing file; frozen
  means frozen.
