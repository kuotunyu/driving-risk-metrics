---
name: running-locked-segmentation-evals
description: Use when running, resuming, or reporting a formal BDD100K semantic segmentation evaluation on the locked validation cohort - freezing cohort manifests, launching one of the nine approved training jobs, scoring a checkpoint, or preparing published numbers from those runs. Also use when asked to reproduce or summarize this project's segmentation scores. Not for generic image classification, generic PyTorch training loops, or datasets outside the frozen protocol.
---

# Running locked segmentation evaluations

## Overview

This project's results are only worth anything because the evaluation cohort is
frozen and untouched. Every rule below protects one of two things: the locked
cohort stays unseen until the moment it is scored, and every published number
traces to a hash somebody can recompute.

The failure mode is not carelessness. It is a careful run in which one value
that could not be measured got filled in from memory, or one record that nobody
would read today never got written. Both produce a result that looks correct and
cannot be defended later.

## The cohorts, and what may touch them

The protocol freezes four cohorts. Confusing them is unrecoverable, because the
locked cohort's value is entirely in never having been fit on.

| Cohort | Size | May be used for | Must never be used for |
| --- | --- | --- | --- |
| `source_train` | 7,000 assigned, 6,996 eligible | Deriving the train/calibration split | Direct training |
| `train` | 6,300 assigned, 6,296 eligible | Gradient updates | Calibration, any selection |
| `calibration` | 700 | Fitting the scalar temperature | Gradient updates, reporting |
| `locked_validation` | 1,000 assigned, 998 eligible | Scoring a finished checkpoint, once | Anything that changes a run |

"Anything that changes a run" is broader than it sounds. It includes choosing a
checkpoint, choosing a threshold, choosing which seed to report, deciding
whether to rerun a job, and deciding when to stop. If a locked-cohort number
influenced a decision, the cohort is spent and the study needs a new protocol
version and a new cohort.

## The sequence

Each phase ends in a decision, not in the next command. Run the check, compare
it against what the private handoff records, and stop if they disagree. A drift
that is investigated costs an hour; a drift that is carried forward costs every
number downstream.

**1. Gate.** Confirm a clean tree, then run the eight-stage verification:

```bash
uv run --frozen python -m drivemetrics.dev verify
```

Stop on any non-zero exit. **Decide:** does HEAD match what the handoff records?

**2. Freeze the cohorts.** `driving-risk data preflight` writes four manifests
and refuses to overwrite an existing one, because a frozen cohort is evidence
and replacing it silently detaches every record that cited its hash. Run it
twice into different output directories and compare the four manifest hashes
pairwise by name, not as a set - a set comparison passes even if two cohorts
swap contents.

**Decide:** do all four hashes match the recorded values? If not, the dataset
drifted. Stop; nothing downstream is meaningful.

**3. Record provenance by measuring it.** See "Provenance is measured" below.

**4. Train.** One job per (architecture, seed) pair, seeds 17, 42 and 73 only,
against `train.json`. The protocol writes one final-step checkpoint; that is the
selection rule, not a storage limitation.

**Decide:** did the run record write `status: succeeded` with a
`final_checkpoint` artifact hash?

**5. Score.** `driving-risk evaluate` against `locked_validation.json`, once per
checkpoint. It re-hashes every file against the manifest before it runs, so a
mismatch stops the job rather than producing numbers for a different cohort.

**6. Index the set.** Nine succeeded runs are not a formal set until
`driving-risk index` has built the immutable run index from them:

```bash
uv run --frozen driving-risk index --runs-root <runs> \n  --config configs/protocols/bdd100k_semseg_v1.yaml \n  --manifest artifacts/manifests/bdd100k_semseg_v1/locked_validation.json \n  --risk-profile configs/risk_profiles/vru_priority.yaml \n  --output <runs>/formal_run_index.json
```

It reads the four records every `<model>/seed-<seed>/` directory must hold,
re-checks every binding (status, protocol hash, seed, the checkpoint hash against
the temperature artifact, the locked manifest against both evaluations, identical
image sets across the two evaluations) and refuses to overwrite. `aggregate`
consumes the index, never the directories, so a run the index rejected cannot
reach a published number by another road.

**Decide:** did the index accept all nine (model, seed) pairs? A missing pair or a
rejected binding stops the study here, before any statistic exists to be tempted by.

**7. Validate before trusting anything.** See "Run the validator" below.

**8. Publish.** Only claims the audit can reproduce reach the report.

## Provenance is measured, never typed

`DRIVEMETRICS_RUN_PROVENANCE` supplies the commit, lock hash and hardware that
every run record carries. The commit and lock hash are computed. **The hardware
string is the one field nothing can check for you**, which makes it the field
most likely to be filled in from memory - and a plausible GPU name typed on the
wrong machine is a synthetic value sitting in a provenance record that the whole
report rests on.

Query it on the machine that will run the job, in the same session:

```bash
python - <<'PY'
import json, subprocess, hashlib, pathlib
gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True, check=True,
).stdout.strip()
print(json.dumps({
    "commit": subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip(),
    "lock_sha256": hashlib.sha256(pathlib.Path("uv.lock").read_bytes()).hexdigest(),
    "hardware": {"gpu": gpu, "runtime": "colab"},
}, separators=(",", ":")))
PY
```

If `nvidia-smi` is not present, the job is not on a GPU and the hardware must say
so. Never carry a provenance value from one machine to another.

## Write to the handoff at four moments

The private handoff lives outside the repository and is the only record that
survives a lost session. A nine-job matrix runs for roughly a day of paid GPU
time; started without a record of what was launched and where its outputs land,
it cannot be resumed or audited by whoever picks the work up next - including
you, after a context compaction.

Write to it at exactly these moments, before doing the thing rather than after:

1. **Before a long or paid job:** what is being launched, the exact command, the
   output directory, and how to tell whether it finished.
2. **After the run:** the observed status, the artifact hashes, and the UTC time.
3. **After verification:** the exit code, the test count, and the coverage.
4. **At commit:** the commit hash and what it contains.

A step that was done but not recorded did not happen, because nobody downstream
can tell it apart from a step that was skipped.

## Prove a fail-closed check actually fails closed

When rehearsing that something rejects bad input, assert on the specific
message, not on a non-zero exit. A rehearsal pointed at a path that does not
exist also exits non-zero, so it passes while proving nothing - and then stands
as evidence of a safety property that was never tested.

Build the broken input explicitly, then require the exact failure.
`tests/fixtures/bdd100k_tiny_manifest_input.json` describes files rather than
containing them, and the preflight resolves all four protocol directories before
it builds anything - so materialize the fixture into the real layout, or the
run fails on a missing directory and proves nothing about pairing.

```python
import json, pathlib, tempfile
from typer.testing import CliRunner
from drivemetrics.cli.app import app

spec = json.loads(pathlib.Path("tests/fixtures/bdd100k_tiny_manifest_input.json").read_text())
root = pathlib.Path(tempfile.mkdtemp())
trees = {
    "train": ("images/10k/train", "labels/sem_seg/masks/train"),
    "val": ("images/10k/val", "labels/sem_seg/masks/val"),
}
for image_dir, label_dir in trees.values():
    (root / image_dir).mkdir(parents=True, exist_ok=True)
    (root / label_dir).mkdir(parents=True, exist_ok=True)
    for sample in spec["samples"]:
        (root / image_dir / pathlib.Path(sample["image_path"]).name).write_text(
            sample["image_content"]
        )
        (root / label_dir / pathlib.Path(sample["label_path"]).name).write_text(
            sample["label_content"]
        )

# The deliberate break: remove one validation label.
(root / trees["val"][1] / pathlib.Path(spec["samples"][0]["label_path"]).name).unlink()

result = CliRunner().invoke(
    app,
    [
        "data",
        "preflight",
        "--config",
        "configs/protocols/bdd100k_semseg_v1.yaml",
        "--data-root",
        str(root),
        "--output",
        str(root / "out"),
    ],
)
assert result.exit_code != 0, "the preflight accepted a broken cohort"
assert "missing label for sample IDs" in result.output, (
    f"nonzero for the wrong reason, so nothing was proven: {result.output}"
)
```

The second assertion is the one that matters. Without it, a typo in the data root
produces the same exit code and the rehearsal reports success. Run as written it
exits 1 with `missing label for sample IDs: ['sample-a']`.

## Run the validator

Before any locked-cohort result is trusted, summarized, or published, run
[the validator](scripts/validate_locked_eval.py):

```bash
uv run python .agents/skills/running-locked-segmentation-evals/scripts/validate_locked_eval.py \
  --config configs/protocols/bdd100k_semseg_v1.yaml \
  --manifest artifacts/manifests/bdd100k_semseg_v1/locked_validation.json \
  --run-record <run-output-dir>/run_record.json \
  --claims docs/claims.yaml
```

It reports every violation at once and exits non-zero on any of: a cohort whose
size disagrees with the protocol, a run record scored under a different protocol
hash or describing a different manifest, a seed outside 17/42/73, a checkpoint
artifact that is not the final step, the locked cohort having been fit on, and a
verified claim about this cohort whose evidence type is not `observed` or
`derived`. Exit zero prints a JSON status naming the cohort and its hashes.

Reading a run record by eye instead is how each of those slips through.

## When there is nothing to report

If the manifests or checkpoints do not exist, the correct answer is that the
evaluation has not been run, plus the exact missing inputs. Never estimate a
score, never carry a number over from a similar model, and never present a
figure from the plan as a result. A fabricated number is the one failure this
project cannot recover from.

## Incomplete input

If a request is missing prerequisites, name them exactly and stop. Do not guess
a data root, invent a hash, or start a job to find out. The prerequisites are:
the protocol config, a frozen manifest for the intended cohort, a checkpoint
with matching protocol provenance, the dataset root, and measured provenance for
the current machine.

## Quick reference

| Question | Answer |
| --- | --- |
| Approved seeds | 17, 42, 73 |
| Checkpoint selection | Final step only |
| Steps / warmup / effective batch | 30,000 / 1,000 / 16 |
| Bootstrap | 5,000 resamples, seed 20260831, 95% |
| Locked cohort size | 1,000 assigned, 998 eligible (amendment A1) |
| Where scores are computed | Source geometry, not padded geometry |
| GPU policy | Colab A100 first; local GPU needs current explicit approval |

## Common mistakes

- **Reporting one seed.** Every number is a mean over 17/42/73 with its paired
  bootstrap interval.
- **Comparing manifest hashes as a set.** Compare pairwise by cohort name.
- **Reusing a provenance blob across machines.** Re-measure it every session.
- **Treating a ranking reversal as the goal.** It is an observation. Stability is
  an equally publishable result, and the report must say which occurred.
- **Calling an image band a distance.** `normalized_image_band` is top, middle
  and bottom thirds of the frame, not depth.
