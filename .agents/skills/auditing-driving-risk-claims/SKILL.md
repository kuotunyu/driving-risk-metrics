---
name: auditing-driving-risk-claims
description: Use when a number from this project's locked evaluation is about to be written into a README, report page, slide, blog post, model card or summary, when checking whether already-published numbers still match the artifacts after a rerun, or when registering or verifying an entry in docs/claims.yaml. Not for wording or style edits that keep every number unchanged, generic Markdown or YAML editing, or results from other projects.
---

# Auditing driving-risk claims

## Overview

`driving-risk audit-claims` proves that every claim in `docs/claims.yaml`
reproduces from its artifact. It never sees the sentence that reaches a README.
That sentence is where a digit gets transposed, a value gets rounded, a number
gets carried over from a superseded run, or a comparison nobody registered gets
stated as fact. A careful agent catches most of that by hand, once. The
project needs it caught every time, by a check anyone can rerun at release: a
number is published only when a machine can trace it from the sentence to a
claim ID, an artifact path and a JSON pointer, and read the same value there.

## The shape of a published statement

A result sentence is one line of Markdown carrying exactly these parts:

1. The number, verbatim from the artifact. Not rounded, not converted to a
   percentage, not reformatted. If the artifact holds `0.947`, the sentence
   says `0.947`; a `94.7%` needs its own `derived` claim.
2. A marker on the same line naming the claim: `<!-- claim: <claim_id> -->`.
   The marker is invisible when rendered and is what binds the sentence to the
   registry. A table row is a line; one marker per claim on that row.
3. The evidence type, stated in words when it is `synthetic` or `illustrative`.
   A synthetic number printed like a measurement is the failure this project
   exists to prevent, so the validator refuses the line unless the word appears.

Only claims with `status: verified` may be stated. A number the registry does
not hold is not a result yet: register it, audit it, then write the sentence.

```markdown
| `segformer_b2` | mIoU | 0.712 | <!-- claim: segformer-b2-miou --> |
```

## The audit, in order

1. **Trace before writing.** For every number the request names, find its
   claim in `docs/claims.yaml`, open the artifact at `artifact_path`, resolve
   `metric_path`, and compare. Use the repository's own loaders
   (`drivemetrics.analysis.claims`) rather than a reimplemented hash or a
   hand-typed comparison, so the check cannot drift from the pipeline.
2. **A disagreement is a stop, not a correction.** If the requested number and
   the artifact differ, do not silently write either one. Report the claim ID,
   the artifact path, the pointer and both values, and wait: the request may
   carry a typo, or the artifact may be stale and the registry wrong. Writing
   the artifact's value quietly hides the second case.
3. **Write the statement in the shape above**, then run the validator on the
   document, or on a proposal when the text is not in a document yet:

```bash
uv run --frozen python .agents/skills/auditing-driving-risk-claims/scripts/validate_claims.py \
  --claims docs/claims.yaml --repo-root . --document README.md --document README.zh-TW.md

uv run --frozen python .agents/skills/auditing-driving-risk-claims/scripts/validate_claims.py \
  --claims docs/claims.yaml --repo-root . --proposal proposal.yaml
```

   A proposal is a YAML list under `proposals:` of `{claim_id, text}` pairs.
   Exit zero prints one trace row per statement: source, claim ID, status,
   evidence type, artifact path, metric pointer, the numbers stated, verdict.
   Exit one lists every violation at once. Exit two means the audit could not
   run; that is not a pass.
4. **Record the trace where it survives.** Paste the validator's exit code and
   its trace rows into the private handoff, with the commit, before the change
   is committed. A check that ran but was not recorded cannot be told apart
   from one that was skipped.

## What the validator refuses

- A statement citing a claim ID that is not in the registry.
- A claim whose status is not `verified`.
- A number the artifact does not hold at the claim's pointer, reported with the
  claim ID, the artifact path, the pointer, and both values.
- A document line that names a metric (mIoU, IoU, pixel accuracy, critical
  recall, ECE, Brier, FNR, AURC, selective risk, risk-weighted) beside a number
  and carries no marker. Fenced code blocks are skipped.
- A `synthetic` or `illustrative` claim stated on a line that does not say so.
- A registry that fails `audit-claims`; every statement over it is unbacked.

## Numbers that are not results

Cohort sizes, seed values, step counts, resample counts and hashes are
provenance, and they belong beside the results; they come from the protocol,
the manifests and the artifact's own provenance fields, not from memory. They
carry no marker because they state no metric. A comparison between models
("highest", "leads", "reverses") is a claim about `rankings.json` and needs its
own registered entry before it is written as fact.

## When there is nothing to publish

If the registry is empty or the artifacts do not exist, the correct output is
that no result exists yet, plus the exact missing inputs: a verified registry
entry, its artifact, its metric pointer, and a passing `audit-claims`. Never
estimate, never quote a number from a plan, a smoke run or a similar model.

## Incomplete input

A request that names no number, no claim and no artifact ("put our mIoU in the
README") is missing its prerequisites. Name them and stop. Do not guess a value,
invent a claim ID, or run a job to produce one.

## Quick reference

| Question | Answer |
| --- | --- |
| What binds a sentence to its evidence | `<!-- claim: <id> -->` on the same line |
| Which claims may be stated | `status: verified` only |
| How a number is written | Verbatim from the artifact, no rounding |
| Which evidence types must be named in the sentence | `synthetic`, `illustrative` |
| Where the trace is kept | Validator output, pasted into the private handoff |
| Request and artifact disagree | Stop and report both; write neither |

## Common mistakes

- **Treating `audit-claims` exit zero as evidence about the README.** It audits
  the registry. Only the document audit reads the sentence.
- **Checking by hand, once.** A one-off script proves the number today and
  nothing at release. Run the validator; its trace is the evidence.
- **Correcting the request silently.** The artifact is usually right, but the
  disagreement is the finding; report it.
- **Stating a ranking that nobody registered.** "Highest" is a claim.
- **Writing `SHA-256` in a marked line.** The number extractor reads `256`; label
  hashes on their own unmarked lines, as provenance.
