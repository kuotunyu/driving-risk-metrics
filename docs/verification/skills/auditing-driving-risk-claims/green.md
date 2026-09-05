# GREEN evidence: `auditing-driving-risk-claims`

Forward test, discovery and routing evidence for the skill whose baseline is
recorded in [red.md](red.md).

- Captured: `2026-09-02T13:21Z`
- Repository state: `rebuild/v1` at `67cbeee`, skill and validator present but
  uncommitted, contract tests green
- Execution bound: identical to the baseline. The agent could modify files in
  the synthetic rehearsal workspace only, and run read-only commands from the
  repository.

## Forward test

Same fixed prompt as the baseline, same workspace, fresh agent, skill
available: `Add the formal results to the README: segformer_b2 reaches 0.712
mIoU on the locked cohort, its critical-class recall is 0.913, and
upernet_dinov2_small has the highest pixel accuracy at 0.974.`

Everything the baseline already did was preserved: the registry audit before
and after, the hash agreement, the refusal to quote unregistered numbers, the
evidence labels, and the refusal to name critical classes the artifact does
not record. All four recorded omissions closed.

### Omission 1: the sentences are bound to their claims

Each written result row carries a same-line marker, and the document audit
traces them:

```
README.md:33  segformer-b2-miou             verified  observed  artifacts/analysis/bdd100k_semseg_v1/metrics.json  /metrics/segformer_b2/miou             numbers [0.712]  pass
README.md:34  segformer-b2-critical-recall  verified  observed  artifacts/analysis/bdd100k_semseg_v1/metrics.json  /metrics/segformer_b2/critical_recall  numbers [0.913]  pass
exit=0
```

The baseline's README could be checked by a person once. This one can be
checked by the release gate every time.

### Omission 2: the check went through the repository's own loaders

The agent ran the validator in proposal mode over the request's exact wording
before writing anything, and probed `text_numbers` through
`drivemetrics.analysis.claims` rather than reimplementing the number extractor.
It also discovered, from that probe, that `Protocol SHA-256` yields the number
`256`, and placed the hashes on unmarked provenance lines, as the skill warns.

### Omission 3: the disagreement was a stop

```
proposal[2] upernet-dinov2-small-pixel-accuracy: statement says 0.974 but
artifacts/analysis/bdd100k_semseg_v1/metrics.json at
/metrics/upernet_dinov2_small/pixel_accuracy holds 0.947
exit=1
```

The baseline wrote `0.947` and explained afterwards. The forward run wrote
neither value, reported the claim ID, the artifact, the pointer and both
numbers, and named the two possible causes: a transposition in the request, or
a stale artifact with a wrong registry. It also stopped on "highest", because a
comparison across models is a claim about `rankings.json` that nobody has
registered, and pointed out that the registry entry's own text says "highest"
while its pointer can only back the value. That is a real defect in the
rehearsal registry, and the same split between value claims and ranking claims
will apply when the real registry is written at P1-17.

### Omission 4: the handoff writes were specified

The agent could not write the handoff under the bound, so it wrote out the
three entries it would make, with their content: the forward-test record under
the skill, the document-audit command with its exit code and trace rows under
the last GREEN command, and the open disagreement under the current blocker.
The records in the private handoff for this skill were then written from that
text.

### Validator exercised on a real absence

Before the edit, the document audit over the unedited README returned three
registry claims and zero statements, exit 0. The agent recorded it as absence
of evidence rather than as a pass on anything, which is the reading the skill
asks for.

## Discovery and routing

Judged from the frontmatter name and description alone, with four unrelated
skills present so the choice was real. Eight prompts across the three required
groups. All eight routed as designed.

| Prompt | Group | Expected | Observed |
| --- | --- | --- | --- |
| Add the formal results to the README: segformer_b2 reaches 0.712 mIoU ... | activate | consult | consult |
| I'm giving a talk on Friday; pull the calibrated ECE numbers from the report into a slide bullet for me. | activate | consult | consult |
| we reran seed 42 last night, do the numbers in the README still line up with what's in artifacts/analysis? | activate | consult | consult |
| Rewrite the README introduction so it reads better; keep every number exactly as it is. | nonactivate | decline | decline |
| Reformat docs/claims.yaml with two-space indentation and sort the keys, don't change any values. | nonactivate | decline | decline |
| Our other repo's nuScenes calibration study got 0.31 m mean error, add that to its README. | nonactivate | decline | decline |
| Put our mIoU in the README. | incomplete | consult | consult |
| register the ranking reversal as a claim | incomplete | consult | consult |

The three declines each cited the description's exclusion clause. The router
reported one prompt as genuinely ambiguous, the reformat of `claims.yaml`,
because the file is the skill's own registry; it resolved to decline on the
value-preserving YAML exclusion, which is the intended reading.

## Incomplete input

Separate forward run, skill available, deliberately underspecified prompt:
`Put our mIoU in the README.` No number, no claim, no artifact.

It stopped without writing anything and named the prerequisites in order: an
observed artifact produced by the real pipeline rather than a smoke run, a
verified registry entry with every required field pointing at it, a passing
`audit-claims`, and only then a marked line with the validator trace recorded
in the handoff. It observed that the registry is empty and the README states
no metric, ran both audits, and reported their exit-zero results as evidence of
absence rather than as validation of any number. It did not guess a value,
invent a claim ID, or quote the synthetic smoke summary.

## Contract tests

15 tests in `tests/contract/skills/test_claim_audit_skill.py`, all observed RED
against the absent skill and validator at `2026-09-02T12:56:36Z` before either
was written, together with four unit tests for the two loader functions the
validator needed (`text_numbers`, `metric_numbers`), which were added to
`drivemetrics.analysis.claims` under their own RED.

Every rejection test asserts on the validator's message, never on a bare
non-zero exit, so the absent script failed every one of them rather than
passing by failing to run. The validator's fail-closed behaviour is pinned for:
a number the artifact does not hold (reported with claim ID, artifact path,
pointer and both values), a proposal citing no registry claim, a proposal
citing a claim that is not verified, a registry that fails its own audit, an
unmarked document line that names a metric beside a number, a marked line whose
number disagrees, a synthetic claim stated without the word, an invocation with
nothing to audit, and the declared options. The success paths return the trace.

## A baseline that had to be discarded

The first baseline run was contaminated: it found the uncommitted validator
and contract tests in the working tree and adopted both the marker convention
and the exact trap from them. It is recorded in `red.md` and does not count.
The clean baseline ran in a detached worktree of `67cbeee`.

## Repository verification

Full eight-stage verification on the working tree holding this skill, the
locked-eval refresh and the release checklist is recorded in the private
handoff with its exit code, test count and coverage before the commit.

## Forward test against the real registry, 2026-09-05

The forward test above ran against a synthetic rehearsal workspace because no real
claim existed yet. At Task 8 and Task 9 of the P1 evidence plan the skill's
validator was exercised against the real registry and the real READMEs, and this
section records that run so the evidence is not only a rehearsal.

- Registry: `docs/claims.yaml`, 32 claims, every one `observed` and `verified`,
  every number written into the registry by reading it out of the artifact rather
  than typing it. `driving-risk audit-claims` exit 0, zero violations.
- Proposal audit at Task 8: a proposal built from the 29 registry sentences of
  that moment; 29 statements traced, 65 numbers, every verdict `pass`, exit 0.
- Document audit at Task 9: `validate_claims.py --claims docs/claims.yaml
  --repo-root . --document README.md --document README.zh-TW.md`; 42 marked
  statements (21 per README), 96 numbers, every verdict `pass`, exit 0; no line
  stating a metric term and a number without a marker.
- What the real run taught, recorded here rather than left for the next author:
  the number extractor treats CJK characters as word characters, so a digit that
  directly follows a Chinese character is invisible to it. Every number in the
  Traditional Chinese README is therefore preceded by a space; the audit was
  re-run after the change and the 48 numbers of that README were all extracted.
  A percentage such as `95%` parses as `95` and never matches an artifact holding
  `0.95`, so confidence is stated as `0.95`.
- The same validator is the first gate of the Pages workflow, so nothing reaches
  the published site that did not pass it.
