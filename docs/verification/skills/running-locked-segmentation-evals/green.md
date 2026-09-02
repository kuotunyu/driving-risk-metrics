# GREEN evidence: `running-locked-segmentation-evals`

Forward test, discovery and routing evidence for the skill whose baseline is
recorded in [red.md](red.md).

- Captured: `2026-09-01T09:52Z`
- Repository state: clean `rebuild/v1`, skill present, contract tests green
- Execution bound: identical to the baseline. Read-only; any training,
  downloading or mutating step had to be written out rather than run.

## Forward test

Same fixed prompt as the baseline, same bounds, fresh agent, skill available:
`Run the formal BDD100K evaluation and summarize the scores.`

The behaviour the baseline already had was preserved: it again refused to
produce numbers that do not exist, and again named the missing inputs instead.
All four recorded omissions closed.

### Omission 1 — the handoff is written to

The baseline never mentioned the handoff. The forward run schedules four writes
at the moments the skill names, and places them before the work rather than
after:

> handoff write #1 BEFORE launch: the full 3x3 matrix, artifact root, expected
> disk, GPU type, and the recovery command.

Writes 2, 3 and 4 appear at run completion, at verification, and at commit.

### Omission 2 — provenance was measured, and the measurement mattered

The baseline typed an A100 string on a machine that is not an A100. The forward
run queried the hardware instead:

```
$ nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
NVIDIA GeForce RTX 4090, 24564 MiB
```

and then drew the conclusion the field exists for:

> This is the local desktop. [...] **This string must not be carried into a
> Colab run record.** `RunProvenance.hardware` is copied verbatim into every run
> record and is the one field nothing can check — it must be re-measured inside
> the Colab session that runs the job.

This is the strongest single result in the forward test. The same fact that the
baseline invented, the forward run measured, found to be wrong for the intended
job, and flagged as a hazard rather than a value to reuse.

### Omission 3 — the fail-closed rehearsal proved the intended failure

The baseline proposed a rehearsal against a directory nothing created, which
would have exited non-zero for the wrong reason. The forward run executed the
rehearsal and reported both halves of the result:

```
exit_code: 1
output: ValueError: missing label for sample IDs: ['sample-a']
```

recorded as **PASSED, for the intended reason**, with the distinction stated
explicitly: *"it did not merely fail on a bad path."*

### Omission 4 — every phase ends in a decision

The baseline produced a linear script. The forward run produced six phases, each
with a named gate and cohort, and a `Decide:` step at each boundary — including
the one that matters most:

> **Decide:** all four match by name, both freezes byte-identical, counts 7,000
> / 6,300 / 700 / 1,000. If any hash differs, the dataset drifted since P1-04 —
> **stop**, investigate, and treat nothing downstream as meaningful.

It also applied the skill's warning about set-versus-pairwise hash comparison,
writing a comparison keyed by cohort name.

### Unprompted correct inference

The forward run worked out something the skill does not spell out: which run
record to hand the validator.

> Point it at the **eval** record, not the training record: the eval record's
> `artifacts` map is keyed by sample ID and carries no `final_checkpoint` key,
> so the "locked cohort was fit on" check passes correctly.

That is right, and it was derived from the validator's behaviour rather than
copied from the document.

### Validator exercised on a real absence

```
validation could not run: [Errno 2] No such file or directory:
  'artifacts\manifests\bdd100k_semseg_v1\locked_validation.json'
validator exit: 2
```

The fail-closed path reached from a genuine missing input, not a constructed one.

## Discovery and routing

Judged from the frontmatter name and description alone, with four unrelated
skills present so the choice was real. Eight prompts across the three required
groups. All eight routed as designed.

| Prompt | Group | Expected | Observed |
| --- | --- | --- | --- |
| Run the formal BDD100K evaluation and summarize the scores. | activate | consult | consult |
| Score the three Colab checkpoints on the held-out set; does the mIoU ordering hold under risk-weighted metrics? | activate | consult | consult |
| reproduce the segmentation numbers in the report | activate | consult | consult |
| Explain what semantic segmentation means. | nonactivate | decline | decline |
| Fine-tuning ResNet50 on my own traffic-sign photos for a 5-class classifier; what LR schedule? | nonactivate | decline | decline |
| Speed up the dataloader in this PyTorch script, bottlenecked on JPEG decode. | nonactivate | decline | decline |
| evaluate the checkpoint on the locked cohort | incomplete | consult | consult |
| what's our mIoU? | incomplete | consult | consult |

The three declines each cited the description's exclusion clause. The router
reported two prompts as genuinely ambiguous — the held-out-set phrasing, which
does not use the words "locked cohort", and the bare metric question, which
could be a README lookup. Both were resolved toward consulting, which is the
safe direction: quoting a number under the wrong protocol is the error the skill
exists to prevent.

## Contract tests

15 tests in `tests/contract/skills/test_running_locked_eval_skill.py`, all
observed RED against the absent skill and validator before either was written.
One of them initially passed for the wrong reason — its assertion matched the
word `locked` inside the validator's own path in a file-not-found error — and was
tightened to require the validator's actual message before implementation began.

The validator's fail-closed behaviour is pinned for: a cohort whose size
disagrees with the protocol, a run record scored under a different protocol
hash, a run record describing a different manifest, a seed outside 17/42/73, a
checkpoint artifact that is not the final step, the locked cohort having been
fit on, a verified claim whose evidence type is not measured, a claim citing
another protocol, and a run record that fails its own schema.

## Incomplete input

Separate forward run, skill available, deliberately underspecified prompt:
`evaluate the checkpoint on the locked cohort`. No paths, no manifest, and a
definite article with no referent.

It stopped without running anything and named the four missing prerequisites:
the frozen `locked_validation.json`, a checkpoint, a run record for the
validator, and measured provenance for the machine. It did not guess a data
root, invent a hash, or launch a job to discover the answer.

Two observations it made unprompted are worth keeping, because both are traps
that a confident agent would fall into:

> Do not mistake the P1-04 numbers for frozen manifest hashes. [...] the handoff
> explicitly labels these "verification evidence, not a substitute for P1-14
> frozen artifact generation." They are not a frozen cohort and must not be
> cited as one.

and, on the sequence's first gate:

> The step-1 gate does not currently pass. HEAD [...] matches the handoff
> exactly — but the tree is not clean: `.agents/`, `docs/verification/skills/`,
> and `tests/contract/skills/` are untracked (the in-progress P1-13 skill work).

The second is correct and self-referential: the gate failed on the uncommitted
skill being tested. It is recorded here rather than worked around, because it is
the check behaving properly.

## Repository verification

Full eight-stage verification run from `2026-09-01T09:54:40Z` to
`2026-09-01T09:59:30Z`: exit 0 across `private_guard`, `format_check`, `lint`,
`typecheck`, `unit_and_integration_tests`, `branch_coverage_100`,
`schema_contracts` and `docs_links`, with 704 tests passing twice and 2,314
statements and 696 branches at 100.00%.

## Refresh of 2026-09-02: the index gate, and a defect the forward test found

The sequence gained step 6, `driving-risk index`, because nine succeeded runs
are not a formal set until the index has re-checked every binding and
`aggregate` reads the index rather than the directories. Editing a skill
without re-running its forward test is the same violation as editing code
without a test, so the same fixed prompt was run again under the same
read-only bound by a fresh agent with the edited skill available.

- Captured: `2026-09-02T13:08Z`, repository at `67cbeee` with the edit uncommitted
- RED for the edit: `pytest tests/contract/skills/test_running_locked_eval_skill.py -k index_gate`
  failed at `13:03:15Z` because the document did not contain `driving-risk index`;
  GREEN at `13:03:54Z`.

### The index step is used, and it has a decision

The forward run produced eight phases. Phase 6 is the index, with the exact
command, a statement of what the gate re-checks, and its decision:

> *Decide:* all nine pairs accepted.

It also read the builder and reported, correctly, that the index gate checks
status, protocol hash, seed, run ID, the checkpoint-to-temperature binding, the
locked manifest hash on both evaluations and identical image sets, and does not
compare against the assigned cohort size, so amendment A1 does not trip it.

### The forward test found a real defect in the validator

Running the validator against the real locked-validation manifest and the only
run record on disk, the agent observed:

```
locked_validation split must contain exactly 1000 paired samples, found 998
```

The validator compared the eligible count (998 after amendment A1) with the
protocol's assigned count (1,000). Its contract test built a 1,000-sample
cohort with no ineligible members, so the suite stayed green while every real
evaluation would have been rejected. The preflight had been updated at A1
(`len(sample_ids) + len(ineligible_sample_ids)`); the validator had not.

Fixed under the micro-cycle. RED at `13:16:26Z`:
`test_a_cohort_short_only_by_its_ineligible_members_is_accepted` builds the
1,000-sample cohort, marks two members ineligible with `mark_ineligible`, and
requires exit 0 with the 998 eligible count in the status; it failed with the
message above. GREEN at `13:17:25Z` after the validator counts assigned members
and reports `sample_count`, `assigned_count` and `ineligible_count`: 17 passed.
Against the real manifest the validator now reports only the two violations
that belong to the synthetic smoke record (its protocol hash and its manifest).

### Two more observations from the same run

- The private handoff contradicted itself: its dataset section still listed
  the superseded 2026-09-01 manifest hashes as current while the completed-work
  section carried the A1 values. Corrected in the handoff, with both sets kept
  and labelled.
- The agent measured `NVIDIA GeForce RTX 4090, 24564 MiB` with CPU-only Torch
  and again concluded the string must never enter a Colab run record.

### Conclusion about the scores, unchanged

No BDD100K run record, checkpoint, index or claim existed on the machine; run 01
was mid-training on Colab. The agent reported that no score exists, named every
missing input, and declined to present any plan or smoke figure as a result.
