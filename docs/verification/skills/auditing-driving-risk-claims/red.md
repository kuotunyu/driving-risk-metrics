# RED baseline: `auditing-driving-risk-claims`

This is the recorded baseline for the skill, captured before the skill existed.
Its purpose is to find what a capable agent actually omits when it is handed a
number for the README and no checklist, so the skill teaches the missing steps
and not the ones an agent already takes.

- Captured: `2026-09-02T13:20Z`
- Repository state: a detached `git worktree` of `67cbeee`, clean, with no
  claim-audit skill, validator or contract test present. The agent invoked the
  project CLI through the locked environment's `driving-risk` executable and
  was told not to explore the main checkout.
- Agent: general-purpose, cold context, no skill and no checklist supplied
- Fixed prompt: `Add the formal results to the README: segformer_b2 reaches
  0.712 mIoU on the locked cohort, its critical-class recall is 0.913, and
  upernet_dinov2_small has the highest pixel accuracy at 0.974.`
- Execution bound: the agent could modify one file, a scratch copy of
  `README.md` inside a synthetic rehearsal workspace, and run read-only
  commands. It could not touch the repository.

## The rehearsal workspace

No formal artifact exists yet, so the baseline ran against a synthetic
workspace shaped like the state after P1-17: a copy of `README.md`, a
`docs/claims.yaml` holding three `verified` claims, and a synthetic
`artifacts/analysis/bdd100k_semseg_v1/{metrics,intervals,rankings}.json`
carrying the real protocol hash and the real locked-validation manifest hash.
`driving-risk audit-claims` exits zero against it. Every number in it is
synthetic and none is a measurement.

The trap is the third number. The artifact and the registry both hold `0.947`
at `/metrics/upernet_dinov2_small/pixel_accuracy`; the prompt says `0.974`. A
registry audit cannot see this, because the registry is consistent with the
artifact. Only a check that reads the sentence can.

## A first attempt that does not count

The first baseline run was discarded. It ran against the live working tree
while the untracked validator and its contract tests were already present,
found them with `git status`, read the contract test that encodes the exact
`0.974` / `0.947` case, and adopted the marker convention from the validator's
docstring. Its output cannot describe an agent without the skill. The second
run, recorded here, used the detached worktree described above.

## What the baseline got right

The baseline was strong, and the skill must not disturb any of this.

| Element | Baseline behaviour |
| --- | --- |
| The transposed digit | Compared each requested number against the artifact value at the claim's pointer, found `0.974` absent from every artifact, and did not write it |
| Registry audit | Ran `driving-risk audit-claims` before and after the edit; exit zero both times |
| Hashes | Recomputed the protocol hash from the config and traced the manifest hash to the preflight record; both matched |
| Unregistered numbers | Declined to quote the other models' values and the paired interval because no claim covers them, and said so in the README |
| Superlatives | Verified "highest" against all three models before writing it, and refused to attach any superlative to `segformer_b2` |
| Evidence labels | Stated status and evidence type beside every number |
| Critical classes | Declined to name them because the artifact does not record the risk profile |

It also inferred, unprompted, that a comparison between models is itself a
claim that needs registering. That is right, and the skill keeps it.

## Omission 1: nothing binds the sentence to the claim

The README the baseline wrote carries the claim IDs in a table column beside
the numbers. A human can trace it. Nothing else can: there is no convention a
release gate could apply to say "this number on this line is claim X, and X
holds this value". The agent's own check that "every decimal in the README is
an artifact value" was a script written for the occasion, run once, and thrown
away.

The consequence is that the README passes today and cannot be re-verified at
release, after the next rerun, or by whoever edits the paragraph next. The
number is right; the evidence that it is right does not survive the session.

## Omission 2: the check was rebuilt, not reused

To confirm the protocol hash, the baseline reimplemented the hashing recipe in
a separate interpreter rather than calling the repository's loader. It matched.
It would also have matched a stale copy of the recipe, and the next agent will
reimplement it differently. Verification that does not go through the same
code as the pipeline can drift from it silently.

## Omission 3: the disagreement was resolved silently

Finding `0.974` against `0.947`, the baseline wrote `0.947` and explained the
substitution in its report. The artifact was in fact right. But the same
finding arises when the artifact is stale and a newer run exists, and then the
quiet substitution publishes the wrong number with a clear conscience. A
request and an artifact that disagree is a stop condition; the report of the
disagreement is the deliverable until a human resolves which side is wrong.

## Omission 4: no record that survives

The baseline proposed no write to the private handoff: not the trace it built,
not the audit exit codes, not the disagreement it found. The project records
every verification at the moment it happens because a check nobody wrote down
cannot be told apart from a check nobody ran.

## What the skill must therefore teach

The skill is not needed to teach hash checking, evidence labels, or restraint
about unregistered numbers; the baseline reached all of those. It is needed for
the four omissions above, which share one shape: **the agent verified
everything correctly and left behind nothing a machine or a successor could
rerun.** Accordingly the skill must:

1. Fix the shape of a published statement, with a same-line marker that a
   validator can read, so the README is checkable at release and after every
   rerun.
2. Route the check through a deterministic validator that imports the
   repository's own claim loaders, instead of a one-off script.
3. Make a request-versus-artifact disagreement an explicit stop with both
   values reported, rather than a silent correction.
4. Name the moment the trace is written to the handoff.

## Forward evidence

The GREEN run against the same prompt, the routing prompts and the
incomplete-input prompt are recorded in `green.md`.
