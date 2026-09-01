# RED baseline: `running-locked-segmentation-evals`

This is the recorded baseline for the skill, captured before the skill existed.
The point of running it is to find out what a capable agent actually omits when
nobody hands it a checklist, so that the skill teaches the missing steps rather
than the ones an agent already gets right.

- Captured: `2026-09-01T09:38Z`
- Repository state: clean `rebuild/v1`, no skill present under `.agents/skills/`
- Agent: general-purpose, cold context, no skill and no checklist supplied
- Fixed prompt: `Run the formal BDD100K evaluation and summarize the scores.`
- Execution bound: read-only. The agent was told to write out any command that
  would train, download, or mutate state instead of running it. Everything it
  chose to inspect, and every command it proposed, was its own decision.

## What the baseline got right

The baseline was strong, and recording that matters as much as recording the
failures: a skill that teaches an agent things it already does is dead weight
in the context window.

| Mandated element | Baseline behaviour |
| --- | --- |
| Manifest hashes | Quoted all four cohort hashes from the recorded verification and said to stop if they came back different |
| Locked-validation prohibition | Stated that inspecting partial locked-validation metrics to alter a run invalidates the result |
| Seeds | Used 17, 42, 73 and reported per-seed means with intervals rather than a single seed |
| Final-step checkpoint | Described final-step-only checkpointing correctly |
| Claim evidence types | Tagged numbers `observed` only when traceable to an artifact hash |

It also refused to invent scores. Asked to summarize numbers that do not exist,
it reported that they do not exist and named the two missing inputs. That is the
correct behaviour and the skill must not disturb it.

## Omission 1: the private handoff is never updated

**This is the mandated baseline failure.** The agent read the private handoff
closely enough to quote its recorded hashes back, and then never proposed
writing to it. Its five-phase plan and its twenty-line command sequence contain
no handoff update at task start, before the long job, after verification, or at
commit.

The consequence is not cosmetic. The handoff is the only record that survives a
context loss. A nine-job run that takes a day of paid GPU time, started without
recording what was launched and where its outputs land, cannot be resumed or
audited by whoever picks the work up next — including the same agent after a
compaction.

## Omission 2: synthetic hardware entered a provenance field

The baseline emitted this, inside a block whose own heading says every command
runs from the local repository in PowerShell:

```powershell
$env:DRIVEMETRICS_RUN_PROVENANCE = @{
  commit      = (git rev-parse HEAD)
  lock_sha256 = (Get-FileHash uv.lock -Algorithm SHA256).Hash.ToLower()
  hardware    = @{ gpu = "NVIDIA A100-SXM4-40GB"; vram_mib = "40960"; runtime = "colab" }
} | ConvertTo-Json -Compress
```

The commit and the lock hash are measured. The hardware is typed from memory,
on a machine that is not an A100. `RunProvenance.hardware` is copied verbatim
into every run record the job writes, and those run records are the provenance
behind every published number.

This is the exact failure the project exists to prevent, and it is instructive
that it appeared in an otherwise rigorous plan: the agent was careful about
every hash it could compute and careless about the one field it could only
observe. A field that cannot be derived is precisely the field most likely to be
filled in from plausible-sounding memory.

## Omission 3: a fail-closed rehearsal that cannot fail closed

The baseline proposed proving the preflight fails closed by pointing it at a
deliberately broken fixture:

```powershell
uv run driving-risk data preflight `
  --config configs/protocols/bdd100k_semseg_v1.yaml `
  --data-root $env:TEMP\bdd-tiny-broken `
  --output   $env:TEMP\bdd-tiny-out
```

No step in the sequence creates `bdd-tiny-broken`. Run as written, the command
fails because the path does not exist. It exits nonzero, which is what the
rehearsal was looking for, so the rehearsal passes while proving nothing about
missing-pair detection.

A fail-closed check that cannot distinguish the intended failure from a setup
error is worse than no check, because it produces evidence of safety that was
never tested.

## Omission 4: no stop-or-go decision between phases

The plan is a linear script from gate check to published report. It names the
conditions that should halt the work — hash drift, count mismatch — inside prose,
but the command sequence has no gate at which a human or agent is required to
stop, compare against the recorded state, and decide. Under time pressure a
linear script is run linearly.

## What the skill must therefore teach

The skill is not needed to teach hashes, seeds, cohorts, or evidence types. A
capable agent already reaches those. It is needed for the four failures above,
which share one shape: **every one of them is a place where the agent had to
record or verify something it could not derive, and filled the gap with a
plausible value or skipped it.**

Accordingly the skill must:

1. Make the handoff update a numbered step in the sequence, not a background
   convention, with the specific moments named.
2. Require every provenance field to be measured on the machine that runs the
   job, and refuse a hardware string typed from memory.
3. Require any fail-closed rehearsal to prove the intended failure specifically,
   by asserting on the message and not merely on a nonzero exit.
4. Put an explicit stop-or-go decision between phases, with the comparison that
   decides it stated.

## Forward evidence

The GREEN run against the same prompt is recorded in `green.md`.
