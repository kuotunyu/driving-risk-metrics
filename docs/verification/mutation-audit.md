# Mutation audit

Mutation testing asks a question coverage cannot: not "was this line executed"
but "would anything have noticed if it were wrong". This project runs it on the
pure core only — `metrics`, `calibration`, `analysis` and `protocol` — because
those are the modules whose arithmetic reaches a published number. Adapters, the
CLI, the report and the training glue are held to the same 100% branch gate but
are not scored here.

The release gate is a score of at least 90%. A mutant that survives must either
gain a test or be shown equivalent, and an equivalence has to be an argument
from the code and its environment, not an assertion that it did not matter.

## How to reproduce

Mutmut rewrites source, so it cannot run on the Windows working tree; it runs on
a Linux clone.

```bash
wsl -d Ubuntu-bench -- bash <scratchpad>/sync_and_mutate.sh
```

That syncs `~/drm-mut` to the current `rebuild/v1` commit, rebuilds the
environment from the lock, and runs `mutmut run` with the `[tool.mutmut]`
configuration committed in `pyproject.toml`. Roughly 40 minutes.

## What the configuration excludes, and why

Two test files are excluded from the mutation sandbox, and the reason is
structural rather than convenience:

- `tests/contract/skills/` reads `.agents/`, which the mutant tree does not copy.
- `tests/contract/test_artifact_bytes.py` SCANS `src/` for writers and asserts
  each one pins its line ending. Mutmut rewrites `src/` to create mutants, so
  inside the sandbox that test describes the sandbox rather than the project.

Any test that reads first-party source rather than running it is incompatible
with mutation testing by construction. This matters for the audit below: the
line-ending guarantee is enforced on every platform by a test that cannot
participate in the mutation run.

## Equivalent mutants

### Family 1: numpy dtype arguments that restate the default

Mutmut removes or nulls `dtype=` arguments. Each of the following was verified
empirically in the project environment rather than argued from memory
(**numpy 2.4.6**, platform default integer `int64`):

| Mutation | Why it cannot change a result |
| --- | --- |
| `astype(np.float64)` → `astype(None)` | `np.dtype(None)` IS float64. The arrays are identical. |
| `np.arange(n, dtype=np.int64)` → `np.arange(n)` | The default integer dtype is `int64`. Identical dtype and values. |
| `np.arange(1, n+1, dtype=np.float64)` → default | The result is only ever used as the divisor of a true division, which produces float64 either way. |
| `np.cumsum(bool_array, dtype=np.float64)` → `dtype=None` | A boolean `cumsum` accumulates in `int64`; the following true division yields the identical float64 for any count below 2**53, which is 9,007,199,254,740,992 and therefore unreachable for a per-image confusion. |

**This family is version-dependent and the version must travel with the proof.**
Before numpy 2, the default integer dtype on Windows was `int32`, and the second
row would then be false. The claim is made for the pinned environment, not for
numpy in general.

### Family 2: `encoding="utf-8"` and `newline="\n"` on Linux

Mutmut nulls these arguments. Under the mutation run's own environment they
cannot change a byte:

- `encoding=None` uses the locale's preferred encoding. The mutation run is a
  Linux container whose preferred encoding is UTF-8, so the read and the write
  are identical.
- `newline=None` enables universal-newline translation, which on Linux writes
  `os.linesep`, which is `"\n"`. Identical bytes.
- The writes in `analysis/aggregate.py` and `calibration/service.py` go through
  `json.dumps`, whose default `ensure_ascii=True` emits pure ASCII, so even a
  non-UTF-8 locale would produce the same bytes for those files.

**These are equivalent only in the environment the score is measured in, and
they are NOT equivalent on Windows** — `newline=None` there writes `\r\n` and
the artifact hash changes. That defect actually occurred in this repository and
was fixed at `a051943`. It is enforced on every platform by
`tests/contract/test_artifact_bytes.py`, which is excluded from the mutation run
for the structural reason given above. The mutants are recorded as equivalent
*for the score* and as guarded *for the project* by a different mechanism.

## Survivors that gained a test instead

Mutation testing pointed at real gaps rather than only at noise. The following
were killed by tests written for their own sake:

- **The selective-risk tie order.** `np.argsort(-confidence, kind="stable")` had
  a documented tie contract, a test named after it, and no protection: the test
  used TWO samples, and NumPy's introsort falls back to stable insertion sort
  below sixteen elements, so it passed with or without the argument. The
  replacement computes its reference with Python's `sorted`, whose stability is
  a language guarantee, and fails for any reordering. Confidence is quantized to
  uint16, so ties are expected rather than exotic.
- **The protocol's bootstrap defaults.** `resamples=5000` and `seed=20260831` are
  arguments with defaults so a caller can vary them, which meant nothing pinned
  what the default IS. A run that quietly used 5,001 resamples would reproduce
  from its own record and not from the protocol.
- **Class ID zero and seed zero.** Guards written `<= 0` instead of `< 0` would
  reject the first class in the taxonomy — BDD100K class 0 is `road`, the
  largest class in the dataset — and would reject a perfectly reproducible seed.
- **Fail-closed message wording.** 168 assertions were changed from a substring
  (`match="probability"`) to an anchor on the message itself
  (`match=r"^probability values must be finite"`). A substring cannot detect
  either message mutation mutmut produces: padding keeps the original text
  inside `"XXprobability...XX"`, and lower-casing leaves an all-lowercase
  substring alone.

## Survivors still outstanding

Recorded rather than hidden. See the handoff for the current counts and the
per-module breakdown; the largest remaining groups are boundary comparisons and
loop-exit changes in `calibration.service`, `analysis.aggregate` and
`calibration.temperature`. Each needs a killing test or an equivalence proof
before the release tag.
