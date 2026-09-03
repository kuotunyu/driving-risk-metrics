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
wsl.exe -d Ubuntu-bench -- bash -lc "bash ~/drm-tools/sync_and_mutate.sh"
```

That syncs `~/drm-mut` to the current `rebuild/v1` commit, rebuilds the
environment from the lock, discards the mutant cache, and runs `mutmut run`
with the `[tool.mutmut]` configuration committed in `pyproject.toml`. Roughly
40 minutes. `mut_summary.sh` alongside it prints the score arithmetic, and
`check_kill.sh <mutant-id>...` re-runs named mutants against the working tree's
current tests, which is how each kill below was verified rather than argued.

The scripts live outside this repository, because they are operator tooling
rather than product code and they read a private working tree.

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

Mutmut nulls these arguments, and also produces the variant that drops the
argument entirely; both forms carry the identical claim, and counting both is
why this family holds 27 mutants rather than the 19 first recorded here. Under
the mutation run's own environment they cannot change a byte:

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

### Families 3 to 18, argued 2026-09-04

Each row states the mutation, the argument, and how the argument was
established. "Measured" means it was executed in this project's environment —
Python 3.11.15, numpy 2.4.6, scipy 1.17.1, pydantic 2.13.5, PyYAML 6.0.3 — and
the version is part of the claim. "Read" means it follows from code that is
quoted in the reason and could be checked by reading the same lines.

| # | Mutation | Why it cannot change a result | Basis |
| --- | --- | --- | --- |
| 3 | `reshape(-1)` → `reshape(-2)` | numpy treats **any** negative dimension as "infer this one". Verified equal in array and shape for `(6,)`, `(2,3)`, `(2,3,4)`, `(1,5)`, `(0,)`, `(0,3)` and for the two-argument form `reshape(-1, n)`. This is undocumented numpy behaviour, so the version is part of the claim. | Measured |
| 4 | `reshape(-1)` → `reshape(None)` **at two sites only** | `reshape(None)` does **not** flatten; it returns the array unchanged. It is harmless only where the next call flattens regardless: `np.flatnonzero` in `sample_pixel_indices`, and `np.packbits` without an axis in `pack_correctness`. Both verified byte-for-byte on a two-dimensional input. The third `reshape(None)` survivor is **not** in this family — see the rejected proposals below. | Measured |
| 5 | `statistic(summed) * signs` → `/ signs` | `signs` holds only `+1.0` and `-1.0`, and IEEE-754 division by ±1.0 is exact. | Read |
| 6 | `models[position + 1 :]` → `models[position - 1 :]` | The inner loop of `_has_strict_pair_flip` still visits every unordered pair at least once. Its predicate `baseline_delta * comparison_delta < 0.0` is symmetric in the pair, and is `0.0` for the self-pair the mutant adds, so the returned boolean is unchanged. | Read |
| 7 | `training=False` → `True` / `None` in `_sample_logits` | `transforms.py:68` is the only use of the flag: `if training and flip_draw < 0.5`. The call site pins `flip_draw=1.0`, so no value of `training` reaches the flip. | Read |
| 8 | pair label `else 1` → `else 2` | Both consumers are label-value blind. `_signed_difference_statistic` tests `label == 0`; `_model_run_groups` partitions with `np.unique`; `_validate_labels` checks length and integer type only, never contiguity. `{0, 2}` partitions exactly as `{0, 1}`. | Read |
| 9 | `matrix.shape[0]` → `matrix.shape[1]` | `_validate_confusion` has already refused any matrix that is not square. | Read |
| 10 | `if valid.size <= pixels` → `<` | At equality the mutant takes the sampling branch, where `choice(valid.size, size=pixels, replace=False)` is a permutation of every index; `np.sort` then restores ascending order, and `np.flatnonzero` had already returned `valid` ascending. Both branches return the identical array. | Read |
| 11 | `zip(..., strict=True)` weakened or dropped | `_ratios` has exactly three callers, all in `summarize_confusion`, all passing vectors derived from one matrix that `_validate_confusion` proved square. The lengths are equal by construction, so `strict` can never fire. | Read (all callers enumerated) |
| 12 | PyYAML `deep` flag on `_construct_unique_mapping` | All three committed risk profiles load to identical documents under `deep=False`, `deep=True` and `deep=None`: `construct_document` drains the generator-based constructors before `load` returns. | Measured |
| 13 | `generator.integers(0, high, …)` → `integers(high, …)` | numpy documents a single positional argument as `high` with `low=0`. Identical streams under one seed. | Measured |
| 14 | `choice(…, replace=False)` → `replace=None` | Identical draws under one seed; numpy treats `None` as `False` here. | Measured |
| 15 | `np.finfo(np.float64)` → `np.finfo(None)` | Equal objects and an identical `.max`. **Note:** numpy emits a `DeprecationWarning` ("finfo() dtype cannot be None … will raise an error in the future", deprecated in 1.25). This project promotes only `RuntimeWarning` to an error, so the mutant survives today; a future numpy will kill it without any change here. | Measured |
| 16 | `np.errstate(over="ignore", invalid="ignore")` with `invalid` nulled or dropped | `apply_temperature` validates the logits finite and the temperature finite and positive **before** the guarded block. IEEE-754 division raises *invalid* only for `0/0` and `inf/inf`, neither of which is reachable. Overflow, which **is** reachable, is caught by the `isfinite` check on the next line — that is what `over="ignore"` is for. | Read |
| 17 | `method="bounded"` nulled, upper-cased or dropped | `method="bounded"`, `method=None` and `method="BOUNDED"` return the identical minimiser: scipy lower-cases the name and selects `bounded` whenever `bounds` is given. **The `options` argument on the neighbouring line is a different matter — see the rejected proposals.** | Measured |
| 18 | `model_dump(mode="json")` → `None` / any other string | Every `RunRecordV1` field is `str`, `int` or `dict[str, str]`, so json mode, python mode, `None` and an unrecognised string all produce the identical dictionary and identical serialised bytes. | Measured |

## Proposals that failed their own check

Three families were proposed from reading the code and then **rejected** when
the check was run. Each is a real gap, and each now has a test. They are
recorded here so that nobody proposes them again, and because a rejected
equivalence argument is more informative than an accepted one.

- **`runs[0]` → `runs[1]` for the cohort's sample-ID list.** The proposal was
  that `validate_formal_run_index` has already proved every run shares one
  cohort. It proves less than that: it compares cohorts as **frozensets**, so
  it establishes the same *set* and says nothing about the *order*.
  `aggregate_runs` takes the order from `runs[0]` and imposes it on every run,
  so adopting a different run's ordering permutes the image axis, which changes
  which images a fixed bootstrap seed draws, and therefore the published
  interval. Recorded for P1-17: the cohort order should be canonical, or the
  validator should compare ordered tuples.
- **scipy's `options={"xatol": 1e-12}` nulled, dropped, or its key renamed, and
  the tolerance widened.** Measured on the planted-temperature objective: the
  real optimiser recovers `T = 2.0` to within `1.587e-08`; nulling, dropping or
  renaming the option lands at `6.775e-07`; widening the tolerance to `1.0`
  lands at `2.3327`, an error of `0.33`. The optimiser tolerance is load-bearing
  and is now pinned by a test asserting `T == 2.0` to `abs=1e-7`, which clears
  the real value by 6.3× and the nearest mutant by 6.8×.
- **`reshape(None)` in `_validated_probabilities`.** Unlike the two sites in
  family 4, this one **returns** the reshaped targets, so multi-dimensional
  targets keep their leading shape instead of being flattened. The tests that
  let it survive all passed one-dimensional targets, where the call is a no-op.

## What reading the survivors found in the product

Two defects were found not by a mutant dying but by asking why one lived. Both
are open at the time of writing, because the formal runs are in flight and the
source they touch is frozen until they finish; both are listed for P1-17.

- **The paired interval is half the difference its key names.** `aggregate_runs`
  builds the "A minus B" interval by handing the bootstrap a signed statistic —
  `+metric` on A's runs, `-metric` on B's — whose estimate is the mean over the
  two groups of their group means, which is `(mean_A - mean_B) / 2`. The sign
  and the zero-crossing are unaffected, so the ranking comparison is right, but
  the reported effect size and both interval ends are half the difference the
  key names. A test pinning the correct estimand is committed as a strict
  `xfail`, so the day the estimator is fixed the marker fails and the test
  becomes the guard.
- **A test named after a property could not detect its own subject.**
  `test_the_tertile_edges_are_the_hand_computed_ranks` used six observations and
  its docstring claimed both rank indices were sensitive there. They are not:
  at six, `(6-1)//3`, `(6-2)//3` and `5//4` are all 1, and `(12-1)//3` and
  `(12-2)//3` are both 3, so none of the three surviving arithmetic mutants
  moves an edge. Counts of seven and eight do separate them, and are now tested.
  This is the second instance of the same failure mode in this repository — the
  first was the sort-stability test below — and the lesson is the same: a
  docstring that names the mutation a test defends against is not evidence that
  the test can detect it.

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

## The score, and what is still outstanding

Recorded rather than hidden. Every number below comes from one run of
`sync_and_mutate.sh`; the commit is stated so a reader can reproduce it.

### At `0481652`, run `2026-09-03T17:11Z` to `17:49Z`

| | Count | Share |
| --- | ---: | ---: |
| Mutants generated | 2,422 | |
| Killed by a test | 2,108 | 87.04% |
| Timed out | 12 | |
| Killed or timed out | 2,120 | 87.53% |
| Proven equivalent (18 families above) | 116 | |
| **Accounted for** | **2,236** | **92.32%** |
| Unexplained survivors | 186 | 7.68% |

A timeout is counted as a detection: the mutant changed the program's behaviour
enough to make the suite hang, which is a difference the tests noticed. The
row above it is given so a reader who rejects that convention can see the
number without it — 91.82% counting the 116 equivalents but not the 12
timeouts, which still clears the gate.

### Where the 186 unexplained survivors are

| Module | Survivors |
| --- | ---: |
| `calibration.temperature` | 38 |
| `analysis.aggregate` | 35 |
| `metrics.instances` | 34 |
| `calibration.service` | 31 |
| `metrics.calibration` | 23 |
| `metrics.confusion` | 13 |
| `analysis.claims` | 8 |
| `metrics.selective` | 7 |
| `protocol.config` | 5 |
| `protocol.risk_profiles` | 2 |
| `protocol.hashing` | 1 |
| `analysis.bootstrap` | 1 |

Roughly half are error-message wording, which is killed by anchoring a
`pytest.raises` pattern on the whole message rather than a substring. The rest
are boundary comparisons and argument defaults, each of which needs a test that
distinguishes the boundary. Neither group may be waved through: an error
message is the artifact a human reads when a fail-closed check fires, and this
project's own claim-audit skill requires such a check to state its reason.
