# Mutation audit

Mutation testing asks a question coverage cannot: not "was this line executed"
but "would anything have noticed if it were wrong". This project runs it on the
pure core only — `metrics`, `calibration`, `analysis` and `protocol` — because
those are the modules whose arithmetic reaches a published number. Adapters, the
CLI, the report and the training glue are held to the same 100% branch gate but
are not scored here.

The release gate is a score of at least 90%, counted as mutants KILLED. A
surviving mutant may also be shown equivalent, and an equivalence has to be an
argument from the code and its environment rather than an assertion that it did
not matter — but the gate deliberately does not depend on any such argument,
because a wrong one inflates the score silently while a kill needs no argument
at all.

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

A surviving mutant may be recorded as equivalent only when the argument is made
from the code or from a measured behaviour of a pinned dependency. Two rules
keep that honest.

**The score does not depend on any of it.** The release-gate number below is
mutants KILLED, with no equivalence credit at all. Everything in this section
explains why the remainder survive; none of it is load-bearing.

**Every family is checked against the kill data.** An equivalent mutant cannot
be killed, so if any mutant a family claims has a non-zero exit code, the family
is wrong and is narrowed until the check is clean. That check scans every
mutant in the tree, not only the survivors, and it currently reports 76 claimed
and zero contradictions.

That check earned its place. Three earlier families claimed mutants that had in
fact been killed, and each was a genuinely different mutation that a textual
pattern had swept up:

- **A `dtype=` argument is not always redundant.** `np.zeros(n, dtype=np.int64)`
  with the dtype nulled yields **float64**, not int64, and the same is true of
  `np.frombuffer(data, dtype=np.uint8)`. Both were correctly killed. The family
  now lists the specific call-and-dtype pairs that were measured, rather than
  matching `dtype=` and hoping.
- **A padded string is not a nulled argument.** mutmut also rewrites `"utf-8"`
  to `"XXutf-8XX"`, which is an invalid encoding name that raises.
- **A dropped argument shifts the following line.** When mutmut removes an
  argument, the first differing line is the one that moved up, so the recorded
  before/after can name two unrelated arguments. Families that could be fooled
  by that claim only the explicit `=None` form.

### The families, and how each was established

"Measured" means it was executed in this project's environment — Python 3.11.15,
numpy 2.4.6, scipy 1.17.1, pydantic 2.13.5, PyYAML 6.0.3 — and the version is
part of the claim. "Read" means it follows from code quoted in the reason.

| Mutation | Why it cannot change a result | Basis |
| --- | --- | --- |
| `encoding="utf-8"` / `newline="\n"` nulled | The scoring run is Linux, where the preferred encoding is UTF-8 and `os.linesep` is `"\n"`. **Not equivalent on Windows**, where `newline=None` writes `\r\n` and the artifact hash changes. That defect actually occurred here and was fixed at `a051943`; it is guarded on every platform by `tests/contract/test_artifact_bytes.py`, which is excluded from the mutation run for the structural reason given above. Recorded as equivalent *for the score* and guarded *for the project* by a different mechanism. | Read |
| `dtype=np.X` nulled on `np.empty`, `np.zeros`, `np.array`, `np.asarray`, `np.full`, `np.arange`, `np.cumsum`, for the specific call-and-dtype pairs measured | Each restates what numpy infers for that call with those inputs: float64 for the allocators and for Python floats, int64 for a tuple of Python ints and for `arange`, uint8 for a PIL image, and float64 after the true division that always follows `arange` and `cumsum` here. | Measured |
| `astype(np.float64)` nulled | `np.dtype(None)` **is** float64. | Measured |
| `astype(dtype, copy=False)` with the copy flag changed or dropped | All four forms return an array of the same dtype holding the same values; only the allocation differs, and nothing here mutates such an array in place. | Measured |
| `reshape(-1)` to `reshape(-2)` | numpy treats **any** negative dimension as "infer this one". Verified equal in array and shape for `(6,)`, `(2,3)`, `(2,3,4)`, `(1,5)`, `(0,)`, `(0,3)` and for the two-argument form. Undocumented numpy behaviour, so the version is part of the claim. | Measured |
| `reshape(-1)` to `reshape(None)` **at two sites only** | `reshape(None)` does **not** flatten; it returns the array unchanged. It is harmless only where the next call flattens regardless: `np.flatnonzero` in `sample_pixel_indices`, and `np.packbits` without an axis in `pack_correctness`. The third such survivor, in `_validated_probabilities`, RETURNS the reshaped targets and is **not** in this family — it now has a test. | Measured |
| `generator.integers(0, high, ...)` with the explicit zero dropped | numpy documents a single positional argument as `high` with `low=0`; identical streams under one seed. Setting `low` to `None` or `1` changes the draw and was correctly killed. | Measured |
| `choice(..., replace=False)` nulled | Identical draws under one seed. | Measured |
| `np.finfo(np.float64)` nulled | Equal objects and an identical `.max`. numpy emits a `DeprecationWarning` here ("finfo() dtype cannot be None ... will raise an error in the future"); this project promotes only `RuntimeWarning` to an error, so the mutant survives today and a future numpy will kill it without any change here. | Measured |
| `np.argsort(..., kind="stable")` upper-cased | numpy normalises the sort-kind string case-insensitively. | Measured |
| `int.from_bytes(digest[:8], "big")` with the byte order dropped | Python 3.11 gives `byteorder` a default of `"big"`. | Measured |
| `method="bounded"` nulled or dropped in `minimize_scalar` | scipy selects `bounded` whenever `bounds` is given, and returns the identical minimiser. **The `options` argument beside it is a different matter** — see the rejected proposals. | Measured |
| `model_dump(mode="json")` given another mode | Every `RunRecordV1` field is `str`, `int` or `dict[str, str]`, so all modes produce the identical dictionary and identical bytes. | Measured |
| PyYAML's `deep` flag on `_construct_unique_mapping` | All three committed risk profiles load to identical documents under `deep=False`, `True` and `None`: `construct_document` drains the generator-based constructors before `load` returns. Nulling the NODE argument beside it destroys the parse and was correctly killed. | Measured |
| `zip(..., strict=True)` weakened | `_ratios` has exactly three callers, all in `summarize_confusion`, all passing vectors derived from one matrix that `_validate_confusion` proved square. The lengths are equal by construction. | Read (all callers enumerated) |
| `matrix.shape[0]` to `shape[1]`, and the emptiness check inside `_validate_confusion` | The square check has already refused every non-square matrix; the emptiness check sits on the very next line. | Read |
| `np.errstate(over="ignore", invalid="ignore")` with `invalid` nulled or dropped | `apply_temperature` validates the logits finite and the temperature finite and positive **before** the guarded block, and IEEE-754 division raises *invalid* only for `0/0` and `inf/inf`. Overflow, which **is** reachable, is caught by the `isfinite` check that follows. Giving `invalid` another string raises and was correctly killed. | Read |
| `statistic(summed) * signs` to `/ signs` | `signs` holds only `+1.0` and `-1.0`, and IEEE-754 division by one is exact. | Read |
| `models[position + 1 :]` to `[position - 1 :]` | The inner loop still visits every unordered pair at least once; the predicate is symmetric in the pair and is `0.0` for the self-pair the mutant adds. | Read |
| `training=False` changed in `_sample_logits` | `transforms.py:68` is the only use of the flag, and the call site pins `flip_draw=1.0`, so no value reaches the flip. | Read |
| pair label `else 1` to `else 2` | Both consumers are label-value blind: one tests `label == 0`, the other partitions with `np.unique`, and the validator checks length and integer type only. | Read |
| `if valid.size <= pixels` to `<` | At equality the mutant permutes every index and `np.sort` restores ascending order, which is what the early return already gives. | Read |
| `if drawn.shape[0] < pixels` to `<=` | At equality the padding count is zero and the concatenation returns the array unchanged. | Read |
| `split_paths(protocol, manifest.split_name)` nulled | `calibrate_checkpoint` refuses any manifest whose split is not `calibration` before this line runs, and `split_paths` returns the validation tree only for the locked-validation split. | Read |

## Proposals that failed their own check

Three families were proposed from reading the code and then **rejected** when
the check was run. Each is a real gap, and each now has a test. They are
recorded here so nobody proposes them again, and because a rejected
equivalence argument is more informative than an accepted one.

- **`runs[0]` to `runs[1]` for the cohort's sample-ID list.** The proposal was
  that `validate_formal_run_index` has already proved every run shares one
  cohort. It proves less: it compares cohorts as **frozensets**, so it
  establishes the same *set* and says nothing about the *order*.
  `aggregate_runs` takes the order from `runs[0]` and imposes it on every run,
  so adopting a different run's ordering permutes the image axis, which changes
  which images a fixed bootstrap seed draws, and therefore the published
  interval. Recorded for P1-17: the cohort order should be canonical, or the
  validator should compare ordered tuples.
- **scipy's `options={"xatol": 1e-12}` nulled, dropped, or its key renamed, and
  the tolerance widened.** Measured on the planted-temperature objective: the
  real optimiser recovers `T = 2.0` to within `1.587e-08`; nulling, dropping or
  renaming the option lands at `6.775e-07`; widening the tolerance to `1.0`
  lands at `2.3327`, an error of `0.33`. The optimiser tolerance is
  load-bearing and is now pinned by a test asserting `T == 2.0` to `abs=1e-7`,
  which clears the real value by 6.3 times and the nearest mutant by 6.8 times.
- **`reshape(None)` in `_validated_probabilities`.** Unlike the two sites in the
  flattened-downstream family, this one **returns** the reshaped targets, so
  multi-dimensional targets keep their leading shape instead of being
  flattened. Every test that let it survive passed one-dimensional targets,
  where the call is a no-op.

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

Every number below comes from one run of `sync_and_mutate.sh`, and the commit
is stated so a reader can reproduce it.

### At `cad5600`, run `2026-09-03T20:56Z` to `21:10Z`

| | Count | Share |
| --- | ---: | ---: |
| Mutants generated | 2,422 | |
| **Killed by a test** | **2,301** | **95.00%** |
| Timed out | 0 | |
| Survived | 121 | 5.00% |
| — of those, proven equivalent above | 76 | |
| — of those, still unexplained | 45 | 1.86% |

**The gate is cleared on kills alone, by 121 mutants.** 90% of 2,422 is 2,180
and the suite kills 2,301, so no equivalence argument is needed to reach the
threshold. Counting the 76 proven equivalents as well would give 98.14%; that
number is reported for completeness and is not what the gate rests on.

The previous run, at `0481652`, killed 2,108 of the same 2,422 and left 302
survivors. The 193 additional kills came from tests written for their own sake,
each verified against the specific mutant before and after: `survived` with the
test absent, `killed` with it present.

### Where the 45 unexplained survivors are

| Module | Survivors |
| --- | ---: |
| `calibration.service` | 12 |
| `analysis.aggregate` | 8 |
| `analysis.claims` | 6 |
| `analysis.bootstrap` | 4 |
| `protocol.risk_profiles` | 4 |
| `metrics.calibration` | 3 |
| `protocol.config` | 3 |
| `metrics.selective` | 2 |
| `metrics.spatial` | 1 |
| `calibration.temperature` | 1 |
| `protocol.hashing` | 1 |

These are recorded rather than hidden. Most are argument defaults and message
wording on paths a test already exercises; each still needs either a test that
distinguishes it or an argument that survives the falsification check before it
may be called equivalent. Two are worth naming because their arguments are
written but not yet checked into a family:

- `sha256_file`'s chunk size nulled. `read(None)` reads to end of file, so the
  loop takes one pass instead of several and the digest is the same. The
  chunked and unchunked digests are now compared by a test, which is the
  guarantee that matters; the mutant survives because both paths are correct.
- `Image.open(...).convert("RGB")` with the mode nulled, in the calibration
  sampler. The fixture images are already RGB, so the conversion is a no-op
  there. That is a property of the fixture rather than of the code, and it is
  the same class of weakness as the three fixture defects found in this round.
