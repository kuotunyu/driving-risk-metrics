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
mutant in the tree, not only the survivors, and it currently reports 72 claimed
and zero contradictions.

That check earned its place. Four earlier claims were withdrawn because of it,
and each was a genuinely different mutation that a textual pattern had swept up:

- **A `dtype=` argument is not always redundant.** `np.zeros(n, dtype=np.int64)`
  with the dtype nulled yields **float64**, not int64, and the same is true of
  `np.frombuffer(data, dtype=np.uint8)`. Both were correctly killed. The family
  now lists the specific call-and-dtype pairs that were measured, rather than
  matching `dtype=` and hoping.
- **A `dtype=` on `np.asarray` is a conversion, not a default.** This one was
  found by running the same check against the sibling project, where
  `np.asarray(existing_weight, dtype=np.float64)` in a learned corrector was
  killed: a float32 stem weight is exactly what that line exists to promote.
  The four `np.asarray` survivors in THIS project are equivalent because their
  inputs already carry the target dtype, but that is a property of each call
  site rather than of the text, so they are no longer claimed by the family and
  are counted among the unexplained.
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
| `dtype=np.X` nulled on `np.empty`, `np.zeros`, `np.array`, `np.full`, `np.arange`, `np.cumsum`, for the specific call-and-dtype pairs measured | Each restates the dtype the CALL ITSELF would produce: float64 for the allocators and for Python floats, int64 for `arange` and for `full` with an integer fill, and float64 after the true division that always follows `arange` and `cumsum` here. **`np.asarray` is deliberately excluded** — its inferred dtype comes from the caller's array, so the argument is a real conversion whenever that array is not already the target type. | Measured |
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
  interval. Fixed at `e3ad6fa`: the image axis is sorted and the runs are
  ordered by the approved model and seed lists, so no published number depends
  on how an index was assembled. Writing the test for it caught a third member
  of the same family that nobody had proposed: the order the models first
  appeared was deciding every pair's ORIENTATION, so one index would publish
  `A minus B` and another `B minus A`; and the order of the seeds inside a model
  is the axis the seed resample draws positions on, so the bounds moved with it
  too. The rule the fix states is that the index's assembly order is not data
  and must not reach a published number.
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
were fixed at `e3ad6fa`, once the nine formal runs had finished and the source
freeze was lifted; each kept its test, and each test failed first.

- **The paired interval is half the difference its key names.** `aggregate_runs`
  builds the "A minus B" interval by handing the bootstrap a signed statistic —
  `+metric` on A's runs, `-metric` on B's — whose estimate is the mean over the
  two groups of their group means, which is `(mean_A - mean_B) / 2`. The sign
  and the zero-crossing are unaffected, so the ranking comparison is right, but
  the reported effect size and both interval ends are half the difference the
  key names. Fixed at `e3ad6fa`: the bootstrap gained a `combine` argument —
  `mean` for a level, where the unweighted average over models is the estimand,
  and `sum` for a CONTRAST, where the caller has already signed one group
  negative — and the pairing passes `sum`, so `+A` and `-B` become `A - B`. The
  strict `xfail` that carried the correct estimand became the guard the day the
  fix landed, exactly as it was written to; the test it marked now asserts every
  pair and every metric rather than one of each.
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

## An assertion the score cannot see

A `pytest.raises(..., match="critical")` looked like a source of easy kills.
mutmut pads a plain string literal to `"XXcriticalXX"`, which that pattern still
matches, so anchoring on the whole message should turn those survivors into
kills. The sibling project tested the prediction directly, re-measuring its core
on a clone of the commit that anchored every one of its loose patterns: the
score did not move at all. mutmut 3.7 does not mutate f-strings, and nearly
every message in these projects is an f-string that names the offending value.

The pass was worth making for a different reason, and this repository is where
the reason showed. `match="critical"` in `test_formal_index.py` matched three
separate refusals - `critical_class_ids must be non-empty and unique`,
`critical_class_ids must be integers`, and `critical class 99 is outside 0..18` -
so a parametrized test with four rows passed whichever one fired, and one of
those rows was exercising a contract nobody had checked. `match="duplicate"` in
`test_splits.py` covered three more. Across this repository 45 assertions could
not tell two contracts apart; the portfolio total was 51.

Every row now names its own message, anchored from the start, and a scan of all
three repositories reports 655 patterns with none ambiguous. The scan is not a
gate here, because a gate on assertion shape would be a gate on style rather
than on behaviour, and the tests that matter are the ones that were added when a
row turned out to be checking nothing.

## The score, and what is still outstanding

Every number below comes from one run of `sync_and_mutate.sh`, and the commit
is stated so a reader can reproduce it.

### At `9d6933f`, run `2026-09-05T05:05:24Z` to `2026-09-05T05:39:56Z` — the release candidate, after the analysis modules were scored for the first time

| | Count | Share |
| --- | ---: | ---: |
| Mutants generated | 3,677 | |
| **Killed by a test** | **3,491** | **94.94%** |
| Timed out | 28 | |
| Survived | 158 | 4.30% |
| — of those, proven equivalent above | 152 | |
| — of those, still unexplained | 6 | 0.16% |

**The gate is cleared on kills alone, by 181 mutants.** 90% of 3,677 is
3,310 and the suite kills 3,491. `audit_families.py` reports
zero contradictions across the whole mutant tree.

### How the number got here: three runs on 2026-09-05

Between `e3ad6fa` and `b2c38ed` the pure core gained `analysis/gallery.py`,
`analysis/extended.py`, the per-class, calibration and separability blocks of
`analysis/aggregate.py`, and the classwise ECE, Brier and histogram finalisers in
`metrics/calibration.py` and `metrics/selective.py` — about 1,280 new mutants,
none of which had ever been scored. Three full runs followed, each on a commit:

| Commit | Mutants | Killed | Killed % | Timed out | Survived | Documented | Unexplained | Margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `b2c38ed` (first score of the new modules) | 3,743 | 3,488 | 93.19% | 24 | 231 | 81 | 150 | 119 |
| `94d37dc` (after the tests below) | 3,740 | 3,551 | 94.95% | 10 | 179 | 130 | 49 | 185 |
| `9d6933f` (after one writer for every document) | 3,677 | 3,491 | 94.94% | 28 | 158 | 152 | 6 | 181 |

The first run's score, 93.19%, was lower than `e3ad6fa`'s 95.05% and its
unexplained survivors tripled, from 50 to 150; 118 of the 150 sat in the three
analysis modules the run was the first to score. **That is the number to read.**
A module that had met real data twice and been corrected twice (`docs/protocol.md`,
"Ground-truth metrics") still carried tests that agreed with a wrong
implementation on more than a dozen separate points.

### What the survivors at `b2c38ed` found in the tests

Every item below was a survivor at `b2c38ed` and has a test now. The test's RED is
that survival; its GREEN is the mutant's verdict at `94d37dc`, where every one of
these was killed.

- **Every model had the same predictions, and so did every seed.** The extended
  fixture's `prediction_grid` depended on the image alone, so `!= model` for
  `== model`, and `calibrated[1]` for `calibrated[0]`, both survived. The fixture
  now adds one top-band error for every model after the first and one
  person-instance error for that model's seed 42 (`x_extended_metrics__mutmut_87`, `_132`).
- **Both images had the same mask, byte for byte**, so `file_sha256[2 * position - 1]`
  read the same digest from the wrong slot. A direct test of
  `_resolve_label_paths` with two masks that differ by one pixel kills it
  (`x__resolve_label_paths__mutmut_20`).
- **No annotation ID had a zero low byte.** `(b2 >> 8) | b3` and `(b3 << 8) | b3`
  kept every ID distinct; ID 256 collapses to background under both
  (`x__instance_block__mutmut_43`, `_44`). `<< 9` is injective over every byte
  pair and is a family, not a gap.
- **Every corroborated instance had at least two pixels**, so reading its class
  from the second pixel never raised; a one-pixel person is the instance the study
  most needs to score (`x__corroborated_instances__mutmut_41`).
- **The unknown-category instance sat on sky**, so a sentinel colliding with train
  ID 1 was never corroborated; it sits on sidewalk now (`x__semantic_class_of_category__mutmut_8`).
- **No curve had exactly one or two points** (`x__selective_block__mutmut_16`, `_17`).
- **The histogram's `correct +=` could be `=`**; the pooled total is asserted
  (`x__confidence_histogram__mutmut_27`).
- **The result objects were never read** — `computed`, `models`, `document_path`
  of `ExtendedMetricsResult`, and every field of `GalleryResult`.
- **No output ever went to a directory that did not exist** (`extended_metrics`
  `_193/_195/_197`, `select_gallery` `_139`).
- **Gate messages were matched by prefix**, so the separator between two
  violations could change. The message is now compared whole against the
  validator's own reasons (`x_extended_metrics__mutmut_22`, `x_select_gallery__mutmut_20`).
- **The gallery fixture orders images identically for every model**, so `!= model`
  chose the same sample IDs; the per-seed mIoU values behind them are recomputed
  from the model's own runs (`x_select_gallery__mutmut_57`).
- `per_model` was tested at 0 and 2, not at 1 or `True` (`x_select_gallery__mutmut_5`, `_2`).
- `_ground_truth_support` was tested with the divergent run in the middle; a
  divergent second run with the message asserted exactly, a divergent first run,
  and an image holding exactly one pixel of a class are tested now
  (`x__ground_truth_support__mutmut_17/23/29/40`).
- `excludes_zero` was computed inline from whatever the bootstrap produced; it is
  a named helper with boundary tests at zero (`x_aggregate_runs__mutmut_290/294/295`).
- `seed_count` as `len(runs) / len(models)` gave `3.0` and the contract coerced it.
  Integer fields are `strict` now (`x_aggregate_runs__mutmut_338`).
- The ECE validator's `np.asarray(..., dtype=...)` lines RETURN int64 and float64,
  and that return type is the contract the finalisers divide on; uint8/float32
  input asserts it (`x__validated_ece__mutmut_3/5/8/10/13/15`); negative positive
  counts were never tried (`_30`).
- The Brier score summed float32 input in float32 when the promotion was dropped;
  nineteen 0.1f values sum differently in the two precisions, and the test checks
  that they do before asserting (`x_multiclass_brier_score__mutmut_3`, `_5`).

Two guards were removed by simplification: the dead per-run `"definition"` key in
`_band_block`, and the four-way `None` guard in `_accumulate_calibration`, replaced
by refusing an empty cohort before anything is read and seeding the sums from the
first artifact.

### What the survivors at `94d37dc` found — in the audit, not the code

The 49 unexplained survivors at `94d37dc` were mostly the audit's own defects:

- **Twenty-five belonged to existing families and were misfiled.** The predicate for
  a DROPPED argument compared `np.zeros(n)` against `np.zeros(n, )` — mutmut leaves
  the comma — and never matched. Fixed; `audit_families.py` confirms every claim.
- **Six were the three producers' multi-line `write_text` calls**, whose dropped
  arguments no textual predicate can claim soundly (the changed line is whatever
  shifted up). Every published document is now written by one function,
  `artifacts.documents.write_document`, which validates, serialises and writes with
  the encoding and newline pinned. The write is one site, outside the scored core,
  and the byte format has its own test.
- **Four `np.asarray` casts in `selective_risk_from_histogram`** are equivalent one
  at a time — every consumer is a comparison, an `int(...sum())`, or a `cumsum`
  with an explicit dtype, and the one dtype-sensitive operator receives an int64
  operand whenever only one cast is removed. Removing both with boolean input
  would raise, and mutmut never removes both. Claimed per site, as the `asarray`
  policy requires, with the boolean-input test pinning that the function accepts them.
- **Two `np.asarray` casts in `analysis.bootstrap`** take a Python float and a
  tuple of Python ints; claimed per site.
- **The only standalone `dtype=np.float64,` line in the scored core** is the
  `np.array` of Python floats in `_statistic_for`; claimed by that uniqueness, which
  the falsification check will contradict if a second such line ever appears.
- `label_paths: dict[str, Path] = {}` -> `None`: every read is guarded by
  `manifest is not None`, which implies the reassignment ran.

### Timeouts are re-executed, never trusted

mutmut reported 24 timeouts at `b2c38ed` and 10 at `94d37dc`, in consecutive
blocks inside `extended_metrics` and `select_gallery`. Re-executing each by name
with `check_kill.sh` gave a verdict for every one: **20 killed and 4 survived**,
then **8 killed and 2 survived**. Several "timeouts" were mutants such as
`runs = None` that fail on first use. The two that survived at `94d37dc` were
exactly the two `audit_families.py` had flagged as contradictions — a timed-out
mutant carries a non-zero exit code and the scan had read that as a kill. Both
survived on re-execution, so both families stand; the scan now lists timed-out
claims apart from contradictions, and the four genuine survivors from the first
round have tests above.

### Families added on 2026-09-05

Twenty-one families were added to `families.py`, each with its measurement (in
the project environment: pydantic 2.13.5, numpy 2.4.6, Python 3.11) or its
reading in the predicate's docstring. The padding rule in `rejected_outright` is
exempted for `model_dump(mode=` alone, with the measurement that pydantic falls
back to python mode rather than raising.

| Mutation | Why it cannot change a result | Basis |
| --- | --- | --- |
| `encoding="utf-8"` upper-cased | Python codec lookup is case-insensitive. | Measured |
| `encoding="utf-8"` / `newline="\n"` nulled or dropped anywhere on a one-line call | The parent family, which matched only lines starting with the argument. Linux defaults; the Windows hazard is guarded by `test_artifact_bytes.py`. | Measured |
| `dtype=` dropped on `np.full`/`np.zeros`/`np.empty`/`np.array`/`np.arange`/`np.cumsum` at the measured pairs, one-line calls only | The parent family's argument for the dropped form; the predicate accepts a drop only when the after-line is the before-line minus that argument (trailing comma normalised). | Measured |
| `np.zeros(n, dtype=np.int64)` -> float64 at the four band/histogram accumulators | Every published value passes through `int(...)` or an int64 `asarray`; 998 × 921,600 pixels is far below 2^53. By site. | Measured |
| `predicted_class.astype(np.int64)` -> `astype(None)` into an int64 array | Cast back on assignment. | Measured |
| `np.asarray(<PIL image>, dtype=np.uint8)` nulled or dropped | RGB and L images convert to uint8. Keyed on the PIL handle. | Measured |
| `tuple(sorted(runs[0][...]))` -> `runs[1]` | Rejected 2026-09-04 for the unsorted code; the sort at `e3ad6fa` makes the tuple a function of the validator-proven set. | Read |
| `and` -> `or` after the XOR guard | Both None or both set at that line. | Read |
| A seed-invariant field read from seed 1 | `pixels`, `defined_at`, `true_rows[0]` after the agreement loop. | Read |
| `range(1, n)` -> `range(n)` in the agreement loop | Run 0 compared with itself. | Read |
| `(b2 << 8) \| b3` -> `<< 9` | Injective over every byte pair; equality only. `>> 8` and the misplaced channel are killed. | Measured |
| Unknown-category sentinel `-1` -> `-2` | Compared only with train IDs 0..18 and 255. `+1` is killed. | Read |
| `zip(positions, finalised, strict=True)` weakened | Equal length by construction. | Read |
| `model_dump(mode="json")` -> any other string | Python mode, identical for str/int/dict fields. | Measured |
| `deep=deep` dropped / `invalid="ignore"` dropped | The removed forms of two existing families. | Measured / Read |
| `stream.read(chunk_size)` -> `read(None)` | Identical digest. | Measured |
| `np.asarray(counts\|correct_counts, dtype=np.int64)` in the histogram curve, one at a time | See above. By site. | Read |
| `np.asarray` of a Python float / a tuple of ints in `bootstrap` | numpy infers float64 / int64. By site. | Read |
| The one standalone `dtype=np.float64,` line | `np.array` of Python floats in `_statistic_for`; unique in the core. | Read |
| `label_paths = {}` -> `None` | Every read guarded by `manifest is not None`. | Read |

### Where the 6 unexplained survivors are, at `9d6933f`

| Module | Survivors |
| --- | ---: |
| `analysis.gallery` | 16 |
| `analysis.extended` | 10 |
| `calibration.service` | 5 |

The count above is over mutants mutmut reported as `survived`. An earlier version
of `count_equivalent.py` also counted family matches among `timeout` mutants as
documented equivalents, which understated the unexplained survivors by the number
of family-claimed timeouts (three at this commit); it counts survivors only now,
and the timeouts are handled below.

- `calibration.service`, five, all pre-existing: `Image.open(...).convert(None)` in
  the calibration sampler survives because the fixture images are already RGB — a
  fixture weakness recorded on 2026-09-04 and still open; and the four dropped
  `encoding`/`newline` arguments of `calibrate_checkpoint`'s two multi-line
  `write_text` calls, equivalent on Linux by the same reading as the io family but
  not claimable by any sound textual predicate, because the changed line is
  whatever shifted up. The analysis producers escaped this shape by writing
  through one function; the calibration service still writes its run record and
  temperature artifact directly and could do the same.
- `analysis.extended`, one: `label_paths: dict[str, Path] = {}` -> `None`. Every read
  of the variable is guarded by `manifest is not None`, which implies the
  reassignment ran, so the initial value is never observed. It stays unexplained
  by policy: `rejected_outright` refuses every whole-right-hand-side `None`, and
  that rule is kept rather than carved for one line.

### The 28 timeouts at `9d6933f`, re-executed

Re-executed by name with `check_kill.sh`: **25 killed,
3 survived.** The survivors are the three mutants the
falsification scan listed as family-claimed timeouts — `encoding=None`,
`encoding="UTF-8"` and `runs[1]` in `select_gallery` — each of which is equivalent
by its family's argument and survived, as an equivalence requires. Counted with
those, the mutants that neither a test nor a family accounts for at this commit
number 6, all named above.

### At `e3ad6fa`, run `2026-09-04T10:04Z` to `10:18Z`

| | Count | Share |
| --- | ---: | ---: |
| Mutants generated | 2,463 | |
| **Killed by a test** | **2,341** | **95.05%** |
| Timed out | 0 | |
| Survived | 122 | 4.95% |
| — of those, proven equivalent above | 72 | |
| — of those, still unexplained | 50 | 2.03% |

**The gate is cleared on kills alone, by 124 mutants.** 90% of 2,463 is 2,217
and the suite kills 2,341, so no equivalence argument is needed to reach the
threshold. Counting the 72 proven equivalents as well would give 97.97%; that
number is reported for completeness and is not what the gate rests on.

This run exists because the estimator was corrected. Fixing the paired estimand
added the `combine` argument and its validation, so the mutant population grew
from 2,422 to 2,463: **a score measured before that change no longer describes
this source.** The 41 new mutants were killed as they were created, which is
what a fix arriving with its own tests should look like, and the score moved
from 95.00% to 95.05% rather than falling.

The run before that, at `cad5600`, killed 2,301 of 2,422 and left 121
survivors; the one before it, at `0481652`, killed 2,108 and left 302. Every
additional kill came from a test written for its own sake and verified against
the specific mutant before and after: `survived` with the test absent, `killed`
with it present.

### Where the 50 unexplained survivors are

| Module | Survivors |
| --- | ---: |
| `calibration.service` | 14 |
| `analysis.aggregate` | 9 |
| `analysis.claims` | 6 |
| `analysis.bootstrap` | 6 |
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
