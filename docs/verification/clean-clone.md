# Clean clone of the release candidate

Section 5 of the [release checklist](release-checklist.md) requires that the
release candidate be cloned into a throwaway directory, installed from the lock,
built, and pushed through the full verification gate there, so that nothing the
release depends on lives only in a working tree. This records the two clones made
on 2026-09-05, in order, because the first one failed and the failure is the
reason this step exists.

## Environment

- WSL 2 distribution `Ubuntu-bench` on the development desktop, Linux, so the
  gate also runs on the platform CI uses rather than only on Windows.
- `uv` 0.11.18, CPython 3.11 managed by uv, `uv sync --frozen --all-groups --extra train`
  from the committed `uv.lock`.
- Script: `clean_clone.sh` from the private tooling. It clones `rebuild/v1` from the
  Windows working tree, syncs, builds, runs `python -m drivemetrics.dev verify` and
  `driving-risk --help`, and leaves the clone for inspection. It never pushes.

## Clone 1: `b0ae5b1`, failed at the private guard

Clone `drm-clean-20260905122830`, HEAD `b0ae5b1ed9e658c253608ca8ccf50e15cc5037b0`.
Sync from the lock succeeded; `uv build` produced
`driving_risk_metrics-1.0.0-py3-none-any.whl` and `driving_risk_metrics-1.0.0.tar.gz`.
`verify` then stopped at its first stage:

```
docs/verification/release-checklist.md: contains private handoff marker
private_guard failed with exit code 1
```

The cause was a change made in that very commit. Section 4 of the checklist had
grepped for half of the private marker, which matched the guard's own source, where
the marker is deliberately built from two halves. The pattern was tightened to the
whole marker, and the checklist thereby came to contain the whole marker itself.
On Windows the eight-stage gate had passed on the working tree before the file was
staged; the guard reads paths and contents from the Git index, so it examined the
previous content, and the leak reached the commit.

Two things follow, both now written into the checklist and the handoff: the grep
pattern brackets its last letter (`COMMI[T]`) so that it matches the marker without
being it, and the pre-commit gate is run on the staged tree, never before staging.

Everything that did not depend on the guard was also exercised in that clone and
passed, which is why the second clone was expected to differ only at the guard:

| Check in clone 1 | Result |
| --- | --- |
| `driving-risk report` rebuilt, `index.html` | `86b0c8a7e5c5ff0042f9739305a28d8b78e4da453ada687042c789492a787129`, identical to the Windows build |
| Four report figure documents | identical to the Windows build |
| `driving-risk figures` rebuilt | `0e6ed867…` and `e6590243…`, identical to the tracked `docs/figures/` |
| `generate-schemas` | no tracked file changed |
| `audit-claims` | 0 violations |
| Document audit, both READMEs | 42 statements, 96 numbers, every verdict `pass` |
| `tests/contract/skills` | 32 passed |
| Wheel SHA-256 | `dbc2d5c44b8800e312d1d6335ca6bef0febf347369afb096a02815cd5300bf30` |
| Sdist SHA-256 | `91cb4b4019c994b53fadb27d4bcd988fedb11179504d1be182ed73d900eb14d0` |

## Clone 2: `c18f2ab`, the release candidate

Clone `drm-clean-20260905123937`, HEAD `c18f2abd660610d9a5f583b6231568338c77afec`,
made after the fix above. Sync from the lock succeeded, `uv build` produced the 1.0.0
wheel and sdist, and the eight-stage gate passed in full on Linux:

```
TOTAL                                          3892      0   1098      0   100%
Required test coverage of 100% reached. Total coverage: 100.00%
1140 passed in 303.39s (0:05:03)
[verify] private_guard
[verify] format_check
[verify] lint
[verify] typecheck
[verify] unit_and_integration_tests
[verify] branch_coverage_100
[verify] schema_contracts
[verify] docs_links
```

`driving-risk --help` answered. The clone was left clean (`git status` empty) and the
same comparisons were then run inside it:

| Check in clone 2 | Result |
| --- | --- |
| `driving-risk report` rebuilt, `index.html` | `86b0c8a7e5c5ff0042f9739305a28d8b78e4da453ada687042c789492a787129`, identical to the Windows build at the same commit |
| Four report figure documents | `7ea1dfc6…`, `44f8755a…`, `ab981e38…`, `c546a9a1…`, identical to the Windows build |
| `driving-risk figures` rebuilt | `0e6ed867580fc66b662ad3fbafd74ff819c7ef85bb53c53861268712fed64589` and `e6590243f49bed6e8f72bc89d6a99dbe517ab503eecdb75f10b9813561108d9e`, identical to the tracked `docs/figures/` |
| `generate-schemas` | no tracked file changed |
| `audit-claims` | 0 violations |
| Document audit, both READMEs | 42 statements, 96 numbers, every verdict `pass` |
| `tests/contract/skills` | 32 passed |
| Wheel SHA-256 | `e66a3594dc59f4ec5a194e479a09d0eab121742ed3c1c9fcef8991932ceb27ad` |
| Sdist SHA-256 | `32e729506446089f89fb93b3539f728a7f5cfd2ba18e809dbbf98c85db708800` |

**One limitation, stated rather than hidden.** The wheel and sdist digests differ from
clone 1's although the package source is identical between the two commits (only a
documentation file changed): the archives carry build timestamps, so two builds of the
same source are not byte-identical. The published documents, the figures and the schemas
are byte-reproducible; the distribution archives are not, and the checksums that count
are the `SHA256SUMS` the Release workflow publishes beside the assets it built. Making
the archives reproducible (a pinned `SOURCE_DATE_EPOCH` in the build) is recorded as a
follow-up, not done here.

## Licence scan, recorded here as section 4 asks

No licence-scanning tool is in the lock, so the licences were read from the
installed packages' metadata with `importlib.metadata` in the locked environment,
and that is what this table is: package metadata, not an independent audit.

| Package | Licence |
| --- | --- |
| numpy | BSD-3-Clause (with 0BSD, MIT, Zlib and CC0 components) |
| scipy | BSD-3-Clause |
| pillow | MIT-CMU |
| pydantic, annotated-types | MIT |
| pyyaml | MIT |
| typer, click, rich, shellingham | MIT, BSD-3-Clause, MIT, ISC |
| jinja2, markupsafe | BSD-3-Clause |
| plotly, narwhals | MIT |
| torch | Apache-2.0 and BSD components |
| torchvision | BSD |
| transformers | Apache-2.0 |
| tqdm | MPL-2.0 and MIT |
| packaging | Apache-2.0 or BSD-2-Clause |

All are compatible with redistribution of this repository under MIT. None is
vendored; every one is declared in `pyproject.toml` and pinned in `uv.lock`.

## What this establishes

The release candidate installs from its lock on a second platform, builds a wheel
and an sdist, passes every stage of the gate, and rebuilds its published report,
its figures and its schemas byte for byte. The one failure the process produced was
a real defect, found before anything was public, which is the only kind of failure
a release checklist is for.
