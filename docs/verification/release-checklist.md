# Release checklist for `v1.0.2`

This is the order of operations for turning a verified `main` commit into a
public tagged release. Every box is a gate: an unchecked box stops the release,
and nothing below it is attempted. Evidence for each box is recorded in the
private handoff and, where the box says so, in a file under this directory.

The checklist exists because a release is the one moment when every earlier
promise is cashed at once. A number that was never measured, a claim that no
artifact reproduces, a test that never failed first, or a private file that
reached the index would each survive an ordinary commit and be caught here, or
not at all.

## 0. Preconditions

- [ ] The nine formal runs succeeded and `driving-risk index` accepted all nine
      (model, seed) pairs. The index file and its SHA-256 are recorded in the
      handoff.
- [ ] `driving-risk aggregate`, the failure gallery, `docs/claims.yaml` and
      `driving-risk audit-claims` are complete (P1-17), and every claim marked
      `verified` is `observed` or `derived`.
- [ ] The private handoff matches the resolved repository path, the branch, HEAD
      and a clean working tree. Existing public releases and tags are preserved.

## 1. Every-phase gate, run once more on the release candidate

```powershell
git status --short --branch
git diff --check
uv run --frozen python -m drivemetrics.dev verify
```

- [ ] `verify` exits 0 across all eight stages: private guard, format, lint,
      mypy, tests, 100% statement and branch coverage, schema contracts, docs
      links.
- [ ] The latest task has recorded RED evidence caused by missing behaviour and
      later GREEN evidence, in the handoff.
- [ ] Both project skills have RED, GREEN, activation, nonactivation and
      forward evidence under [`skills/`](skills/).
- [ ] No locked-validation result influenced any config, checkpoint, threshold
      or gallery selection. If one did, the study needs a new protocol version
      and a new locked cohort; it cannot be released.

## 2. Mutation score on the pure core

`mutmut` does not run natively on Windows. It runs on Linux (the local WSL
distribution, or CI) against a clone of the release candidate, never against the
Windows working tree, so the `mutants/` directory and the mutation cache never
touch the repository.

```bash
uv sync --frozen --all-groups
uv run --frozen mutmut run
uv run --frozen mutmut results
```

- [ ] Score over `metrics`, `calibration`, `analysis` and `protocol` is at least
      90%.
- [ ] Every surviving mutant is either killed by a new test or proven equivalent
      from an invariant in the design specification. The proof, the command,
      the commit and the score are written to `mutation-audit.md`.

## 3. Claims and the two READMEs

```powershell
uv run --frozen driving-risk audit-claims --claims docs/claims.yaml
uv run --frozen python .agents/skills/auditing-driving-risk-claims/scripts/validate_claims.py `
  --claims docs/claims.yaml --document README.md --document README.en.md --document docs/release-notes/v1.0.2.md
```

- [ ] The registry audit exits 0.
- [ ] The document audit exits 0: every result sentence in both READMEs carries
      a `<!-- claim: <id> -->` marker, every marked number is held at its
      claim's JSON pointer, and no synthetic or illustrative number is printed
      like a measurement.
- [ ] The English and Traditional Chinese READMEs state the same limitations:
      one protocol, one frozen cohort, three models trained under this protocol
      only, image bands are not distances, intervals are not hypothesis tests,
      a ranking reversal is an observation and not a success criterion.
- [ ] Dataset card, model card, experiment card, license, citation, security
      and privacy notes and the reproducibility section are complete, and the
      dataset card still names the instance-annotation source as a mirror
      rather than an official checksum match.

## 4. Repository hygiene

```powershell
uv run --frozen python -m drivemetrics.private_guard
git grep -nE "PRIVATE HANDOFF - DO NOT COMMI[T]|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|hf_[A-Za-z0-9]{34}|-----BEGIN" -- . ":!docs/verification/release-checklist.md"
git ls-files "*.ipynb"
git ls-files -z | ForEach-Object { Get-Item $_ } | Sort-Object Length -Descending | Select-Object -First 15 Length, FullName
uv run --frozen python -m drivemetrics.dev schema-contracts
uv run --frozen python -m drivemetrics.dev docs-links
git diff --check
```

- [ ] The private guard reports zero violations.
- [ ] The secret grep finds nothing outside this file. The pattern matches the
      FULL private marker on purpose: the guard and its tests build the marker
      from two halves precisely so that a grep for the whole string cannot match
      them, and a grep for half of it would report the guard as a leak. The last
      letter is bracketed so that this file does not itself carry the marker
      string; the 2026-09-05 clean clone caught exactly that leak.
- [ ] No notebook is tracked. Colab notebooks live in the private transfer
      package, never in the repository.
- [ ] No tracked file is larger than the aggregated evidence needs; no raw
      dataset, checkpoint, prediction artifact or `.zip` is tracked.
- [ ] Every JSON schema regenerates byte-identically, every document under `docs/evidence/` validates against the contract its `schema_version` names, and the cross-repository
      envelope fixture conforms.
- [ ] License scan: every runtime dependency in `uv.lock` carries a license
      compatible with MIT redistribution. Record the tool and its output in the
      handoff; if no tool is in the lock, inspect the lock by hand and say so.

## 5. Clean clone

```powershell
$clone = Join-Path $env:TEMP ("drm-clean-" + (Get-Date -Format yyyyMMddHHmmss))
git clone --quiet --no-local --branch main . $clone
Set-Location $clone
git rev-parse HEAD
uv sync --frozen --all-groups --extra train
uv lock --check
$env:SOURCE_DATE_EPOCH = git log -1 --format=%ct
uv run --frozen python -m drivemetrics.release backend
uv run --frozen python -m build --no-isolation
uv run --frozen python -m drivemetrics.release normalize-sdist --dist-dir dist --epoch $env:SOURCE_DATE_EPOCH
uv run --frozen python -m drivemetrics.release verify --dist-dir dist --tag v1.0.2
uv run --frozen python -m drivemetrics.dev verify
uv run --frozen driving-risk --help
```

- [ ] The clone is at the release-candidate commit, installs from the lock,
      builds a wheel and an sdist, passes `verify`, and runs the synthetic
      end-to-end chain (`tests/integration/test_formal_chain.py`) inside
      `verify`.
- [ ] The report rebuilds from the committed evidence in the clone and the
      claims audit passes there.
- [ ] The exact commit, the commands and their outputs are written to
      `clean-clone.md`.
- [ ] The workspace-only `clean_clone.sh REPO [BRANCH] [PACKAGE]` is tested with
      the intended local source. `REPO` is required; defaults are `main` and
      `drivemetrics`. P2 must pass its own repository, branch and `bevcalib`.
      Keep complete output and the actual command exit, including failed runs.

## 6. Public repository, CI and Pages `[GitHub UI]`

P1 already has its public repository, CI and Pages. The next push and annotated
tag require this release's explicit human authorization `推`. Local verification
does not authorize either operation, and the existing tags are never moved.

- [ ] Repository description, topics, social preview, Pages source, release
      notes and the artifact allowlist are drafted in the handoff.
- [ ] The human authorizes the push of the verified `main` candidate.
- [ ] Remote CI passes on the release commit. Pages builds from the committed
      evidence only: no restricted media, no credentials, no backend.

## 7. Tag and verify

```powershell
git tag -a v1.0.2 -m "driving-risk-metrics v1.0.2"
git push origin v1.0.2
```

- [ ] The annotated tag points at the commit remote CI passed on.
- [ ] Before publishing, build the same committed source in two independent
      clean directories with Python 3.11, uv 0.11.18, the same lock and commit
      epoch. Both `uv lock --check` and `drivemetrics.release backend` pass;
      setuptools 84.0.0 and wheel 0.45.1 come from the frozen environment.
- [ ] In each directory, run `python -m build --no-isolation`, then
      `drivemetrics.release normalize-sdist --dist-dir dist --epoch EPOCH` and
      `drivemetrics.release verify --dist-dir dist --tag v1.0.2` through
      `uv run --frozen`. Record the actual source commit, epoch, runtime,
      commands and matching wheel/sdist SHA-256 values below.
- [ ] Install each wheel into a fresh environment. Installed metadata and
      `drivemetrics.__version__` identify 1.0.2, with imports resolved outside
      the source checkout. Build a wheel from the normalized sdist and verify
      that it installs and imports with the same identity.
- [ ] `SHA256SUMS` contains only the wheel and sdist basenames, with LF endings.
      Check it from each distribution directory using `sha256sum -c SHA256SUMS`.
      The workflow uploads exactly `*.whl`, `*.tar.gz` and `SHA256SUMS`, and uses
      `gh release create --verify-tag` so a missing remote tag is not created.
- [ ] After the authorized publication, download the assets and verify their
      portable checksums. Record the actual remote run and tag; local evidence
      does not prove that a future remote job succeeded.
- [ ] A public clean clone of the tag installs and passes `verify`.
- [ ] The handoff status becomes `released`, with the public URL, the tag and
      the commit, and the handoff itself is still in no Git index anywhere.

### v1.0.2 local preparation evidence

The initial Python 3.11.15 candidate probe built twice at one fixed commit epoch:
the wheels matched, but the source archives differed. Archive inspection found
no file-payload changes; variation was confined to the gzip timestamp and TAR
member/PAX `mtime` fields, including generated metadata and directories. The
normalizer changes those timestamps and the gzip filename header only. Tests
preserve file bytes, names, order, mode, owner/group and non-time PAX fields and
check idempotence. This probe used an uncommitted candidate; it is diagnosis,
not the same-commit release proof required above.

Focused release checks passed 29 tests with 100% statement and branch coverage
on the new helper. They reject tag, installed/runtime, filename, wheel metadata
and canonical sdist metadata mismatches before writing checksums. The shared
clone helper passed four local fixture-clone tests, including explicit P2
arguments and propagation of verifier exit 42 without a success message; its
dependency commands were replaced at the external UV boundary for those tests.
Full staged verification, committed-source build evidence and the real P1
clone are separate gates and must be recorded before claiming local completion.

These are v1.0.2 engineering corrections. The historical
[`clean-clone.md`](clean-clone.md) remains evidence for the releases it names;
it is not retroactively changed into proof of reproducible older artifacts.
