# Legacy repository audit at `da35026`

## Purpose and boundary

This document records the immutable pre-v1 state of `driving-risk-metrics`
before the approved BDD100K rebuild. It is an evidence archive, not a claim that
the v1 system exists or that the legacy outputs are production results.

P1-01 did not change production code, download a dataset, train a model, run a
GPU workload, create a remote, or publish anything. The approved v1 protocol is
Colab-first for training, batch inference, and prediction-artifact generation.

## Source identity

| Field | Freshly observed value |
|---|---|
| Audit date (UTC) | 2026-08-31 |
| Starting branch | `master` |
| Starting commit | `da350261f3f3e2c95f2915dd2a62f234fb6a2c87` |
| Starting tree | `5eb8b8d9489e1488e1df632957a45ef694cc870a` |
| Starting worktree | Clean |
| Tracked files | 45 |
| Archive tag | annotated `legacy-v0-da35026`, peeled target `da350261f3f3e2c95f2915dd2a62f234fb6a2c87` |
| Rebuild branch | `rebuild/v1`, created from the same commit |

The tag archives the pre-v1 commit. This audit is intentionally committed on
`rebuild/v1`, so it is not part of the archived legacy tree.

## Reproduction log

Every required legacy command was executed from the repository root without
installing or modifying the environment first.

| Command | UTC | Exit | Observed result | Consequence | v1 disposition |
|---|---:|---:|---|---|---|
| `git ls-files` | 2026-08-31T04:17:56Z | 0 | 45 tracked paths | Establishes the complete inspection scope below. | Preserve this manifest as the legacy boundary. |
| `git check-ignore -v src/drivemetrics/data/bdd100k.py` | 2026-08-31T04:18:03Z | 0 | `.gitignore:13:data/` matched the package path | A future source module under `src/drivemetrics/data/` would be silently ignored. | P1-02 must replace the broad rule with root-anchored dataset/cache rules before adding package data code. |
| `python -m pytest -q` | 2026-08-31T04:18:09Z–04:19:01Z | 0 | 95 legacy tests passed | The CamVid/synthetic prototype is internally testable; this does not exercise approved v1 contracts. | Retain only useful unit behavior and add v1 contract tests before implementation. |
| `python -m build` | 2026-08-31T04:19:07Z | 1 | `D:\anaconda3\python.exe: No module named build` | The current environment cannot execute the declared package-build gate. | P1-02 must lock/install the approved Python 3.11 dev toolchain and rerun the build. |
| `ruff format --check .` | 2026-08-31T04:19:12Z | 1 | PowerShell could not find `ruff` | Formatting compliance was not measured because the formatter is absent. | P1-02 must lock/install Ruff, rerun this exact command, and then fix only measured diffs. |

The first PowerShell timing wrapper around the Ruff command incorrectly exposed
an empty `$LASTEXITCODE` as success. The command was rerun directly; PowerShell's
actual process exit was 1. This table records the direct result. No formatting
claim is inferred from an unavailable executable.

## Required defect disposition

### 1. Broad `data/` ignore rule — confirmed

`.gitignore` line 13 contains `data/`, and Git reports that rule as the reason
`src/drivemetrics/data/bdd100k.py` would be ignored. No tracked
`src/drivemetrics/data/` directory exists at the legacy commit.

**Disposition:** P1-02 must narrow ignores to explicit repository-root dataset,
cache, checkpoint, and artifact locations. Package source paths must remain
trackable.

### 2. Declared CLI target is missing — confirmed

`pyproject.toml` declares
`drivemetrics = "drivemetrics.cli:main"`, while `git ls-files` contains no
`src/drivemetrics/cli.py` or equivalent tracked module. A built console script
would therefore target a module that is absent.

**Disposition:** P1-02 must add the tested v1 CLI contract; package/build smoke
tests must import and invoke the installed entry point.

### 3. CI branch contract differs from the repository — confirmed

The legacy repository starts on `master`, but `.github/workflows/ci.yml` only
lists `branches: [main]` for pushes. The workflow runs legacy Ruff lint and
pytest coverage, but does not request branch coverage or `--fail-under=100`.

**Disposition:** P1-02 must make the workflow's branch triggers match the actual
branch policy and install from the lockfile. The approved gate must test Python
3.11, enforce 100% statement and branch coverage for in-scope code, build the
package, and smoke-test the installed CLI.

### 4. Training/build dependency contract is incomplete — confirmed

The base dependency list contains only NumPy. The `train` extra lists PyTorch,
Torchvision, Pillow, and tqdm, but the SegFormer builder imports
`transformers`, which is not declared. The training script also imports
`drivemetrics.data.dataset`, which is absent because no data package is tracked.
The required build command additionally fails because the current interpreter
does not have `build`. There is no `uv.lock`.

**Disposition:** P1-02 must define and lock the complete Python 3.11 dependency
graph. Training dependencies must be explicit and reproducible; P1-03 and later
tasks must not rely on packages that happen to exist in a notebook runtime.

### 5. Dataset, model set, and training protocol are obsolete — confirmed

The tracked implementation is centered on CamVid-11. Its registry/notebook uses
four names: `fcn8s`, `deeplabv3_resnet50`, `setr_pup`, and `segformer_b0`.
`scripts/train.py` defaults to 40 epochs and seed 0, selects `best.pt` using
validation mIoU, and applies AdamW plus cosine scheduling across the legacy
model loop.

The approved v1 experiment instead uses the official BDD100K 10K semantic
split, exactly FCN-ResNet50, DeepLabV3-ResNet50, and SegFormer-B0, 30,000 steps,
seeds 17/42/73, and final-step checkpoint selection. Legacy protocol code and
numbers cannot be relabelled as v1 results.

**Disposition:** Replace the legacy experiment through the approved task
sequence. Do not incrementally train or publish the old four-model CamVid setup.

### 6. Formatting gate failure — confirmed; file diffs not yet measured

The required `ruff format --check .` gate fails because Ruff is not installed.
That is fresh evidence of an incomplete development environment, but it is not
evidence identifying which files Ruff would reformat.

**Disposition:** P1-02 must install Ruff from the lock and rerun the exact gate.
Record and correct the resulting file-level differences then; do not claim the
legacy tree is formatted or unformatted before that measurement.

## Additional observed gaps

- `src/drivemetrics/models/registry.py` contains `# pragma: no cover`; approved
  v1 code may not use coverage exclusions to satisfy the gate.
- Committed comparison/evaluation JSON explicitly marks its results as
  synthetic. Those numbers exercise the legacy code path and are not trained
  model evidence.
- `reports/dataset_stats.json` records a machine-relative CamVid path. It is not
  a portable dataset contract.
- `notebooks/train_colab.ipynb` is valid JSON with no stored cell outputs, but
  its setup and commands implement the obsolete CamVid/four-model protocol.
- The README states that its actual four-model comparison was not run. Numeric
  legacy prose and committed reports therefore remain historical or synthetic
  evidence unless separately reproduced.

## Inspection method

All 45 paths in the source tree were included in the audit scope. Text and
configuration files were read directly; every tracked Python file parsed with
Python's AST; every tracked JSON and notebook file parsed as JSON; targeted
searches then traced dataset, model, seed, checkpoint, CI, dependency, and
provenance contracts. No parse errors were found. Parsing establishes structural
readability, not v1 correctness.

The exact source manifest follows. Object identifiers are Git blob IDs from the
immutable tree named above; byte counts are from `git ls-tree -r --long HEAD`.

```text
100644 17f5a8673ba599a2df67491b03f068f07bad39e0      258 .gitattributes
100644 128f93cc8ca9bf1bde39edd9fea7c2b5f45cd36f     2000 .github/workflows/ci.yml
100644 2e8b271bb6ddd24e4f1bf7a2d34f3fe156bf6b12      849 .gitignore
100644 c1fc62a1db9325e338efcb15332f1461bda48caa     1067 LICENSE
100644 5ddce68490f6f92abe379f186191fea42c843711    11655 README.md
100644 c7f419eabcafe045c8ff5e92f666a71762e6d7bd     6563 docs/AUDIT.md
100644 6ed48773685cebac50a7cc9867948d565f461ab0     8841 notebooks/train_colab.ipynb
100644 8654efdf2101d081f01318346395feb012b30586     2378 pyproject.toml
100644 fdc35cfd930965d3238e7cd262566023113db789     5620 reports/comparison.json
100644 c7406359bd10f964e47c54281a3c6b9223e4ef5e     5162 reports/dataset_stats.json
100644 2d28148f755233c1f871e600de522e672a3c78c2    20710 reports/eval/synth-background-biased.json
100644 ace8a2e55c4d17b2267cb73d12ec50023d467255    18363 reports/eval/synth-balanced.json
100644 7903d8f6f40c8e9a3b8441cf88bbbc7882f9207b    18405 reports/eval/synth-hazard-priority.json
100644 4930c38c5e3a7854af26420ad13bddf82d57f7fa     3255 reports/notebook_audit.json
100644 cfa68ac9b5ba4d7870c0700d78bfe68c7b7a5019   160506 reports/split_manifest.json
100644 09f8fc33884a7a909548c5f7ba53aff3b7dec3f6     4567 scripts/analyse_dataset.py
100644 8b98e62e228ec0c1fad1d10cbe5b3707728851b4     7191 scripts/audit_notebooks.py
100644 fd3151e3a7763ca6b792e6892ec3929e82aaf729     7618 scripts/evaluate.py
100644 5069dd6f7a28151630e555eeffc511ee07464216     3408 scripts/report.py
100644 2ece2a19f605237cbfe271d96f25ecfa0dfbda53    10875 scripts/train.py
100644 bcbaf71a7ae2c48ea9408a69e14b24826d0242ba     2472 src/drivemetrics/__init__.py
100644 691eec3000b1d506acacf17ddba8f1e4922cc1e6    10895 src/drivemetrics/evaluate.py
100644 a2bd0795816559cb304abe0e3ef9c85feb66074d       34 src/drivemetrics/geometry/__init__.py
100644 e3cafb4af156e40a84ccb044c2bb5b454776a64a     7908 src/drivemetrics/geometry/ipm.py
100644 a2bd0795816559cb304abe0e3ef9c85feb66074d       34 src/drivemetrics/metrics/__init__.py
100644 9caea40d7c38b73b53c4b2e8638ae144553f798e     6683 src/drivemetrics/metrics/blindspot.py
100644 504fde96551f117ec32a57fa883ac22ac8c0980a    10097 src/drivemetrics/metrics/confusion.py
100644 301cbdbba1aa951ccea1cd9e81217b22c64afd96     9351 src/drivemetrics/metrics/risk.py
100644 1f6b79d57fd0a6dd7579f6f8c0c21e20a6d2b57c     4815 src/drivemetrics/metrics/stratified.py
100644 a2bd0795816559cb304abe0e3ef9c85feb66074d       34 src/drivemetrics/models/__init__.py
100644 a52d761e84a20f60edd9433efd9962611a6130ae     3241 src/drivemetrics/models/fcn.py
100644 0115d143e11ba7621cbbb3dbbdfe0f32a61fac02     4846 src/drivemetrics/models/registry.py
100644 d66333e614ce63cb0ccbd3ea26f7382d44b93519     3097 src/drivemetrics/models/setr.py
100644 a2bd0795816559cb304abe0e3ef9c85feb66074d       34 src/drivemetrics/report/__init__.py
100644 a995f83c664fdbc27ebf6468be4e413700b830a0    11048 src/drivemetrics/report/figures.py
100644 6a06e085630a6786b1710bc5bf73b6df569ded68    15914 src/drivemetrics/report/html.py
100644 d7aa004fef5649c8d4b61233016080bc7c73acce     6872 src/drivemetrics/sensitivity.py
100644 824a074632f4f74cb616bee925aadc555e0a69d4     8038 src/drivemetrics/synthetic.py
100644 de88975e0b118e61bf1c6247948a76ded7cdb12f    11032 src/drivemetrics/taxonomy.py
100644 a95b1c949377259d79793cf3d6e619e04f83f26d     6214 tests/test_blindspot.py
100644 2788a0bb31bfdfea3a89322351c176aff705ea19     7829 tests/test_confusion.py
100644 44b8f07db1296e0c3da854454e0dff46611946f7     8273 tests/test_geometry.py
100644 79bfd94344c98b505702367184bab831a2bc60e7     9878 tests/test_pipeline.py
100644 05b539fb0eaf17bc87d608642c25235f7a588947     6076 tests/test_report.py
100644 f8c09b36c373fe76ec29bc4d2f0e738b55450647    11021 tests/test_risk.py
```

## Gate to P1-02

P1-01 archives and classifies the prototype; it does not repair it. P1-02 may
start only from `rebuild/v1` after this audit is the sole committed tree change
above `da35026`, the archive tag still peels to `da35026`, and the worktree is
clean. P1-02's first scope is packaging, locked tooling, narrow ignore rules,
the tested CLI skeleton, and CI gates—not model training.
