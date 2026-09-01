# Repository working agreement

## Scope

This repository contains the installable, data-independent core and adapters for
`driving-risk-metrics`. Keep raw datasets, predictions, checkpoints, rendered
artifacts, credentials, and private progress handoffs outside this Git repository.

## Required workflow

1. Work on the assigned feature branch; do not rewrite history or publish without
   explicit human authorization.
2. For every behavior change, add one focused test first and observe the intended
   failure before writing production code.
3. Use Python 3.11 and the committed uv lock. Run
   `uv sync --frozen --all-groups --extra train` for the development environment.
   The training extra is required by the coverage gate, because the Torch
   training and evaluation backends are exercised against the real framework
   rather than against fakes.
4. Run `uv run python -m drivemetrics.dev verify` before a local checkpoint.
5. Keep first-party statement and branch coverage at 100%. Do not use coverage
   exclusions or `pragma: no cover` in first-party executable code.
6. Keep package code under `src/drivemetrics`; never place unique production logic
   in notebooks, workflows, or top-level wrapper scripts.

## Project skills

Before running, resuming, or reporting any formal BDD100K evaluation — freezing
cohort manifests, launching a training job, scoring a checkpoint, or preparing
published numbers — read and follow
[`.agents/skills/running-locked-segmentation-evals/SKILL.md`](.agents/skills/running-locked-segmentation-evals/SKILL.md).
Its validator must exit zero before any locked-cohort result is trusted.

## Data and compute safety

- `/data/` and `/artifacts/` are repository-local ignore boundaries; package paths
  such as `src/drivemetrics/data/` must remain trackable.
- Formal training, batch inference, and prediction-artifact generation are
  Colab-A100-first. Do not start a long or unattended local GPU job without current
  human approval.
- Unit, contract, packaging, formatting, linting, type checking, and documentation
  checks are CPU-only and must not read the external BDD100K dataset.

## Public/private boundary

Run `uv run python -m drivemetrics.private_guard` before committing. The guard
must inspect the Git index and fail closed for private handoff material,
environment-secret files, credential-shaped text, and restricted dataset/model
artifacts. A documented `.env.example` template is allowed.
