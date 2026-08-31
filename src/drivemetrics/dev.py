"""Deterministic, fail-fast repository verification commands."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from urllib.parse import unquote, urlsplit

VERIFY_STAGES: tuple[str, ...] = (
    "private_guard",
    "format_check",
    "lint",
    "typecheck",
    "unit_and_integration_tests",
    "branch_coverage_100",
    "schema_contracts",
    "docs_links",
)

StageRunner = Callable[[str, Sequence[str], Path], int]
_MARKDOWN_LINK = re.compile(r"!?\[[^]]*\]\((?P<target><[^>]+>|[^)\s]+)")


def _stage_commands() -> dict[str, tuple[str, ...]]:
    python = sys.executable
    return {
        "private_guard": (python, "-m", "drivemetrics.private_guard"),
        "format_check": (python, "-m", "ruff", "format", "--check", "."),
        "lint": (python, "-m", "ruff", "check", "."),
        "typecheck": (python, "-m", "mypy", "src", "tests"),
        "unit_and_integration_tests": (python, "-m", "pytest", "--no-cov"),
        "branch_coverage_100": (
            python,
            "-m",
            "pytest",
            "--cov=drivemetrics",
            "--cov-branch",
            "--cov-report=term-missing",
            "--cov-report=json:coverage.json",
            "--cov-fail-under=100",
        ),
        "schema_contracts": (python, "-m", "drivemetrics.dev", "schema-contracts"),
        "docs_links": (python, "-m", "drivemetrics.dev", "docs-links"),
    }


def subprocess_runner(stage: str, command: Sequence[str], cwd: Path) -> int:
    """Run one verification subprocess and preserve its exact exit code."""

    del stage
    return subprocess.run(command, cwd=cwd, check=False).returncode


def verify_repository(repo_root: Path, runner: StageRunner = subprocess_runner) -> int:
    """Run every fixed verification stage, stopping at the first non-zero exit."""

    commands = _stage_commands()
    for stage in VERIFY_STAGES:
        print(f"[verify] {stage}")
        exit_code = runner(stage, commands[stage], repo_root)
        if exit_code != 0:
            print(f"{stage} failed with exit code {exit_code}", file=sys.stderr)
            return exit_code
    return 0


def verify_schema_contracts(repo_root: Path) -> int:
    """Parse every JSON schema and report all malformed files."""

    invalid: list[Path] = []
    for path in sorted((repo_root / "schemas").glob("**/*.json")):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            invalid.append(path)

    for path in invalid:
        print(f"invalid JSON schema: {path.relative_to(repo_root).as_posix()}", file=sys.stderr)
    return 1 if invalid else 0


def _iter_markdown_files(repo_root: Path) -> list[Path]:
    excluded = {".git", ".venv", "build", "dist", "htmlcov"}
    return [
        path
        for path in sorted(repo_root.glob("**/*.md"))
        if not excluded.intersection(path.relative_to(repo_root).parts)
    ]


def verify_docs_links(repo_root: Path) -> int:
    """Report every missing local file referenced by tracked-style Markdown."""

    broken: list[tuple[Path, str]] = []
    for markdown_path in _iter_markdown_files(repo_root):
        text = markdown_path.read_text(encoding="utf-8")
        for match in _MARKDOWN_LINK.finditer(text):
            raw_target = match.group("target").strip("<>")
            parsed = urlsplit(raw_target)
            if not parsed.path or parsed.scheme or raw_target.startswith("#"):
                continue

            decoded_path = unquote(parsed.path)
            if decoded_path.startswith("/"):
                target = repo_root / decoded_path.lstrip("/")
            else:
                target = markdown_path.parent / decoded_path
            if not target.exists():
                broken.append((markdown_path, raw_target))

    for source, missing_target in broken:
        relative_source = source.relative_to(repo_root).as_posix()
        print(
            f"broken local link: {relative_source} -> {missing_target}",
            file=sys.stderr,
        )
    return 1 if broken else 0


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: StageRunner = subprocess_runner,
) -> int:
    """Dispatch the public developer verification commands."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("verify", "schema-contracts", "docs-links", "generate-schemas"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()

    if args.command == "verify":
        return verify_repository(repo_root, runner=runner)
    if args.command == "schema-contracts":
        return verify_schema_contracts(repo_root)
    if args.command == "generate-schemas":
        from drivemetrics.artifacts.schemas import write_contract_schemas

        write_contract_schemas(repo_root)
        return 0
    return verify_docs_links(repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
