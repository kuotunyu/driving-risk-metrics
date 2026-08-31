"""Detect private state, credentials, and restricted artifacts in Git paths."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

_PRIVATE_MARKER = "PRIVATE HANDOFF" + " - DO NOT COMMIT"
_CREDENTIAL_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    re.compile(r"sk-[A-Za-z0-9]{32,}"),
    re.compile(r"hf_[A-Za-z0-9]{34,}"),
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{82}"),
)
_FORBIDDEN_ARTIFACT_SUFFIXES = (".npz", ".onnx", ".pt", ".pth", ".tar.gz", ".zip")


class GitIndexError(RuntimeError):
    """Raised when the guard cannot inspect the repository index safely."""


def _normalise_path(tracked_path: str) -> str:
    normalised = tracked_path.replace("\\", "/")
    while normalised.startswith("./"):
        normalised = normalised[2:]
    return PurePosixPath(normalised).as_posix()


def find_forbidden_tracked_files(
    repo_root: Path,
    tracked_paths: Sequence[str],
    tracked_text: dict[str, str],
) -> tuple[str, ...]:
    """Return every tracked-file violation without reading files or invoking Git."""

    violations: set[str] = set()

    for tracked_path in tracked_paths:
        normalised = _normalise_path(tracked_path)
        lowered = normalised.casefold()
        parts = PurePosixPath(normalised).parts
        basename = parts[-1].casefold() if parts else ""

        text = tracked_text.get(tracked_path, tracked_text.get(normalised, ""))
        if _PRIVATE_MARKER in text:
            violations.add(f"{normalised}: contains private handoff marker")
        if any(pattern.search(text) for pattern in _CREDENTIAL_PATTERNS):
            violations.add(f"{normalised}: possible credential")

        if basename == ".env" or (basename.startswith(".env.") and basename != ".env.example"):
            violations.add(f"{normalised}: forbidden environment file")
        if "handoff" in (part.casefold() for part in parts[:-1]) and lowered.endswith(".md"):
            violations.add(f"{normalised}: forbidden handoff file")
        if lowered.endswith(_FORBIDDEN_ARTIFACT_SUFFIXES):
            violations.add(f"{normalised}: forbidden data or model artifact")

    return tuple(sorted(violations))


def _run_git(repo_root: Path, arguments: Sequence[str]) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise GitIndexError(detail or f"git exited with code {result.returncode}")
    return result.stdout


def check_repository(repo_root: Path) -> tuple[str, ...]:
    """Read paths and contents from Git's index, then apply the pure guard."""

    try:
        tracked_paths = tuple(
            path
            for path in _run_git(repo_root, ["ls-files", "-z"]).decode("utf-8").split("\0")
            if path
        )
    except UnicodeDecodeError as exc:
        raise GitIndexError("tracked path is not valid UTF-8") from exc

    tracked_text: dict[str, str] = {}
    for tracked_path in tracked_paths:
        blob = _run_git(repo_root, ["show", f":./{tracked_path}"])
        try:
            tracked_text[tracked_path] = blob.decode("utf-8")
        except UnicodeDecodeError:
            continue

    return find_forbidden_tracked_files(repo_root, tracked_paths, tracked_text)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the guard against one repository and return a process exit code."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)

    try:
        violations = check_repository(args.repo_root.resolve())
    except GitIndexError as exc:
        print(f"unable to read Git index: {exc}", file=sys.stderr)
        return 2

    for violation in violations:
        print(violation)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
