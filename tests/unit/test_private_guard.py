"""Behavior tests for the tracked-file privacy and artifact guard."""

from __future__ import annotations

import runpy
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

import pytest

Guard = Callable[[Path, Sequence[str], dict[str, str]], tuple[str, ...]]


def load_guard_module() -> ModuleType:
    from drivemetrics import private_guard

    return private_guard


def load_guard() -> Guard:
    """Import inside the test so the missing module is a purposeful RED failure."""

    try:
        from drivemetrics.private_guard import find_forbidden_tracked_files
    except ModuleNotFoundError:
        pytest.fail("drivemetrics.private_guard is missing", pytrace=False)
    return find_forbidden_tracked_files


@pytest.mark.parametrize(
    ("tracked_path", "tracked_text", "expected"),
    [
        (
            "notes.md",
            "PRIVATE HANDOFF" + " - DO NOT COMMIT\n",
            ("notes.md: contains private handoff marker",),
        ),
        (
            ".env",
            "SERVICE_TOKEN=replace-me\n",
            (".env: forbidden environment file",),
        ),
        (
            "handoff/driving-risk-metrics.md",
            "private progress\n",
            ("handoff/driving-risk-metrics.md: forbidden handoff file",),
        ),
        (
            "settings.txt",
            "TOKEN=" + "ghp_" + "a" * 36,
            ("settings.txt: possible credential",),
        ),
        (
            "./settings.txt",
            "TOKEN=" + "github_pat_" + "a" * 82,
            ("settings.txt: possible credential",),
        ),
    ],
)
def test_reports_private_or_credential_violation(
    tmp_path: Path,
    tracked_path: str,
    tracked_text: str,
    expected: tuple[str, ...],
) -> None:
    """Removing any private/credential check must expose its specific fixture."""

    guard = load_guard()

    assert guard(tmp_path, [tracked_path], {tracked_path: tracked_text}) == expected


@pytest.mark.parametrize(
    "tracked_path",
    [
        "data/raw.zip",
        "data/raw.tar.gz",
        "predictions/output.npz",
        "checkpoints/model.pt",
        "checkpoints/model.pth",
        "exports/model.onnx",
    ],
)
def test_reports_forbidden_dataset_or_model_artifact(
    tmp_path: Path,
    tracked_path: str,
) -> None:
    """Removing an artifact suffix check must allow restricted bytes into Git."""

    guard = load_guard()

    assert guard(tmp_path, [tracked_path], {}) == (
        f"{tracked_path}: forbidden data or model artifact",
    )


def test_allows_documented_environment_example(tmp_path: Path) -> None:
    """Treating every env-shaped filename as secret would reject safe templates."""

    guard = load_guard()

    assert (
        guard(
            tmp_path,
            [".env.example"],
            {".env.example": "SERVICE_TOKEN=replace-me\n"},
        )
        == ()
    )


def test_allows_clean_index(tmp_path: Path) -> None:
    """An empty result is required for ordinary tracked source and prose."""

    guard = load_guard()

    assert (
        guard(
            tmp_path,
            ["src/drivemetrics/example.py", "README.md"],
            {
                "src/drivemetrics/example.py": "VALUE = 1\n",
                "README.md": "Public project documentation.\n",
            },
        )
        == ()
    )


def test_reports_every_violation_in_stable_order(tmp_path: Path) -> None:
    """Returning after the first hit would hide later leaks from the operator."""

    guard = load_guard()
    marker = "PRIVATE HANDOFF" + " - DO NOT COMMIT"

    assert guard(
        tmp_path,
        ["weights.pt", ".env"],
        {".env": marker},
    ) == (
        ".env: contains private handoff marker",
        ".env: forbidden environment file",
        "weights.pt: forbidden data or model artifact",
    )


def initialise_git_repository(repo_root: Path) -> None:
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )


def track_file(repo_root: Path, relative_path: str, content: str | bytes) -> None:
    path = repo_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    subprocess.run(
        ["git", "add", "--force", "--", relative_path],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )


def test_check_repository_reads_only_the_git_index(tmp_path: Path) -> None:
    """Scanning the working directory would falsely reject an untracked local env."""

    guard_module = load_guard_module()
    assert hasattr(guard_module, "check_repository"), "Git index adapter is missing"
    initialise_git_repository(tmp_path)
    track_file(tmp_path, "README.md", "Public documentation.\n")
    (tmp_path / ".env").write_text("LOCAL_ONLY=1\n", encoding="utf-8")

    assert guard_module.check_repository(tmp_path) == ()


def test_check_repository_skips_non_utf8_text_but_checks_its_path(tmp_path: Path) -> None:
    """Binary tracked files must not crash UTF-8 scanning or bypass path checks."""

    guard_module = load_guard_module()
    assert hasattr(guard_module, "check_repository"), "Git index adapter is missing"
    initialise_git_repository(tmp_path)
    track_file(tmp_path, "blob.bin", b"\xff\xfe")
    track_file(tmp_path, "weights.pt", b"\xff\xfe")

    assert guard_module.check_repository(tmp_path) == (
        "weights.pt: forbidden data or model artifact",
    )


def test_check_repository_fails_closed_for_non_utf8_git_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An undecodable tracked path cannot safely be treated as a clean index."""

    guard_module = load_guard_module()

    def invalid_path_output(repo_root: Path, arguments: Sequence[str]) -> bytes:
        del repo_root, arguments
        return b"\xff\0"

    monkeypatch.setattr(guard_module, "_run_git", invalid_path_output)

    with pytest.raises(guard_module.GitIndexError, match=r"^tracked path is not valid"):
        guard_module.check_repository(tmp_path)


def test_guard_cli_reports_every_violation_and_returns_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Returning success or printing only one hit could permit a leaking commit."""

    guard_module = load_guard_module()
    assert hasattr(guard_module, "main"), "private guard CLI is missing"
    initialise_git_repository(tmp_path)
    marker = "PRIVATE HANDOFF" + " - DO NOT COMMIT"
    track_file(tmp_path, ".env", marker)
    track_file(tmp_path, "weights.pt", b"model")

    assert guard_module.main([str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        ".env: contains private handoff marker",
        ".env: forbidden environment file",
        "weights.pt: forbidden data or model artifact",
    ]
    assert captured.err == ""


def test_guard_cli_returns_zero_for_clean_index(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A clean tracked index is the success contract consumed by CI."""

    guard_module = load_guard_module()
    assert hasattr(guard_module, "main"), "private guard CLI is missing"
    initialise_git_repository(tmp_path)
    track_file(tmp_path, "README.md", "Public documentation.\n")

    assert guard_module.main([str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_guard_cli_returns_two_when_git_index_cannot_be_read(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Treating Git adapter failure as clean would fail open."""

    guard_module = load_guard_module()
    assert hasattr(guard_module, "main"), "private guard CLI is missing"

    assert guard_module.main([str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unable to read Git index" in captured.err


def test_private_guard_module_entrypoint_returns_clean_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module command used by CI must invoke the same guarded main path."""

    guard_module = load_guard_module()
    assert guard_module.__file__ is not None
    initialise_git_repository(tmp_path)
    track_file(tmp_path, "README.md", "Public documentation.\n")
    monkeypatch.setattr(sys, "argv", ["private_guard.py", str(tmp_path)])

    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(Path(guard_module.__file__)), run_name="__main__")

    assert raised.value.code == 0
