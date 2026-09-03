"""Artifact bytes must not depend on the platform that wrote them.

Every published number in this project is defended by pointing at an artifact
and inviting somebody to recompute it. `Path.write_text` with the default
`newline` translates every `\n` to `\r\n` on Windows, so the same document
written by the same command on the desktop and in CI differs byte for byte.
The content would match and the hashes would not, which is the one failure a
reproducibility claim cannot survive.

The check is structural on purpose: the writers live behind fixtures that cost
GPU time or a dataset to build, and a rule that is only enforced where a cheap
fixture happens to exist is a rule that decays.
"""

from __future__ import annotations

import re
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "drivemetrics"
WRITE_TEXT = re.compile(r"\.write_text\((?P<arguments>.*?)\)\s*$", re.DOTALL | re.MULTILINE)


def write_text_calls() -> list[tuple[Path, str]]:
    """Return every ``write_text`` call in first-party source, with its arguments."""

    calls: list[tuple[Path, str]] = []
    for path in sorted(SOURCE_ROOT.glob("**/*.py")):
        text = path.read_text(encoding="utf-8")
        for match in WRITE_TEXT.finditer(text):
            calls.append((path, match.group("arguments")))
    return calls


def test_the_search_finds_the_writers_it_is_meant_to_guard() -> None:
    """A regex that silently matches nothing would pass this file forever."""

    assert len(write_text_calls()) >= 8


def test_every_written_artifact_pins_its_line_ending() -> None:
    """CRLF translation makes the same content hash differently per platform."""

    offenders = [
        path.relative_to(SOURCE_ROOT).as_posix()
        for path, arguments in write_text_calls()
        if "newline=" not in arguments
    ]

    assert offenders == [], (
        "these writers inherit the platform line ending, so their bytes are not "
        f"reproducible across operating systems: {offenders}"
    )
