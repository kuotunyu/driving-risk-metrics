"""Release wiring must deliver locked inputs to the actual build command."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_the_release_workflow_pins_the_build_epoch() -> None:
    steps = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())["jobs"]["release"][
        "steps"
    ]
    epoch_exported = False
    builds = 0
    for step in steps:
        command = step.get("run", "")
        for line in command.splitlines():
            if "git log -1 --format=%ct" in line and "GITHUB_ENV" in line:
                epoch_exported = "SOURCE_DATE_EPOCH=" in line
            if "uv build" in line or "python -m build" in line:
                builds += 1
                assert epoch_exported, "the distribution build receives no tagged-commit epoch"
                assert "--no-isolation" in line, "build must use the frozen backend environment"
    assert builds >= 1


def test_release_checks_lock_and_backend_before_build_and_uploads_only_distributions() -> None:
    steps = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())["jobs"]["release"][
        "steps"
    ]
    commands = [step.get("run", "") for step in steps]
    assert any("python -m build" in command for command in commands), "no locked backend build"
    build_index = next(i for i, command in enumerate(commands) if "python -m build" in command)
    assert any("uv lock --check" in command for command in commands[:build_index])
    assert any("drivemetrics.release backend" in command for command in commands[:build_index])
    upload = next(command for command in commands if "gh release create" in command)
    assert "dist/* " not in upload
    assert all(asset in upload for asset in ("dist/*.whl", "dist/*.tar.gz", "dist/SHA256SUMS"))
    assert "--verify-tag" in upload
    assert any("drivemetrics.release verify" in command for command in commands[build_index:])
