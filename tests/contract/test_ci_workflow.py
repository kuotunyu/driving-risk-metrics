"""CI must retain an honest failure receipt for the exact reviewed source."""

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def workflow():
    return yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))


def test_ci_build_uses_locked_backend_without_isolated_dependency_resolution():
    build = next(
        step
        for step in workflow()["jobs"]["verify"]["steps"]
        if step["name"] == "Build wheel and source distribution"
    )
    # uv build does not accept --frozen; run the locked build frontend instead.
    assert shlex.split(build["run"]) == [
        "uv",
        "run",
        "--frozen",
        "python",
        "-m",
        "build",
        "--no-isolation",
    ]


def test_ci_pins_reviewed_source_runtime_and_safe_failure_artifacts():
    job = workflow()["jobs"]["verify"]
    steps = job["steps"]
    checkout = next(step for step in steps if "actions/checkout@" in step.get("uses", ""))
    assert checkout["with"]["ref"] == "${{ github.event.pull_request.head.sha || github.sha }}"
    assert checkout["with"]["persist-credentials"] is False
    setup = next(step for step in steps if "setup-uv@" in step.get("uses", ""))
    assert setup["with"]["python-version"] == "3.11.15"
    assert job["env"]["CUDA_VISIBLE_DEVICES"] == ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert job["env"][name] == "1"
    upload = next(step for step in steps if "actions/upload-artifact@" in step.get("uses", ""))
    assert re.fullmatch(r"actions/upload-artifact@[a-f0-9]{40}", upload["uses"])
    assert upload["if"] == "always()"
    assert set(upload["with"]["path"].splitlines()) == {
        "ci-logs/provenance.txt",
        "ci-logs/clock.tsv",
        "ci-logs/verify.log",
        "ci-logs/exit-code.txt",
    }


@pytest.mark.parametrize("exit_code", [0, 23])
def test_ci_gate_preserves_command_exit_and_diagnostics(tmp_path, exit_code):
    gate = next(
        step
        for step in workflow()["jobs"]["verify"]["steps"]
        if step["name"] == "Run deterministic verification gate"
    )
    bash = (
        str(Path("C:/Program Files/Git/bin/bash.exe")) if os.name == "nt" else shutil.which("bash")
    )
    assert bash
    # Replace only the external environment/tool boundary; execute the actual CI shell.
    prefix = f"""uv() {{
      if [[ "$*" == "run --frozen python -m drivemetrics.dev verify" ]]; then
        echo gate-stdout; echo gate-stderr >&2; return {exit_code}
      fi
      echo runtime-fixture
    }}
    git() {{ echo source-fixture; }}
    """
    (tmp_path / "uv.lock").write_text("lock-fixture\n", encoding="utf-8")
    result = subprocess.run(
        [bash, "-eo", "pipefail", "-c", prefix + gate["run"]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == exit_code, result.stderr
    logs = tmp_path / "ci-logs"
    assert (logs / "exit-code.txt").read_text().strip() == str(exit_code)
    assert "gate-stdout" in (logs / "verify.log").read_text()
    assert "gate-stderr" in (logs / "verify.log").read_text()
    provenance = (logs / "provenance.txt").read_text()
    assert "source-fixture" in provenance
    assert "uv.lock" in provenance
    assert "runtime-fixture" in provenance
    assert len((logs / "clock.tsv").read_text().splitlines()) >= 2
