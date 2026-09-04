"""Contracts for assembling the immutable formal run index from nine run directories.

The index is what the formal-set gate validates and what aggregation consumes.
Nothing else in the pipeline is allowed to reconstruct it by hand, because a
hand-written index is exactly where a wrong seed, a stale checkpoint hash, or a
temperature fitted for different weights slips into the published numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from drivemetrics.artifacts.formal_set import validate_formal_run_index

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = REPO_ROOT / "configs" / "protocols" / "bdd100k_semseg_v1.yaml"

MODELS = ("segformer_b2", "upernet_convnextv2_tiny", "upernet_dinov2_small")
SEEDS = (17, 42, 73)
SAMPLES = ("v0001", "v0002", "v0003")
CRITICAL = (11, 12, 17, 18)
COMMIT = "1" * 40
LOCK = "2" * 64


def load_index_module() -> ModuleType:
    try:
        from drivemetrics.artifacts import formal_index
    except ImportError:
        pytest.fail("drivemetrics.artifacts.formal_index is missing", pytrace=False)
    return formal_index


def run_record(
    run_id: str,
    seed: int,
    protocol_hash: str,
    manifest_hash: str,
    artifacts: dict[str, str],
    status: str = "succeeded",
) -> dict[str, Any]:
    return {
        "schema_version": "driving-risk-run/v1",
        "run_id": run_id,
        "commit": COMMIT,
        "config_sha256": "3" * 64,
        "protocol_sha256": protocol_hash,
        "dataset_manifest_sha256": manifest_hash,
        "lock_sha256": LOCK,
        "hardware": {"gpu": "NVIDIA A100-SXM4-40GB", "runtime": "colab"},
        "seed": seed,
        "started_at_utc": "2026-09-02T00:00:00Z",
        "finished_at_utc": "2026-09-02T08:00:00Z",
        "status": status,
        "artifacts": artifacts,
    }


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")


def build_runs(
    root: Path,
    protocol_hash: str,
    locked_hash: str,
    calibration_hash: str,
    *,
    skip: tuple[str, int] | None = None,
    mutate: Any = None,
) -> Path:
    """Lay out nine complete runs in the directory convention the builder reads."""

    for model in MODELS:
        for seed in SEEDS:
            if skip == (model, seed):
                continue
            run_id = f"{model}-seed-{seed}"
            run = root / model / f"seed-{seed}"
            checkpoint_sha = f"{model}{seed}".encode().hex().ljust(64, "a")[:64]

            train = run_record(
                run_id, seed, protocol_hash, "4" * 64, {"final_checkpoint": checkpoint_sha}
            )
            temperature = {
                "schema_version": "drivemetrics-temperature/v1",
                "temperature": 1.3,
                "protocol_sha256": protocol_hash,
                "dataset_manifest_sha256": calibration_hash,
                "checkpoint_sha256": checkpoint_sha,
                "run_id": run_id,
                "seed": seed,
                "sampled_images": 700,
                "pixels_per_image": 2048,
                "sampling_seed": 20260901,
            }
            per_sample = dict.fromkeys(SAMPLES, "d" * 64)
            eval_raw = run_record(f"eval-{run_id}", seed, protocol_hash, locked_hash, per_sample)
            eval_cal = run_record(f"eval-{run_id}", seed, protocol_hash, locked_hash, per_sample)

            documents = {
                "train": train,
                "temperature": temperature,
                "eval_raw": eval_raw,
                "eval_cal": eval_cal,
            }
            if mutate is not None:
                mutate(model, seed, documents)

            write_json(run / "train" / "run_record.json", documents["train"])
            write_json(run / "calibration" / "temperature.json", documents["temperature"])
            write_json(run / "eval" / "run_record.json", documents["eval_raw"])
            write_json(run / "eval_calibrated" / "run_record.json", documents["eval_cal"])
    return root


@pytest.fixture
def hashes(tmp_path: Path) -> dict[str, Any]:
    from drivemetrics.data.manifest import build_paired_manifest, save_manifest
    from drivemetrics.protocol.config import load_protocol

    def cohort(name: str, split: str, count: int) -> Path:
        images = tmp_path / name / "images"
        labels = tmp_path / name / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index in range(count):
            (images / f"v{index:04d}.jpg").write_bytes(b"i")
            (labels / f"v{index:04d}_train_id.png").write_bytes(b"l")
        manifest = build_paired_manifest(images, labels, split)
        path = tmp_path / f"{split}.json"
        save_manifest(manifest, path)
        return path

    locked = cohort("locked", "locked_validation", 1000)
    calibration = cohort("calib", "calibration", 700)
    from drivemetrics.data.manifest import load_manifest

    return {
        "protocol": load_protocol(PROTOCOL).protocol_sha256,
        "locked_path": locked,
        "locked": load_manifest(locked).manifest_sha256,
        "calibration": load_manifest(calibration).manifest_sha256,
    }


def build(tmp_path: Path, hashes: dict[str, Any], **layout: Any) -> Any:
    index = load_index_module()
    runs = build_runs(
        tmp_path / "runs", hashes["protocol"], hashes["locked"], hashes["calibration"], **layout
    )
    return index.build_formal_run_index(
        runs,
        PROTOCOL,
        hashes["locked_path"],
        runs / "formal_run_index.json",
        critical_class_ids=CRITICAL,
    )


def test_a_complete_matrix_produces_an_index_the_gate_accepts(
    tmp_path: Path, hashes: dict[str, Any]
) -> None:
    """The builder and the gate must agree, or every downstream stage argues with itself."""

    result = build(tmp_path, hashes)
    document = json.loads(result.index_path.read_text(encoding="utf-8"))

    assert validate_formal_run_index(document) == ()
    assert len(document["runs"]) == 9
    assert document["protocol_sha256"] == hashes["protocol"]
    assert document["dataset_manifest_sha256"] == hashes["locked"]
    assert document["num_classes"] == 19
    assert document["critical_class_ids"] == list(CRITICAL)
    assert document["cohort"] == "locked_validation"


def test_each_entry_carries_its_own_temperature_and_checkpoint(
    tmp_path: Path, hashes: dict[str, Any]
) -> None:
    """Copying one run's values into another would pass the gate and corrupt the result."""

    result = build(tmp_path, hashes)
    document = json.loads(result.index_path.read_text(encoding="utf-8"))

    entry = next(e for e in document["runs"] if e["run_id"] == "segformer_b2-seed-42")
    assert entry["temperature"] == pytest.approx(1.3)
    assert entry["checkpoint_sha256"] == b"segformer_b242".hex().ljust(64, "a")[:64]
    assert entry["final_step"] == 30000
    assert entry["artifacts_dir"] == "segformer_b2/seed-42/eval"
    assert entry["calibrated_artifacts_dir"] == "segformer_b2/seed-42/eval_calibrated"
    assert entry["uncalibrated_sample_ids"] == list(SAMPLES)


def test_a_missing_run_is_named_not_skipped(tmp_path: Path, hashes: dict[str, Any]) -> None:
    """An index silently built from eight runs is the failure the gate exists to catch."""

    with pytest.raises(
        FileNotFoundError, match=r"^missing run_record\.json for upernet_dinov2_small seed-73:"
    ):
        build(tmp_path, hashes, skip=("upernet_dinov2_small", 73))


def test_a_run_that_did_not_succeed_is_refused(tmp_path: Path, hashes: dict[str, Any]) -> None:
    """A failed training run has no final checkpoint to index, whatever its directory holds."""

    def mutate(model: str, seed: int, documents: dict[str, Any]) -> None:
        if (model, seed) == ("segformer_b2", 17):
            documents["train"]["status"] = "failed"

    with pytest.raises(ValueError, match=r"^training run"):
        build(tmp_path, hashes, mutate=mutate)


def test_a_temperature_fitted_for_other_weights_is_refused(
    tmp_path: Path, hashes: dict[str, Any]
) -> None:
    """A temperature belongs to the exact checkpoint it was fitted on."""

    def mutate(model: str, seed: int, documents: dict[str, Any]) -> None:
        if (model, seed) == ("upernet_convnextv2_tiny", 42):
            documents["temperature"]["checkpoint_sha256"] = "f" * 64

    with pytest.raises(ValueError, match=r"^temperature for"):
        build(tmp_path, hashes, mutate=mutate)


def test_an_evaluation_on_another_cohort_is_refused(tmp_path: Path, hashes: dict[str, Any]) -> None:
    """Pooling a run scored on a different cohort produces a number about nothing."""

    def mutate(model: str, seed: int, documents: dict[str, Any]) -> None:
        if (model, seed) == ("upernet_dinov2_small", 17):
            documents["eval_raw"]["dataset_manifest_sha256"] = "e" * 64

    with pytest.raises(ValueError, match=r"^uncalibrated evaluation of"):
        build(tmp_path, hashes, mutate=mutate)


def test_calibrated_and_uncalibrated_evaluations_must_cover_the_same_images(
    tmp_path: Path, hashes: dict[str, Any]
) -> None:
    """The two artifact sets are compared image by image; a mismatch is not comparable."""

    def mutate(model: str, seed: int, documents: dict[str, Any]) -> None:
        if (model, seed) == ("segformer_b2", 73):
            documents["eval_cal"]["artifacts"] = {"v0001": "d" * 64}

    with pytest.raises(ValueError, match=r"^calibrated and uncalibrated evaluations of"):
        build(tmp_path, hashes, mutate=mutate)


def test_the_index_is_immutable_once_written(tmp_path: Path, hashes: dict[str, Any]) -> None:
    """Rewriting the index detaches every analysis and claim that cited it."""

    build(tmp_path, hashes)

    with pytest.raises(FileExistsError, match=r"^formal_run_index\.json already exists:"):
        build(tmp_path, hashes)


def test_a_manifest_that_is_not_the_locked_cohort_is_refused(
    tmp_path: Path, hashes: dict[str, Any]
) -> None:
    """Indexing against the training cohort would score every model on data it saw."""

    index = load_index_module()
    runs = build_runs(
        tmp_path / "runs", hashes["protocol"], hashes["locked"], hashes["calibration"]
    )
    wrong = tmp_path / "calibration.json"

    with pytest.raises(ValueError, match=r"^the index must be built against the locked_validation"):
        index.build_formal_run_index(
            runs, PROTOCOL, wrong, runs / "formal_run_index.json", critical_class_ids=CRITICAL
        )


def test_the_package_exports_the_builder() -> None:
    """The command line consumes the builder through the package entry point."""

    import drivemetrics.artifacts as artifacts

    index = load_index_module()
    assert artifacts.build_formal_run_index is index.build_formal_run_index


def _mutating(target: tuple[str, int], key: str, field: str, value: Any) -> Any:
    def mutate(model: str, seed: int, documents: dict[str, Any]) -> None:
        if (model, seed) == target:
            documents[key][field] = value

    return mutate


@pytest.mark.parametrize(
    ("key", "field", "value", "message"),
    [
        ("train", "seed", 42, r"^training run segformer_b2-seed-17 carries seed 42, expected 17$"),
        (
            "train",
            "run_id",
            "someone-else",
            r"^training run_id 'someone-else' does not match 'segformer_b2-seed-17'$",
        ),
        (
            "train",
            "protocol_sha256",
            "9" * 64,
            r"^training run segformer_b2-seed-17 was produced under a different protocol hash$",
        ),
        (
            "train",
            "artifacts",
            {},
            r"^training run segformer_b2-seed-17 recorded no final_checkpoint artifact$",
        ),
        (
            "temperature",
            "schema_version",
            "other/v9",
            r"^temperature for segformer_b2-seed-17 has the wrong schema_version$",
        ),
        (
            "temperature",
            "protocol_sha256",
            "9" * 64,
            r"^temperature for segformer_b2-seed-17 was fitted under another protocol$",
        ),
        (
            "temperature",
            "temperature",
            0.0,
            r"^temperature for segformer_b2-seed-17 must be finite and positive$",
        ),
        (
            "eval_raw",
            "status",
            "aborted",
            r"^uncalibrated evaluation of segformer_b2-seed-17 status must be succeeded, got 'aborted'$",
        ),
        (
            "eval_raw",
            "protocol_sha256",
            "9" * 64,
            r"^uncalibrated evaluation of segformer_b2-seed-17 was produced under a different protocol hash$",
        ),
        (
            "eval_cal",
            "seed",
            73,
            r"^calibrated evaluation of segformer_b2-seed-17 carries seed 73, expected 17$",
        ),
    ],
)
def test_every_binding_in_a_run_directory_is_checked(
    key: str,
    field: str,
    value: Any,
    message: str,
    tmp_path: Path,
    hashes: dict[str, Any],
) -> None:
    """Each record is bound to the others by hash and seed; one loose binding is one bad row."""

    with pytest.raises(ValueError, match=message):
        build(tmp_path, hashes, mutate=_mutating(("segformer_b2", 17), key, field, value))


def test_a_run_record_that_fails_its_schema_is_refused(
    tmp_path: Path, hashes: dict[str, Any]
) -> None:
    """A record the run-record schema rejects cannot anchor an index entry."""

    with pytest.raises(
        ValueError,
        match=r"is not a valid run record: 1 validation error for RunRecordV1\ncommit\n  String should match pattern '\^\[0-9a-f\]\{40\}\$'",
    ):
        build(tmp_path, hashes, mutate=_mutating(("segformer_b2", 17), "train", "commit", "x"))


def test_a_run_file_that_is_not_an_object_is_refused(
    tmp_path: Path, hashes: dict[str, Any]
) -> None:
    """A JSON list where a record belongs would reach the schema as positional garbage."""

    runs = build_runs(
        tmp_path / "runs", hashes["protocol"], hashes["locked"], hashes["calibration"]
    )
    (runs / "segformer_b2" / "seed-17" / "calibration" / "temperature.json").write_text(
        "[1.3]", encoding="utf-8"
    )

    with pytest.raises(TypeError, match=r"must contain a JSON object$"):
        load_index_module().build_formal_run_index(
            runs, PROTOCOL, hashes["locked_path"], runs / "x.json", critical_class_ids=CRITICAL
        )


@pytest.mark.parametrize(
    ("critical", "expected"),
    [
        ((), r"^critical_class_ids must be non-empty and unique$"),
        ((11, 11), r"^critical_class_ids must be non-empty and unique$"),
        ((True,), r"^critical_class_ids must be integers$"),
        ((99,), r"^critical class 99 is outside 0\.\.18$"),
    ],
)
def test_invalid_critical_class_ids_are_refused(
    critical: tuple[Any, ...], expected: str, tmp_path: Path, hashes: dict[str, Any]
) -> None:
    """Critical recall is defined over exact class ids; a wrong set is a wrong metric."""

    runs = build_runs(
        tmp_path / "runs", hashes["protocol"], hashes["locked"], hashes["calibration"]
    )

    with pytest.raises(ValueError, match=expected):
        load_index_module().build_formal_run_index(
            runs, PROTOCOL, hashes["locked_path"], runs / "x.json", critical_class_ids=critical
        )


def test_runs_that_do_not_share_one_cohort_are_refused_by_the_gate(
    tmp_path: Path, hashes: dict[str, Any]
) -> None:
    """Every per-run binding can hold while the matrix as a whole is still unpaired.

    The builder checks each run against the protocol and the locked manifest; only
    the gate sees all nine at once and can notice that one run scored different
    images. The builder must surface that refusal rather than write the index.
    """

    def mutate(model: str, seed: int, documents: dict[str, Any]) -> None:
        if (model, seed) == ("upernet_dinov2_small", 42):
            other = {"v0001": "d" * 64, "v9999": "d" * 64}
            documents["eval_raw"]["artifacts"] = dict(other)
            documents["eval_cal"]["artifacts"] = dict(other)

    with pytest.raises(ValueError, match=r"^assembled index failed its own gate: runs do not"):
        build(tmp_path, hashes, mutate=mutate)


def test_the_manifest_module_imports_cleanly_in_a_fresh_interpreter() -> None:
    """Importing the manifest module first must not trip a cycle through this package.

    The cycle only appears when `drivemetrics.data.manifest` is the first thing
    imported, so an in-process test that already loaded the package cannot see
    it. A fresh interpreter can.
    """

    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import drivemetrics.data.manifest"],
        check=False,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
