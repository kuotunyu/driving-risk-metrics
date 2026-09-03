"""Behavior tests for metric-sufficient prediction artifacts."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_predictions() -> ModuleType:
    try:
        return importlib.import_module("drivemetrics.artifacts.predictions")
    except ImportError:
        pytest.fail("drivemetrics.artifacts.predictions is missing", pytrace=False)


def load_calibration_metrics() -> ModuleType:
    try:
        return importlib.import_module("drivemetrics.metrics.calibration")
    except ImportError:
        pytest.fail("drivemetrics.metrics.calibration is missing", pytrace=False)


def valid_record() -> Any:
    prediction = load_predictions()
    metrics = load_calibration_metrics()
    correct = np.array([True, False, True, True, False, True], dtype=np.bool_)
    return prediction.PredictionRecord(
        sample_id="sample-001",
        predicted_class=np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8),
        top1_confidence_q16=np.array(
            [[65535, 49151, 32768], [60000, 40000, 50000]],
            dtype=np.uint16,
        ),
        correctness_bitset=metrics.pack_correctness(correct),
        confusion=np.array([[2, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.int64),
        brier_sum_by_class=np.array([0.12, 0.34, 0.56], dtype=np.float64),
        valid_pixel_count=6,
    )


def valid_ece_stats() -> Any:
    metrics = load_calibration_metrics()
    return metrics.ECEBinSufficientStatistics(
        counts=np.array([[3, 3], [4, 2], [5, 1]], dtype=np.int64),
        confidence_sums=np.array([[0.6, 2.1], [0.8, 1.2], [0.8, 0.5]], dtype=np.float64),
        positive_counts=np.array([[1, 1], [1, 1], [2, 0]], dtype=np.int64),
    )


def write_valid_artifact(path: Path) -> Any:
    prediction = load_predictions()
    return prediction.write_prediction_artifact(
        path,
        valid_record(),
        valid_ece_stats(),
        protocol_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
    )


def test_prediction_artifact_round_trip_preserves_every_metric_sufficient_value(
    tmp_path: Path,
) -> None:
    """Dropping or coercing any stored statistic would prevent exact downstream recomputation."""

    prediction = load_predictions()
    path = tmp_path / "sample-001.json"

    manifest = write_valid_artifact(path)
    loaded_manifest, loaded_record, loaded_ece = prediction.read_prediction_artifact(path)

    assert loaded_manifest == manifest
    assert manifest.confidence_scale == 65535
    assert manifest.payload_file == f"sample-001.{manifest.payload_sha256}.npz"
    assert loaded_record.sample_id == "sample-001"
    assert loaded_record.valid_pixel_count == 6
    np.testing.assert_array_equal(loaded_record.predicted_class, valid_record().predicted_class)
    np.testing.assert_array_equal(
        loaded_record.top1_confidence_q16,
        valid_record().top1_confidence_q16,
    )
    assert loaded_record.correctness_bitset == valid_record().correctness_bitset
    np.testing.assert_array_equal(loaded_record.confusion, valid_record().confusion)
    np.testing.assert_array_equal(
        loaded_record.brier_sum_by_class, valid_record().brier_sum_by_class
    )
    np.testing.assert_array_equal(loaded_ece.counts, valid_ece_stats().counts)
    np.testing.assert_array_equal(loaded_ece.confidence_sums, valid_ece_stats().confidence_sums)
    np.testing.assert_array_equal(loaded_ece.positive_counts, valid_ece_stats().positive_counts)


def test_metric_accepted_near_normalized_probabilities_produce_a_valid_artifact(
    tmp_path: Path,
) -> None:
    """Public metric output must remain serializable at its declared row-sum tolerance."""

    prediction = load_predictions()
    metrics = load_calibration_metrics()
    probabilities = np.array([[0.49999999999955, 0.49999999999955]], dtype=np.float64)
    targets = np.array([0], dtype=np.int64)
    predicted_class = np.argmax(probabilities, axis=1).astype(np.uint8)
    correct = predicted_class == targets
    record = prediction.PredictionRecord(
        sample_id="near-normalized",
        predicted_class=predicted_class,
        top1_confidence_q16=metrics.quantize_confidence(probabilities.max(axis=1)),
        correctness_bitset=metrics.pack_correctness(correct),
        confusion=np.array([[1, 0], [0, 0]], dtype=np.int64),
        brier_sum_by_class=metrics.multiclass_brier_sums(
            probabilities,
            targets,
            num_classes=2,
        ),
        valid_pixel_count=1,
    )
    ece = metrics.classwise_ece_sufficient_statistics(
        probabilities,
        targets,
        num_classes=2,
        bin_count=2,
    )

    manifest = prediction.write_prediction_artifact(
        tmp_path / "near-normalized.json",
        record,
        ece,
        protocol_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
    )

    assert manifest.valid_pixel_count == 1


def test_prediction_manifest_is_canonical_json_with_hashes_and_array_descriptors(
    tmp_path: Path,
) -> None:
    """An ambiguous manifest would make artifact review and payload verification unreliable."""

    prediction = load_predictions()
    path = tmp_path / "artifact.json"

    manifest = write_valid_artifact(path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    assert path.read_bytes() == prediction.canonical_json_bytes(raw) + b"\n"
    assert manifest.protocol_sha256 == "a" * 64
    assert manifest.dataset_manifest_sha256 == "b" * 64
    assert len(manifest.payload_sha256) == 64
    assert manifest.arrays.predicted_class.dtype == "uint8"
    assert manifest.arrays.correctness_bitset.shape == (1,)
    assert manifest.arrays.ece_counts.shape == (3, 2)


def test_hard_mask_png_alone_fails_prediction_schema_validation() -> None:
    """A hard PNG cannot recover confidence, correctness, Brier, or ECE evidence."""

    prediction = load_predictions()

    with pytest.raises(ValidationError):
        prediction.PredictionArtifactV1.model_validate(
            {
                "schema_version": "driving-risk-prediction-artifact/v1",
                "sample_id": "sample-001",
                "payload_file": "mask.png",
                "payload_sha256": "c" * 64,
                "protocol_sha256": "a" * 64,
                "dataset_manifest_sha256": "b" * 64,
            }
        )


def test_array_descriptor_rejects_empty_or_negative_shape() -> None:
    """A descriptor without a realizable array shape cannot validate an NPZ member."""

    prediction = load_predictions()

    for shape in [(), (-1,)]:
        with pytest.raises(ValidationError):
            prediction.ArrayDescriptorV1(dtype="uint8", shape=shape)


def test_prediction_manifest_requires_every_metric_sufficient_array(tmp_path: Path) -> None:
    """A manifest missing one payload member must fail even when every other field is valid."""

    prediction = load_predictions()
    path = tmp_path / "artifact.json"
    manifest = write_valid_artifact(path)
    values = manifest.model_dump(mode="json")
    del values["arrays"]["ece_counts"]

    with pytest.raises(ValidationError):
        prediction.PredictionArtifactV1.model_validate(values)


def test_exported_schema_requires_fixed_arrays_and_nonnegative_nonempty_shapes() -> None:
    """External JSON Schema consumers must enforce the same payload completeness as runtime."""

    schemas = importlib.import_module("drivemetrics.artifacts.schemas")
    schema = json.loads(schemas.contract_schema_documents()["prediction_artifact_v1.json"])
    arrays_definition = schema["$defs"]["PredictionArraysV1"]

    assert arrays_definition["additionalProperties"] is False
    assert set(arrays_definition["required"]) == {
        "predicted_class",
        "top1_confidence_q16",
        "correctness_bitset",
        "confusion",
        "brier_sum_by_class",
        "ece_counts",
        "ece_confidence_sums",
        "ece_positive_counts",
    }
    expected_dtypes = {
        "predicted_class": "uint8",
        "top1_confidence_q16": "uint16",
        "correctness_bitset": "uint8",
        "confusion": "int64",
        "brier_sum_by_class": "float64",
        "ece_counts": "int64",
        "ece_confidence_sums": "float64",
        "ece_positive_counts": "int64",
    }
    for field_name, expected_dtype in expected_dtypes.items():
        descriptor_reference = arrays_definition["properties"][field_name]["$ref"]
        specialized_definition = schema["$defs"][descriptor_reference.rsplit("/", 1)[-1]]
        assert specialized_definition["properties"]["dtype"]["const"] == expected_dtype
        assert specialized_definition["properties"]["shape"]["minItems"] == 1
        assert specialized_definition["properties"]["shape"]["items"]["minimum"] == 0


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "top1_confidence_q16",
            np.array([1, 2], dtype=np.uint16),
            "predicted_class and top1_confidence_q16 shapes",
        ),
        ("valid_pixel_count", 5, "valid_pixel_count"),
        ("correctness_bitset", b"", "byte length"),
        ("confusion", np.zeros((2, 3), dtype=np.int64), "confusion shape"),
        ("brier_sum_by_class", np.zeros(2, dtype=np.float64), "brier_sum_by_class shape"),
    ],
)
def test_writer_rejects_shape_count_or_truncated_bitset_corruption(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    """Structurally inconsistent payloads must fail before a manifest is published."""

    prediction = load_predictions()
    corrupt = replace(valid_record(), **{field: replacement})

    with pytest.raises(ValueError, match=message):
        prediction.write_prediction_artifact(
            tmp_path / "bad.json",
            corrupt,
            valid_ece_stats(),
            protocol_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
        )


def test_writer_rejects_ece_shape_mismatch(tmp_path: Path) -> None:
    """ECE tables for a different class set cannot accompany the same prediction record."""

    prediction = load_predictions()
    metrics = load_calibration_metrics()
    corrupt_ece = metrics.ECEBinSufficientStatistics(
        counts=np.zeros((2, 2), dtype=np.int64),
        confidence_sums=np.zeros((2, 2), dtype=np.float64),
        positive_counts=np.zeros((2, 2), dtype=np.int64),
    )

    with pytest.raises(ValueError, match=r"^ECE shape must be num_classes by at least one bin"):
        prediction.write_prediction_artifact(
            tmp_path / "bad.json",
            valid_record(),
            corrupt_ece,
            protocol_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
        )


def test_writer_rejects_manifest_payload_path_collision(tmp_path: Path) -> None:
    """A manifest path must never be allowed to overwrite its own binary payload."""

    prediction = load_predictions()

    with pytest.raises(ValueError, match=r"\.json"):
        prediction.write_prediction_artifact(
            tmp_path / "artifact.npz",
            valid_record(),
            valid_ece_stats(),
            protocol_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
        )


def changed_record() -> Any:
    return replace(
        valid_record(),
        brier_sum_by_class=np.array([0.22, 0.44, 0.66], dtype=np.float64),
    )


def assert_existing_artifact_unchanged(
    prediction: ModuleType,
    manifest_path: Path,
    manifest_bytes: bytes,
    payload_path: Path,
    payload_bytes: bytes,
) -> None:
    assert manifest_path.read_bytes() == manifest_bytes
    assert payload_path.read_bytes() == payload_bytes
    _, record, _ = prediction.read_prediction_artifact(manifest_path)
    np.testing.assert_array_equal(record.brier_sum_by_class, valid_record().brier_sum_by_class)


def test_invalid_metadata_does_not_damage_existing_artifact(tmp_path: Path) -> None:
    """Manifest validation failure must occur without replacing either published file."""

    prediction = load_predictions()
    manifest_path = tmp_path / "artifact.json"
    manifest = write_valid_artifact(manifest_path)
    payload_path = tmp_path / manifest.payload_file
    manifest_bytes = manifest_path.read_bytes()
    payload_bytes = payload_path.read_bytes()

    with pytest.raises(ValidationError):
        prediction.write_prediction_artifact(
            manifest_path,
            changed_record(),
            valid_ece_stats(),
            protocol_sha256="invalid",
            dataset_manifest_sha256="b" * 64,
        )

    assert_existing_artifact_unchanged(
        prediction,
        manifest_path,
        manifest_bytes,
        payload_path,
        payload_bytes,
    )


def test_payload_write_failure_does_not_damage_existing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure after binary serialization must affect only an unpublished temporary file."""

    prediction = load_predictions()
    manifest_path = tmp_path / "artifact.json"
    manifest = write_valid_artifact(manifest_path)
    payload_path = tmp_path / manifest.payload_file
    manifest_bytes = manifest_path.read_bytes()
    payload_bytes = payload_path.read_bytes()
    real_savez = prediction.np.savez

    def write_then_fail(path: object, **arrays: np.ndarray) -> None:
        real_savez(path, **arrays)
        raise OSError("simulated payload failure")

    monkeypatch.setattr(prediction.np, "savez", write_then_fail)

    with pytest.raises(OSError, match=r"^simulated payload failure"):
        prediction.write_prediction_artifact(
            manifest_path,
            changed_record(),
            valid_ece_stats(),
            protocol_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
        )

    assert_existing_artifact_unchanged(
        prediction,
        manifest_path,
        manifest_bytes,
        payload_path,
        payload_bytes,
    )


def test_manifest_publish_failure_does_not_damage_existing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publishing a new payload without its manifest must leave the old pair consistent."""

    prediction = load_predictions()
    manifest_path = tmp_path / "artifact.json"
    manifest = write_valid_artifact(manifest_path)
    payload_path = tmp_path / manifest.payload_file
    manifest_bytes = manifest_path.read_bytes()
    payload_bytes = payload_path.read_bytes()
    real_replace = os.replace

    def fail_manifest_replace(
        source: str | os.PathLike[str], destination: str | os.PathLike[str]
    ) -> None:
        if Path(destination) == manifest_path:
            raise OSError("simulated manifest publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_manifest_replace)

    with pytest.raises(OSError, match=r"^simulated manifest publication failure"):
        prediction.write_prediction_artifact(
            manifest_path,
            changed_record(),
            valid_ece_stats(),
            protocol_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
        )

    assert_existing_artifact_unchanged(
        prediction,
        manifest_path,
        manifest_bytes,
        payload_path,
        payload_bytes,
    )


def test_identical_write_reuses_verified_content_addressed_payload(tmp_path: Path) -> None:
    """Retrying identical output must validate and reuse the immutable payload bytes."""

    manifest_path = tmp_path / "artifact.json"
    first = write_valid_artifact(manifest_path)
    payload_path = tmp_path / first.payload_file
    payload_bytes = payload_path.read_bytes()

    second = write_valid_artifact(manifest_path)

    assert second.payload_file == first.payload_file
    assert payload_path.read_bytes() == payload_bytes


def test_writer_rejects_corrupt_existing_content_addressed_payload(tmp_path: Path) -> None:
    """An occupied hash-derived name with different bytes must never be overwritten."""

    manifest_path = tmp_path / "artifact.json"
    first = write_valid_artifact(manifest_path)
    payload_path = tmp_path / first.payload_file
    payload_path.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match=r"^existing content-addressed payload hash mismatch"):
        write_valid_artifact(manifest_path)


@pytest.mark.parametrize(
    ("record_update", "message"),
    [
        ({"sample_id": ""}, "sample_id"),
        ({"predicted_class": np.zeros(6, dtype=np.int64)}, "predicted_class must have dtype"),
        (
            {"top1_confidence_q16": np.zeros(6, dtype=np.int64)},
            "top1_confidence_q16 must have dtype",
        ),
        (
            {"confusion": np.array([[2, -1, 0], [0, 2, 1], [0, 1, 1]], dtype=np.int64)},
            "confusion counts",
        ),
        (
            {"predicted_class": np.array([[3, 1, 2], [2, 1, 0]], dtype=np.uint8)},
            "outside confusion shape",
        ),
        ({"brier_sum_by_class": np.array([0.1, np.nan, 0.2])}, "finite nonnegative"),
    ],
)
def test_writer_rejects_invalid_record_values(
    tmp_path: Path,
    record_update: dict[str, object],
    message: str,
) -> None:
    """Wrong dtypes, ranges, and counts must not be serialized as valid evidence."""

    prediction = load_predictions()

    with pytest.raises(ValueError, match=message):
        prediction.write_prediction_artifact(
            tmp_path / "bad.json",
            replace(valid_record(), **record_update),
            valid_ece_stats(),
            protocol_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
        )


def test_writer_rejects_more_than_uint8_class_capacity(tmp_path: Path) -> None:
    """A class table larger than uint8 storage can represent must fail closed."""

    prediction = load_predictions()
    metrics = load_calibration_metrics()
    confusion = np.zeros((257, 257), dtype=np.int64)
    confusion[0, 0] = 1
    record = prediction.PredictionRecord(
        sample_id="sample-001",
        predicted_class=np.array([0], dtype=np.uint8),
        top1_confidence_q16=np.array([65535], dtype=np.uint16),
        correctness_bitset=metrics.pack_correctness(np.array([True], dtype=np.bool_)),
        confusion=confusion,
        brier_sum_by_class=np.zeros(257, dtype=np.float64),
        valid_pixel_count=1,
    )
    ece = metrics.ECEBinSufficientStatistics(
        counts=np.ones((257, 1), dtype=np.int64),
        confidence_sums=np.zeros((257, 1), dtype=np.float64),
        positive_counts=np.zeros((257, 1), dtype=np.int64),
    )

    with pytest.raises(ValueError, match=r"^confusion shape must define between"):
        prediction.write_prediction_artifact(
            tmp_path / "bad.json",
            record,
            ece,
            protocol_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    ("ece_update", "message"),
    [
        ({"confidence_sums": np.zeros((3, 3), dtype=np.float64)}, "ECE shape"),
        ({"counts": np.array([[-1, 7], [4, 2], [5, 1]], dtype=np.int64)}, "nonnegative"),
        (
            {"positive_counts": np.array([[4, 1], [1, 1], [2, 0]], dtype=np.int64)},
            "must not exceed",
        ),
        (
            {"confidence_sums": np.array([[np.nan, 2.1], [0.8, 1.2], [0.8, 0.5]])},
            "finite and nonnegative",
        ),
        (
            {"confidence_sums": np.array([[3.1, 2.1], [0.8, 1.2], [0.8, 0.5]])},
            "must not exceed counts",
        ),
        (
            {"counts": np.array([[2, 3], [4, 2], [5, 1]], dtype=np.int64)},
            "sum to valid_pixel_count",
        ),
    ],
)
def test_writer_rejects_invalid_ece_values(
    tmp_path: Path,
    ece_update: dict[str, object],
    message: str,
) -> None:
    """Inconsistent ECE sufficient statistics must fail before artifact publication."""

    prediction = load_predictions()

    with pytest.raises(ValueError, match=message):
        prediction.write_prediction_artifact(
            tmp_path / "bad.json",
            valid_record(),
            replace(valid_ece_stats(), **ece_update),
            protocol_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
        )


def test_writer_rejects_predicted_histogram_that_disagrees_with_confusion(
    tmp_path: Path,
) -> None:
    """Predicted masks and confusion columns must describe the same class totals."""

    prediction = load_predictions()
    corrupt = replace(
        valid_record(),
        predicted_class=np.array([[0, 0, 0], [2, 1, 0]], dtype=np.uint8),
    )

    with pytest.raises(ValueError, match=r"^predicted-class histogram must equal confusion column"):
        prediction.write_prediction_artifact(
            tmp_path / "bad.json",
            corrupt,
            valid_ece_stats(),
            protocol_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
        )


def test_writer_rejects_correctness_popcount_that_disagrees_with_confusion(
    tmp_path: Path,
) -> None:
    """Correctness bits and the confusion diagonal must yield one exact correct count."""

    prediction = load_predictions()
    metrics = load_calibration_metrics()
    corrupt = replace(
        valid_record(),
        correctness_bitset=metrics.pack_correctness(np.ones(6, dtype=np.bool_)),
    )

    with pytest.raises(
        ValueError, match=r"^correctness popcount must equal the confusion diagonal"
    ):
        prediction.write_prediction_artifact(
            tmp_path / "bad.json",
            corrupt,
            valid_ece_stats(),
            protocol_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
        )


def test_writer_rejects_ece_targets_that_disagree_with_confusion(tmp_path: Path) -> None:
    """Classwise ECE positives and confusion rows must describe identical target support."""

    prediction = load_predictions()
    corrupt = replace(
        valid_ece_stats(),
        positive_counts=np.array([[1, 1], [1, 0], [2, 0]], dtype=np.int64),
    )

    with pytest.raises(ValueError, match=r"^ECE positive totals must equal confusion row totals"):
        prediction.write_prediction_artifact(
            tmp_path / "bad.json",
            valid_record(),
            corrupt,
            protocol_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
        )


def test_writer_rejects_impossible_brier_or_top1_confidence(tmp_path: Path) -> None:
    """Per-pixel squared errors and top-1 probabilities have strict mathematical bounds."""

    prediction = load_predictions()

    for corrupt, message in [
        (
            replace(
                valid_record(),
                brier_sum_by_class=np.array([7.0, 0.2, 0.3], dtype=np.float64),
            ),
            "Brier",
        ),
        (
            replace(
                valid_record(),
                top1_confidence_q16=np.array(
                    [[100, 49151, 32768], [60000, 40000, 50000]], dtype=np.uint16
                ),
            ),
            "top1 confidence",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            prediction.write_prediction_artifact(
                tmp_path / "bad.json",
                corrupt,
                valid_ece_stats(),
                protocol_sha256="a" * 64,
                dataset_manifest_sha256="b" * 64,
            )


def test_writer_rejects_ece_confidence_totals_that_cannot_sum_to_probabilities(
    tmp_path: Path,
) -> None:
    """Across classes, ECE confidence sums must equal one probability mass per pixel."""

    prediction = load_predictions()
    corrupt = replace(
        valid_ece_stats(),
        confidence_sums=np.array([[0.6, 2.1], [0.8, 1.2], [1.0, 0.5]]),
    )

    with pytest.raises(
        ValueError, match=r"^ECE global confidence sum must equal valid_pixel_count"
    ):
        prediction.write_prediction_artifact(
            tmp_path / "bad.json",
            valid_record(),
            corrupt,
            protocol_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
        )


def test_writer_rejects_ece_confidence_sums_outside_declared_bins(tmp_path: Path) -> None:
    """A fixed-bin aggregate must be achievable by values inside that bin's interval."""

    prediction = load_predictions()
    corrupt = replace(
        valid_ece_stats(),
        confidence_sums=np.array([[1.6, 1.1], [0.8, 1.2], [0.8, 0.5]]),
    )

    with pytest.raises(
        ValueError, match=r"^ECE confidence_sums must be achievable within declared"
    ):
        prediction.write_prediction_artifact(
            tmp_path / "bad.json",
            valid_record(),
            corrupt,
            protocol_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
        )


def test_writer_rejects_impossible_total_multiclass_brier_sum(tmp_path: Path) -> None:
    """Multiclass Brier error for one pixel cannot exceed two."""

    prediction = load_predictions()
    corrupt = replace(
        valid_record(),
        brier_sum_by_class=np.array([5.0, 5.0, 5.0], dtype=np.float64),
    )

    with pytest.raises(
        ValueError, match=r"^total Brier sum must not exceed twice valid_pixel_count"
    ):
        prediction.write_prediction_artifact(
            tmp_path / "bad.json",
            corrupt,
            valid_ece_stats(),
            protocol_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
        )


def test_reader_rejects_wrong_payload_hash(tmp_path: Path) -> None:
    """Any byte-level NPZ change must invalidate the manifest before array loading."""

    prediction = load_predictions()
    manifest_path = tmp_path / "artifact.json"
    manifest = write_valid_artifact(manifest_path)
    payload_path = tmp_path / manifest.payload_file
    payload_path.write_bytes(payload_path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match=r"^payload SHA-256 mismatch"):
        prediction.read_prediction_artifact(manifest_path)


def test_reader_loads_the_same_payload_bytes_whose_hash_was_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacing the payload path after hashing must not change returned metric arrays."""

    prediction = load_predictions()
    manifest_path = tmp_path / "artifact.json"
    manifest = write_valid_artifact(manifest_path)
    payload_path = tmp_path / manifest.payload_file
    alternate_arrays = loaded_payload_arrays(payload_path)
    alternate_arrays["brier_sum_by_class"] = np.array([0.22, 0.44, 0.66])
    alternate_stream = io.BytesIO()
    np.savez(alternate_stream, **alternate_arrays)  # type: ignore[arg-type]
    alternate_bytes = alternate_stream.getvalue()
    real_load = prediction.np.load

    def replace_path_then_load(source: object, **kwargs: object) -> Any:
        payload_path.write_bytes(alternate_bytes)
        return real_load(source, **kwargs)

    monkeypatch.setattr(prediction.np, "load", replace_path_then_load)

    _, record, _ = prediction.read_prediction_artifact(manifest_path)

    np.testing.assert_array_equal(record.brier_sum_by_class, valid_record().brier_sum_by_class)


def rewrite_manifest(path: Path, **updates: object) -> None:
    values = json.loads(path.read_text(encoding="utf-8"))
    values.update(updates)
    path.write_text(json.dumps(values), encoding="utf-8")


def rewrite_payload(path: Path, arrays: dict[str, np.ndarray]) -> str:
    np.savez(path, **arrays)  # type: ignore[arg-type]
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loaded_payload_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files}


def test_reader_rejects_missing_npz_member_with_matching_tampered_hash(tmp_path: Path) -> None:
    """Rehashing an incomplete payload must not make it structurally valid."""

    prediction = load_predictions()
    manifest_path = tmp_path / "artifact.json"
    manifest = write_valid_artifact(manifest_path)
    payload_path = tmp_path / manifest.payload_file
    arrays = loaded_payload_arrays(payload_path)
    del arrays["ece_counts"]
    rewrite_manifest(manifest_path, payload_sha256=rewrite_payload(payload_path, arrays))

    with pytest.raises(
        ValueError, match=r"^NPZ members do not match the manifest payload contract"
    ):
        prediction.read_prediction_artifact(manifest_path)


def test_reader_rejects_array_descriptor_mismatch(tmp_path: Path) -> None:
    """A descriptor that lies about payload shape must fail after hash verification."""

    prediction = load_predictions()
    manifest_path = tmp_path / "artifact.json"
    write_valid_artifact(manifest_path)
    values = json.loads(manifest_path.read_text(encoding="utf-8"))
    values["arrays"]["predicted_class"]["shape"] = [6]
    manifest_path.write_text(json.dumps(values), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^NPZ member 'predicted_class' does not match its array"):
        prediction.read_prediction_artifact(manifest_path)


def test_reader_rejects_manifest_dimension_mismatch(tmp_path: Path) -> None:
    """Manifest dimensions must agree with already hash-verified payload arrays."""

    prediction = load_predictions()
    manifest_path = tmp_path / "artifact.json"
    write_valid_artifact(manifest_path)
    rewrite_manifest(manifest_path, num_classes=2)

    with pytest.raises(ValueError, match=r"^manifest dimensions do not match the verified NPZ"):
        prediction.read_prediction_artifact(manifest_path)


@pytest.mark.parametrize("field", ["protocol_sha256", "dataset_manifest_sha256", "payload_sha256"])
def test_prediction_manifest_rejects_wrong_hash_shape(field: str) -> None:
    """A truncated provenance hash is not an acceptable artifact identity."""

    prediction = load_predictions()
    values = {
        "schema_version": "driving-risk-prediction-artifact/v1",
        "sample_id": "sample-001",
        "protocol_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "payload_file": "sample-001.npz",
        "payload_sha256": "c" * 64,
        "confidence_scale": 65535,
        "valid_pixel_count": 1,
        "num_classes": 1,
        "ece_bin_count": 1,
        "arrays": {},
    }
    values[field] = "0" * 63

    with pytest.raises(ValidationError):
        prediction.PredictionArtifactV1.model_validate(values)


def test_reader_rejects_payload_path_escape(tmp_path: Path) -> None:
    """A manifest must not make the reader open a payload outside its artifact directory."""

    prediction = load_predictions()
    values = {
        "schema_version": "driving-risk-prediction-artifact/v1",
        "sample_id": "sample-001",
        "protocol_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "payload_file": "../outside.npz",
        "payload_sha256": "c" * 64,
        "confidence_scale": 65535,
        "valid_pixel_count": 1,
        "num_classes": 1,
        "ece_bin_count": 1,
        "arrays": {},
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(values), encoding="utf-8")

    with pytest.raises(ValidationError):
        prediction.read_prediction_artifact(path)


def test_checked_in_prediction_schema_matches_generated_model_bytes() -> None:
    """A hand-edited schema must not drift from the runtime artifact validator."""

    schemas = importlib.import_module("drivemetrics.artifacts.schemas")

    assert (REPO_ROOT / "schemas" / "prediction_artifact_v1.json").read_bytes() == (
        schemas.contract_schema_documents()["prediction_artifact_v1.json"]
    )
