"""Hash-verified metric-sufficient prediction manifest and NPZ payload."""

from __future__ import annotations

import hashlib
import io
import os
import re
import secrets
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from drivemetrics.metrics.calibration import (
    PROBABILITY_ROW_SUM_ATOLERANCE,
    ECEBinSufficientStatistics,
    unpack_correctness,
)

from .envelope import canonical_json_bytes

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]
UInt8Array = npt.NDArray[np.uint8]
UInt16Array = npt.NDArray[np.uint16]

_PAYLOAD_KEYS = frozenset(
    {
        "predicted_class",
        "top1_confidence_q16",
        "correctness_bitset",
        "confusion",
        "brier_sum_by_class",
        "ece_counts",
        "ece_confidence_sums",
        "ece_positive_counts",
    }
)
_SAFE_STEM_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
NonNegativeDimension = Annotated[int, Field(ge=0)]


@dataclass(frozen=True)
class PredictionRecord:
    """Metric-sufficient arrays retained for one evaluated image."""

    sample_id: str
    predicted_class: UInt8Array
    top1_confidence_q16: UInt16Array
    correctness_bitset: bytes
    confusion: Int64Array
    brier_sum_by_class: Float64Array
    valid_pixel_count: int


class ArrayDescriptorV1(BaseModel):
    """Expected dtype and shape of one named NPZ member."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dtype: Literal["uint8", "uint16", "int64", "float64"]
    shape: tuple[NonNegativeDimension, ...] = Field(min_length=1)


class UInt8ArrayDescriptorV1(ArrayDescriptorV1):
    """Descriptor whose payload member must be uint8."""

    dtype: Literal["uint8"]


class UInt16ArrayDescriptorV1(ArrayDescriptorV1):
    """Descriptor whose payload member must be uint16."""

    dtype: Literal["uint16"]


class Int64ArrayDescriptorV1(ArrayDescriptorV1):
    """Descriptor whose payload member must be int64."""

    dtype: Literal["int64"]


class Float64ArrayDescriptorV1(ArrayDescriptorV1):
    """Descriptor whose payload member must be float64."""

    dtype: Literal["float64"]


class PredictionArraysV1(BaseModel):
    """Fixed descriptors for every metric-sufficient NPZ member."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    predicted_class: UInt8ArrayDescriptorV1
    top1_confidence_q16: UInt16ArrayDescriptorV1
    correctness_bitset: UInt8ArrayDescriptorV1
    confusion: Int64ArrayDescriptorV1
    brier_sum_by_class: Float64ArrayDescriptorV1
    ece_counts: Int64ArrayDescriptorV1
    ece_confidence_sums: Float64ArrayDescriptorV1
    ece_positive_counts: Int64ArrayDescriptorV1


class PredictionArtifactV1(BaseModel):
    """Strict JSON manifest for one prediction NPZ payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["driving-risk-prediction-artifact/v1"]
    sample_id: str = Field(min_length=1)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_file: str = Field(pattern=r"^[A-Za-z0-9._-]+\.npz$")
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confidence_scale: Literal[65535]
    valid_pixel_count: int = Field(gt=0)
    num_classes: int = Field(ge=1, le=256)
    ece_bin_count: int = Field(gt=0)
    arrays: PredictionArraysV1


def _require_array(
    name: str,
    value: np.ndarray,
    dtype: np.dtype[np.generic],
) -> None:
    if not isinstance(value, np.ndarray) or value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype.name}")


def _validate_prediction_values(
    record: PredictionRecord,
    ece: ECEBinSufficientStatistics,
) -> tuple[int, int]:
    if not record.sample_id:
        raise ValueError("sample_id must not be empty")
    _require_array("predicted_class", record.predicted_class, np.dtype(np.uint8))
    _require_array(
        "top1_confidence_q16",
        record.top1_confidence_q16,
        np.dtype(np.uint16),
    )
    if record.predicted_class.shape != record.top1_confidence_q16.shape:
        raise ValueError("predicted_class and top1_confidence_q16 shapes must match")
    if record.valid_pixel_count <= 0 or record.predicted_class.size != record.valid_pixel_count:
        raise ValueError("valid_pixel_count must equal the stored prediction count")
    correctness = unpack_correctness(record.correctness_bitset, record.valid_pixel_count)

    _require_array("confusion", record.confusion, np.dtype(np.int64))
    if record.confusion.ndim != 2 or record.confusion.shape[0] != record.confusion.shape[1]:
        raise ValueError("confusion shape must be square")
    num_classes = record.confusion.shape[0]
    if num_classes < 1 or num_classes > 256:
        raise ValueError("confusion shape must define between 1 and 256 classes")
    if np.any(record.confusion < 0) or int(record.confusion.sum()) != record.valid_pixel_count:
        raise ValueError("confusion counts must be nonnegative and sum to valid_pixel_count")
    if np.any(record.predicted_class >= num_classes):
        raise ValueError("predicted_class contains a class outside confusion shape")
    predicted_histogram = np.bincount(
        record.predicted_class.reshape(-1),
        minlength=num_classes,
    )
    if not np.array_equal(predicted_histogram, record.confusion.sum(axis=0)):
        raise ValueError("predicted-class histogram must equal confusion column totals")
    if int(np.count_nonzero(correctness)) != int(np.trace(record.confusion)):
        raise ValueError("correctness popcount must equal the confusion diagonal total")
    minimum_top1_probability = (1.0 - PROBABILITY_ROW_SUM_ATOLERANCE) / num_classes
    minimum_top1_q16 = int(np.rint(65535 * minimum_top1_probability))
    if np.any(record.top1_confidence_q16 < minimum_top1_q16):
        raise ValueError("top1 confidence is below the multiclass probability lower bound")

    _require_array("brier_sum_by_class", record.brier_sum_by_class, np.dtype(np.float64))
    if record.brier_sum_by_class.shape != (num_classes,):
        raise ValueError("brier_sum_by_class shape must match num_classes")
    if not np.all(np.isfinite(record.brier_sum_by_class)) or np.any(
        record.brier_sum_by_class < 0.0
    ):
        raise ValueError("brier_sum_by_class must contain finite nonnegative values")
    if np.any(record.brier_sum_by_class > record.valid_pixel_count):
        raise ValueError("Brier class sums must not exceed valid_pixel_count")
    numerical_tolerance = 64.0 * np.finfo(np.float64).eps * max(1, record.valid_pixel_count)
    probability_mass_tolerance = (
        PROBABILITY_ROW_SUM_ATOLERANCE * record.valid_pixel_count + numerical_tolerance
    )
    if float(record.brier_sum_by_class.sum()) > (
        2.0 * record.valid_pixel_count + numerical_tolerance
    ):
        raise ValueError("total Brier sum must not exceed twice valid_pixel_count")

    _require_array("ECE counts", ece.counts, np.dtype(np.int64))
    _require_array("ECE confidence_sums", ece.confidence_sums, np.dtype(np.float64))
    _require_array("ECE positive_counts", ece.positive_counts, np.dtype(np.int64))
    if ece.counts.ndim != 2 or ece.counts.shape[0] != num_classes or ece.counts.shape[1] < 1:
        raise ValueError("ECE shape must be num_classes by at least one bin")
    if (
        ece.confidence_sums.shape != ece.counts.shape
        or ece.positive_counts.shape != ece.counts.shape
    ):
        raise ValueError("ECE shape must match for every sufficient-statistic array")
    if np.any(ece.counts < 0) or np.any(ece.positive_counts < 0):
        raise ValueError("ECE counts must be nonnegative")
    if np.any(ece.positive_counts > ece.counts):
        raise ValueError("ECE positive_counts must not exceed counts")
    if not np.all(np.isfinite(ece.confidence_sums)) or np.any(ece.confidence_sums < 0.0):
        raise ValueError("ECE confidence_sums must be finite and nonnegative")
    if np.any(ece.confidence_sums > ece.counts.astype(np.float64) + 1e-12):
        raise ValueError("ECE confidence_sums must not exceed counts")
    ece_bin_count = ece.counts.shape[1]
    lower_edges = np.arange(ece_bin_count, dtype=np.float64) / ece_bin_count
    upper_edges = (np.arange(ece_bin_count, dtype=np.float64) + 1.0) / ece_bin_count
    lower_confidence_sums = ece.counts * lower_edges[np.newaxis, :]
    upper_confidence_sums = ece.counts * upper_edges[np.newaxis, :]
    if np.any(ece.confidence_sums < lower_confidence_sums - numerical_tolerance) or np.any(
        ece.confidence_sums > upper_confidence_sums + numerical_tolerance
    ):
        raise ValueError("ECE confidence_sums must be achievable within declared bin intervals")
    if not np.isclose(
        float(ece.confidence_sums.sum()),
        record.valid_pixel_count,
        rtol=0.0,
        atol=probability_mass_tolerance,
    ):
        raise ValueError("ECE global confidence sum must equal valid_pixel_count")
    if np.any(ece.counts.sum(axis=1) != record.valid_pixel_count):
        raise ValueError("ECE class counts must each sum to valid_pixel_count")
    if not np.array_equal(ece.positive_counts.sum(axis=1), record.confusion.sum(axis=1)):
        raise ValueError("ECE positive totals must equal confusion row totals")
    return num_classes, ece_bin_count


def write_prediction_artifact(
    manifest_path: Path,
    record: PredictionRecord,
    ece: ECEBinSufficientStatistics,
    *,
    protocol_sha256: str,
    dataset_manifest_sha256: str,
) -> PredictionArtifactV1:
    """Atomically publish a canonical manifest after its immutable NPZ payload."""

    num_classes, ece_bin_count = _validate_prediction_values(record, ece)
    if manifest_path.suffix != ".json" or _SAFE_STEM_PATTERN.fullmatch(manifest_path.stem) is None:
        raise ValueError("manifest_path must have a safe filename ending in .json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload_arrays: dict[str, np.ndarray] = {
        "predicted_class": record.predicted_class,
        "top1_confidence_q16": record.top1_confidence_q16,
        "correctness_bitset": np.frombuffer(record.correctness_bitset, dtype=np.uint8),
        "confusion": record.confusion,
        "brier_sum_by_class": record.brier_sum_by_class,
        "ece_counts": ece.counts,
        "ece_confidence_sums": ece.confidence_sums,
        "ece_positive_counts": ece.positive_counts,
    }
    payload_handle, payload_temporary_name = tempfile.mkstemp(
        dir=manifest_path.parent,
        prefix=f".{manifest_path.stem}.",
        suffix=".npz.tmp",
    )
    os.close(payload_handle)
    payload_temporary_path = Path(payload_temporary_name)
    manifest_temporary_path: Path | None = None
    try:
        with payload_temporary_path.open("wb") as payload_stream:
            np.savez(payload_stream, **payload_arrays)  # type: ignore[arg-type]
        payload_sha256 = hashlib.sha256(payload_temporary_path.read_bytes()).hexdigest()
        payload_path = manifest_path.with_name(f"{manifest_path.stem}.{payload_sha256}.npz")
        manifest = PredictionArtifactV1(
            schema_version="driving-risk-prediction-artifact/v1",
            sample_id=record.sample_id,
            protocol_sha256=protocol_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
            payload_file=payload_path.name,
            payload_sha256=payload_sha256,
            confidence_scale=65535,
            valid_pixel_count=record.valid_pixel_count,
            num_classes=num_classes,
            ece_bin_count=ece_bin_count,
            arrays=PredictionArraysV1(
                predicted_class=UInt8ArrayDescriptorV1(
                    dtype="uint8", shape=payload_arrays["predicted_class"].shape
                ),
                top1_confidence_q16=UInt16ArrayDescriptorV1(
                    dtype="uint16", shape=payload_arrays["top1_confidence_q16"].shape
                ),
                correctness_bitset=UInt8ArrayDescriptorV1(
                    dtype="uint8", shape=payload_arrays["correctness_bitset"].shape
                ),
                confusion=Int64ArrayDescriptorV1(
                    dtype="int64", shape=payload_arrays["confusion"].shape
                ),
                brier_sum_by_class=Float64ArrayDescriptorV1(
                    dtype="float64", shape=payload_arrays["brier_sum_by_class"].shape
                ),
                ece_counts=Int64ArrayDescriptorV1(
                    dtype="int64", shape=payload_arrays["ece_counts"].shape
                ),
                ece_confidence_sums=Float64ArrayDescriptorV1(
                    dtype="float64", shape=payload_arrays["ece_confidence_sums"].shape
                ),
                ece_positive_counts=Int64ArrayDescriptorV1(
                    dtype="int64", shape=payload_arrays["ece_positive_counts"].shape
                ),
            ),
        )

        manifest_handle, manifest_temporary_name = tempfile.mkstemp(
            dir=manifest_path.parent,
            prefix=f".{manifest_path.stem}.",
            suffix=".json.tmp",
        )
        os.close(manifest_handle)
        manifest_temporary_path = Path(manifest_temporary_name)
        manifest_temporary_path.write_bytes(
            canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
        )

        if payload_path.exists():
            existing_hash = hashlib.sha256(payload_path.read_bytes()).hexdigest()
            if not secrets.compare_digest(existing_hash, payload_sha256):
                raise ValueError("existing content-addressed payload hash mismatch")
            payload_temporary_path.unlink()
        else:
            os.replace(payload_temporary_path, payload_path)
        os.replace(manifest_temporary_path, manifest_path)
        return manifest
    finally:
        if payload_temporary_path.exists():
            payload_temporary_path.unlink()
        if manifest_temporary_path is not None and manifest_temporary_path.exists():
            manifest_temporary_path.unlink()


def read_prediction_artifact(
    manifest_path: Path,
) -> tuple[PredictionArtifactV1, PredictionRecord, ECEBinSufficientStatistics]:
    """Verify hashes, descriptors, and cross-array invariants before returning values."""

    manifest = PredictionArtifactV1.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    payload_path = manifest_path.parent / manifest.payload_file
    payload_bytes = payload_path.read_bytes()
    actual_hash = hashlib.sha256(payload_bytes).hexdigest()
    if not secrets.compare_digest(actual_hash, manifest.payload_sha256):
        raise ValueError("payload SHA-256 mismatch")

    with np.load(io.BytesIO(payload_bytes), allow_pickle=False) as payload:
        if frozenset(payload.files) != _PAYLOAD_KEYS:
            raise ValueError("NPZ members do not match the manifest payload contract")
        arrays = {name: payload[name].copy() for name in payload.files}

    for name in _PAYLOAD_KEYS:
        descriptor = getattr(manifest.arrays, name)
        value = arrays[name]
        if value.dtype.name != descriptor.dtype or value.shape != descriptor.shape:
            raise ValueError(f"NPZ member {name!r} does not match its array descriptor")

    record = PredictionRecord(
        sample_id=manifest.sample_id,
        predicted_class=arrays["predicted_class"],
        top1_confidence_q16=arrays["top1_confidence_q16"],
        correctness_bitset=arrays["correctness_bitset"].tobytes(),
        confusion=arrays["confusion"],
        brier_sum_by_class=arrays["brier_sum_by_class"],
        valid_pixel_count=manifest.valid_pixel_count,
    )
    ece = ECEBinSufficientStatistics(
        counts=arrays["ece_counts"],
        confidence_sums=arrays["ece_confidence_sums"],
        positive_counts=arrays["ece_positive_counts"],
    )
    num_classes, ece_bin_count = _validate_prediction_values(record, ece)
    if num_classes != manifest.num_classes or ece_bin_count != manifest.ece_bin_count:
        raise ValueError("manifest dimensions do not match the verified NPZ payload")
    return manifest, record, ece
