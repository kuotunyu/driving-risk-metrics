"""Tests for byte-exact files, canonical manifests, and semantic protocols."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMITTED_PROTOCOL = REPO_ROOT / "configs" / "protocols" / "bdd100k_semseg_v1.yaml"


def load_hashing_module() -> ModuleType:
    try:
        from drivemetrics.protocol import hashing
    except ImportError:
        pytest.fail("drivemetrics.protocol.hashing is missing", pytrace=False)
    return hashing


def load_config_module() -> ModuleType:
    try:
        from drivemetrics.protocol import config
    except ImportError:
        pytest.fail("drivemetrics.protocol.config is missing", pytrace=False)
    return config


def test_sha256_file_hashes_raw_bytes_independently_of_chunk_size(tmp_path: Path) -> None:
    hashing = load_hashing_module()
    path = tmp_path / "路況.bin"
    content = b"abc\x00def" * 19
    path.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    assert hashing.sha256_file(path, chunk_size=1) == expected
    assert hashing.sha256_file(path, chunk_size=17) == expected


def test_sha256_file_rejects_nonpositive_chunk_size(tmp_path: Path) -> None:
    hashing = load_hashing_module()
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match=r"^chunk_size must be positive"):
        hashing.sha256_file(path, chunk_size=0)


def test_canonical_manifest_hash_is_order_independent_and_content_sensitive() -> None:
    hashing = load_hashing_module()
    first = {"split": "train", "ids": ["b", "a"]}
    reordered = {"ids": ["b", "a"], "split": "train"}
    drifted = {"split": "train", "ids": ["a", "b"]}

    assert hashing.canonical_manifest_sha256(first) == hashing.canonical_manifest_sha256(reordered)
    assert hashing.canonical_manifest_sha256(first) != hashing.canonical_manifest_sha256(drifted)


def protocol_yaml(*, extra: str = "", formatting: str = "block") -> str:
    if formatting == "flow":
        return f"""schema_version: bdd100k-semseg-protocol/v1
dataset: {{name: bdd100k, version: 10k-semantic-v1}}
splits: {{source_train: 7000, train: 6300, calibration: 700, locked_validation: 1000, unlabeled_test: 2000}}
paths:
  train_images: images/10k/train
  train_labels: labels/sem_seg/masks/train
  validation_images: images/10k/val
  validation_labels: labels/sem_seg/masks/val
input: {{resize_height: 512, resize_width: 910, padded_height: 512, padded_width: 1024, image_pad_value_after_normalization: 0.0, mask_pad_value: 255}}
training: {{steps: 30000, warmup_steps: 1000, effective_batch_size: 16, horizontal_flip_probability: 0.5, checkpoint_selection: final_step_only}}
models:
  upernet_convnextv2_tiny: {{optimizer: adamw, learning_rate: 0.0001, weight_decay: 0.05}}
  upernet_dinov2_small: {{optimizer: adamw, learning_rate: 0.0001, weight_decay: 0.05}}
  segformer_b2: {{optimizer: adamw, learning_rate: 0.00006, weight_decay: 0.01}}
calibration: {{method: scalar_temperature, objective: multiclass_nll}}
statistics: {{bootstrap_resamples: 5000, bootstrap_seed: 20260831, confidence: 0.95}}
{extra}"""
    return f"""schema_version: bdd100k-semseg-protocol/v1
dataset:
  name: bdd100k
  version: 10k-semantic-v1
splits:
  source_train: 7000
  train: 6300
  calibration: 700
  locked_validation: 1000
  unlabeled_test: 2000
paths:
  train_images: images/10k/train
  train_labels: labels/sem_seg/masks/train
  validation_images: images/10k/val
  validation_labels: labels/sem_seg/masks/val
input:
  resize_height: 512
  resize_width: 910
  padded_height: 512
  padded_width: 1024
  image_pad_value_after_normalization: 0.0
  mask_pad_value: 255
training:
  steps: 30000
  warmup_steps: 1000
  effective_batch_size: 16
  horizontal_flip_probability: 0.5
  checkpoint_selection: final_step_only
models:
  upernet_convnextv2_tiny:
    optimizer: adamw
    learning_rate: 0.0001
    weight_decay: 0.05
  upernet_dinov2_small:
    optimizer: adamw
    learning_rate: 0.0001
    weight_decay: 0.05
  segformer_b2:
    optimizer: adamw
    learning_rate: 0.00006
    weight_decay: 0.01
calibration:
  method: scalar_temperature
  objective: multiclass_nll
statistics:
  bootstrap_resamples: 5000
  bootstrap_seed: 20260831
  confidence: 0.95
{extra}"""


def test_protocol_hash_uses_validated_semantics_not_yaml_formatting(tmp_path: Path) -> None:
    config = load_config_module()
    block = tmp_path / "block.yaml"
    flow = tmp_path / "flow.yaml"
    block.write_text(protocol_yaml(), encoding="utf-8")
    flow.write_text(protocol_yaml(formatting="flow"), encoding="utf-8")

    loaded_block = config.load_protocol(block)
    loaded_flow = config.load_protocol(flow)

    assert loaded_block.protocol == loaded_flow.protocol
    assert loaded_block.protocol_sha256 == loaded_flow.protocol_sha256
    assert len(loaded_block.protocol_sha256) == 64


def test_committed_protocol_is_strictly_loadable_and_frozen() -> None:
    config = load_config_module()

    loaded = config.load_protocol(COMMITTED_PROTOCOL)

    assert loaded.protocol.splits.train == 6300
    assert loaded.protocol.splits.calibration == 700
    assert loaded.protocol.splits.locked_validation == 1000
    assert loaded.protocol.training.checkpoint_selection == "final_step_only"


def test_protocol_loader_rejects_unknown_keys(tmp_path: Path) -> None:
    config = load_config_module()
    path = tmp_path / "unknown.yaml"
    path.write_text(protocol_yaml(extra="unexpected: true\n"), encoding="utf-8")

    with pytest.raises(
        ValidationError,
        match=r"^1 validation error for BDD100KSemanticProtocolV1\nunexpected\n  Extra inputs are not permitted",
    ):
        config.load_protocol(path)


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("train_images: images/10k/train", "train_images: ../train", "train_images"),
        ("train_images: images/10k/train", "train_images: .", "train_images"),
        ("learning_rate: 0.0001", "learning_rate: 0.0002", "learning_rate"),
    ],
)
def test_protocol_loader_rejects_unsafe_path_or_changed_fixed_model(
    tmp_path: Path,
    old: str,
    new: str,
    expected: str,
) -> None:
    config = load_config_module()
    path = tmp_path / "changed.yaml"
    path.write_text(protocol_yaml().replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ValidationError, match=expected):
        config.load_protocol(path)


@pytest.mark.parametrize(
    "document",
    ["- not-a-mapping\n", "schema_version: bdd100k-semseg-protocol/v2\n"],
)
def test_protocol_loader_rejects_wrong_document_shape_or_version(
    tmp_path: Path,
    document: str,
) -> None:
    config = load_config_module()
    path = tmp_path / "invalid.yaml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises((TypeError, ValidationError)):
        config.load_protocol(path)


def test_the_protocol_hash_is_the_digest_of_the_validated_semantics(tmp_path: Path) -> None:
    """The hash is recomputed here independently, because it is the study's identity.

    Every artifact carries this digest and every consumer refuses anything that
    disagrees with it. Computed over a constant instead of over the loaded
    protocol, every protocol version would share one digest, every
    cross-protocol guard in the repository would pass, and results from two
    different studies would pool without complaint.

    Recomputing it from the model dump also pins WHAT is hashed: the validated
    semantics, not the YAML text, which is what makes two differently formatted
    files agree.
    """

    from drivemetrics.artifacts.envelope import canonical_json_bytes

    config = load_config_module()
    path = tmp_path / "protocol.yaml"
    path.write_text(protocol_yaml(), encoding="utf-8")

    loaded = config.load_protocol(path)

    expected = hashlib.sha256(
        canonical_json_bytes(loaded.protocol.model_dump(mode="json"))
    ).hexdigest()
    assert loaded.protocol_sha256 == expected
    assert len(loaded.protocol_sha256) == 64


def test_a_protocol_document_that_is_not_a_mapping_says_so(tmp_path: Path) -> None:
    """The shape error is reported before validation, and it must name the shape.

    A YAML list parses cleanly and then fails somewhere inside pydantic with a
    message about missing fields, which sends an operator looking for a typo in
    a field name rather than at the top-level structure. This check exists to
    say the real thing, so its wording is part of the contract.
    """

    config = load_config_module()
    path = tmp_path / "sequence.yaml"
    path.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(TypeError, match=r"^protocol document must be a mapping$"):
        config.load_protocol(path)


def test_reading_a_file_in_chunks_gives_the_same_digest_as_reading_it_whole(
    tmp_path: Path,
) -> None:
    """The chunk size is a memory bound, never a change to the digest.

    Checkpoints are gigabytes, so the hash is streamed. A reader that streams
    must agree with one that does not, or every hash in the study would depend
    on a buffer size nobody records.
    """

    hashing = load_hashing_module()
    payload = bytes(range(256)) * 8
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)

    assert hashing.sha256_file(path) == hashlib.sha256(payload).hexdigest()
    assert hashing.sha256_file(path, chunk_size=7) == hashlib.sha256(payload).hexdigest()
    assert hashing.sha256_file(path, chunk_size=len(payload)) == hashlib.sha256(payload).hexdigest()
