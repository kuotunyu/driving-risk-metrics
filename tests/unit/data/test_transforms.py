"""Behavior tests for deterministic segmentation preprocessing geometry."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import ModuleType

import numpy as np
import pytest


def load_transforms_module() -> ModuleType:
    try:
        from drivemetrics.data import transforms
    except ImportError:
        pytest.fail("drivemetrics.data.transforms is missing", pytrace=False)
    return transforms


def test_bdd_geometry_resizes_720x1280_to_512x910_then_pads_57_each_side() -> None:
    transforms = load_transforms_module()
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    mask = np.zeros((720, 1280), dtype=np.uint8)

    prepared = transforms.prepare_sample(image, mask, training=False, flip_draw=0.0)

    assert prepared.image_chw.shape == (3, 512, 1024)
    assert prepared.image_chw.dtype == np.float32
    assert prepared.mask_hw.shape == (512, 1024)
    assert prepared.mask_hw.dtype == np.int64
    assert (prepared.pad_left, prepared.pad_right) == (57, 57)
    assert prepared.original_height == 720
    assert prepared.original_width == 1280
    assert np.all(prepared.image_chw[:, :, :57] == 0.0)
    assert np.all(prepared.image_chw[:, :, -57:] == 0.0)
    assert np.all(prepared.mask_hw[:, :57] == 255)
    assert np.all(prepared.mask_hw[:, -57:] == 255)


def test_aspect_preserving_resize_splits_odd_horizontal_padding_by_at_most_one() -> None:
    transforms = load_transforms_module()
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    mask = np.zeros((3, 4), dtype=np.uint8)

    prepared = transforms.prepare_sample(image, mask, training=False, flip_draw=1.0)

    assert prepared.pad_left == 170
    assert prepared.pad_right == 171
    assert abs(prepared.pad_left - prepared.pad_right) <= 1


def test_rgb_is_bilinear_mask_is_nearest_and_imagenet_normalized() -> None:
    transforms = load_transforms_module()
    image = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
    mask = np.array([[1, 9]], dtype=np.uint8)

    prepared = transforms.prepare_sample(image, mask, training=False, flip_draw=0.25)
    black_expected = np.array(
        [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        dtype=np.float32,
    )
    white_expected = np.array(
        [(1.0 - 0.485) / 0.229, (1.0 - 0.456) / 0.224, (1.0 - 0.406) / 0.225],
        dtype=np.float32,
    )
    reconstructed_red = (prepared.image_chw[0] * 0.229 + 0.485) * 255.0

    assert prepared.pad_left == prepared.pad_right == 0
    np.testing.assert_allclose(prepared.image_chw[:, 0, 0], black_expected, rtol=1e-6)
    np.testing.assert_allclose(prepared.image_chw[:, 0, -1], white_expected, rtol=1e-6)
    assert np.any((reconstructed_red > 0.0) & (reconstructed_red < 255.0))
    assert set(np.unique(prepared.mask_hw)) == {1, 9}


@pytest.mark.parametrize(
    ("training", "flip_draw", "expected_first", "expected_last"),
    [(True, 0.499, 9, 1), (True, 0.5, 1, 9), (False, 0.0, 1, 9)],
)
def test_flip_occurs_only_during_training_below_half(
    training: bool,
    flip_draw: float,
    expected_first: int,
    expected_last: int,
) -> None:
    transforms = load_transforms_module()
    image = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
    mask = np.array([[1, 9]], dtype=np.uint8)

    prepared = transforms.prepare_sample(
        image,
        mask,
        training=training,
        flip_draw=flip_draw,
    )

    assert prepared.mask_hw[0, 0] == expected_first
    assert prepared.mask_hw[0, -1] == expected_last


def test_restore_prediction_crops_padding_and_returns_original_shape() -> None:
    transforms = load_transforms_module()
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    mask = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype=np.uint8)
    prepared = transforms.prepare_sample(image, mask, training=False, flip_draw=0.0)

    restored = transforms.restore_prediction(prepared.mask_hw.astype(np.uint8), prepared)

    assert restored.shape == (3, 4)
    assert restored.dtype == np.uint8
    np.testing.assert_array_equal(restored, mask)

    with pytest.raises(FrozenInstanceError):
        prepared.pad_left = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("image", "mask", "flip_draw", "expected"),
    [
        (np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8), 0.0, "image"),
        (
            np.zeros((2, 2, 3), dtype=np.float32),
            np.zeros((2, 2), dtype=np.uint8),
            0.0,
            "uint8",
        ),
        (
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2, 1), dtype=np.uint8),
            0.0,
            "mask",
        ),
        (
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.float32),
            0.0,
            "mask must have uint8 dtype",
        ),
        (
            np.zeros((2, 3, 3), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.uint8),
            0.0,
            "spatial",
        ),
        (
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.uint8),
            -0.1,
            "flip_draw",
        ),
        (
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.uint8),
            1.1,
            "flip_draw",
        ),
        (
            np.zeros((0, 2, 3), dtype=np.uint8),
            np.zeros((0, 2), dtype=np.uint8),
            0.0,
            "spatial dimensions",
        ),
        (
            np.zeros((1, 3, 3), dtype=np.uint8),
            np.zeros((1, 3), dtype=np.uint8),
            0.0,
            "aspect ratio",
        ),
        (
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.uint8),
            float("nan"),
            "flip_draw",
        ),
    ],
)
def test_prepare_sample_rejects_invalid_arrays_or_flip_draw(
    image: np.ndarray,
    mask: np.ndarray,
    flip_draw: float,
    expected: str,
) -> None:
    transforms = load_transforms_module()

    with pytest.raises(ValueError, match=expected):
        transforms.prepare_sample(image, mask, training=False, flip_draw=flip_draw)


@pytest.mark.parametrize(
    "prediction",
    [
        np.zeros((512, 1024), dtype=np.int64),
        np.zeros((512, 1023), dtype=np.uint8),
        np.zeros((1, 512, 1024), dtype=np.uint8),
    ],
)
def test_restore_prediction_rejects_wrong_dtype_or_shape(prediction: np.ndarray) -> None:
    transforms = load_transforms_module()
    prepared = transforms.PreparedSample(
        image_chw=np.zeros((3, 512, 1024), dtype=np.float32),
        mask_hw=np.zeros((512, 1024), dtype=np.int64),
        original_height=3,
        original_width=4,
        pad_left=170,
        pad_right=171,
    )

    with pytest.raises(ValueError, match="prediction"):
        transforms.restore_prediction(prediction, prepared)


def test_the_restore_index_map_agrees_exactly_with_the_restored_prediction() -> None:
    """Two different nearest mappings would misalign confidence from its own class."""

    module = load_transforms_module()
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    mask = rng.integers(0, 19, size=(720, 1280), dtype=np.uint8)
    prepared = module.prepare_sample(image, mask, training=False, flip_draw=1.0)
    canvas = rng.integers(0, 19, size=(512, 1024), dtype=np.uint8)

    index_map = module.restore_index_map(prepared)
    content = canvas[:, prepared.pad_left : 1024 - prepared.pad_right]

    assert index_map.dtype == np.int64
    assert index_map.shape == (720, 1280)
    np.testing.assert_array_equal(
        content.reshape(-1)[index_map],
        module.restore_prediction(canvas, prepared),
    )


def test_the_restore_index_map_only_addresses_unpadded_content() -> None:
    """An index reaching into the padded columns would score invented pixels."""

    module = load_transforms_module()
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    mask = np.zeros((720, 1280), dtype=np.uint8)
    prepared = module.prepare_sample(image, mask, training=False, flip_draw=1.0)

    index_map = module.restore_index_map(prepared)
    content_width = 1024 - prepared.pad_left - prepared.pad_right

    assert int(index_map.min()) >= 0
    assert int(index_map.max()) < 512 * content_width
