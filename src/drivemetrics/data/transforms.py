"""Pure deterministic preprocessing for road-scene semantic segmentation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image

UInt8Array = npt.NDArray[np.uint8]

TARGET_HEIGHT = 512
CANVAS_WIDTH = 1024
MASK_PAD_VALUE = 255
_IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(3, 1, 1)


@dataclass(frozen=True)
class PreparedSample:
    """Model-ready tensors plus the geometry needed to restore a prediction."""

    image_chw: npt.NDArray[np.float32]
    mask_hw: npt.NDArray[np.int64]
    original_height: int
    original_width: int
    pad_left: int
    pad_right: int


def _validate_inputs(image_hwc: UInt8Array, mask_hw: UInt8Array, flip_draw: float) -> None:
    if image_hwc.ndim != 3 or image_hwc.shape[2] != 3:
        raise ValueError("image must have shape [height, width, 3]")
    if image_hwc.dtype != np.uint8:
        raise ValueError("image must have uint8 dtype")
    if mask_hw.ndim != 2:
        raise ValueError("mask must have shape [height, width]")
    if mask_hw.dtype != np.uint8:
        raise ValueError("mask must have uint8 dtype")
    if image_hwc.shape[:2] != mask_hw.shape:
        raise ValueError("image and mask spatial shapes must match")
    if image_hwc.shape[0] == 0 or image_hwc.shape[1] == 0:
        raise ValueError("image and mask spatial dimensions must be positive")
    if not math.isfinite(flip_draw) or not 0.0 <= flip_draw <= 1.0:
        raise ValueError("flip_draw must be finite and within [0, 1]")


def _resized_width(original_height: int, original_width: int) -> int:
    width = round(original_width * TARGET_HEIGHT / original_height)
    if width <= 0 or width > CANVAS_WIDTH:
        raise ValueError("input aspect ratio does not fit the 512x1024 canvas")
    return width


def prepare_sample(
    image_hwc: UInt8Array,
    mask_hw: UInt8Array,
    *,
    training: bool,
    flip_draw: float,
) -> PreparedSample:
    """Resize, optionally flip, normalize, and horizontally pad one sample."""

    _validate_inputs(image_hwc, mask_hw, flip_draw)
    original_height, original_width = mask_hw.shape
    if training and flip_draw < 0.5:
        image_hwc = np.ascontiguousarray(np.flip(image_hwc, axis=1))
        mask_hw = np.ascontiguousarray(np.flip(mask_hw, axis=1))

    resized_width = _resized_width(original_height, original_width)
    resized_image = np.asarray(
        Image.fromarray(image_hwc).resize(
            (resized_width, TARGET_HEIGHT),
            resample=Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )
    resized_mask = np.asarray(
        Image.fromarray(mask_hw).resize(
            (resized_width, TARGET_HEIGHT),
            resample=Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    )

    total_padding = CANVAS_WIDTH - resized_width
    pad_left = total_padding // 2
    pad_right = total_padding - pad_left
    image_chw = np.transpose(resized_image.astype(np.float32) / np.float32(255.0), (2, 0, 1))
    normalized = (image_chw - _IMAGENET_MEAN) / _IMAGENET_STD
    padded_image = np.pad(
        normalized,
        ((0, 0), (0, 0), (pad_left, pad_right)),
        mode="constant",
        constant_values=0.0,
    ).astype(np.float32, copy=False)
    padded_mask = np.pad(
        resized_mask.astype(np.int64),
        ((0, 0), (pad_left, pad_right)),
        mode="constant",
        constant_values=MASK_PAD_VALUE,
    )
    return PreparedSample(
        image_chw=padded_image,
        mask_hw=padded_mask,
        original_height=original_height,
        original_width=original_width,
        pad_left=pad_left,
        pad_right=pad_right,
    )


def restore_prediction(
    prediction_hw: UInt8Array,
    prepared: PreparedSample,
) -> UInt8Array:
    """Remove model-canvas padding and nearest-resize labels to source geometry."""

    if prediction_hw.dtype != np.uint8 or prediction_hw.shape != (TARGET_HEIGHT, CANVAS_WIDTH):
        raise ValueError("prediction must be a uint8 array with shape [512, 1024]")
    content = prediction_hw[:, prepared.pad_left : CANVAS_WIDTH - prepared.pad_right]
    restored = Image.fromarray(content).resize(
        (prepared.original_width, prepared.original_height),
        resample=Image.Resampling.NEAREST,
    )
    return np.asarray(restored, dtype=np.uint8)
