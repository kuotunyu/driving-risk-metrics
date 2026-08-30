"""Flat-ground inverse perspective mapping, used to turn image rows into metres.

Why this is here: "the model has 0.17 IoU on pedestrians" is much less useful
than "the model has 0.44 IoU on pedestrians within 15 m and 0.02 beyond 40 m",
because only the first kind of statement can be acted on. Distance is what
converts a perception error into a stopping-distance problem.

CamVid does not publish camera intrinsics or extrinsics. This module therefore
does **not** pretend to recover true metric depth. It applies a flat-ground
pinhole model under three explicitly declared parameters — horizon row, focal
length in pixels, and camera height — and every function that consumes it
carries those parameters in its output. ``scripts/sensitivity.py`` sweeps them,
and any conclusion that does not survive the sweep is reported as not surviving
it.

The parameterisation is by *horizon row* rather than pitch angle on purpose. The
horizon is directly visible in an image and can be checked by eye against a
render; pitch is not, and an error in it is silent.

Model
-----
For a pinhole camera with the ground plane flat and the optical axis level with
the horizon at image row ``v_h``, a pixel at row ``v > v_h`` sees the ground at
a depression angle below horizontal of::

    alpha = arctan((v - v_h) / f)

and therefore at a longitudinal ground distance of::

    Z = h / tan(alpha)

Rows at or above the horizon do not intersect the ground plane at all and are
assigned infinite distance. Lateral offset follows as ``X = (u - cx) * Z / f``.

Everything above the horizon, and everything on a non-flat road, is outside this
model. That is a real limitation, stated here rather than buried.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CameraModel",
    "CAMVID_DEFAULT_CAMERA",
    "DEFAULT_BANDS",
    "row_to_distance",
    "distance_map",
    "band_map",
    "band_labels",
]


@dataclass(frozen=True)
class CameraModel:
    """Flat-ground pinhole parameters, all of them assumptions for CamVid.

    Attributes
    ----------
    horizon_row:
        Image row of the horizon, in pixels from the top, in the resolution the
        masks are evaluated at.
    focal_px:
        Focal length in pixels at that same resolution.
    height_m:
        Camera height above the road surface, in metres.
    image_height / image_width:
        Resolution these parameters are expressed in. Stored so that a model
        used at the wrong resolution fails loudly instead of silently scaling.
    source:
        Free text recording where the numbers came from. For CamVid this says
        "assumed"; for any dataset with real calibration it should say so.
    """

    horizon_row: float
    focal_px: float
    height_m: float
    image_height: int
    image_width: int
    source: str = "assumed"

    def __post_init__(self) -> None:
        if self.focal_px <= 0:
            raise ValueError(f"focal_px must be positive, got {self.focal_px}")
        if self.height_m <= 0:
            raise ValueError(f"height_m must be positive, got {self.height_m}")
        if not (0 <= self.horizon_row < self.image_height):
            raise ValueError(
                f"horizon_row {self.horizon_row} is outside the image "
                f"(0..{self.image_height})"
            )

    def scaled_to(self, height: int, width: int) -> CameraModel:
        """Rescale to another resolution, preserving the geometry."""
        sy = height / self.image_height
        sx = width / self.image_width
        # A non-uniform resize changes the aspect ratio and with it the implied
        # focal length; refuse rather than guess which axis to trust.
        if abs(sy - sx) > 1e-6:
            raise ValueError(
                f"non-uniform rescale ({sx:.4f} x, {sy:.4f} y) would distort the "
                "camera model; resize masks with a uniform factor"
            )
        return CameraModel(
            horizon_row=self.horizon_row * sy,
            focal_px=self.focal_px * sy,
            height_m=self.height_m,
            image_height=height,
            image_width=width,
            source=f"{self.source} (rescaled x{sy:.4f})",
        )

    def as_dict(self) -> dict:
        return {
            "horizon_row": float(self.horizon_row),
            "focal_px": float(self.focal_px),
            "height_m": float(self.height_m),
            "image_height": int(self.image_height),
            "image_width": int(self.image_width),
            "source": self.source,
        }


#: Defaults for CamVid at the 360x480 resolution the notebooks train on.
#:
#: These are **assumptions**, not measurements. The horizon row was taken from
#: where the road/sky boundary sits in the CamVid sequences; the focal length is
#: consistent with a ~60 degree horizontal field of view, typical of the
#: forward-facing cameras used for this kind of capture; the height is a normal
#: windscreen mounting. Distances derived from them are approximate and
#: comparative — good enough to rank near against far, not to certify a range.
CAMVID_DEFAULT_CAMERA = CameraModel(
    horizon_row=138.0,
    focal_px=420.0,
    height_m=1.35,
    image_height=360,
    image_width=480,
    source="assumed (CamVid publishes no calibration; see docs/GEOMETRY.md)",
)

#: Distance bands in metres. The near band is roughly the stopping distance at
#: urban speed, so a miss inside it is the one that cannot be recovered from.
DEFAULT_BANDS: tuple[float, ...] = (0.0, 15.0, 40.0, float("inf"))


def band_labels(bands: Sequence[float] = DEFAULT_BANDS) -> tuple[str, ...]:
    """Human-readable names for the bands, e.g. ``('0-15m', '15-40m', '40m+')``."""
    out = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        if np.isinf(hi):
            out.append(f"{lo:g}m+")
        else:
            out.append(f"{lo:g}-{hi:g}m")
    return tuple(out)


def row_to_distance(rows: np.ndarray, camera: CameraModel) -> np.ndarray:
    """Longitudinal ground distance in metres for each image row.

    Rows at or above the horizon return ``inf``: those rays never meet the
    ground plane, so no finite distance is defined for them.
    """
    rows = np.asarray(rows, dtype=np.float64)
    delta = rows - camera.horizon_row
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.arctan(delta / camera.focal_px)
        dist = np.where(alpha > 0, camera.height_m / np.tan(alpha), np.inf)
    return dist


def distance_map(camera: CameraModel) -> np.ndarray:
    """A ``(H, W)`` array of ground distance in metres for every pixel.

    Constant along each row, because the flat-ground model makes distance a
    function of row alone. Returned as a full 2-D array anyway so it can be
    masked and indexed alongside label maps without broadcasting bugs.
    """
    rows = np.arange(camera.image_height, dtype=np.float64)
    per_row = row_to_distance(rows, camera)
    return np.repeat(per_row[:, None], camera.image_width, axis=1)


def band_map(
    camera: CameraModel, bands: Sequence[float] = DEFAULT_BANDS
) -> np.ndarray:
    """A ``(H, W)`` array of band indices, or ``-1`` where no band applies.

    Pixels above the horizon fall in no band and are excluded from every
    distance-stratified statistic rather than being lumped into the far band —
    sky is not "far away road".
    """
    if len(bands) < 2:
        raise ValueError("need at least two edges to define one band")
    if not all(b < a for b, a in zip(bands[:-1], bands[1:])):
        raise ValueError(f"band edges must be strictly increasing, got {tuple(bands)}")

    dist = distance_map(camera)
    out = np.full(dist.shape, -1, dtype=np.int8)
    for i, (lo, hi) in enumerate(zip(bands[:-1], bands[1:])):
        mask = (dist >= lo) & (dist < hi) & np.isfinite(dist)
        out[mask] = i
    return out
