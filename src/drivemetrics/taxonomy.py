"""CamVid 11-class taxonomy and the harm model that risk-weighted metrics are built on.

The harm model is the one genuinely subjective part of this package. It is kept
here, in one place, as plain data — never hard-coded inside a metric — so that
every number this package reports can be recomputed under a different harm model
by swapping a single object. `scripts/sensitivity.py` does exactly that.

Nothing in this module imports torch. It is pure data plus small dataclasses so
that the harm model can be inspected, serialised, and diffed in review.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "CAMVID_11",
    "CLASS_NAMES",
    "NUM_CLASSES",
    "IGNORE_INDEX",
    "ClassSpec",
    "HarmModel",
    "DEFAULT_HARM",
    "UNIFORM_HARM",
    "VRU_CLASSES",
    "DYNAMIC_CLASSES",
]

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

# CamVid's widely used 11-class reduction. Index 11 ("Unlabelled") is the void
# label and is excluded from every metric in this package, matching the
# convention used by Cityscapes and by the SegNet CamVid benchmark.
IGNORE_INDEX = 11


@dataclass(frozen=True)
class ClassSpec:
    """One semantic class and the properties the harm model reasons about."""

    index: int
    name: str
    #: Can this class move under its own power into the ego path?
    dynamic: bool
    #: Is this a Vulnerable Road User — an unprotected human?
    vru: bool
    #: Colour used in figures. CamVid's own label colormap, so that renders in
    #: this repo are visually comparable to every other CamVid paper.
    color: tuple[int, int, int]


CAMVID_11: tuple[ClassSpec, ...] = (
    ClassSpec(0, "Sky", dynamic=False, vru=False, color=(128, 128, 128)),
    ClassSpec(1, "Building", dynamic=False, vru=False, color=(128, 0, 0)),
    ClassSpec(2, "Pole", dynamic=False, vru=False, color=(192, 192, 128)),
    ClassSpec(3, "Road", dynamic=False, vru=False, color=(128, 64, 128)),
    ClassSpec(4, "Pavement", dynamic=False, vru=False, color=(60, 40, 222)),
    ClassSpec(5, "Tree", dynamic=False, vru=False, color=(128, 128, 0)),
    ClassSpec(6, "SignSymbol", dynamic=False, vru=False, color=(192, 128, 128)),
    ClassSpec(7, "Fence", dynamic=False, vru=False, color=(64, 64, 128)),
    ClassSpec(8, "Car", dynamic=True, vru=False, color=(64, 0, 128)),
    ClassSpec(9, "Pedestrian", dynamic=True, vru=True, color=(64, 64, 0)),
    ClassSpec(10, "Bicyclist", dynamic=True, vru=True, color=(0, 128, 192)),
)

CLASS_NAMES: tuple[str, ...] = tuple(c.name for c in CAMVID_11)
NUM_CLASSES = len(CAMVID_11)

VRU_CLASSES: tuple[int, ...] = tuple(c.index for c in CAMVID_11 if c.vru)
DYNAMIC_CLASSES: tuple[int, ...] = tuple(c.index for c in CAMVID_11 if c.dynamic)


# ---------------------------------------------------------------------------
# Harm model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarmModel:
    """Assigns a cost to each kind of confusion a segmentation model can make.

    Two quantities are declared per class:

    ``miss_cost``
        Cost of failing to recognise this class — predicting something else
        where this class actually is. For a pedestrian this is the cost of not
        seeing a person who is there.

    ``phantom_cost``
        Cost of hallucinating this class where it is not. For a pedestrian this
        is the cost of an unnecessary emergency stop: real, but not comparable
        to running someone over.

    Costs are *relative* and unitless. Only their ratios matter, and every
    headline number in this repo is reported alongside a sensitivity sweep over
    them, because no single choice of ratio is objectively correct.

    The asymmetry between the two is the whole point. A metric that treats a
    missed pedestrian and a phantom pedestrian as one unit of error each cannot
    express the thing that actually matters about driving perception.
    """

    name: str
    miss_cost: Mapping[int, float]
    phantom_cost: Mapping[int, float]
    notes: str = ""

    def __post_init__(self) -> None:
        for label, table in (("miss_cost", self.miss_cost), ("phantom_cost", self.phantom_cost)):
            missing = set(range(NUM_CLASSES)) - set(table)
            if missing:
                names = ", ".join(CLASS_NAMES[i] for i in sorted(missing))
                raise ValueError(f"{self.name}: {label} has no entry for: {names}")
            for idx, value in table.items():
                if value < 0:
                    raise ValueError(
                        f"{self.name}: {label}[{CLASS_NAMES[idx]}] is {value}; costs cannot be negative"
                    )

    def cost_matrix(self) -> list[list[float]]:
        """Build the ``C[true][pred]`` cost matrix these costs imply.

        The cost of predicting ``p`` where the truth is ``t`` is::

            C[t][p] = max(0, miss(t) - miss(p))                 # severity not signalled
                    + phantom(p) if miss(p) > miss(t) else 0    # nuisance from over-calling

        In words: **you are charged for the danger you failed to report, plus a
        smaller nuisance charge when you report danger that is not there.**

        This is not the obvious ``miss(t) + phantom(p)`` decomposition, and the
        difference matters. Under that simpler rule, labelling a pedestrian as a
        bicyclist would cost *more* than labelling them as road, because it adds
        a phantom charge on top of the full miss charge. That is backwards: a
        planner brakes for a bicyclist exactly as it brakes for a pedestrian, so
        the substitution is harmless, while calling a person "road" is the
        failure that kills.

        The subtractive form gets this right. Confusing two classes of equal
        declared severity costs exactly zero, and confusing a pedestrian for a
        car costs only the severity difference, crediting the model for having
        at least seen *something* it must not hit.

        Two consequences worth stating plainly:

        * This is a **safety-consequence metric, not a classification metric.**
          A model that labels every cyclist a pedestrian scores perfectly here.
          That is intended — but it is why mIoU is always reported alongside,
          since only mIoU will catch that confusion.
        * The matrix is asymmetric by construction, which is the point:
          ``C[Pedestrian][Road]`` and ``C[Road][Pedestrian]`` are different
          numbers because a missed person and an unnecessary brake are
          different events.
        """
        matrix: list[list[float]] = []
        for t in range(NUM_CLASSES):
            row = []
            for p in range(NUM_CLASSES):
                if t == p:
                    row.append(0.0)
                    continue
                miss_t = self.miss_cost[t]
                miss_p = self.miss_cost[p]
                under_call = max(0.0, miss_t - miss_p)
                over_call = self.phantom_cost[p] if miss_p > miss_t else 0.0
                row.append(under_call + over_call)
            matrix.append(row)
        return matrix

    def class_weights(self) -> list[float]:
        """Per-class weights for the simpler risk-weighted-IoU aggregation."""
        return [float(self.miss_cost[i]) for i in range(NUM_CLASSES)]

    def scaled(self, factor: float, classes: Sequence[int]) -> HarmModel:
        """Return a copy with ``miss_cost`` scaled for ``classes``.

        Used by the sensitivity sweep to answer "how hard would we have to lean
        on the VRU weight before the model ranking changes?".
        """
        miss = dict(self.miss_cost)
        for c in classes:
            miss[c] = miss[c] * factor
        return HarmModel(
            name=f"{self.name}×{factor:g}@{'+'.join(CLASS_NAMES[c] for c in classes)}",
            miss_cost=miss,
            phantom_cost=dict(self.phantom_cost),
            notes=f"derived from {self.name!r}",
        )


def _harm(
    *,
    vru: float,
    vehicle: float,
    obstacle: float,
    surface: float,
    background: float,
    phantom_vru: float,
    phantom_vehicle: float,
    phantom_static: float,
) -> tuple[dict[int, float], dict[int, float]]:
    """Expand five harm tiers into the per-class tables ``HarmModel`` wants."""
    miss = {
        9: vru,  # Pedestrian
        10: vru,  # Bicyclist
        8: vehicle,  # Car
        2: obstacle,  # Pole — small, rigid, and genuinely dangerous to hit
        6: obstacle,  # SignSymbol
        7: obstacle,  # Fence
        3: surface,  # Road
        4: surface,  # Pavement
        1: background,  # Building
        5: background,  # Tree
        0: background,  # Sky
    }
    phantom = {
        9: phantom_vru,
        10: phantom_vru,
        8: phantom_vehicle,
        2: phantom_static,
        6: phantom_static,
        7: phantom_static,
        3: phantom_static,
        4: phantom_static,
        1: phantom_static,
        5: phantom_static,
        0: phantom_static,
    }
    return miss, phantom


_DEFAULT_MISS, _DEFAULT_PHANTOM = _harm(
    vru=100.0,
    vehicle=30.0,
    obstacle=10.0,
    surface=3.0,
    background=1.0,
    phantom_vru=5.0,
    phantom_vehicle=3.0,
    phantom_static=1.0,
)

#: The harm model used for headline numbers. Every claim made with it is
#: reported together with the sensitivity sweep in ``reports/sensitivity.json``.
DEFAULT_HARM = HarmModel(
    name="default-v1",
    miss_cost=_DEFAULT_MISS,
    phantom_cost=_DEFAULT_PHANTOM,
    notes=(
        "Ordinal tiers, not calibrated against any injury statistic. VRU:vehicle:"
        "obstacle:surface:background = 100:30:10:3:1, and missing a class is "
        "20x costlier than hallucinating it for VRUs. Chosen to be defensible in "
        "ordering rather than precise in magnitude; conclusions that do not "
        "survive the sensitivity sweep are reported as not surviving it."
    ),
)

_UNIFORM_MISS, _UNIFORM_PHANTOM = _harm(
    vru=1.0,
    vehicle=1.0,
    obstacle=1.0,
    surface=1.0,
    background=1.0,
    phantom_vru=1.0,
    phantom_vehicle=1.0,
    phantom_static=1.0,
)

#: The null hypothesis: every error is equally bad. Recovers conventional
#: accuracy-style behaviour and exists so the effect of the harm model can be
#: isolated rather than assumed.
UNIFORM_HARM = HarmModel(
    name="uniform",
    miss_cost=_UNIFORM_MISS,
    phantom_cost=_UNIFORM_PHANTOM,
    notes="All errors weighted equally — the implicit model behind plain mIoU.",
)


def class_index(name: str) -> int:
    """Look up a class index by name, case-insensitively."""
    lowered = name.strip().lower()
    for spec in CAMVID_11:
        if spec.name.lower() == lowered:
            return spec.index
    raise KeyError(f"unknown class {name!r}; expected one of {', '.join(CLASS_NAMES)}")
