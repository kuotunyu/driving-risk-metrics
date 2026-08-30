"""Tests for the evaluation runner, the sensitivity sweep and the fixtures.

The property that matters most here is the one the whole repository argues for:
that a model can rank better on mIoU and worse on risk. It is asserted against
the synthetic fixtures so that the codepath is exercised on every CI run rather
than only when a real training run happens to produce an inversion.
"""

from __future__ import annotations

import numpy as np
import pytest

from drivemetrics import (
    CAMVID_DEFAULT_CAMERA,
    DEFAULT_HARM,
    PROFILES,
    ConfusionMatrix,
    dataset_iou,
    degrade,
    evaluate_pairs,
    expected_risk,
    rank_stability,
    safety_recall,
    sweep_harm_model,
)
from drivemetrics.taxonomy import class_index

PED = class_index("Pedestrian")
ROAD = class_index("Road")
SKY = class_index("Sky")
BUILDING = class_index("Building")
POLE = class_index("Pole")


def _scene(h=48, w=64):
    """A small synthetic road scene with a pedestrian and a pole in it.

    Kept at CamVid's 4:3 aspect ratio so the default camera model can be
    rescaled onto it; `CameraModel.scaled_to` deliberately refuses a
    ratio-changing resize rather than silently distorting the geometry.
    """
    t = np.full((h, w), ROAD, dtype=np.int64)
    t[: h // 3] = SKY
    t[h // 3 : h // 2] = BUILDING
    t[h // 2 : h // 2 + 14, 8:20] = PED
    t[h // 2 : h // 2 + 20, 40:44] = POLE
    return t


def _pairs(profile, n=6, seed=0):
    rng = np.random.default_rng(seed)
    base = _scene()
    for i in range(n):
        yield f"img{i}", base, degrade(base, profile, rng)


# ---------------------------------------------------------------------------
# evaluate_pairs
# ---------------------------------------------------------------------------


def test_bundle_carries_its_assumptions():
    """A number must never be quotable without the model that produced it."""
    b = evaluate_pairs(_pairs(PROFILES["synth-balanced"]), model="m", split="val")
    prov = b.provenance
    assert prov["harm_model"]["name"] == DEFAULT_HARM.name
    assert prov["camera"]["source"].startswith("assumed")
    assert prov["blind_spot_thresholds"]["recall_threshold"] == 0.10
    assert prov["package_version"]
    assert b.n_samples == 6


def test_both_iou_protocols_are_reported():
    b = evaluate_pairs(_pairs(PROFILES["synth-balanced"]), model="m")
    assert b.dataset_iou["aggregation"] == "dataset"
    assert b.per_image_iou["aggregation"] == "per_image_nanmean"
    assert b.protocol_gap is not None


def test_headline_is_a_strict_subset_of_the_detail():
    b = evaluate_pairs(_pairs(PROFILES["synth-balanced"]), model="m")
    d = b.as_dict()
    assert d["headline"]["mean_iou"] == d["iou"]["dataset_protocol"]["mean_iou"]
    assert d["headline"]["risk_skill"] == d["risk"]["risk_skill"]
    assert d["headline"]["vru_recall"] == d["safety_recall"]["pooled"]


def test_shape_mismatch_is_an_error_not_a_skip():
    def bad():
        yield "a", np.zeros((8, 8), dtype=np.int64), np.zeros((4, 4), dtype=np.int64)

    with pytest.raises(ValueError, match="!="):
        evaluate_pairs(bad(), model="m")


def test_empty_input_is_an_error():
    with pytest.raises(ValueError, match="no samples"):
        evaluate_pairs(iter([]), model="m")


def test_geometry_can_be_disabled():
    b = evaluate_pairs(_pairs(PROFILES["synth-balanced"]), model="m", camera=None)
    assert b.stratified == {}
    assert b.provenance["camera"] is None


def test_camera_is_rescaled_to_the_mask_size():
    """The fixture is 48x64; the default camera is 360x480 and must adapt."""
    b = evaluate_pairs(_pairs(PROFILES["synth-balanced"]), model="m")
    assert b.stratified["camera"]["image_height"] == 48
    assert "rescaled" in b.stratified["camera"]["source"]


def test_mismatched_aspect_ratio_is_refused_with_a_useful_message():
    """Masks at a different aspect ratio would silently distort the geometry."""

    def square():
        t = np.full((40, 40), ROAD, dtype=np.int64)
        yield "a", t, t.copy()

    with pytest.raises(ValueError, match="aspect ratio|non-uniform"):
        evaluate_pairs(square(), model="m")


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def test_degrade_is_deterministic_for_a_given_seed():
    t = _scene()
    a = degrade(t, PROFILES["synth-balanced"], np.random.default_rng(3))
    b = degrade(t, PROFILES["synth-balanced"], np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_dropped_classes_are_never_predicted():
    t = _scene()
    prof = PROFILES["synth-background-biased"]
    # Noise can reintroduce a stray pixel, so check the profile's own mechanism
    # rather than the noisy output.
    clean = type(prof)(
        name=prof.name, description="", drop=prof.drop,
        erode=prof.erode, confused_with=prof.confused_with, noise=0.0,
    )
    pred = degrade(t, clean, np.random.default_rng(0))
    for cls in prof.drop:
        assert not np.any(pred == cls), f"{cls} was dropped but still predicted"


def test_void_pixels_pass_through_unchanged():
    from drivemetrics import IGNORE_INDEX

    t = _scene()
    t[0, :] = IGNORE_INDEX
    pred = degrade(t, PROFILES["synth-balanced"], np.random.default_rng(0))
    assert np.all(pred[0, :] == IGNORE_INDEX)


def test_background_biased_profile_is_blind_to_pedestrians():
    """The fixture must reproduce the failure it claims to imitate."""
    cm = ConfusionMatrix()
    for _, t, p in _pairs(PROFILES["synth-background-biased"]):
        cm.update(t, p)
    assert cm.per_class_recall()[PED] < 0.05


# ---------------------------------------------------------------------------
# The rank inversion
# ---------------------------------------------------------------------------


def _matrices():
    out = {}
    for name, profile in PROFILES.items():
        cm = ConfusionMatrix()
        for _, t, p in _pairs(profile, n=8):
            cm.update(t, p)
        out[name] = cm
    return out


def test_miou_and_risk_disagree_on_the_fixtures():
    """The claim the package exists to support, as an executable assertion.

    `synth-hazard-priority` is sloppy on background and so scores no better than
    `synth-background-biased` on mIoU -- yet it recovers the pedestrians the
    other one never sees. Any metric worth using for driving has to separate
    those two, and mIoU does not.
    """
    mats = _matrices()
    biased = mats["synth-background-biased"]
    hazard = mats["synth-hazard-priority"]

    # Safety behaviour is wildly different...
    r_biased = safety_recall(biased)["pooled"]
    r_hazard = safety_recall(hazard)["pooled"]
    assert r_hazard > 2 * r_biased

    # ...and the risk metric reflects that...
    assert (
        expected_risk(hazard, DEFAULT_HARM).risk_skill
        > expected_risk(biased, DEFAULT_HARM).risk_skill
    )

    # ...while mIoU does not put the safer model ahead.
    assert dataset_iou(hazard).mean <= dataset_iou(biased).mean + 0.02


def test_sweep_detects_the_disagreement():
    sweep = sweep_harm_model(_matrices())
    assert sweep.disagreement_fraction > 0.0
    assert len(sweep.factors) == len(next(iter(sweep.rank_by_model.values())))


def test_ranking_is_invariant_to_a_uniform_rescale_of_all_costs():
    """The control: costs are relative, so scaling all of them changes nothing.

    This isolates the disagreement with mIoU as coming from the *shape* of the
    harm model rather than from its magnitude or from a bug in the ranking code.
    A metric whose ordering moved under a pure rescale would be meaningless.
    """
    mats = _matrices()
    base = sweep_harm_model(mats, factors=(1.0,))

    all_classes = list(range(len(DEFAULT_HARM.miss_cost)))
    scaled = DEFAULT_HARM.scaled(1000.0, all_classes)
    rescaled = sweep_harm_model(mats, factors=(1.0,), base=scaled)

    assert base.modal_risk_ranking == rescaled.modal_risk_ranking


def test_raising_the_vru_weight_helps_the_hazard_prioritising_model():
    """The harm model must actually be the thing driving the ranking."""
    mats = _matrices()
    low = sweep_harm_model(mats, factors=(0.01,))
    high = sweep_harm_model(mats, factors=(50.0,))

    def gap(sweep):
        i = sweep.factors.index(sweep.factors[0])
        return (
            sweep.skill_by_model["synth-hazard-priority"][i]
            - sweep.skill_by_model["synth-background-biased"][i]
        )

    assert gap(high) > gap(low)


def test_rank_stability_flags_movement():
    stability = rank_stability(sweep_harm_model(_matrices()))
    assert set(stability) == set(PROFILES)
    for entry in stability.values():
        assert entry["best_rank"] <= entry["worst_rank"]
        assert isinstance(entry["stable"], bool)


def test_sweep_requires_models():
    with pytest.raises(ValueError, match="no models"):
        sweep_harm_model({})


def test_sweep_costs_no_extra_traversal():
    """The sweep must work from confusion matrices alone.

    If it ever needed the images again, a hundred-point sweep would become a
    hundred passes over the dataset and would quietly stop being run.
    """
    mats = _matrices()
    sweep = sweep_harm_model(mats, factors=(0.5, 1.0, 2.0, 5.0))
    assert len(sweep.factors) == 4
    assert all(len(v) == 4 for v in sweep.skill_by_model.values())


# ---------------------------------------------------------------------------
# Camera sweep
# ---------------------------------------------------------------------------


def test_camera_variants_stay_inside_the_image():
    from drivemetrics import camera_variants

    variants = camera_variants(CAMVID_DEFAULT_CAMERA)
    assert len(variants) > 10
    for cam in variants:
        assert 0 <= cam.horizon_row < cam.image_height
        assert cam.focal_px > 0
        assert "swept" in cam.source
