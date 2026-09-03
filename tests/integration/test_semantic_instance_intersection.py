"""Filesystem-level contracts for BDD100K semantic/instance cohort audits."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest


def load_bdd100k_module() -> ModuleType:
    try:
        from drivemetrics.data import bdd100k
    except ImportError:
        pytest.fail("drivemetrics.data.bdd100k is missing", pytrace=False)
    return bdd100k


def touch_labels(root: Path, relative_paths: tuple[str, ...]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for relative_path in relative_paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        paths.append(path)
    return tuple(paths)


def test_semantic_instance_mismatch_fails_closed_with_auditable_reason(
    tmp_path: Path,
) -> None:
    bdd100k = load_bdd100k_module()
    semantic = touch_labels(
        tmp_path,
        (
            "semantic/a_train_id.png",
            "semantic/b_train_id.png",
        ),
    )
    instance = touch_labels(tmp_path, ("instance/a.png",))

    with pytest.raises(bdd100k.SemanticInstanceMismatchError) as caught:
        bdd100k.semantic_instance_intersection(semantic, instance)

    assert [(drop.sample_id, drop.reason) for drop in caught.value.dropped] == [
        ("b", "missing_instance_annotation")
    ]
    assert "allow_audited_intersection=True" in str(caught.value)


def test_explicit_audited_intersection_returns_exact_pairs_and_drops(
    tmp_path: Path,
) -> None:
    bdd100k = load_bdd100k_module()
    semantic = touch_labels(
        tmp_path,
        (
            "semantic/b_train_id.png",
            "semantic/a_train_id.png",
        ),
    )
    instance = touch_labels(
        tmp_path,
        (
            "instance/c.png",
            "instance/a.png",
        ),
    )

    result = bdd100k.semantic_instance_intersection(
        semantic,
        instance,
        allow_audited_intersection=True,
    )

    assert result.retained_count == 1
    assert tuple(pair.sample_id for pair in result.pairs) == ("a",)
    assert result.pairs[0].semantic_label_path == semantic[1]
    assert result.pairs[0].instance_label_path == instance[1]
    assert [(drop.sample_id, drop.reason) for drop in result.dropped] == [
        ("b", "missing_instance_annotation"),
        ("c", "missing_semantic_annotation"),
    ]


def test_exact_intersection_needs_no_opt_in_and_has_no_drops(tmp_path: Path) -> None:
    bdd100k = load_bdd100k_module()
    semantic = touch_labels(tmp_path, ("semantic/a_train_id.png",))
    instance = touch_labels(tmp_path, ("instance/a.png",))

    result = bdd100k.semantic_instance_intersection(semantic, instance)

    assert result.retained_count == 1
    assert result.dropped == ()


def test_instance_filename_parser_rejects_non_png_artifacts() -> None:
    bdd100k = load_bdd100k_module()

    with pytest.raises(ValueError, match="instance label filename"):
        bdd100k.instance_label_sample_id(Path("sample-a.jpg"))


@pytest.mark.parametrize("duplicate_side", ["semantic", "instance"])
def test_intersection_rejects_duplicate_sample_ids(tmp_path: Path, duplicate_side: str) -> None:
    bdd100k = load_bdd100k_module()
    semantic = touch_labels(tmp_path, ("semantic/a_train_id.png",))
    instance = touch_labels(tmp_path, ("instance/a.png",))
    if duplicate_side == "semantic":
        semantic += touch_labels(tmp_path, ("semantic-copy/a_train_id.png",))
    else:
        instance += touch_labels(tmp_path, ("instance-copy/a.png",))

    with pytest.raises(ValueError, match=f"duplicate {duplicate_side}"):
        bdd100k.semantic_instance_intersection(semantic, instance)


def test_audited_intersection_rejects_zero_retained_samples(tmp_path: Path) -> None:
    bdd100k = load_bdd100k_module()
    semantic = touch_labels(tmp_path, ("semantic/a_train_id.png",))
    instance = touch_labels(tmp_path, ("instance/b.png",))

    with pytest.raises(ValueError, match="at least one"):
        bdd100k.semantic_instance_intersection(
            semantic,
            instance,
            allow_audited_intersection=True,
        )


def test_default_disjoint_intersection_preserves_every_drop_reason(tmp_path: Path) -> None:
    bdd100k = load_bdd100k_module()
    semantic = touch_labels(tmp_path, ("semantic/a_train_id.png",))
    instance = touch_labels(tmp_path, ("instance/b.png",))

    with pytest.raises(bdd100k.SemanticInstanceMismatchError) as caught:
        bdd100k.semantic_instance_intersection(semantic, instance)

    assert [(drop.sample_id, drop.reason) for drop in caught.value.dropped] == [
        ("a", "missing_instance_annotation"),
        ("b", "missing_semantic_annotation"),
    ]


@pytest.mark.parametrize("allow", [1, "yes", None])
def test_intersection_requires_explicit_boolean_opt_in(tmp_path: Path, allow: object) -> None:
    bdd100k = load_bdd100k_module()
    semantic = touch_labels(tmp_path, ("semantic/a_train_id.png",))
    instance = touch_labels(tmp_path, ("instance/a.png",))

    with pytest.raises(TypeError, match=r"^allow_audited_intersection must be a boolean"):
        bdd100k.semantic_instance_intersection(
            semantic,
            instance,
            allow_audited_intersection=allow,  # type: ignore[arg-type]
        )


def test_data_package_exports_intersection_contract() -> None:
    import drivemetrics.data as data

    bdd100k = load_bdd100k_module()
    assert data.SemanticInstanceIntersection is bdd100k.SemanticInstanceIntersection
    assert data.SemanticInstanceMismatchError is bdd100k.SemanticInstanceMismatchError
    assert data.semantic_instance_intersection is bdd100k.semantic_instance_intersection
