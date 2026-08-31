"""Strict versioned risk-profile YAML contracts."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILE_ROOT = REPO_ROOT / "configs" / "risk_profiles"


def load_profiles_module() -> ModuleType:
    try:
        from drivemetrics.protocol import risk_profiles
    except ImportError:
        pytest.fail("drivemetrics.protocol.risk_profiles is missing", pytrace=False)
    return risk_profiles


def profile_yaml(
    *,
    schema_version: str = "bdd100k-risk-profile/v1",
    name: str = "tiny",
    taxonomy: str = "bdd100k-semantic-train-id/v1",
    sensitivity: float = 1.0,
    costs: str = """  - {class_id: 0, class_name: road, cost: 0.5}
  - {class_id: 1, class_name: sidewalk, cost: 1.0}
  - {class_id: 2, class_name: building, cost: 1.5}""",
    critical: str = "[2]",
    extra: str = "",
) -> str:
    return f"""schema_version: {schema_version}
name: {name}
taxonomy: {taxonomy}
sensitivity: {sensitivity}
class_costs:
{costs}
critical_class_ids: {critical}
{extra}"""


def write_profile(tmp_path: Path, content: str, *, name: str = "tiny") -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_all_committed_profiles_load_with_complete_normalized_taxonomy() -> None:
    profiles = load_profiles_module()
    expected_critical = {
        "balanced": (),
        "vru_priority": (11, 12, 17, 18),
        "drivable_boundary": (0, 1),
    }

    for name, critical_ids in expected_critical.items():
        loaded = profiles.load_risk_profile(PROFILE_ROOT / f"{name}.yaml")
        assert loaded.name == name
        assert loaded.sensitivity == 1.0
        assert loaded.critical_class_ids == critical_ids
        assert tuple(loaded.class_cost) == tuple(range(19))
        assert sum(loaded.class_cost.values()) / 19 == pytest.approx(1.0, abs=1e-12)

    balanced = profiles.load_risk_profile(PROFILE_ROOT / "balanced.yaml")
    vru = profiles.load_risk_profile(PROFILE_ROOT / "vru_priority.yaml")
    boundary = profiles.load_risk_profile(PROFILE_ROOT / "drivable_boundary.yaml")
    assert set(balanced.class_cost.values()) == {1.0}
    vru_ids = {11, 12, 17, 18}
    assert len({vru.class_cost[class_id] for class_id in vru_ids}) == 1
    assert len({vru.class_cost[class_id] for class_id in set(range(19)) - vru_ids}) == 1
    assert vru.class_cost[11] > vru.class_cost[0]
    assert boundary.class_cost[0] > boundary.class_cost[2]
    assert boundary.class_cost[0] == boundary.class_cost[1]
    assert len({boundary.class_cost[class_id] for class_id in range(2, 19)}) == 1


def test_loader_accepts_valid_partial_taxonomy_profile(tmp_path: Path) -> None:
    profiles = load_profiles_module()
    path = write_profile(tmp_path, profile_yaml())

    loaded = profiles.load_risk_profile(path)

    assert loaded.name == "tiny"
    assert dict(loaded.class_cost) == {0: 0.5, 1: 1.0, 2: 1.5}
    assert loaded.critical_class_ids == (2,)
    assert loaded.sensitivity == 1.0


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("- not\n- a\n- mapping\n", "mapping"),
        (profile_yaml(extra="name: tiny"), "duplicate mapping key"),
        (profile_yaml(extra="unexpected: true"), "unexpected"),
        (profile_yaml(schema_version="bdd100k-risk-profile/v2"), "schema_version"),
        (profile_yaml(taxonomy="other"), "taxonomy"),
        (profile_yaml(sensitivity=1.5), "sensitivity"),
        (
            profile_yaml(
                costs="""  - {class_id: 0, class_name: road, cost: 1.0, extra: true}
  - {class_id: 1, class_name: sidewalk, cost: 1.0}""",
                critical="[]",
            ),
            "extra",
        ),
        (
            profile_yaml(
                costs="""  - {class_id: 0, class_name: road, cost: 1.0, cost: 1.0}""",
                critical="[]",
            ),
            "duplicate mapping key",
        ),
        (
            profile_yaml(
                costs="""  - {class_id: 0, class_name: road, cost: 1.0}
  - {class_id: 0, class_name: road, cost: 1.0}""",
                critical="[]",
            ),
            "unique",
        ),
        (
            profile_yaml(
                costs="""  - {class_id: 19, class_name: road, cost: 1.0}""",
                critical="[]",
            ),
            "taxonomy",
        ),
        (
            profile_yaml(
                costs="""  - {class_id: 0, class_name: sidewalk, cost: 1.0}""",
                critical="[]",
            ),
            "name",
        ),
        (
            profile_yaml(
                costs="""  - {class_id: 0, class_name: road, cost: -1.0}
  - {class_id: 1, class_name: sidewalk, cost: 3.0}""",
                critical="[]",
            ),
            "nonnegative",
        ),
        (
            profile_yaml(
                costs="""  - {class_id: 0, class_name: road, cost: .inf}""",
                critical="[]",
            ),
            "finite",
        ),
        (
            profile_yaml(
                costs="""  - {class_id: 0, class_name: road, cost: 0.0}
  - {class_id: 1, class_name: sidewalk, cost: 0.0}""",
                critical="[]",
            ),
            "positive",
        ),
        (
            profile_yaml(
                costs="""  - {class_id: 0, class_name: road, cost: 1.0}
  - {class_id: 1, class_name: sidewalk, cost: 2.0}""",
                critical="[]",
            ),
            "mean non-zero",
        ),
        (profile_yaml(critical="[2, 2]"), "unique"),
        (profile_yaml(critical="[3]"), "declared"),
    ],
)
def test_loader_rejects_invalid_documents(tmp_path: Path, content: str, expected: str) -> None:
    profiles = load_profiles_module()
    path = write_profile(tmp_path, content)

    with pytest.raises((TypeError, ValueError, ValidationError), match=expected):
        profiles.load_risk_profile(path)


def test_loader_rejects_filename_name_mismatch(tmp_path: Path) -> None:
    profiles = load_profiles_module()
    path = write_profile(tmp_path, profile_yaml(name="different"))

    with pytest.raises(ValueError, match="filename"):
        profiles.load_risk_profile(path)


def test_loader_accepts_only_yaml_extension(tmp_path: Path) -> None:
    profiles = load_profiles_module()
    path = tmp_path / "tiny.txt"
    path.write_text(profile_yaml(), encoding="utf-8")

    with pytest.raises(ValueError, match=r"\.yaml"):
        profiles.load_risk_profile(path)
