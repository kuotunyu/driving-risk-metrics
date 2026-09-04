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
        ("- not\n- a\n- mapping\n", r"^risk profile document must be a mapping$"),
        (profile_yaml(extra="name: tiny"), r"^duplicate mapping key: 'name'$"),
        (
            profile_yaml(extra="unexpected: true"),
            r"^1 validation error for RiskProfileDocument\nunexpected\n  Extra inputs are not permitted",
        ),
        (
            profile_yaml(schema_version="bdd100k-risk-profile/v2"),
            r"^1 validation error for RiskProfileDocument\nschema_version\n  Input should be 'bdd100k-risk-profile/v1'",
        ),
        (
            profile_yaml(taxonomy="other"),
            r"^1 validation error for RiskProfileDocument\ntaxonomy\n  Input should be 'bdd100k-semantic-train-id/v1'",
        ),
        (
            profile_yaml(sensitivity=1.5),
            r"^1 validation error for RiskProfileDocument\nsensitivity\n  Value error, sensitivity must be one of 0\.5, 1\.0, or 2\.0",
        ),
        (
            profile_yaml(
                costs="""  - {class_id: 0, class_name: road, cost: 1.0, extra: true}
  - {class_id: 1, class_name: sidewalk, cost: 1.0}""",
                critical="[]",
            ),
            r"^1 validation error for RiskProfileDocument\nclass_costs\.0\.extra\n  Extra inputs are not permitted",
        ),
        (
            profile_yaml(
                costs="""  - {class_id: 0, class_name: road, cost: 1.0, cost: 1.0}""",
                critical="[]",
            ),
            r"^duplicate mapping key: 'cost'$",
        ),
        (
            profile_yaml(
                costs="""  - {class_id: 0, class_name: road, cost: 1.0}
  - {class_id: 0, class_name: road, cost: 1.0}""",
                critical="[]",
            ),
            r"^1 validation error for RiskProfileDocument\n  Value error, class cost IDs must be unique",
        ),
        (
            profile_yaml(
                costs="""  - {class_id: 19, class_name: road, cost: 1.0}""",
                critical="[]",
            ),
            r"^1 validation error for RiskProfileDocument\nclass_costs\.0\.class_id\n  Value error, class ID is outside the BDD100K taxonomy",
        ),
        (
            profile_yaml(
                costs="""  - {class_id: 0, class_name: sidewalk, cost: 1.0}""",
                critical="[]",
            ),
            r"^1 validation error for RiskProfileDocument\nclass_costs\.0\n  Value error, class name must be 'road' for ID 0",
        ),
        (
            profile_yaml(
                costs="""  - {class_id: 0, class_name: road, cost: -1.0}
  - {class_id: 1, class_name: sidewalk, cost: 3.0}""",
                critical="[]",
            ),
            r"^1 validation error for RiskProfileDocument\nclass_costs\.0\n  Value error, class costs must be nonnegative",
        ),
        (
            profile_yaml(
                costs="""  - {class_id: 0, class_name: road, cost: .inf}""",
                critical="[]",
            ),
            r"^1 validation error for RiskProfileDocument\nclass_costs\.0\n  Value error, class costs must be finite",
        ),
        (
            profile_yaml(
                costs="""  - {class_id: 0, class_name: road, cost: 0.0}
  - {class_id: 1, class_name: sidewalk, cost: 0.0}""",
                critical="[]",
            ),
            r"^risk profile must contain at least one positive cost$",
        ),
        (
            profile_yaml(
                costs="""  - {class_id: 0, class_name: road, cost: 1.0}
  - {class_id: 1, class_name: sidewalk, cost: 2.0}""",
                critical="[]",
            ),
            r"^mean non-zero class cost must equal 1 within 1e-12$",
        ),
        (
            profile_yaml(critical="[2, 2]"),
            r"^1 validation error for RiskProfileDocument\n  Value error, critical class IDs must be unique",
        ),
        (
            profile_yaml(critical="[3]"),
            r"^1 validation error for RiskProfileDocument\n  Value error, critical class IDs must be declared in class_costs",
        ),
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

    with pytest.raises(ValueError, match=r"^risk profile filename must match its declared name"):
        profiles.load_risk_profile(path)


def test_loader_accepts_only_yaml_extension(tmp_path: Path) -> None:
    profiles = load_profiles_module()
    path = tmp_path / "tiny.txt"
    path.write_text(profile_yaml(), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^risk profile path must use the \.yaml extension$"):
        profiles.load_risk_profile(path)


def test_the_declared_class_count_matches_every_committed_risk_profile() -> None:
    """A taxonomy that drifts from the profiles would silently mis-index every cost."""

    import yaml

    from drivemetrics.data.bdd100k import NUM_TRAIN_CLASSES

    profiles = sorted((REPO_ROOT / "configs" / "risk_profiles").glob("*.yaml"))
    assert profiles

    for path in profiles:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert len(document["class_costs"]) == NUM_TRAIN_CLASSES


@pytest.mark.parametrize(
    ("split_name", "expected"),
    [
        ("locked_validation", ("images/10k/val", "labels/sem_seg/masks/val")),
        ("train", ("images/10k/train", "labels/sem_seg/masks/train")),
        ("calibration", ("images/10k/train", "labels/sem_seg/masks/train")),
        ("source_train", ("images/10k/train", "labels/sem_seg/masks/train")),
    ],
)
def test_each_cohort_resolves_to_its_own_protocol_directories(
    split_name: str,
    expected: tuple[str, str],
) -> None:
    """Reading the calibration cohort from the validation tree would score the wrong files."""

    from drivemetrics.protocol.config import load_protocol, split_paths

    protocol = load_protocol(REPO_ROOT / "configs" / "protocols" / "bdd100k_semseg_v1.yaml")

    assert split_paths(protocol.protocol, split_name) == expected
