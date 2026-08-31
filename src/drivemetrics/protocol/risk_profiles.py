"""Strict BDD100K class-cost profile loading and validation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from yaml.nodes import MappingNode

BDD100K_SEMANTIC_CLASS_NAMES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)
ALLOWED_SENSITIVITIES = (0.5, 1.0, 2.0)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects last-key-wins ambiguity."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate mapping key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class RiskProfile:
    """Normalized base costs and a declared critical-class sensitivity."""

    name: str
    class_cost: Mapping[int, float]
    critical_class_ids: tuple[int, ...]
    sensitivity: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("risk profile name must not be empty")

        copied_costs: dict[int, float] = {}
        for class_id, cost in self.class_cost.items():
            if isinstance(class_id, bool) or not isinstance(class_id, int):
                raise TypeError("class cost IDs must be integers")
            if class_id < 0 or class_id >= len(BDD100K_SEMANTIC_CLASS_NAMES):
                raise ValueError("class cost ID is outside the BDD100K taxonomy")
            numeric_cost = float(cost)
            if not math.isfinite(numeric_cost):
                raise ValueError("class costs must be finite")
            if numeric_cost < 0:
                raise ValueError("class costs must be nonnegative")
            copied_costs[class_id] = numeric_cost

        positive_costs = tuple(value for value in copied_costs.values() if value > 0)
        if not positive_costs:
            raise ValueError("risk profile must contain at least one positive cost")
        mean_nonzero = sum(positive_costs) / len(positive_costs)
        if not math.isclose(mean_nonzero, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("mean non-zero class cost must equal 1 within 1e-12")

        canonical_critical_ids = tuple(self.critical_class_ids)
        if len(set(canonical_critical_ids)) != len(canonical_critical_ids):
            raise ValueError("critical class IDs must be unique")
        for class_id in canonical_critical_ids:
            if isinstance(class_id, bool) or not isinstance(class_id, int):
                raise TypeError("critical class IDs must be integers")
            if class_id < 0 or class_id >= len(BDD100K_SEMANTIC_CLASS_NAMES):
                raise ValueError("critical class ID is outside the BDD100K taxonomy")
            if class_id not in copied_costs:
                raise ValueError("critical class ID must have a declared class cost")
        if type(self.sensitivity) is not float:
            raise TypeError("sensitivity must be a built-in float")
        if self.sensitivity not in ALLOWED_SENSITIVITIES:
            raise ValueError("sensitivity must be one of 0.5, 1.0, or 2.0")

        canonical_costs = dict(sorted(copied_costs.items()))
        object.__setattr__(self, "class_cost", MappingProxyType(canonical_costs))
        object.__setattr__(self, "critical_class_ids", canonical_critical_ids)
        object.__setattr__(self, "sensitivity", float(self.sensitivity))


class StrictRiskModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ClassCostDocument(StrictRiskModel):
    class_id: int
    class_name: str
    cost: float

    @field_validator("class_id")
    @classmethod
    def validate_class_id(cls, value: int) -> int:
        if value < 0 or value >= len(BDD100K_SEMANTIC_CLASS_NAMES):
            raise ValueError("class ID is outside the BDD100K taxonomy")
        return value

    @model_validator(mode="after")
    def validate_name_and_cost(self) -> ClassCostDocument:
        expected_name = BDD100K_SEMANTIC_CLASS_NAMES[self.class_id]
        if self.class_name != expected_name:
            raise ValueError(f"class name must be {expected_name!r} for ID {self.class_id}")
        if not math.isfinite(self.cost):
            raise ValueError("class costs must be finite")
        if self.cost < 0:
            raise ValueError("class costs must be nonnegative")
        return self


class RiskProfileDocument(StrictRiskModel):
    schema_version: Literal["bdd100k-risk-profile/v1"]
    name: str
    taxonomy: Literal["bdd100k-semantic-train-id/v1"]
    sensitivity: float
    class_costs: list[ClassCostDocument]
    critical_class_ids: list[int]

    @field_validator("sensitivity")
    @classmethod
    def validate_sensitivity(cls, value: float) -> float:
        if value not in ALLOWED_SENSITIVITIES:
            raise ValueError("sensitivity must be one of 0.5, 1.0, or 2.0")
        return value

    @model_validator(mode="after")
    def validate_ids(self) -> RiskProfileDocument:
        class_ids = tuple(entry.class_id for entry in self.class_costs)
        if len(set(class_ids)) != len(class_ids):
            raise ValueError("class cost IDs must be unique")
        if len(set(self.critical_class_ids)) != len(self.critical_class_ids):
            raise ValueError("critical class IDs must be unique")
        if any(class_id not in class_ids for class_id in self.critical_class_ids):
            raise ValueError("critical class IDs must be declared in class_costs")
        return self


def load_risk_profile(path: Path) -> RiskProfile:
    """Load a versioned YAML profile and reject semantic or naming drift."""

    if path.suffix != ".yaml":
        raise ValueError("risk profile path must use the .yaml extension")
    raw_value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    if not isinstance(raw_value, dict):
        raise TypeError("risk profile document must be a mapping")
    document = RiskProfileDocument.model_validate(raw_value)
    if path.stem != document.name:
        raise ValueError("risk profile filename must match its declared name")
    return RiskProfile(
        name=document.name,
        class_cost={entry.class_id: entry.cost for entry in document.class_costs},
        critical_class_ids=tuple(document.critical_class_ids),
        sensitivity=document.sensitivity,
    )
