"""Contracts for the approved segmentation model registry."""

from __future__ import annotations

import subprocess
import sys
from types import ModuleType
from typing import Any

import pytest


def load_registry_module() -> ModuleType:
    try:
        from drivemetrics.models import registry
    except ImportError:
        pytest.fail("drivemetrics.models.registry is missing", pytrace=False)
    return registry


class FakeWeight:
    """A parameter stand-in that only needs a comparable shape."""

    def __init__(self, shape: tuple[int, ...] = (1,)) -> None:
        self.shape = shape


def install_fake_transformers(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Install a Transformers stand-in that records how each architecture is built."""

    calls: dict[str, list[Any]] = {
        "segformer_config": [],
        "from_pretrained": [],
        "backbone_config": [],
        "upernet_labels": [],
        "backbone_loaded": [],
    }

    class FakeSegformerConfig:
        def __init__(self, **keywords: Any) -> None:
            calls["segformer_config"].append(keywords)
            self.keywords = keywords
            self.num_labels = keywords.get("num_labels")

    class FakeSegformer:
        def __init__(self, config: Any) -> None:
            self.config = config

        @classmethod
        def from_pretrained(cls, checkpoint: str, **keywords: Any) -> Any:
            calls["from_pretrained"].append((checkpoint, keywords))
            return cls(FakeSegformerConfig(**keywords))

    class FakeBackboneConfig:
        def __init__(self, **keywords: Any) -> None:
            calls["backbone_config"].append((type(self).__name__, keywords))
            self.keywords = keywords

    class FakeConvNextV2Config(FakeBackboneConfig):
        pass

    class FakeDinov2Config(FakeBackboneConfig):
        pass

    class FakePretrainedBackbone:
        @classmethod
        def from_pretrained(cls, checkpoint: str) -> Any:
            calls["from_pretrained"].append((checkpoint, {}))
            return cls()

        def state_dict(self) -> dict[str, FakeWeight]:
            # One parameter the backbone shares, and a classification head it does not.
            return {"encoder.weight": FakeWeight(), "classifier.weight": FakeWeight((2,))}

    class FakeConvNextV2Model(FakePretrainedBackbone):
        pass

    class FakeDinov2Model(FakePretrainedBackbone):
        pass

    class FakeBackbone:
        def state_dict(self) -> dict[str, FakeWeight]:
            return {"encoder.weight": FakeWeight(), "hidden_states_norms.weight": FakeWeight()}

        def load_state_dict(self, values: dict[str, Any], strict: bool = True) -> None:
            calls["backbone_loaded"].append((sorted(values), strict))

    class FakeUperNetConfig:
        def __init__(self, backbone_config: Any) -> None:
            self.backbone_config = backbone_config
            self.num_labels: int | None = None

    class FakeUperNet:
        def __init__(self, config: Any) -> None:
            calls["upernet_labels"].append(config.num_labels)
            self.config = config
            self.backbone = FakeBackbone()

    transformers = ModuleType("transformers")
    for name, value in (
        ("SegformerConfig", FakeSegformerConfig),
        ("SegformerForSemanticSegmentation", FakeSegformer),
        ("ConvNextV2Config", FakeConvNextV2Config),
        ("ConvNextV2Model", FakeConvNextV2Model),
        ("Dinov2Config", FakeDinov2Config),
        ("Dinov2Model", FakeDinov2Model),
        ("UperNetConfig", FakeUperNetConfig),
        ("UperNetForSemanticSegmentation", FakeUperNet),
    ):
        setattr(transformers, name, value)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return calls


def test_registry_exposes_exactly_the_three_approved_models() -> None:
    """A fourth architecture would silently change the locked comparison protocol."""

    registry = load_registry_module()

    assert registry.APPROVED_MODEL_NAMES == (
        "segformer_b2",
        "upernet_convnextv2_tiny",
        "upernet_dinov2_small",
    )


def test_an_unapproved_model_name_fails_closed() -> None:
    """Falling back to a default architecture would publish an unlabelled model."""

    registry = load_registry_module()

    with pytest.raises(ValueError, match="approved"):
        registry.create_model("setr", 19, False)


@pytest.mark.parametrize(
    ("name", "expected_config"),
    [
        ("upernet_convnextv2_tiny", "FakeConvNextV2Config"),
        ("upernet_dinov2_small", "FakeDinov2Config"),
    ],
)
def test_each_upernet_variant_builds_its_own_backbone_at_the_requested_class_count(
    name: str,
    expected_config: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two variants sharing one decoder must still differ in exactly the backbone."""

    registry = load_registry_module()
    calls = install_fake_transformers(monkeypatch)

    registry.create_model(name, 19, False)

    assert [entry[0] for entry in calls["backbone_config"]] == [expected_config]
    assert calls["upernet_labels"] == [19]


def test_pretraining_loads_the_backbone_only_and_leaves_the_decoder_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A segmentation-pretrained decoder would confound ranking with pretraining data.

    Only parameters the classification model and the segmentation backbone share
    are copied. The per-stage output norms the wrapper adds have no counterpart
    and must start fresh, and the classification head must never be copied.
    """

    registry = load_registry_module()
    calls = install_fake_transformers(monkeypatch)

    registry.create_model("upernet_convnextv2_tiny", 19, True)

    assert calls["from_pretrained"] == [(registry.CONVNEXTV2_BACKBONE_CHECKPOINT, {})]
    loaded, strict = calls["backbone_loaded"][0]
    assert loaded == ["encoder.weight"]
    assert strict is False


def test_no_weights_are_fetched_when_pretraining_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hidden download would break the offline smoke run and the no-weights test."""

    registry = load_registry_module()
    calls = install_fake_transformers(monkeypatch)

    registry.create_model("upernet_dinov2_small", 19, False)

    assert calls["from_pretrained"] == []
    assert calls["backbone_loaded"] == []


def test_pretrained_segformer_loads_only_the_imagenet_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Cityscapes-finetuned checkpoint would leak segmentation supervision into the study."""

    registry = load_registry_module()
    calls = install_fake_transformers(monkeypatch)

    registry.create_model("segformer_b2", 19, True)

    assert len(calls["from_pretrained"]) == 1
    checkpoint, keywords = calls["from_pretrained"][0]
    assert checkpoint == registry.SEGFORMER_ENCODER_CHECKPOINT
    assert keywords["num_labels"] == 19


def test_segformer_without_pretraining_is_built_from_configuration_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hidden download would break the offline smoke run and the no-weights integration test.

    The label count is set on the configuration attribute rather than passed as a
    constructor keyword, because ``num_labels`` reaches ``SegformerConfig`` only
    through untyped ``PretrainedConfig`` keyword arguments.
    """

    registry = load_registry_module()
    calls = install_fake_transformers(monkeypatch)

    model = registry.create_model("segformer_b2", 19, False)

    assert calls["from_pretrained"] == []
    assert calls["segformer_config"] == [registry.SEGFORMER_B2_GEOMETRY]
    assert model.module.config.num_labels == 19  # type: ignore[attr-defined]


@pytest.mark.parametrize("num_classes", [1, 0, -1, True, 2.0])
def test_num_classes_below_two_fails_closed(num_classes: object) -> None:
    """A single-class head cannot express a confusion matrix or any risk-weighted metric."""

    registry = load_registry_module()

    with pytest.raises(ValueError, match="at least two"):
        registry.create_model("upernet_convnextv2_tiny", num_classes, False)  # type: ignore[arg-type]


@pytest.mark.parametrize("pretrained", ["yes", 1, None])
def test_a_non_boolean_pretrained_flag_fails_closed(pretrained: object) -> None:
    """A truthy string would silently select pretraining that the run record cannot describe."""

    registry = load_registry_module()

    with pytest.raises(TypeError, match="boolean"):
        registry.create_model("upernet_convnextv2_tiny", 19, pretrained)  # type: ignore[arg-type]


def test_importing_the_pure_metric_core_never_loads_torch() -> None:
    """Eager Torch imports would make the CPU metric core unusable in the base environment."""

    load_registry_module()
    program = (
        "import sys; import drivemetrics.metrics, drivemetrics.analysis; "
        "sys.exit(1 if any(name.split('.')[0] in {'torch', 'torchvision', 'transformers'} "
        "for name in sys.modules) else 0)"
    )

    result = subprocess.run([sys.executable, "-c", program], check=False, capture_output=True)

    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")


def test_models_package_exports_the_public_entry_points() -> None:
    """Training, evaluation, and report stages consume these through the package entry point."""

    import drivemetrics.models as models

    registry = load_registry_module()
    assert models.create_model is registry.create_model
    assert models.APPROVED_MODEL_NAMES == registry.APPROVED_MODEL_NAMES


def test_a_backbone_that_matches_no_pretrained_parameter_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silently training from scratch would look like pretraining and rank differently.

    `load_state_dict` is called with `strict=False` so the wrapper's own norms can
    start fresh. That same leniency would accept a checkpoint sharing nothing at
    all, and the run record would still claim the backbone was pretrained.
    """

    registry = load_registry_module()
    install_fake_transformers(monkeypatch)
    transformers = sys.modules["transformers"]

    class Foreign(transformers.ConvNextV2Model):  # type: ignore[misc, name-defined]
        def state_dict(self) -> dict[str, Any]:
            return {"nothing.in.common": FakeWeight()}

    monkeypatch.setattr(transformers, "ConvNextV2Model", Foreign)

    with pytest.raises(ValueError, match="no pretrained parameter matched"):
        registry.create_model("upernet_convnextv2_tiny", 19, True)
