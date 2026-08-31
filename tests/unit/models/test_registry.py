"""Contracts for the approved segmentation model registry."""

from __future__ import annotations

import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def load_registry_module() -> ModuleType:
    try:
        from drivemetrics.models import registry
    except ImportError:
        pytest.fail("drivemetrics.models.registry is missing", pytrace=False)
    return registry


class RecordingBuilder:
    """Stand in for one torchvision segmentation constructor and record its keywords."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **keywords: Any) -> SimpleNamespace:
        self.calls.append(keywords)
        return SimpleNamespace(architecture=self.name, keywords=keywords)


def install_fake_torchvision(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, RecordingBuilder]:
    """Install a torchvision stand-in so builder keywords are observable on CPU."""

    builders = {
        "fcn_resnet50": RecordingBuilder("fcn_resnet50"),
        "deeplabv3_resnet50": RecordingBuilder("deeplabv3_resnet50"),
    }
    segmentation = ModuleType("torchvision.models.segmentation")
    for name, builder in builders.items():
        setattr(segmentation, name, builder)
    models = ModuleType("torchvision.models")
    models.segmentation = segmentation  # type: ignore[attr-defined]
    torchvision = ModuleType("torchvision")
    torchvision.models = models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.models", models)
    monkeypatch.setitem(sys.modules, "torchvision.models.segmentation", segmentation)
    return builders


def install_fake_transformers(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Install a Transformers stand-in that records how SegFormer is constructed."""

    calls: dict[str, list[Any]] = {"config": [], "from_pretrained": []}

    class FakeSegformerConfig:
        def __init__(self, **keywords: Any) -> None:
            calls["config"].append(keywords)
            self.keywords = keywords
            self.num_labels = keywords.get("num_labels")

    class FakeSegformer:
        def __init__(self, config: FakeSegformerConfig) -> None:
            self.config = config

        @classmethod
        def from_pretrained(cls, checkpoint: str, **keywords: Any) -> FakeSegformer:
            calls["from_pretrained"].append((checkpoint, keywords))
            return cls(FakeSegformerConfig(**keywords))

    transformers = ModuleType("transformers")
    transformers.SegformerConfig = FakeSegformerConfig  # type: ignore[attr-defined]
    transformers.SegformerForSemanticSegmentation = FakeSegformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return calls


def test_registry_exposes_exactly_the_three_approved_models() -> None:
    """A fourth architecture would silently change the locked comparison protocol."""

    registry = load_registry_module()

    assert registry.APPROVED_MODEL_NAMES == (
        "fcn_resnet50",
        "deeplabv3_resnet50",
        "segformer_b0",
    )


def test_an_unapproved_model_name_fails_closed() -> None:
    """Falling back to a default architecture would publish an unlabelled model."""

    registry = load_registry_module()

    with pytest.raises(ValueError, match="approved"):
        registry.create_model("setr", 19, False)


@pytest.mark.parametrize("name", ["fcn_resnet50", "deeplabv3_resnet50"])
def test_torchvision_models_get_a_fresh_head_at_the_requested_class_count(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keeping the 21-class COCO head would score BDD100K classes through the wrong logits."""

    registry = load_registry_module()
    builders = install_fake_torchvision(monkeypatch)

    registry.create_model(name, 19, False)

    assert len(builders[name].calls) == 1
    keywords = builders[name].calls[0]
    assert keywords["num_classes"] == 19
    assert keywords["weights"] is None
    assert keywords["aux_loss"] is False


@pytest.mark.parametrize(
    ("pretrained", "expected_backbone"),
    [(True, "DEFAULT"), (False, None)],
)
def test_pretraining_only_initializes_the_torchvision_backbone(
    pretrained: bool,
    expected_backbone: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Segmentation-pretrained decoders would confound the ranking with pretraining data."""

    registry = load_registry_module()
    builders = install_fake_torchvision(monkeypatch)

    registry.create_model("fcn_resnet50", 19, pretrained)

    keywords = builders["fcn_resnet50"].calls[0]
    assert keywords["weights"] is None
    assert keywords["weights_backbone"] == expected_backbone


def test_pretrained_segformer_loads_only_the_imagenet_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Cityscapes-finetuned checkpoint would leak segmentation supervision into the study."""

    registry = load_registry_module()
    calls = install_fake_transformers(monkeypatch)

    registry.create_model("segformer_b0", 19, True)

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

    model = registry.create_model("segformer_b0", 19, False)

    assert calls["from_pretrained"] == []
    assert calls["config"] == [{}]
    assert model.module.config.num_labels == 19  # type: ignore[attr-defined]


@pytest.mark.parametrize("num_classes", [1, 0, -1, True, 2.0])
def test_num_classes_below_two_fails_closed(
    num_classes: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-class head cannot express a confusion matrix or any risk-weighted metric."""

    registry = load_registry_module()
    install_fake_torchvision(monkeypatch)

    with pytest.raises(ValueError, match="at least two"):
        registry.create_model("fcn_resnet50", num_classes, False)  # type: ignore[arg-type]


@pytest.mark.parametrize("pretrained", ["yes", 1, None])
def test_a_non_boolean_pretrained_flag_fails_closed(
    pretrained: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truthy string would silently select pretraining that the run record cannot describe."""

    registry = load_registry_module()
    install_fake_torchvision(monkeypatch)

    with pytest.raises(TypeError, match="boolean"):
        registry.create_model("fcn_resnet50", 19, pretrained)  # type: ignore[arg-type]


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
