"""Real-framework training backend for the locked protocol.

Every random choice belongs to the engine: this backend receives the sample IDs
and the augmentation draw of each micro batch and never draws anything itself.
Torch is imported lazily inside the methods, so importing the package stays free
of the training extra.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from drivemetrics.data.bdd100k import NUM_TRAIN_CLASSES
from drivemetrics.data.manifest import DatasetManifest, load_manifest
from drivemetrics.data.transforms import prepare_sample
from drivemetrics.models.adapters import SegmentationAdapter
from drivemetrics.models.registry import ModelName, create_model
from drivemetrics.protocol.config import (
    BDD100KSemanticProtocolV1,
    load_protocol,
    split_paths,
)
from drivemetrics.training.losses import IGNORE_INDEX


@dataclass
class TorchTrainingState:
    """The framework objects one run mutates, plus its accumulation counter."""

    adapter: SegmentationAdapter
    optimizer: Any
    pending_micro_batches: int = 0


@dataclass(frozen=True)
class PreparedBatch:
    """One stacked micro batch already on the model canvas."""

    images: np.ndarray
    masks: np.ndarray


class TorchTrainingBackend:
    """Execute one locked training run on Torch, one micro batch at a time.

    Gradients accumulate across a window and are divided by the number of micro
    batches in that window before the single optimizer step, which reproduces the
    mean loss of the whole effective batch whenever the micro batches carry the
    same number of valid pixels.
    """

    def __init__(
        self,
        data_root: Path,
        manifest: DatasetManifest,
        protocol: BDD100KSemanticProtocolV1,
        *,
        device: str = "cpu",
        pretrained: bool = True,
    ) -> None:
        image_root_name, label_root_name = split_paths(protocol, manifest.split_name)
        image_root = data_root / image_root_name
        label_root = data_root / label_root_name
        self._paths = {
            sample_id: (
                image_root / manifest.relative_image_paths[position],
                label_root / manifest.relative_label_paths[position],
            )
            for position, sample_id in enumerate(manifest.sample_ids)
        }
        self._device = device
        self._pretrained = pretrained

    def seed_all(self, seed: int) -> None:
        """Seed every framework random source before any state is created."""

        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def create_training_state(
        self,
        model_name: ModelName,
        optimizer: Mapping[str, float | str],
    ) -> TorchTrainingState:
        """Build the approved architecture and exactly the pinned optimizer."""

        import torch

        adapter = create_model(model_name, NUM_TRAIN_CLASSES, self._pretrained)
        parameters = adapter.trainable_parameters()
        if optimizer["optimizer"] == "sgd":
            built: Any = torch.optim.SGD(
                parameters,
                lr=float(optimizer["learning_rate"]),
                momentum=float(optimizer["momentum"]),
                weight_decay=float(optimizer["weight_decay"]),
            )
        else:
            built = torch.optim.AdamW(
                parameters,
                lr=float(optimizer["learning_rate"]),
                weight_decay=float(optimizer["weight_decay"]),
            )
        return TorchTrainingState(adapter=adapter, optimizer=built)

    def load_batch(self, batch: Sequence[tuple[str, float]]) -> PreparedBatch:
        """Resolve, read, and preprocess one micro batch using the engine draws."""

        images: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for sample_id, flip_draw in batch:
            if sample_id not in self._paths:
                raise KeyError(f"sample {sample_id!r} is not in the frozen manifest")
            image_path, label_path = self._paths[sample_id]
            with Image.open(image_path) as handle:
                image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
            with Image.open(label_path) as handle:
                mask = np.asarray(handle, dtype=np.uint8)
            prepared = prepare_sample(image, mask, training=True, flip_draw=flip_draw)
            images.append(prepared.image_chw)
            masks.append(prepared.mask_hw)
        return PreparedBatch(images=np.stack(images), masks=np.stack(masks))

    def run_step(
        self,
        state: Any,
        batch: Any,
        learning_rate: float,
        *,
        apply_update: bool,
    ) -> float:
        """Accumulate one micro batch and, when the window closes, update once.

        ``state`` is a :class:`TorchTrainingState` and ``batch`` is a sequence of
        ``(sample_id, flip_draw)`` pairs. Both are annotated loosely because the
        engine treats them as opaque, which is what the injected-backend
        contract requires.
        """

        import torch

        prepared = self.load_batch(batch)
        images = torch.from_numpy(prepared.images).to(self._device)
        masks = torch.from_numpy(prepared.masks).to(self._device)
        if not bool((masks != IGNORE_INDEX).any()):
            raise ValueError("micro batch has no valid pixel to learn from")

        module = state.adapter.module
        module.train()
        for group in state.optimizer.param_groups:
            group["lr"] = learning_rate

        logits = state.adapter.extract_logits(module(images))
        logits = torch.nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        loss = torch.nn.functional.cross_entropy(logits, masks, ignore_index=IGNORE_INDEX)
        loss.backward()
        state.pending_micro_batches += 1

        if apply_update:
            scale = float(state.pending_micro_batches)
            for parameter in module.parameters():
                if parameter.grad is not None:
                    parameter.grad /= scale
            state.optimizer.step()
            state.optimizer.zero_grad(set_to_none=True)
            state.pending_micro_batches = 0
        return float(loss.detach())

    def save_checkpoint(
        self,
        state: Any,
        path: Path,
        metadata: Mapping[str, object],
    ) -> str:
        """Persist the final weights beside their metadata and hash the file."""

        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": state.adapter.module.state_dict(), "metadata": dict(metadata)},
            path,
        )
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def load_checkpoint(self, path: Path, expected_metadata: Mapping[str, object]) -> Any:
        """Load one of our own checkpoints, failing closed on a metadata mismatch."""

        import torch

        payload = torch.load(path, map_location=self._device, weights_only=False)
        if payload["metadata"] != dict(expected_metadata):
            raise ValueError("checkpoint metadata does not match the expected run")
        return payload


def build_training_backend(
    config_path: Path,
    manifest_path: Path,
    data_root: Path,
    *,
    device: str = "cpu",
    pretrained: bool = True,
) -> TorchTrainingBackend:
    """Construct the Torch backend described by one training run configuration.

    The command line never resolves protocols or manifests itself; it calls this
    factory so the same run configuration drives both the engine and its backend.
    """

    from drivemetrics.training.engine import load_run_config

    run_config, _ = load_run_config(config_path)
    protocol = load_protocol(config_path.parent / run_config.protocol_path).protocol
    return TorchTrainingBackend(
        data_root,
        load_manifest(manifest_path),
        protocol,
        device=device,
        pretrained=pretrained,
    )
