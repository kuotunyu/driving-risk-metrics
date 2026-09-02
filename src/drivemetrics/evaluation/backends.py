"""Real-framework evaluation backend that restores one final checkpoint.

Torch is imported lazily inside the method, so importing the package stays free
of the training extra.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from drivemetrics.data.bdd100k import NUM_TRAIN_CLASSES
from drivemetrics.models.adapters import SegmentationAdapter
from drivemetrics.models.registry import ModelName, create_model


@dataclass(frozen=True)
class TorchEvaluationBackend:
    """Rebuild the architecture a checkpoint names and restore exactly its weights.

    ``device`` has no default. With one, every locked-cohort evaluation would run
    on the CPU beside an idle accelerator, with no error and no symptom beyond
    being slow, which is exactly what the first formal run did in training.
    """

    device: str

    def load_model(
        self,
        checkpoint_path: Path,
    ) -> tuple[SegmentationAdapter, Mapping[str, object]]:
        """Restore the model and the metadata recorded with its checkpoint.

        Only checkpoints this project wrote are loadable here, because the
        metadata block is required and must name an approved architecture. The
        weights are never trusted to imply an architecture on their own.
        """

        import torch

        payload: dict[str, Any] = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        metadata = dict(payload["metadata"])
        if "model" not in metadata:
            raise ValueError("checkpoint metadata must name the model it was trained for")

        adapter = create_model(cast(ModelName, metadata["model"]), NUM_TRAIN_CLASSES, False)
        adapter.module.load_state_dict(payload["model"])
        # Weights are restored on the CPU and only then moved, because moving
        # first would leave nothing for load_state_dict to copy out of.
        adapter.module.to(self.device)
        return adapter, metadata
