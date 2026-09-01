"""Dataset manifests, frozen cohorts, and data adapters."""

from .bdd100k import (
    NUM_TRAIN_CLASSES,
    SemanticInstanceIntersection,
    SemanticInstanceMismatchError,
    semantic_instance_intersection,
)
from .camvid import CamVidAdapter, CamVidSmokeConfig, load_camvid_config
from .manifest import (
    DatasetManifest,
    build_paired_manifest,
    load_manifest,
    save_manifest,
    subset_manifest,
)
from .preflight import PreflightResult, run_preflight
from .splits import freeze_bdd100k_split, validate_locked_split
from .transforms import (
    PreparedSample,
    prepare_sample,
    restore_index_map,
    restore_prediction,
)

__all__ = [
    "NUM_TRAIN_CLASSES",
    "CamVidAdapter",
    "CamVidSmokeConfig",
    "DatasetManifest",
    "PreflightResult",
    "PreparedSample",
    "SemanticInstanceIntersection",
    "SemanticInstanceMismatchError",
    "build_paired_manifest",
    "freeze_bdd100k_split",
    "load_camvid_config",
    "load_manifest",
    "prepare_sample",
    "restore_index_map",
    "restore_prediction",
    "run_preflight",
    "save_manifest",
    "semantic_instance_intersection",
    "subset_manifest",
    "validate_locked_split",
]
