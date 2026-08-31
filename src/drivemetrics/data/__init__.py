"""Dataset manifests, frozen cohorts, and data adapters."""

from .camvid import CamVidAdapter, CamVidSmokeConfig, load_camvid_config
from .manifest import DatasetManifest, build_paired_manifest
from .splits import freeze_bdd100k_split, validate_locked_split
from .transforms import PreparedSample, prepare_sample, restore_prediction

__all__ = [
    "CamVidAdapter",
    "CamVidSmokeConfig",
    "DatasetManifest",
    "PreparedSample",
    "build_paired_manifest",
    "freeze_bdd100k_split",
    "load_camvid_config",
    "prepare_sample",
    "restore_prediction",
    "validate_locked_split",
]
