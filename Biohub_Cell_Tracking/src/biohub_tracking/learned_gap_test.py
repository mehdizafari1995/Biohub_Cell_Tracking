"""Test-time inference helpers for learned_unet_gap_test_v1.

Applies locked ``learned_unet_infer_v1`` / ``learned_unet_holdout_v1``
detection plus the locked ``learned_unet_gap_v1`` overlay to every
discovered ``test/*.zarr``. Never opens ``train/`` volumes. Writes
``submission.csv`` + ``run_manifest.json``. Do not overwrite the no-gap
``learned_unet_test_v1`` slug. Do not retune. Do not score frozen ``6bba``
again. Not a competition submission until a human submits from the
Kaggle UI.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .learned_gap import LOCKED_GAP_HASH, LOCKED_GATES
from .learned_holdout import LOCKED_HOLDOUT_HASH
from .learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from .learned_split import KERNEL_SLUG as TRAIN_KERNEL_SLUG
from .learned_test import (
    ESTIMATE_HIDDEN_VOLUMES,
    HOLDOUT_KERNEL_SLUG,
    HOLDOUT_PEAK_MEMORY_MB,
    INFER_KERNEL_SLUG,
    KAGGLE_GPU_LIMIT_HOURS,
    SCALE_ZYX,
    TEST_KERNEL_SLUG,
    WEIGHTS_NAME,
    list_test_zarrs,
    refuse_hardcoded_public_stems,
    refuse_train_volume,
)

GAP_TEST_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-gap-test-v1"
LOCKED_TEST_HASH = "7a8725b818fc609b"
NOGAP_HOLDOUT_SEC_PER_VOLUME = 92.2
OVERLAY_SEC_PER_TWO_HOLDOUT = 9.687
OVERLAY_SEC_PER_VOLUME = OVERLAY_SEC_PER_TWO_HOLDOUT / 2.0
GAP_SEC_PER_VOLUME = NOGAP_HOLDOUT_SEC_PER_VOLUME + OVERLAY_SEC_PER_VOLUME


def gap_test_config_payload() -> dict[str, Any]:
    """Locked infer + locked gap overlay on discovered test volumes."""
    return {
        "label": "learned_unet_gap_test_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "holdout_label": "learned_unet_holdout_v1",
        "gap_label": "learned_unet_gap_v1",
        "test_label": "learned_unet_test_v1",
        "association_label": "association_v2",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "holdout_config_hash": LOCKED_HOLDOUT_HASH,
        "gap_config_hash": LOCKED_GAP_HASH,
        "test_parent_hash": LOCKED_TEST_HASH,
        "discovers_test_zarrs": True,
        "hardcoded_test_stems": [],
        "opens_train_volumes": False,
        "det_threshold": 0.5,
        "det_tta": True,
        "pool_kernel_um": 5.0,
        "linker": "association_v2_gap_overlay",
        "gate_um": float(LOCKED_GATES["gate_um"]),
        "max_gap_frames": int(LOCKED_GATES["max_gap_frames"]),
        "gap_scale": str(LOCKED_GATES["gap_scale"]),
        "assignment": str(LOCKED_GATES["assignment"]),
        "interpolate_gaps": bool(LOCKED_GATES["interpolate_gaps"]),
        "max_parents_per_node": int(LOCKED_GATES["max_parents_per_node"]),
        "max_children_per_node": int(LOCKED_GATES["max_children_per_node"]),
        "intensity_weight": float(LOCKED_GATES["intensity_weight"]),
        "downsample": [1, 4, 4],
        "normalize": "window_minmax",
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": True,
        "writes_run_manifest": True,
        "one_shot_holdout": False,
        "kernel_slug": GAP_TEST_KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
        "holdout_kernel_slug": HOLDOUT_KERNEL_SLUG,
        "test_kernel_slug": TEST_KERNEL_SLUG,
        "weights_name": WEIGHTS_NAME,
    }


def gap_test_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or gap_test_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_GAP_TEST_HASH = gap_test_config_hash(gap_test_config_payload())
assert LOCKED_GAP_TEST_HASH == "85379c198ebc7241"


def runtime_estimate(
    n_volumes: int,
    sec_per_volume: float = GAP_SEC_PER_VOLUME,
    limit_hours: float = KAGGLE_GPU_LIMIT_HOURS,
    peak_memory_mb: float = HOLDOUT_PEAK_MEMORY_MB,
) -> dict[str, Any]:
    """Estimate GPU time from holdout infer plus measured overlay cost."""
    n = int(n_volumes)
    sec = float(n) * float(sec_per_volume)
    hours = sec / 3600.0
    return {
        "n_volumes": n,
        "sec_per_volume": float(sec_per_volume),
        "nogap_sec_per_volume": NOGAP_HOLDOUT_SEC_PER_VOLUME,
        "overlay_sec_per_volume": OVERLAY_SEC_PER_VOLUME,
        "estimate_sec": sec,
        "estimate_hours": hours,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_observed": float(peak_memory_mb),
        "source": (
            "learned_unet_holdout_v1 v1 184.4s / 2 volumes + "
            "learned_unet_gap_v1 overlay 9.687s / 2 holdout graphs"
        ),
    }


def refuse_overwrite_nogap_slug(slug: str) -> None:
    if slug == TEST_KERNEL_SLUG:
        raise ValueError("refusing to overwrite locked learned_unet_test_v1 slug")


def write_estimate_json(path: str | Path, n_public: int = 4, n_hidden: int = ESTIMATE_HIDDEN_VOLUMES) -> dict[str, Any]:
    dest = Path(path)
    payload = {
        "label": "learned_unet_gap_test_v1",
        "config_hash": LOCKED_GAP_TEST_HASH,
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "holdout_config_hash": LOCKED_HOLDOUT_HASH,
        "gap_config_hash": LOCKED_GAP_HASH,
        "test_parent_hash": LOCKED_TEST_HASH,
        "sec_per_volume": GAP_SEC_PER_VOLUME,
        "source": runtime_estimate(1)["source"],
        "peak_memory_mb_observed": HOLDOUT_PEAK_MEMORY_MB,
        "public_test": {k: v for k, v in runtime_estimate(n_public).items() if k != "source"},
        "hidden_test_train_sized": {
            k: v for k, v in runtime_estimate(n_hidden).items() if k != "source"
        },
        "kaggle_gpu_limit_hours": KAGGLE_GPU_LIMIT_HOURS,
        "is_leaderboard_score": False,
        "note": (
            "Estimate only. Not a Kaggle run and not a leaderboard score. "
            "Do not overwrite mahdimzafari/biohub-learned-unet-test-v1."
        ),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


__all__ = [
    "GAP_TEST_KERNEL_SLUG",
    "LOCKED_GAP_TEST_HASH",
    "gap_test_config_hash",
    "gap_test_config_payload",
    "list_test_zarrs",
    "refuse_hardcoded_public_stems",
    "refuse_overwrite_nogap_slug",
    "refuse_train_volume",
    "runtime_estimate",
    "write_estimate_json",
]
