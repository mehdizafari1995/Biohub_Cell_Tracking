"""Test-time inference helpers for learned_unet_prune_test_v1.

Applies locked ``learned_unet_infer_v1`` / ``learned_unet_holdout_v1``
detection, the locked ``learned_unet_gap_v1`` overlay, then the locked
``learned_unet_prune_v1`` degree-0 drop to every discovered
``test/*.zarr``. Never opens ``train/`` volumes. Writes
``submission.csv`` + ``run_manifest.json``. Do not overwrite the locked
``learned_unet_gap_test_v1`` or no-gap ``learned_unet_test_v1`` slugs.
Do not retune. Do not score frozen ``6bba`` again. Not a competition
submission until a human submits from the Kaggle UI.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .learned_gap import LOCKED_GAP_HASH
from .learned_gap_test import (
    GAP_SEC_PER_VOLUME,
    GAP_TEST_KERNEL_SLUG,
    LOCKED_GAP_TEST_HASH,
    LOCKED_TEST_HASH,
    NOGAP_HOLDOUT_SEC_PER_VOLUME,
    OVERLAY_SEC_PER_VOLUME,
)
from .learned_holdout import LOCKED_HOLDOUT_HASH
from .learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from .learned_prune import LOCKED_GATES, LOCKED_PRUNE_HASH
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

PRUNE_TEST_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-prune-test-v1"
PRUNE_OVERHEAD_SEC_PER_VOLUME = 0.5
PRUNE_SEC_PER_VOLUME = GAP_SEC_PER_VOLUME + PRUNE_OVERHEAD_SEC_PER_VOLUME
GAP_TEST_MEASURED_PUBLIC_SEC = 330.0157964229584
GAP_TEST_MEASURED_PUBLIC_VOLUMES = 4
GAP_TEST_MEASURED_PEAK_MEMORY_MB = 1184.3046875


def prune_test_config_payload() -> dict[str, Any]:
    """Locked infer + locked gap overlay + drop degree-0 on test volumes."""
    return {
        "label": "learned_unet_prune_test_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "holdout_label": "learned_unet_holdout_v1",
        "gap_label": "learned_unet_gap_v1",
        "prune_label": "learned_unet_prune_v1",
        "test_label": "learned_unet_test_v1",
        "gap_test_label": "learned_unet_gap_test_v1",
        "association_label": "association_v2",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "holdout_config_hash": LOCKED_HOLDOUT_HASH,
        "gap_config_hash": LOCKED_GAP_HASH,
        "prune_config_hash": LOCKED_PRUNE_HASH,
        "test_parent_hash": LOCKED_TEST_HASH,
        "gap_test_parent_hash": LOCKED_GAP_TEST_HASH,
        "discovers_test_zarrs": True,
        "hardcoded_test_stems": [],
        "opens_train_volumes": False,
        "det_threshold": 0.5,
        "det_tta": True,
        "pool_kernel_um": 5.0,
        "linker": "gap_then_drop_degree0",
        "gate_um": float(LOCKED_GATES["gate_um"]),
        "max_gap_frames": int(LOCKED_GATES["max_gap_frames"]),
        "gap_scale": str(LOCKED_GATES["gap_scale"]),
        "assignment": str(LOCKED_GATES["assignment"]),
        "interpolate_gaps": bool(LOCKED_GATES["interpolate_gaps"]),
        "drop_degree0": bool(LOCKED_GATES["drop_degree0"]),
        "max_parents_per_node": int(LOCKED_GATES["max_parents_per_node"]),
        "max_children_per_node": int(LOCKED_GATES["max_children_per_node"]),
        "intensity_weight": float(LOCKED_GATES["intensity_weight"]),
        "downsample": [1, 4, 4],
        "normalize": "window_minmax",
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": True,
        "writes_run_manifest": True,
        "one_shot_holdout": False,
        "kernel_slug": PRUNE_TEST_KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
        "holdout_kernel_slug": HOLDOUT_KERNEL_SLUG,
        "test_kernel_slug": TEST_KERNEL_SLUG,
        "gap_test_kernel_slug": GAP_TEST_KERNEL_SLUG,
        "weights_name": WEIGHTS_NAME,
    }


def prune_test_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or prune_test_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_PRUNE_TEST_HASH = prune_test_config_hash(prune_test_config_payload())
assert LOCKED_PRUNE_TEST_HASH == "074a71dec68304e8"
assert LOCKED_PRUNE_HASH == "115753ea16aaf021"
assert LOCKED_GAP_TEST_HASH == "85379c198ebc7241"


def runtime_estimate(
    n_volumes: int,
    sec_per_volume: float = PRUNE_SEC_PER_VOLUME,
    limit_hours: float = KAGGLE_GPU_LIMIT_HOURS,
    peak_memory_mb: float = HOLDOUT_PEAK_MEMORY_MB,
) -> dict[str, Any]:
    """Estimate GPU time from measured gap-test plus isolate-drop overhead."""
    n = int(n_volumes)
    sec = float(n) * float(sec_per_volume)
    hours = sec / 3600.0
    return {
        "n_volumes": n,
        "sec_per_volume": float(sec_per_volume),
        "nogap_sec_per_volume": NOGAP_HOLDOUT_SEC_PER_VOLUME,
        "overlay_sec_per_volume": OVERLAY_SEC_PER_VOLUME,
        "prune_overhead_sec_per_volume": PRUNE_OVERHEAD_SEC_PER_VOLUME,
        "gap_sec_per_volume": GAP_SEC_PER_VOLUME,
        "estimate_sec": sec,
        "estimate_hours": hours,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_observed": float(peak_memory_mb),
        "source": (
            "learned_unet_gap_test_v1 estimate 97.0435 s/volume + "
            "0.5 s isolate drop; public gap-test measured 330.0 s / 4 volumes"
        ),
    }


def hidden_projection_from_rehearsal(
    public_runtime_sec: float,
    n_public: int = GAP_TEST_MEASURED_PUBLIC_VOLUMES,
    n_hidden: int = ESTIMATE_HIDDEN_VOLUMES,
    peak_memory_mb: float = GAP_TEST_MEASURED_PEAK_MEMORY_MB,
    limit_hours: float = KAGGLE_GPU_LIMIT_HOURS,
) -> dict[str, Any]:
    """Project hidden-test wall time from a completed public rehearsal."""
    n_pub = int(n_public)
    sec_per = float(public_runtime_sec) / max(n_pub, 1)
    hidden = runtime_estimate(int(n_hidden), sec_per_volume=sec_per, peak_memory_mb=peak_memory_mb)
    fail_closed = (not hidden["under_limit"]) or hidden["estimate_hours"] >= 0.9 * float(limit_hours)
    return {
        "public_runtime_sec": float(public_runtime_sec),
        "n_public": n_pub,
        "sec_per_volume": sec_per,
        "hidden_test": hidden,
        "fail_closed_approaches_12h": fail_closed,
        "is_leaderboard_score": False,
    }


def refuse_overwrite_locked_slugs(slug: str) -> None:
    if slug == TEST_KERNEL_SLUG:
        raise ValueError("refusing to overwrite locked learned_unet_test_v1 slug")
    if slug == GAP_TEST_KERNEL_SLUG:
        raise ValueError("refusing to overwrite locked learned_unet_gap_test_v1 slug")


def write_estimate_json(path: str | Path, n_public: int = 4, n_hidden: int = ESTIMATE_HIDDEN_VOLUMES) -> dict[str, Any]:
    dest = Path(path)
    payload = {
        "label": "learned_unet_prune_test_v1",
        "config_hash": LOCKED_PRUNE_TEST_HASH,
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "holdout_config_hash": LOCKED_HOLDOUT_HASH,
        "gap_config_hash": LOCKED_GAP_HASH,
        "prune_config_hash": LOCKED_PRUNE_HASH,
        "test_parent_hash": LOCKED_TEST_HASH,
        "gap_test_parent_hash": LOCKED_GAP_TEST_HASH,
        "sec_per_volume": PRUNE_SEC_PER_VOLUME,
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
            "Do not overwrite mahdimzafari/biohub-learned-unet-gap-test-v1 "
            "or mahdimzafari/biohub-learned-unet-test-v1."
        ),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def compare_prune_vs_gap_counts(
    prune: Mapping[str, Any],
    gap: Mapping[str, Any],
) -> dict[str, Any]:
    """Prune may only drop degree-0 nodes. Edges and gap counts must match."""
    errors: list[str] = []
    prune_edges = int(prune.get("n_edges") or 0)
    gap_edges = int(gap.get("n_edges") or 0)
    if prune_edges != gap_edges:
        errors.append(f"edges changed: prune {prune_edges} vs gap {gap_edges}")
    for key in ("n_gap_links", "n_interpolated", "n_predicted_divisions"):
        if int(prune.get(key) or 0) != int(gap.get(key) or 0):
            errors.append(f"{key} changed: prune {prune.get(key)} vs gap {gap.get(key)}")
    prune_nodes = int(prune.get("n_nodes") or 0)
    gap_nodes = int(gap.get("n_nodes") or 0)
    dropped = int(prune.get("n_dropped_isolates") or (gap_nodes - prune_nodes))
    if prune_nodes > gap_nodes:
        errors.append(f"prune added nodes: {prune_nodes} > {gap_nodes}")
    if dropped < 0:
        errors.append(f"negative isolate drop {dropped}")
    if prune_nodes != gap_nodes - dropped and "n_dropped_isolates" in prune:
        errors.append(
            f"node accounting: prune {prune_nodes} != gap {gap_nodes} - dropped {dropped}"
        )
    return {
        "ok": not errors,
        "errors": errors,
        "n_nodes_gap": gap_nodes,
        "n_nodes_prune": prune_nodes,
        "n_edges_gap": gap_edges,
        "n_edges_prune": prune_edges,
        "n_dropped_isolates": dropped,
    }


__all__ = [
    "GAP_TEST_KERNEL_SLUG",
    "LOCKED_PRUNE_TEST_HASH",
    "PRUNE_TEST_KERNEL_SLUG",
    "compare_prune_vs_gap_counts",
    "hidden_projection_from_rehearsal",
    "list_test_zarrs",
    "prune_test_config_hash",
    "prune_test_config_payload",
    "refuse_hardcoded_public_stems",
    "refuse_overwrite_locked_slugs",
    "refuse_train_volume",
    "runtime_estimate",
    "write_estimate_json",
]
