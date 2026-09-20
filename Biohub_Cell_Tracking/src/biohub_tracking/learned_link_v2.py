"""44b6-only transformer-linker helpers for learned_unet_link_v2.

Keeps locked ``learned_unet_infer_v1`` detections and the trained
``SimpleNodeTransformer`` scores from ``learned_unet_v1``. Replaces the
rejected v1 ``prob > 0.5`` gate with greedy 1-1 assignment on descending
scores and the locked association_v2 12 µm spatial gate. Never opens
``6bba``. Does not write ``submission.csv``. Not a competition submission.
Do not overwrite rejected ``learned_unet_link_v1``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .learned_infer import (  # noqa: F401 — re-export tuning guards / graph I/O
    ASSOC_LOCKED_TUNE,
    INFER_KERNEL_SLUG,
    INFER_STEMS,
    LOCKED_INFER_HASH,
    TRAIN_CONFIG_HASH,
    TRAIN_KERNEL_SLUG,
    WEIGHTS_NAME,
    FROZEN_HOLDOUT_STEMS,
    HOLDOUT_SPECIMENS,
    SCALE_ZYX,
    TUNING_SPECIMEN,
    HoldoutLeakError,
    assert_no_submission_csv,
    assert_tuning_only,
    find_weights,
    graph_from_payload,
    graph_payload,
    load_graph_json,
    refuse_non_infer_stems,
    verify_train_manifest,
    write_graph_json,
)
from .learned_link import (  # noqa: F401 — rejected v1 stays locked
    LOCKED_LINK_HASH as REJECTED_LINK_V1_HASH,
    assign_transformer_links as assign_transformer_links_v1,
)

LINK_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-link-v2"
REJECTED_LINK_V1_SLUG = "mahdimzafari/biohub-learned-unet-link-v1"
INFER_LOCKED_TUNE = 0.6557507105877639
MAX_LINK_UM = 12.0


def link_config_payload() -> dict[str, Any]:
    """Second transformer-linker operating point. Do not retune from holdout."""
    return {
        "label": "learned_unet_link_v2",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "infer_stems": list(INFER_STEMS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "det_threshold": 0.5,
        "det_tta": True,
        "pool_kernel_um": 5.0,
        "linker": "transformer_softmax_greedy_1to1_spatial",
        "edge_activation": "softmax",
        "edge_threshold": None,
        "max_link_um": MAX_LINK_UM,
        "max_parents_per_node": 1,
        "max_children_per_node": 1,
        "max_gap_frames": 0,
        "downsample": [1, 4, 4],
        "normalize": "window_minmax",
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": False,
        "kernel_slug": LINK_KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
        "rejected_link_v1_slug": REJECTED_LINK_V1_SLUG,
        "rejected_link_v1_hash": REJECTED_LINK_V1_HASH,
        "weights_name": WEIGHTS_NAME,
    }


def link_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or link_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_LINK_V2_HASH = link_config_hash()


def pairwise_distances_um(
    src_zyx: np.ndarray,
    tgt_zyx: np.ndarray,
    scale_zyx: Sequence[float] = SCALE_ZYX,
) -> np.ndarray:
    """Pairwise physical distances in µm from integer voxel centroids."""
    src = np.asarray(src_zyx, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(tgt_zyx, dtype=np.float64).reshape(-1, 3)
    if src.size == 0 or tgt.size == 0:
        return np.empty((src.shape[0], tgt.shape[0]), dtype=np.float64)
    scale = np.asarray(scale_zyx, dtype=np.float64).reshape(1, 3)
    src_um = src * scale
    tgt_um = tgt * scale
    diff = src_um[:, None, :] - tgt_um[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def assign_transformer_links_v2(
    scores: np.ndarray,
    distances_um: np.ndarray | None = None,
    max_link_um: float = MAX_LINK_UM,
    max_parents_per_node: int = 1,
    max_children_per_node: int = 1,
) -> list[tuple[int, int, float]]:
    """Greedy 1-1 on descending scores. No probability cutoff.

    Optional ``distances_um`` applies the locked association_v2 12 µm gate so
    a long-range softmax winner cannot link. This is the documented v2 fix
    for rejected ``learned_unet_link_v1`` (softmax + ``prob > 0.5``).
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        return []
    n_src, n_tgt = values.shape
    if distances_um is None:
        allowed = np.ones((n_src, n_tgt), dtype=bool)
    else:
        dist = np.asarray(distances_um, dtype=np.float64)
        if dist.shape != values.shape:
            raise ValueError("distances_um must match scores shape")
        allowed = dist <= float(max_link_um)
    i_idx, j_idx = np.nonzero(allowed)
    if i_idx.size == 0:
        return []
    order = np.argsort(-values[i_idx, j_idx], kind="stable")
    children_count: dict[int, int] = {}
    parents_count: dict[int, int] = {}
    edges: list[tuple[int, int, float]] = []
    for idx in order:
        i = int(i_idx[idx])
        j = int(j_idx[idx])
        if children_count.get(i, 0) >= int(max_children_per_node):
            continue
        if parents_count.get(j, 0) >= int(max_parents_per_node):
            continue
        edges.append((i, j, float(values[i, j])))
        children_count[i] = children_count.get(i, 0) + 1
        parents_count[j] = parents_count.get(j, 0) + 1
    return edges


def runtime_estimate(
    n_volumes: int = 2,
    infer_sec_per_volume: float = 93.5,
    linker_overhead: float = 1.65,
    limit_hours: float = 12.0,
    peak_memory_mb: float = 2500.0,
) -> dict[str, Any]:
    """Estimate GPU time from locked infer_v1 plus transformer pair scoring."""
    n = int(n_volumes)
    sec_per = float(infer_sec_per_volume) * float(linker_overhead)
    sec = float(n) * sec_per
    return {
        "n_volumes": n,
        "infer_sec_per_volume": float(infer_sec_per_volume),
        "linker_overhead": float(linker_overhead),
        "sec_per_volume": sec_per,
        "estimate_sec": sec,
        "estimate_hours": sec / 3600.0,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_budget": float(peak_memory_mb),
        "source": (
            "learned_unet_infer_v1 v3 187.0s / 2 volumes * 1.65 transformer "
            "pair-score + 12 µm gate overhead; pair MLP row-chunked"
        ),
    }
