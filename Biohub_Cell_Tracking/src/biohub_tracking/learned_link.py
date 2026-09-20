"""44b6-only transformer-linker helpers for learned_unet_link_v1.

Uses locked specimen-separated ``learned_unet_v1`` weights and the same
detections as ``learned_unet_infer_v1``. Replaces the greedy 12 µm gate
with trained ``SimpleNodeTransformer`` pairwise scores. Never opens
``6bba``. Does not write ``submission.csv``. Not a competition submission.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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

LINK_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-link-v1"
INFER_LOCKED_TUNE = 0.6557507105877639
_POS_EMBED_DIM = 8


def link_config_payload() -> dict[str, Any]:
    """First transformer-linker operating point. Do not retune from holdout."""
    return {
        "label": "learned_unet_link_v1",
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
        "linker": "transformer_softmax_1to1",
        "edge_activation": "softmax",
        "edge_threshold": 0.5,
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
        "weights_name": WEIGHTS_NAME,
    }


def link_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or link_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_LINK_HASH = link_config_hash()


def extract_pos_features(coords: np.ndarray, image_shape: Sequence[int]) -> np.ndarray:
    """Sinusoidal positional features matching learned_unet_v1 training."""
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 4)
    if coords.size == 0:
        return np.empty((0, 4 * _POS_EMBED_DIM), dtype=np.float32)
    t, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    norms = [c / max(float(s), 1.0) for c, s in zip([t, z, y, x], image_shape)]

    def _embed(vals: np.ndarray) -> np.ndarray:
        freqs = 2 ** np.arange(_POS_EMBED_DIM // 2)
        angles = vals[:, None] * freqs * np.pi
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)

    return np.concatenate([_embed(n) for n in norms], axis=1).astype(np.float32)


def assign_transformer_links(
    probs: np.ndarray,
    threshold: float = 0.5,
    max_parents_per_node: int = 1,
    max_children_per_node: int = 1,
) -> list[tuple[int, int, float]]:
    """Greedy 1-1 assignment on transformer probabilities (descending)."""
    scores = np.asarray(probs, dtype=np.float64)
    if scores.ndim != 2 or scores.size == 0:
        return []
    n_src, n_tgt = scores.shape
    candidates = [
        (float(scores[i, j]), int(i), int(j))
        for i in range(n_src)
        for j in range(n_tgt)
        if float(scores[i, j]) > float(threshold)
    ]
    candidates.sort(reverse=True)
    children_count: dict[int, int] = {}
    parents_count: dict[int, int] = {}
    edges: list[tuple[int, int, float]] = []
    for prob, i, j in candidates:
        if children_count.get(i, 0) >= int(max_children_per_node):
            continue
        if parents_count.get(j, 0) >= int(max_parents_per_node):
            continue
        edges.append((i, j, prob))
        children_count[i] = children_count.get(i, 0) + 1
        parents_count[j] = parents_count.get(j, 0) + 1
    return edges


def runtime_estimate(
    n_volumes: int = 2,
    infer_sec_per_volume: float = 93.5,
    linker_overhead: float = 1.6,
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
            "learned_unet_infer_v1 v3 187.0s / 2 volumes * 1.6 transformer "
            "pair-score overhead; pair MLP row-chunked"
        ),
    }
