"""44b6-only inference helpers for learned_unet_infer_v1.

Uses specimen-separated ``learned_unet_v1`` weights. Never opens ``6bba``.
Does not write ``submission.csv``. Not a competition submission.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .learned_split import (  # noqa: F401 — re-export find_weights / verify_train_manifest
    FROZEN_HOLDOUT_STEMS,
    HOLDOUT_SPECIMENS,
    INTERNAL_VAL_STEMS,
    KERNEL_SLUG as TRAIN_KERNEL_SLUG,
    TRAIN_CONFIG_HASH,
    TUNING_SPECIMEN,
    WEIGHTS_NAME,
    HoldoutLeakError,
    find_weights,
    refuse_holdout_stems,
    verify_train_manifest,
)

INFER_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-infer-v1"
INFER_STEMS = INTERNAL_VAL_STEMS
LOCKED_INFER_HASH = "d7578b5f64ddac08"
ASSOC_LOCKED_TUNE = 0.6324600072329115
SCALE_ZYX = (1.625, 0.40625, 0.40625)


def infer_config_payload() -> dict[str, Any]:
    """First infer operating point. Do not retune from holdout."""
    return {
        "label": "learned_unet_infer_v1",
        "train_label": "learned_unet_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "infer_stems": list(INFER_STEMS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "det_threshold": 0.5,
        "det_tta": True,
        "pool_kernel_um": 5.0,
        "linker": "greedy_gate_12um",
        "gate_um": 12.0,
        "max_gap_frames": 0,
        "max_parents_per_node": 1,
        "max_children_per_node": 1,
        "downsample": [1, 4, 4],
        "normalize": "window_minmax",
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": False,
        "kernel_slug": INFER_KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
        "weights_name": WEIGHTS_NAME,
    }


def infer_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or infer_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def assert_tuning_only(role: str) -> None:
    if role != "tuning":
        raise SystemExit(
            "refusing to score holdout; learned_unet_infer_v1 is 44b6-only until a documented lock"
        )


def refuse_non_infer_stems(stems: Iterable[str]) -> None:
    """Allow only the two frozen-tuning sequences. Holdout is always refused."""
    cleaned: list[str] = []
    for raw in stems:
        stem = Path(raw).name
        for suffix in (".zarr", ".geff", ".json"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        if stem not in cleaned:
            cleaned.append(stem)
    refuse_holdout_stems(cleaned)
    extra = [s for s in cleaned if s not in INFER_STEMS]
    if extra:
        raise HoldoutLeakError(
            "learned_unet_infer_v1 only opens frozen-tuning stems; got " + ", ".join(extra)
        )


def assert_no_submission_csv(working: str | Path) -> None:
    banned = Path(working) / "submission.csv"
    if banned.exists():
        banned.unlink()
        raise RuntimeError("learned_unet_infer_v1 must not write submission.csv")


def pool_kernel_from_um(um: float, voxel_size: Sequence[float]) -> tuple[int, ...]:
    kernel: list[int] = []
    for size in voxel_size:
        k = max(1, round(float(um) / float(size)))
        if k % 2 == 0:
            k += 1
        kernel.append(k)
    return tuple(kernel)


def detect_cells_pooled(
    logits: np.ndarray,
    t: int,
    det_threshold: float,
    pool_kernel: Sequence[int],
) -> np.ndarray:
    """Max-pool local-max peaks. ``logits`` is (Z, Y, X). Returns (N, 4) [t,z,y,x]."""
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 3:
        raise ValueError(f"expected (Z,Y,X) logits, got {logits.shape}")
    pad = tuple(int(k) // 2 for k in pool_kernel)
    padded = np.pad(logits, [(p, p) for p in pad], mode="constant", constant_values=-np.inf)
    windows = np.lib.stride_tricks.sliding_window_view(padded, tuple(int(k) for k in pool_kernel))
    pooled = windows.max(axis=(-3, -2, -1))
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    is_peak = (logits == pooled) & (probs > float(det_threshold))
    zyx = np.argwhere(is_peak)
    if zyx.size == 0:
        return np.empty((0, 4), dtype=np.int16)
    t_col = np.full((len(zyx), 1), int(t), dtype=np.int16)
    return np.concatenate([t_col, zyx.astype(np.int16)], axis=1)


def scale_coords_to_original(coords: np.ndarray, downsample: Sequence[int]) -> np.ndarray:
    out = np.asarray(coords, dtype=np.float32).copy()
    if out.size == 0:
        return np.empty((0, 4), dtype=np.int16)
    ds = np.asarray(downsample, dtype=np.float32)
    out[:, 1:] *= ds
    return np.rint(out).astype(np.int16)


def greedy_gate_links(
    coords_t: np.ndarray,
    coords_t1: np.ndarray,
    scale_zyx: Sequence[float] = SCALE_ZYX,
    gate_um: float = 12.0,
) -> list[tuple[int, int, float]]:
    """1-1 greedy links inside a physical gate. Indices are per-frame."""
    a = np.asarray(coords_t, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(coords_t1, dtype=np.float64).reshape(-1, 3)
    if a.size == 0 or b.size == 0:
        return []
    scale = np.asarray(scale_zyx, dtype=np.float64)
    pa = a * scale
    pb = b * scale
    delta = pa[:, None, :] - pb[None, :, :]
    dist = np.sqrt(np.sum(delta * delta, axis=-1))
    candidates = [
        (float(dist[i, j]), int(i), int(j))
        for i in range(a.shape[0])
        for j in range(b.shape[0])
        if float(dist[i, j]) <= float(gate_um)
    ]
    candidates.sort()
    used_i: set[int] = set()
    used_j: set[int] = set()
    edges: list[tuple[int, int, float]] = []
    for d_um, i, j in candidates:
        if i in used_i or j in used_j:
            continue
        edges.append((i, j, d_um))
        used_i.add(i)
        used_j.add(j)
    return edges


def graph_payload(
    dataset: str,
    coords: np.ndarray,
    edges: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 4)
    node_id = np.arange(1, len(coords) + 1, dtype=np.int64)
    return {
        "dataset": dataset,
        "node_id": node_id.tolist(),
        "t": coords[:, 0].astype(np.int64).tolist() if len(coords) else [],
        "z": coords[:, 1].astype(np.int64).tolist() if len(coords) else [],
        "y": coords[:, 2].astype(np.int64).tolist() if len(coords) else [],
        "x": coords[:, 3].astype(np.int64).tolist() if len(coords) else [],
        "edges": [[int(s), int(d)] for s, d in edges],
    }


def graph_from_payload(payload: Mapping[str, Any]):
    from .graphs import TrackGraph

    refuse_non_infer_stems([str(payload["dataset"])])
    edges = payload.get("edges") or []
    return TrackGraph(
        dataset=str(payload["dataset"]),
        node_id=np.asarray(payload.get("node_id") or [], dtype=np.int64),
        t=np.asarray(payload.get("t") or [], dtype=np.int64),
        z=np.asarray(payload.get("z") or [], dtype=np.int64),
        y=np.asarray(payload.get("y") or [], dtype=np.int64),
        x=np.asarray(payload.get("x") or [], dtype=np.int64),
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2) if edges else np.zeros((0, 2), dtype=np.int64),
    )


def write_graph_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    refuse_non_infer_stems([str(payload["dataset"])])
    dest.write_text(json.dumps(payload, indent=2))
    return dest


def load_graph_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    refuse_non_infer_stems([str(payload["dataset"])])
    return payload


