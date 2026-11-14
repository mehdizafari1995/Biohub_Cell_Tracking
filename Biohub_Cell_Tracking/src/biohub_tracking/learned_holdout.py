"""One-shot frozen-holdout inference helpers for learned_unet_holdout_v1.

Uses locked ``learned_unet_infer_v1`` operating point and specimen-separated
``learned_unet_v1`` weights. Opens only ``6bba_05db0fb1`` and
``6bba_c328f2fd``. Never opens ``44b6`` volumes. Does not write
``submission.csv``. Not a competition submission. Do not retune. Do not
score holdout twice.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .learned_infer import (  # noqa: F401 — re-export detection/linking helpers
    LOCKED_INFER_HASH,
    TRAIN_CONFIG_HASH,
    TRAIN_KERNEL_SLUG,
    detect_cells_pooled,
    find_weights,
    greedy_gate_links,
    pool_kernel_from_um,
    scale_coords_to_original,
    verify_train_manifest,
)
from .learned_split import (
    FROZEN_HOLDOUT_STEMS,
    HOLDOUT_SPECIMENS,
    INTERNAL_VAL_STEMS,
    TUNING_SPECIMEN,
    HoldoutLeakError,
    specimen_of,
)

HOLDOUT_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-holdout-v1"
INFER_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-infer-v1"
HOLDOUT_STEMS = FROZEN_HOLDOUT_STEMS
TUNING_INFER_STEMS = INTERNAL_VAL_STEMS
LOCKED_HOLDOUT_HASH = "d374a21f03dddc04"
ASSOC_LOCKED_HOLDOUT = 0.5915697467955445
SCALE_ZYX = (1.625, 0.40625, 0.40625)
WEIGHTS_NAME = "learned_unet_v1_best.pt"


def _clean_stem(raw: str) -> str:
    stem = Path(raw).name
    for suffix in (".zarr", ".geff", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def holdout_config_payload() -> dict[str, Any]:
    """Locked infer operating point applied to frozen holdout only."""
    return {
        "label": "learned_unet_holdout_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "infer_stems": list(HOLDOUT_STEMS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "refused_tuning_stems": list(TUNING_INFER_STEMS),
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
        "one_shot_holdout": True,
        "kernel_slug": HOLDOUT_KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
        "weights_name": WEIGHTS_NAME,
    }


def holdout_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or holdout_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def assert_holdout_only(role: str) -> None:
    if role != "validation":
        raise SystemExit(
            "learned_unet_holdout_v1 scores frozen holdout only; role must be 'validation'"
        )


def refuse_tuning_stems(stems: Iterable[str]) -> None:
    leaked: list[str] = []
    seen: set[str] = set()
    for raw in stems:
        stem = _clean_stem(str(raw))
        if stem in seen:
            continue
        if specimen_of(stem) == TUNING_SPECIMEN or stem in TUNING_INFER_STEMS:
            leaked.append(stem)
            seen.add(stem)
    if leaked:
        raise HoldoutLeakError(
            "learned_unet_holdout_v1 refuses 44b6 tuning stems: " + ", ".join(leaked)
        )


def refuse_non_holdout_stems(stems: Iterable[str]) -> None:
    """Allow only the two frozen-holdout sequences. Tuning volumes are refused."""
    cleaned: list[str] = []
    for raw in stems:
        stem = _clean_stem(str(raw))
        if stem not in cleaned:
            cleaned.append(stem)
    refuse_tuning_stems(cleaned)
    extra = [s for s in cleaned if s not in HOLDOUT_STEMS]
    if extra:
        raise HoldoutLeakError(
            "learned_unet_holdout_v1 only opens frozen-holdout stems; got " + ", ".join(extra)
        )


def assert_no_submission_csv(working: str | Path) -> None:
    banned = Path(working) / "submission.csv"
    if banned.exists():
        banned.unlink()
        raise RuntimeError("learned_unet_holdout_v1 must not write submission.csv")


def refuse_existing_holdout_score(path: str | Path) -> None:
    """Do not official-score frozen 6bba a second time."""
    dest = Path(path)
    if not dest.exists():
        return
    payload = json.loads(dest.read_text())
    if payload.get("status") == "holdout_scored" and payload.get("learned_combined") is not None:
        raise SystemExit(
            f"refusing to score frozen holdout twice; existing decision={payload.get('decision')}"
        )


def graph_payload(
    dataset: str,
    coords: np.ndarray,
    edges: Iterable[tuple[int, int]],
) -> dict[str, Any]:
    refuse_non_holdout_stems([dataset])
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

    refuse_non_holdout_stems([str(payload["dataset"])])
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
    refuse_non_holdout_stems([str(payload["dataset"])])
    dest.write_text(json.dumps(payload, indent=2))
    return dest


def load_graph_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    refuse_non_holdout_stems([str(payload["dataset"])])
    return payload

