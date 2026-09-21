"""Drop leftover degree-0 detections after locked learned_unet_gap_v1.

Applies the locked ``learned_unet_gap_v1`` overlay, then removes nodes
that still have in-degree 0 and out-degree 0. Consecutive dt=1 links and
gap gates are not retuned. This is not another 12 µm assignment: leftover
isolates inflate ``T_pred`` without contributing edges. Not a competition
submission. A local official score is not a leaderboard number. Do not
overwrite locked train/infer/holdout/test/gap-test slugs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .graphs import TrackGraph
from .learned_gap import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LEARNED_GAP_LOCKED_HOLDOUT,
    LOCKED_GAP_HASH,
    apply_locked_overlay as apply_gap_overlay,
)
from .learned_holdout import LOCKED_HOLDOUT_HASH
from .learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from .learned_split import (
    FROZEN_HOLDOUT_STEMS,
    HOLDOUT_SPECIMENS,
    INTERNAL_VAL_STEMS,
    TUNING_SPECIMEN,
    specimen_of,
)

# Official locked learned_unet_gap_v1 combined scores. Do not retune gap.
LEARNED_GAP_LOCKED_TUNE = 0.6584515375141551
# Official one-shot scores of this overlay. Do not retune. Do not rescore.
LEARNED_PRUNE_LOCKED_TUNE = 0.6620813548547098
LEARNED_PRUNE_LOCKED_HOLDOUT = 0.7013778258203534
SCALE_ZYX = (1.625, 0.40625, 0.40625)

LOCKED_GATES = {
    "gate_um": 12.0,
    "max_gap_frames": 2,
    "gap_scale": "sqrt_dt",
    "assignment": "greedy",
    "interpolate_gaps": True,
    "drop_degree0": True,
    "max_parents_per_node": 1,
    "max_children_per_node": 1,
    "intensity_weight": 0.0,
}


def prune_config_payload() -> dict[str, Any]:
    """First isolate-prune operating point. Do not retune from holdout."""
    return {
        "label": "learned_unet_prune_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "holdout_label": "learned_unet_holdout_v1",
        "gap_label": "learned_unet_gap_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "holdout_config_hash": LOCKED_HOLDOUT_HASH,
        "gap_config_hash": LOCKED_GAP_HASH,
        "source_graphs": "locked_learned_unet_graphs",
        "retunes_detection": False,
        "retunes_consecutive_links": False,
        "retunes_gap_gates": False,
        "uses_intensity": False,
        "uses_transformer": False,
        "linker": "gap_then_drop_degree0",
        "gate_um": LOCKED_GATES["gate_um"],
        "max_gap_frames": LOCKED_GATES["max_gap_frames"],
        "gap_scale": LOCKED_GATES["gap_scale"],
        "assignment": LOCKED_GATES["assignment"],
        "interpolate_gaps": LOCKED_GATES["interpolate_gaps"],
        "drop_degree0": LOCKED_GATES["drop_degree0"],
        "max_parents_per_node": LOCKED_GATES["max_parents_per_node"],
        "max_children_per_node": LOCKED_GATES["max_children_per_node"],
        "intensity_weight": LOCKED_GATES["intensity_weight"],
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": False,
    }


def prune_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or prune_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_PRUNE_HASH = prune_config_hash(prune_config_payload())
assert LOCKED_PRUNE_HASH == "115753ea16aaf021"
assert LOCKED_GAP_HASH == "f4d888d190c45a4a"


def _clean_stem(raw: str) -> str:
    stem = Path(raw).name
    for suffix in (".zarr", ".geff", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def refuse_non_tuning_stems(stems: Iterable[str]) -> None:
    for raw in stems:
        stem = _clean_stem(str(raw))
        if specimen_of(stem) != TUNING_SPECIMEN or stem not in INFER_STEMS:
            raise ValueError(f"learned_unet_prune_v1 tuning refuses stem {stem}")


def refuse_non_holdout_stems(stems: Iterable[str]) -> None:
    for raw in stems:
        stem = _clean_stem(str(raw))
        if specimen_of(stem) in HOLDOUT_SPECIMENS or stem in HOLDOUT_STEMS:
            if stem not in HOLDOUT_STEMS:
                raise ValueError(f"learned_unet_prune_v1 holdout refuses extra stem {stem}")
            continue
        raise ValueError(f"learned_unet_prune_v1 holdout refuses non-holdout stem {stem}")


def graph_from_payload(payload: Mapping[str, Any]) -> TrackGraph:
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


def load_graph_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def drop_degree0(graph: TrackGraph) -> TrackGraph:
    """Remove nodes with no incident edges. Keep all existing edges."""
    linked: set[int] = set()
    edges = [(int(s), int(d)) for s, d in np.asarray(graph.edges).reshape(-1, 2)] if graph.n_edges else []
    for src, dst in edges:
        linked.add(src)
        linked.add(dst)
    nids = [int(x) for x in graph.node_id]
    if not nids:
        out = TrackGraph(
            dataset=graph.dataset,
            node_id=np.zeros((0,), dtype=np.int64),
            t=np.zeros((0,), dtype=np.int64),
            z=np.zeros((0,), dtype=np.int64),
            y=np.zeros((0,), dtype=np.int64),
            x=np.zeros((0,), dtype=np.int64),
            edges=np.zeros((0, 2), dtype=np.int64),
            estimated_n_nodes=graph.estimated_n_nodes,
        )
        out.metadata = dict(graph.metadata)
        out.metadata["n_dropped_isolates"] = 0
        return out
    keep = np.array([nid in linked for nid in nids], dtype=bool)
    n_dropped = int((~keep).sum())
    out = TrackGraph(
        dataset=graph.dataset,
        node_id=graph.node_id[keep],
        t=graph.t[keep],
        z=graph.z[keep],
        y=graph.y[keep],
        x=graph.x[keep],
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2) if edges else np.zeros((0, 2), dtype=np.int64),
        estimated_n_nodes=graph.estimated_n_nodes,
    )
    out.metadata = dict(graph.metadata)
    out.metadata["n_dropped_isolates"] = n_dropped
    out.metadata["n_nodes_before_prune"] = int(graph.n_nodes)
    leftover = set(int(x) for x in out.node_id)
    if any(src not in leftover or dst not in leftover for src, dst in edges):
        raise RuntimeError("prune dropped a linked node; refusing")
    return out


def apply_locked_overlay(graph: TrackGraph) -> TrackGraph:
    """Locked gap overlay, then drop leftover isolates. Do not retune gates."""
    if prune_config_hash() != LOCKED_PRUNE_HASH:
        raise RuntimeError("learned_unet_prune_v1 hash drifted; refusing to retune")
    if LOCKED_GATES["gap_scale"] != "sqrt_dt" or float(LOCKED_GATES["gate_um"]) != 12.0:
        raise RuntimeError("learned_unet_prune_v1 would retune gap gates; refusing")
    gapped = apply_gap_overlay(graph)
    out = drop_degree0(gapped)
    out.metadata["overlay"] = "learned_unet_prune_v1"
    out.metadata["config_hash"] = LOCKED_PRUNE_HASH
    out.metadata["parent_infer_hash"] = LOCKED_INFER_HASH
    out.metadata["parent_gap_hash"] = LOCKED_GAP_HASH
    out.metadata["uses_intensity"] = False
    out.metadata["uses_transformer"] = False
    out.metadata["retunes_detection"] = False
    out.metadata["retunes_consecutive_links"] = False
    out.metadata["retunes_gap_gates"] = False
    out.metadata["drop_degree0"] = True
    return out


def overlay_summary(before: TrackGraph, after: TrackGraph) -> dict[str, Any]:
    return {
        "dataset": after.dataset,
        "n_nodes_before": before.n_nodes,
        "n_nodes_after": after.n_nodes,
        "n_edges_before": before.n_edges,
        "n_edges_after": after.n_edges,
        "n_extra_nodes": int(after.n_nodes - before.n_nodes),
        "n_extra_edges": int(after.n_edges - before.n_edges),
        "n_dropped_isolates": int(after.metadata.get("n_dropped_isolates") or 0),
        "n_gap_links": int(after.metadata.get("n_gap_links") or 0),
        "n_interpolated": int(after.metadata.get("n_interpolated") or 0),
        "n_nodes_before_prune": int(after.metadata.get("n_nodes_before_prune") or after.n_nodes),
    }


def load_role_graphs(graph_dir: str | Path, stems: Sequence[str]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for stem in stems:
        path = Path(graph_dir) / f"{stem}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing learned graph {path}; do not invent a score")
        payload = load_graph_json(path)
        if str(payload.get("dataset")) != stem:
            raise ValueError(f"{path} dataset {payload.get('dataset')!r} != {stem}")
        payloads.append(payload)
    return payloads


def refuse_existing_holdout_score(path: str | Path) -> None:
    dest = Path(path)
    if not dest.exists():
        return
    payload = json.loads(dest.read_text())
    if payload.get("status") == "holdout_scored" and payload.get("learned_prune_combined") is not None:
        raise SystemExit(
            f"refusing to score frozen holdout twice; existing decision={payload.get('decision')}"
        )


def require_tuning_keep(path: str | Path) -> dict[str, Any]:
    dest = Path(path)
    if not dest.exists():
        raise SystemExit("refusing holdout; tuning report missing. Score 44b6 first.")
    payload = json.loads(dest.read_text())
    decision = str(payload.get("decision") or "")
    if not decision.startswith("keep_"):
        raise SystemExit(f"refusing holdout; tuning decision is {decision}")
    return payload


def tuning_decision(learned_combined: float, n_dropped: int) -> str:
    if float(learned_combined) < LEARNED_GAP_LOCKED_TUNE:
        return "reject_learned_unet_prune_v1"
    if int(n_dropped) == 0:
        return "keep_as_noop_learned_unet_prune_v1"
    return "keep_candidate_learned_unet_prune_v1"


def holdout_decision(learned_combined: float) -> str:
    if float(learned_combined) < LEARNED_GAP_LOCKED_HOLDOUT:
        return "reject_learned_unet_prune_v1"
    return "keep_candidate_learned_unet_prune_v1"


def runtime_estimate(
    n_volumes: int = 2,
    gap_sec_per_volume: float = 4.8,
    prune_overhead: float = 0.5,
    limit_hours: float = 12.0,
    peak_memory_mb: float = 900.0,
) -> dict[str, Any]:
    """CPU gap overlay plus isolate drop. No new UNet pass."""
    n = int(n_volumes)
    sec = float(n) * (float(gap_sec_per_volume) + float(prune_overhead))
    return {
        "n_volumes": n,
        "gap_sec_per_volume": float(gap_sec_per_volume),
        "prune_overhead": float(prune_overhead),
        "estimate_sec": sec,
        "estimate_hours": sec / 3600.0,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_budget": float(peak_memory_mb),
        "source": (
            "CPU locked gap overlay then drop degree-0 leftovers on locked "
            "learned infer graphs; no transformer; no new UNet pass"
        ),
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
    }


# Re-export stems used by the scoring script.
__all__ = [
    "FROZEN_HOLDOUT_STEMS",
    "HOLDOUT_STEMS",
    "INFER_STEMS",
    "INTERNAL_VAL_STEMS",
    "LEARNED_GAP_LOCKED_HOLDOUT",
    "LEARNED_GAP_LOCKED_TUNE",
    "LEARNED_PRUNE_LOCKED_HOLDOUT",
    "LEARNED_PRUNE_LOCKED_TUNE",
    "LOCKED_GAP_HASH",
    "LOCKED_PRUNE_HASH",
    "TUNING_SPECIMEN",
    "apply_locked_overlay",
    "drop_degree0",
    "graph_from_payload",
    "holdout_decision",
    "load_role_graphs",
    "overlay_summary",
    "prune_config_hash",
    "prune_config_payload",
    "refuse_existing_holdout_score",
    "refuse_non_holdout_stems",
    "refuse_non_tuning_stems",
    "require_tuning_keep",
    "runtime_estimate",
    "tuning_decision",
]
