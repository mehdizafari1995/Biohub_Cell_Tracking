"""Drop short temporally inconsistent fragments after locked orgdet+gap+prune.

Official 44b6 error decomposition of locked ``learned_unet_orgdet_gap_prune_v1``
showed association-dominated FN edges (115 vs 27 detection). Assignment
overlays stay closed. Unsupported t±1 leftovers are already 0 after prune.
Fragments of size ≤7 had 0 sparse-GT matches on both frozen-tuning graphs
and still inflate ``T_pred``. Dropping those short components is
temporal-consistency detection calibration, not another 12 µm assignment.

Never opens ``6bba`` during tuning. Does not write ``submission.csv``.
Not a competition submission. A local official score is not a
leaderboard number. Do not overwrite locked slugs. Do not retune locked
parent hashes. Do not use public 0.602 / 0.612 / 0.553.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .graphs import TrackGraph
from .learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from .learned_orgdet import LOCKED_ORGDET_HASH
from .learned_orgdet_error_decomp import (
    MIN_KEEP_COMPONENT_NODES,
    SAFE_SHORT_TRACK_MAX_NODES,
    TUNE_KEEP_THRESHOLD,
    TUNE_MARGIN,
    component_sizes,
)
from .learned_orgdet_gap import LOCKED_ORGDET_GAP_HASH
from .learned_orgdet_gap_prune import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LEARNED_ORGDET_GAP_PRUNE_LOCKED_HOLDOUT,
    LEARNED_ORGDET_GAP_PRUNE_LOCKED_TUNE,
    LOCKED_ORGDET_GAP_PRUNE_HASH,
    LOCKED_ORGDET_HOLDOUT_NODE_COUNTS,
    LOCKED_ORGDET_NODE_COUNTS,
    apply_orgdet_gap_prune_overlay,
    ensure_orgdet_tuning_graph,
    load_role_graphs,
    refuse_non_holdout_stems as refuse_prune_holdout,
    refuse_non_orgdet_node_count,
    refuse_non_tuning_stems as refuse_prune_tuning,
)
from .learned_orgop import (
    LOCKED_ORGOP_HASH as REJECTED_ORGOP_HASH,
    ORGANIZER_DET_THRESHOLD,
    ORGANIZER_POOL_KERNEL_UM,
)
from .learned_split import HOLDOUT_SPECIMENS, TUNING_SPECIMEN, specimen_of

SCALE_ZYX = (1.625, 0.40625, 0.40625)
ORGDET_GAP_PRUNE_LOCKED_TUNE = LEARNED_ORGDET_GAP_PRUNE_LOCKED_TUNE
ORGDET_GAP_PRUNE_LOCKED_HOLDOUT = LEARNED_ORGDET_GAP_PRUNE_LOCKED_HOLDOUT

LOCKED_GATES = {
    "min_component_nodes": MIN_KEEP_COMPONENT_NODES,
    "safe_short_track_max_nodes": SAFE_SHORT_TRACK_MAX_NODES,
    "drop_short_fragments": True,
    "assignment": "unchanged_locked_orgdet_gap_prune",
}


def orgdet_tcf_config_payload() -> dict[str, Any]:
    """First short-fragment TCF operating point. Lock before the 44b6 score."""
    return {
        "label": "learned_unet_orgdet_tcf_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "orgdet_label": "learned_unet_orgdet_v1",
        "orgdet_gap_label": "learned_unet_orgdet_gap_v1",
        "orgdet_gap_prune_label": "learned_unet_orgdet_gap_prune_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "orgdet_config_hash": LOCKED_ORGDET_HASH,
        "orgdet_gap_config_hash": LOCKED_ORGDET_GAP_HASH,
        "orgdet_gap_prune_config_hash": LOCKED_ORGDET_GAP_PRUNE_HASH,
        "source_graphs": "locked_learned_unet_orgdet_graphs",
        "retunes_detection": True,
        "retunes_consecutive_links": False,
        "retunes_gap_gates": False,
        "uses_intensity": False,
        "uses_transformer": False,
        "uses_velocity": False,
        "uses_ilp": False,
        "linker": "orgdet_gap_prune_then_drop_short_fragments",
        "min_component_nodes": LOCKED_GATES["min_component_nodes"],
        "safe_short_track_max_nodes": LOCKED_GATES["safe_short_track_max_nodes"],
        "drop_short_fragments": LOCKED_GATES["drop_short_fragments"],
        "assignment": LOCKED_GATES["assignment"],
        "tune_keep_threshold": TUNE_KEEP_THRESHOLD,
        "tune_margin": TUNE_MARGIN,
        "det_threshold": ORGANIZER_DET_THRESHOLD,
        "det_tta": True,
        "pool_kernel_um": ORGANIZER_POOL_KERNEL_UM,
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": False,
    }


def orgdet_tcf_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(
        payload or orgdet_tcf_config_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGDET_TCF_HASH = orgdet_tcf_config_hash(orgdet_tcf_config_payload())
assert LOCKED_ORGDET_TCF_HASH == "a6b46dfcf410478f"
assert LOCKED_ORGDET_GAP_PRUNE_HASH == "862e3e7f7ac0330e"
assert REJECTED_ORGOP_HASH == "250c5bfacd4a8198"


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
            raise ValueError(f"learned_unet_orgdet_tcf_v1 tuning refuses stem {stem}")


def refuse_non_holdout_stems(stems: Iterable[str]) -> None:
    for raw in stems:
        stem = _clean_stem(str(raw))
        if specimen_of(stem) in HOLDOUT_SPECIMENS or stem in HOLDOUT_STEMS:
            if stem not in HOLDOUT_STEMS:
                raise ValueError(f"learned_unet_orgdet_tcf_v1 holdout refuses extra stem {stem}")
            continue
        raise ValueError(f"learned_unet_orgdet_tcf_v1 holdout refuses non-holdout stem {stem}")


def drop_short_components(graph: TrackGraph, min_nodes: int = MIN_KEEP_COMPONENT_NODES) -> TrackGraph:
    """Drop connected components smaller than ``min_nodes``. Keep remaining edges."""
    if int(min_nodes) != MIN_KEEP_COMPONENT_NODES:
        raise RuntimeError("learned_unet_orgdet_tcf_v1 would retune min_component_nodes; refusing")
    sizes = component_sizes(graph)
    nids = [int(x) for x in graph.node_id]
    keep = np.array([sizes.get(nid, 0) >= int(min_nodes) for nid in nids], dtype=bool)
    n_dropped = int((~keep).sum())
    keep_ids = {nid for nid, ok in zip(nids, keep) if ok}
    edges = [(int(s), int(d)) for s, d in np.asarray(graph.edges).reshape(-1, 2)] if graph.n_edges else []
    kept_edges = [(s, d) for s, d in edges if s in keep_ids and d in keep_ids]
    dropped_edge = [e for e in edges if e[0] not in keep_ids or e[1] not in keep_ids]
    if any(s in keep_ids and d in keep_ids for s, d in dropped_edge):
        raise RuntimeError("tcf dropped an edge inside a kept component; refusing")
    out = TrackGraph(
        dataset=graph.dataset,
        node_id=graph.node_id[keep] if nids else np.zeros((0,), dtype=np.int64),
        t=graph.t[keep] if nids else np.zeros((0,), dtype=np.int64),
        z=graph.z[keep] if nids else np.zeros((0,), dtype=np.int64),
        y=graph.y[keep] if nids else np.zeros((0,), dtype=np.int64),
        x=graph.x[keep] if nids else np.zeros((0,), dtype=np.int64),
        edges=np.asarray(kept_edges, dtype=np.int64).reshape(-1, 2) if kept_edges else np.zeros((0, 2), dtype=np.int64),
        estimated_n_nodes=graph.estimated_n_nodes,
    )
    out.metadata = dict(graph.metadata)
    out.metadata["n_dropped_short_nodes"] = n_dropped
    out.metadata["n_dropped_short_edges"] = len(dropped_edge)
    out.metadata["n_nodes_before_tcf"] = int(graph.n_nodes)
    out.metadata["min_component_nodes"] = int(min_nodes)
    leftover = set(int(x) for x in out.node_id)
    if any(src not in leftover or dst not in leftover for src, dst in kept_edges):
        raise RuntimeError("tcf kept a dangling edge; refusing")
    return out


def apply_orgdet_tcf_overlay(graph: TrackGraph) -> TrackGraph:
    """Locked orgdet+gap+prune, then drop short fragments. Do not retune."""
    if orgdet_tcf_config_hash() != LOCKED_ORGDET_TCF_HASH:
        raise RuntimeError("learned_unet_orgdet_tcf_v1 hash drifted; refusing to retune")
    if int(LOCKED_GATES["min_component_nodes"]) != MIN_KEEP_COMPONENT_NODES:
        raise RuntimeError("learned_unet_orgdet_tcf_v1 would retune fragment length; refusing")
    if LOCKED_GATES["assignment"] != "unchanged_locked_orgdet_gap_prune":
        raise RuntimeError("learned_unet_orgdet_tcf_v1 would retune assignment; refusing")
    pruned = apply_orgdet_gap_prune_overlay(graph)
    out = drop_short_components(pruned, min_nodes=LOCKED_GATES["min_component_nodes"])
    out.metadata["overlay"] = "learned_unet_orgdet_tcf_v1"
    out.metadata["config_hash"] = LOCKED_ORGDET_TCF_HASH
    out.metadata["parent_orgdet_hash"] = LOCKED_ORGDET_HASH
    out.metadata["parent_orgdet_gap_hash"] = LOCKED_ORGDET_GAP_HASH
    out.metadata["parent_orgdet_gap_prune_hash"] = LOCKED_ORGDET_GAP_PRUNE_HASH
    out.metadata["rejected_orgop_hash"] = REJECTED_ORGOP_HASH
    out.metadata["uses_intensity"] = False
    out.metadata["uses_transformer"] = False
    out.metadata["uses_ilp"] = False
    out.metadata["retunes_detection"] = True
    out.metadata["retunes_consecutive_links"] = False
    out.metadata["retunes_gap_gates"] = False
    out.metadata["is_12um_assignment"] = False
    return out


def overlay_summary(before: TrackGraph, after: TrackGraph) -> dict[str, Any]:
    return {
        "dataset": after.dataset,
        "n_nodes_before": before.n_nodes,
        "n_nodes_after": after.n_nodes,
        "n_edges_before": before.n_edges,
        "n_edges_after": after.n_edges,
        "n_dropped_short_nodes": int(after.metadata.get("n_dropped_short_nodes") or 0),
        "n_dropped_short_edges": int(after.metadata.get("n_dropped_short_edges") or 0),
        "n_nodes_before_tcf": int(after.metadata.get("n_nodes_before_tcf") or after.n_nodes),
        "min_component_nodes": int(after.metadata.get("min_component_nodes") or MIN_KEEP_COMPONENT_NODES),
        "overlay": "learned_unet_orgdet_tcf_v1",
        "config_hash": LOCKED_ORGDET_TCF_HASH,
    }


def refuse_existing_holdout_score(path: str | Path) -> None:
    dest = Path(path)
    if not dest.exists():
        return
    payload = json.loads(dest.read_text())
    if payload.get("status") == "holdout_scored" and payload.get("learned_orgdet_tcf_combined") is not None:
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
    if float(learned_combined) < float(TUNE_KEEP_THRESHOLD):
        return "reject_learned_unet_orgdet_tcf_v1"
    if int(n_dropped) == 0:
        return "keep_as_noop_learned_unet_orgdet_tcf_v1"
    return "keep_candidate_learned_unet_orgdet_tcf_v1"


def holdout_decision(learned_combined: float) -> str:
    if float(learned_combined) <= float(ORGDET_GAP_PRUNE_LOCKED_HOLDOUT):
        return "reject_learned_unet_orgdet_tcf_v1"
    return "keep_candidate_learned_unet_orgdet_tcf_v1"


def runtime_estimate(
    n_volumes: int = 2,
    overlay_sec_per_volume: float = 6.0,
    tcf_overhead: float = 0.5,
    limit_hours: float = 12.0,
    peak_memory_mb: float = 900.0,
) -> dict[str, Any]:
    n = int(n_volumes)
    sec = float(n) * (float(overlay_sec_per_volume) + float(tcf_overhead))
    return {
        "n_volumes": n,
        "overlay_sec_per_volume": float(overlay_sec_per_volume),
        "tcf_overhead": float(tcf_overhead),
        "estimate_sec": sec,
        "estimate_hours": sec / 3600.0,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_budget": float(peak_memory_mb),
        "source": (
            "CPU locked orgdet+gap+prune then drop components with fewer than "
            f"{MIN_KEEP_COMPONENT_NODES} nodes; no transformer; no ILP; no new UNet"
        ),
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
    }
