"""Drop long consecutive dt=1 edges after locked orgdet+gap+prune.

Official 44b6 error decomposition of locked ``learned_unet_orgdet_gap_prune_v1``
was association-dominated (115 vs 27 detection FN). Assignment overlays stay
closed. Short-fragment TCF missed the +0.002000 gate on T_pred only.

On both frozen-tuning prune graphs, every official-matched TP consecutive
edge was shorter than 8.0 µm (max 7.6219 µm on ``44b6_267148e4``, max
5.8590 µm on ``44b6_12dfb391``), while 26 conflict FPs exceeded 8.0 µm.
Dropping dt=1 edges longer than 8.0 µm after locked prune, then dropping
newly created degree-0 leftovers, is an edge-length filter. It does not
reassign, does not shrink the 12 µm linker/gap gates, and is not a
short-fragment retune.

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
from .learned_orgdet_error_decomp import TUNE_KEEP_THRESHOLD, TUNE_MARGIN
from .learned_orgdet_gap import LOCKED_ORGDET_GAP_HASH
from .learned_orgdet_gap_prune import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LEARNED_ORGDET_GAP_PRUNE_LOCKED_HOLDOUT,
    LEARNED_ORGDET_GAP_PRUNE_LOCKED_TUNE,
    LOCKED_ORGDET_GAP_PRUNE_HASH,
    apply_orgdet_gap_prune_overlay,
    ensure_orgdet_tuning_graph,
    load_role_graphs,
    refuse_non_orgdet_node_count,
)
from .learned_orgop import (
    LOCKED_ORGOP_HASH as REJECTED_ORGOP_HASH,
    ORGANIZER_DET_THRESHOLD,
    ORGANIZER_POOL_KERNEL_UM,
)
from .learned_prune import drop_degree0
from .learned_split import HOLDOUT_SPECIMENS, TUNING_SPECIMEN, specimen_of
from .link import physical_coords

SCALE_ZYX = (1.625, 0.40625, 0.40625)
ORGDET_GAP_PRUNE_LOCKED_TUNE = LEARNED_ORGDET_GAP_PRUNE_LOCKED_TUNE
ORGDET_GAP_PRUNE_LOCKED_HOLDOUT = LEARNED_ORGDET_GAP_PRUNE_LOCKED_HOLDOUT
# Official one-shot scores of this overlay. Do not retune. Do not rescore.
LEARNED_ORGDET_ELF_LOCKED_TUNE = 0.7098380747225755
LEARNED_ORGDET_ELF_LOCKED_HOLDOUT = 0.7377873461563521
# Locked from 44b6 prune TP lengths only. Do not retune after the official score.
MAX_DT1_EDGE_UM = 8.0

LOCKED_GATES = {
    "max_dt1_edge_um": MAX_DT1_EDGE_UM,
    "drop_long_dt1_edges": True,
    "drop_degree0_after": True,
    "assignment": "unchanged_locked_orgdet_gap_prune",
}


def orgdet_elf_config_payload() -> dict[str, Any]:
    """First long-dt1 edge-length filter. Lock before the 44b6 score."""
    return {
        "label": "learned_unet_orgdet_elf_v1",
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
        "retunes_detection": False,
        "retunes_consecutive_links": False,
        "retunes_gap_gates": False,
        "uses_intensity": False,
        "uses_transformer": False,
        "uses_velocity": False,
        "uses_ilp": False,
        "linker": "orgdet_gap_prune_then_drop_long_dt1_edges",
        "max_dt1_edge_um": LOCKED_GATES["max_dt1_edge_um"],
        "drop_long_dt1_edges": LOCKED_GATES["drop_long_dt1_edges"],
        "drop_degree0_after": LOCKED_GATES["drop_degree0_after"],
        "assignment": LOCKED_GATES["assignment"],
        "tune_keep_threshold": TUNE_KEEP_THRESHOLD,
        "tune_margin": TUNE_MARGIN,
        "det_threshold": ORGANIZER_DET_THRESHOLD,
        "det_tta": True,
        "pool_kernel_um": ORGANIZER_POOL_KERNEL_UM,
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": False,
    }


def orgdet_elf_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(
        payload or orgdet_elf_config_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGDET_ELF_HASH = orgdet_elf_config_hash(orgdet_elf_config_payload())
assert LOCKED_ORGDET_ELF_HASH == "292c4f1f968f948d"
assert LOCKED_ORGDET_GAP_PRUNE_HASH == "862e3e7f7ac0330e"
assert REJECTED_ORGOP_HASH == "250c5bfacd4a8198"
assert LEARNED_ORGDET_ELF_LOCKED_TUNE == 0.7098380747225755
assert LEARNED_ORGDET_ELF_LOCKED_HOLDOUT == 0.7377873461563521


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
            raise ValueError(f"learned_unet_orgdet_elf_v1 tuning refuses stem {stem}")


def refuse_non_holdout_stems(stems: Iterable[str]) -> None:
    for raw in stems:
        stem = _clean_stem(str(raw))
        if specimen_of(stem) in HOLDOUT_SPECIMENS or stem in HOLDOUT_STEMS:
            if stem not in HOLDOUT_STEMS:
                raise ValueError(f"learned_unet_orgdet_elf_v1 holdout refuses extra stem {stem}")
            continue
        raise ValueError(f"learned_unet_orgdet_elf_v1 holdout refuses non-holdout stem {stem}")


def drop_long_dt1_edges(graph: TrackGraph, max_um: float = MAX_DT1_EDGE_UM) -> TrackGraph:
    """Drop consecutive dt=1 edges longer than ``max_um``. Keep remaining edges."""
    if float(max_um) != float(MAX_DT1_EDGE_UM):
        raise RuntimeError("learned_unet_orgdet_elf_v1 would retune max_dt1_edge_um; refusing")
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
        out.metadata["n_dropped_long_dt1_edges"] = 0
        out.metadata["n_kept_non_dt1_edges"] = 0
        out.metadata["max_dt1_edge_um"] = float(max_um)
        return out
    xyz = physical_coords(graph.coords_zyx(), SCALE_ZYX)
    index = graph.id_to_index()
    kept: list[tuple[int, int]] = []
    dropped = 0
    kept_non_dt1 = 0
    for src, dst in np.asarray(graph.edges).reshape(-1, 2) if graph.n_edges else []:
        s, d = int(src), int(dst)
        if s not in index or d not in index:
            raise RuntimeError("elf saw a dangling edge; refusing")
        i, j = index[s], index[d]
        dt = int(graph.t[j]) - int(graph.t[i])
        if dt != 1:
            kept.append((s, d))
            kept_non_dt1 += 1
            continue
        dist = float(np.linalg.norm(xyz[i] - xyz[j]))
        if dist > float(max_um):
            dropped += 1
            continue
        kept.append((s, d))
    out = TrackGraph(
        dataset=graph.dataset,
        node_id=graph.node_id,
        t=graph.t,
        z=graph.z,
        y=graph.y,
        x=graph.x,
        edges=np.asarray(kept, dtype=np.int64).reshape(-1, 2) if kept else np.zeros((0, 2), dtype=np.int64),
        estimated_n_nodes=graph.estimated_n_nodes,
    )
    out.metadata = dict(graph.metadata)
    out.metadata["n_dropped_long_dt1_edges"] = int(dropped)
    out.metadata["n_kept_non_dt1_edges"] = int(kept_non_dt1)
    out.metadata["n_nodes_before_elf"] = int(graph.n_nodes)
    out.metadata["n_edges_before_elf"] = int(graph.n_edges)
    out.metadata["max_dt1_edge_um"] = float(max_um)
    leftover = set(int(x) for x in out.node_id)
    if any(src not in leftover or dst not in leftover for src, dst in kept):
        raise RuntimeError("elf kept a dangling edge; refusing")
    return out


def apply_orgdet_elf_overlay(graph: TrackGraph) -> TrackGraph:
    """Locked orgdet+gap+prune, then drop long dt=1 edges and new isolates."""
    if orgdet_elf_config_hash() != LOCKED_ORGDET_ELF_HASH:
        raise RuntimeError("learned_unet_orgdet_elf_v1 hash drifted; refusing to retune")
    if float(LOCKED_GATES["max_dt1_edge_um"]) != float(MAX_DT1_EDGE_UM):
        raise RuntimeError("learned_unet_orgdet_elf_v1 would retune max_dt1_edge_um; refusing")
    if LOCKED_GATES["assignment"] != "unchanged_locked_orgdet_gap_prune":
        raise RuntimeError("learned_unet_orgdet_elf_v1 would retune assignment; refusing")
    if not LOCKED_GATES["drop_long_dt1_edges"] or not LOCKED_GATES["drop_degree0_after"]:
        raise RuntimeError("learned_unet_orgdet_elf_v1 required filters disabled; refusing")
    pruned = apply_orgdet_gap_prune_overlay(graph)
    filtered = drop_long_dt1_edges(pruned, max_um=LOCKED_GATES["max_dt1_edge_um"])
    n_dropped_edges = int(filtered.metadata.get("n_dropped_long_dt1_edges") or 0)
    out = drop_degree0(filtered)
    out.metadata["overlay"] = "learned_unet_orgdet_elf_v1"
    out.metadata["config_hash"] = LOCKED_ORGDET_ELF_HASH
    out.metadata["parent_orgdet_hash"] = LOCKED_ORGDET_HASH
    out.metadata["parent_orgdet_gap_hash"] = LOCKED_ORGDET_GAP_HASH
    out.metadata["parent_orgdet_gap_prune_hash"] = LOCKED_ORGDET_GAP_PRUNE_HASH
    out.metadata["rejected_orgop_hash"] = REJECTED_ORGOP_HASH
    out.metadata["n_dropped_long_dt1_edges"] = n_dropped_edges
    out.metadata["uses_intensity"] = False
    out.metadata["uses_transformer"] = False
    out.metadata["uses_ilp"] = False
    out.metadata["retunes_detection"] = False
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
        "n_dropped_long_dt1_edges": int(after.metadata.get("n_dropped_long_dt1_edges") or 0),
        "n_dropped_isolates": int(after.metadata.get("n_dropped_isolates") or 0),
        "n_nodes_before_elf": int(after.metadata.get("n_nodes_before_elf") or after.n_nodes),
        "n_edges_before_elf": int(after.metadata.get("n_edges_before_elf") or after.n_edges),
        "max_dt1_edge_um": float(after.metadata.get("max_dt1_edge_um") or MAX_DT1_EDGE_UM),
        "overlay": "learned_unet_orgdet_elf_v1",
        "config_hash": LOCKED_ORGDET_ELF_HASH,
    }


def refuse_existing_holdout_score(path: str | Path) -> None:
    dest = Path(path)
    if not dest.exists():
        return
    payload = json.loads(dest.read_text())
    if payload.get("status") == "holdout_scored" and payload.get("learned_orgdet_elf_combined") is not None:
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


def tuning_decision(learned_combined: float, n_dropped_edges: int) -> str:
    if float(learned_combined) < float(TUNE_KEEP_THRESHOLD):
        return "reject_learned_unet_orgdet_elf_v1"
    if int(n_dropped_edges) == 0:
        return "keep_as_noop_learned_unet_orgdet_elf_v1"
    return "keep_candidate_learned_unet_orgdet_elf_v1"


def holdout_decision(learned_combined: float) -> str:
    if float(learned_combined) <= float(ORGDET_GAP_PRUNE_LOCKED_HOLDOUT):
        return "reject_learned_unet_orgdet_elf_v1"
    return "keep_candidate_learned_unet_orgdet_elf_v1"


def runtime_estimate(
    n_volumes: int = 2,
    overlay_sec_per_volume: float = 6.0,
    elf_overhead: float = 0.5,
    limit_hours: float = 12.0,
    peak_memory_mb: float = 900.0,
) -> dict[str, Any]:
    n = int(n_volumes)
    sec = float(n) * (float(overlay_sec_per_volume) + float(elf_overhead))
    return {
        "n_volumes": n,
        "overlay_sec_per_volume": float(overlay_sec_per_volume),
        "elf_overhead": float(elf_overhead),
        "estimate_sec": sec,
        "estimate_hours": sec / 3600.0,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_budget": float(peak_memory_mb),
        "source": (
            "CPU locked orgdet+gap+prune then drop dt=1 edges longer than "
            f"{MAX_DT1_EDGE_UM} µm and new degree-0 leftovers; no transformer; "
            "no ILP; no new UNet"
        ),
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
    }
