"""Conservative claimed-child reassignment on locked FNN graphs.

Applies locked ``division_v2`` geometry to steal an already-claimed FNN
child when that child is closer to a one-child parent than to its
current 1-1 parent. Does not retune detection, the 12 µm linker, gap,
ELF, FNN, or division numeric gates. Does not relaunch rejected
``learned_unet_orgdet_fnn_div_v1`` (that overlay only attached unused
children). Learned graphs have no intensity. Not a competition
submission. A local official score is not a leaderboard number. Do not
overwrite locked slugs. Do not use public 0.602 / 0.612 / 0.553.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.spatial.distance import cdist

from .graphs import TrackGraph
from .division import sibling_alignment_cosine
from .learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from .learned_orgdet import LOCKED_ORGDET_HASH
from .learned_orgdet_elf import LOCKED_ORGDET_ELF_HASH
from .learned_orgdet_fnn import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LEARNED_ORGDET_FNN_LOCKED_HOLDOUT,
    LEARNED_ORGDET_FNN_LOCKED_TUNE,
    LOCKED_ORGDET_FNN_HASH,
    apply_orgdet_fnn_overlay,
    ensure_orgdet_tuning_graph,
    load_role_graphs,
    refuse_non_orgdet_node_count,
)
from .learned_orgdet_fnn_div import LOCKED_ORGDET_FNN_DIV_HASH
from .learned_orgdet_gap import LOCKED_ORGDET_GAP_HASH
from .learned_orgdet_gap_prune import LOCKED_ORGDET_GAP_PRUNE_HASH
from .learned_orgop import (
    LOCKED_ORGOP_HASH as REJECTED_ORGOP_HASH,
    ORGANIZER_DET_THRESHOLD,
    ORGANIZER_POOL_KERNEL_UM,
)
from .learned_split import HOLDOUT_SPECIMENS, TUNING_SPECIMEN, specimen_of
from .link import physical_coords

DIVISION_CONFIG_HASH = "0fa756963e5d42d1"
SCALE_ZYX = (1.625, 0.40625, 0.40625)
ORGDET_FNN_LOCKED_TUNE = LEARNED_ORGDET_FNN_LOCKED_TUNE
ORGDET_FNN_LOCKED_HOLDOUT = LEARNED_ORGDET_FNN_LOCKED_HOLDOUT
TUNE_MARGIN = 0.002000
TUNE_KEEP_THRESHOLD = ORGDET_FNN_LOCKED_TUNE + TUNE_MARGIN

# Locked division_v2 geometry from 44b6 GEFF priors. Numeric gates stay
# frozen. The new rule is claimed-child reassignment, not unused-child
# attachment and not a 12 µm retune.
LOCKED_GATES = {
    "max_parent_child_um": 10.0,
    "min_parent_child_um": 0.0,
    "min_child_separation_um": 8.3,
    "max_sibling_alignment": 0.05,
    "require_parent_predecessor": True,
    "require_no_alternative_parent": False,
    "require_closer_than_current_parent": True,
    "require_current_parent_outdeg_1": True,
    "require_current_child_indeg_1": True,
    "alternative_parent_gate_um": 12.0,
    "min_displacement_um": 1.0,
    "max_children": 2,
    "intensity_sum_tolerance": 0.45,
    "gate_um": 12.0,
    "assignment": "reassign_claimed_fnn_child_when_division_v2_geometry",
}


def orgdet_fnn_reas_config_payload() -> dict[str, Any]:
    """First FNN claimed-child reassignment. Lock before the 44b6 score."""
    return {
        "label": "learned_unet_orgdet_fnn_reas_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "orgdet_label": "learned_unet_orgdet_v1",
        "orgdet_gap_label": "learned_unet_orgdet_gap_v1",
        "orgdet_gap_prune_label": "learned_unet_orgdet_gap_prune_v1",
        "orgdet_elf_label": "learned_unet_orgdet_elf_v1",
        "orgdet_fnn_label": "learned_unet_orgdet_fnn_v1",
        "rejected_fnn_div_label": "learned_unet_orgdet_fnn_div_v1",
        "division_label": "division_v2",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "orgdet_config_hash": LOCKED_ORGDET_HASH,
        "orgdet_gap_config_hash": LOCKED_ORGDET_GAP_HASH,
        "orgdet_gap_prune_config_hash": LOCKED_ORGDET_GAP_PRUNE_HASH,
        "orgdet_elf_config_hash": LOCKED_ORGDET_ELF_HASH,
        "orgdet_fnn_config_hash": LOCKED_ORGDET_FNN_HASH,
        "rejected_fnn_div_hash": LOCKED_ORGDET_FNN_DIV_HASH,
        "division_config_hash": DIVISION_CONFIG_HASH,
        "source_graphs": "locked_learned_unet_orgdet_fnn_graphs",
        "retunes_detection": False,
        "retunes_consecutive_links": False,
        "retunes_gap_gates": False,
        "retunes_max_dt1_edge_um": False,
        "retunes_nn_tie_eps": False,
        "retunes_division_gates": False,
        "uses_intensity": False,
        "uses_transformer": False,
        "uses_velocity": False,
        "uses_ilp": False,
        "linker": "orgdet_fnn_then_locked_division_v2_claimed_child_reassign",
        "allow_divisions": True,
        "max_parent_child_um": LOCKED_GATES["max_parent_child_um"],
        "min_parent_child_um": LOCKED_GATES["min_parent_child_um"],
        "min_child_separation_um": LOCKED_GATES["min_child_separation_um"],
        "max_sibling_alignment": LOCKED_GATES["max_sibling_alignment"],
        "require_parent_predecessor": LOCKED_GATES["require_parent_predecessor"],
        "require_no_alternative_parent": LOCKED_GATES["require_no_alternative_parent"],
        "require_closer_than_current_parent": LOCKED_GATES["require_closer_than_current_parent"],
        "require_current_parent_outdeg_1": LOCKED_GATES["require_current_parent_outdeg_1"],
        "require_current_child_indeg_1": LOCKED_GATES["require_current_child_indeg_1"],
        "alternative_parent_gate_um": LOCKED_GATES["alternative_parent_gate_um"],
        "min_displacement_um": LOCKED_GATES["min_displacement_um"],
        "max_children": LOCKED_GATES["max_children"],
        "intensity_sum_tolerance": LOCKED_GATES["intensity_sum_tolerance"],
        "gate_um": LOCKED_GATES["gate_um"],
        "assignment": LOCKED_GATES["assignment"],
        "tune_keep_threshold": TUNE_KEEP_THRESHOLD,
        "tune_margin": TUNE_MARGIN,
        "det_threshold": ORGANIZER_DET_THRESHOLD,
        "det_tta": True,
        "pool_kernel_um": ORGANIZER_POOL_KERNEL_UM,
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": False,
    }


def orgdet_fnn_reas_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(
        payload or orgdet_fnn_reas_config_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGDET_FNN_REAS_HASH = orgdet_fnn_reas_config_hash(orgdet_fnn_reas_config_payload())
assert LOCKED_ORGDET_FNN_REAS_HASH == "39fb20d675e21d0e"
assert LOCKED_ORGDET_FNN_HASH == "95e9300eb84ba71c"
assert LOCKED_ORGDET_FNN_DIV_HASH == "e184e6ce5bd90e46"
assert DIVISION_CONFIG_HASH == "0fa756963e5d42d1"
assert TUNE_KEEP_THRESHOLD == 0.7266526336456757
assert REJECTED_ORGOP_HASH == "250c5bfacd4a8198"
assert LOCKED_ORGDET_FNN_REAS_HASH != LOCKED_ORGDET_FNN_DIV_HASH
assert LOCKED_ORGDET_FNN_REAS_HASH != LOCKED_ORGDET_FNN_HASH


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
            raise ValueError(f"learned_unet_orgdet_fnn_reas_v1 tuning refuses stem {stem}")


def refuse_non_holdout_stems(stems: Iterable[str]) -> None:
    for raw in stems:
        stem = _clean_stem(str(raw))
        if specimen_of(stem) in HOLDOUT_SPECIMENS or stem in HOLDOUT_STEMS:
            if stem not in HOLDOUT_STEMS:
                raise ValueError(f"learned_unet_orgdet_fnn_reas_v1 holdout refuses extra stem {stem}")
            continue
        raise ValueError(f"learned_unet_orgdet_fnn_reas_v1 holdout refuses non-holdout stem {stem}")


def n_predicted_forks(graph: TrackGraph) -> int:
    return int(sum(1 for children in graph.out_edges().values() if len(children) >= 2))


def reassign_claimed_fnn_children(graph: TrackGraph) -> TrackGraph:
    """Steal at most one already-claimed child per parent when geometry holds."""
    if orgdet_fnn_reas_config_hash() != LOCKED_ORGDET_FNN_REAS_HASH:
        raise RuntimeError("learned_unet_orgdet_fnn_reas_v1 hash drifted; refusing to retune")
    if LOCKED_GATES["max_parent_child_um"] != 10.0 or LOCKED_GATES["min_child_separation_um"] != 8.3:
        raise RuntimeError("learned_unet_orgdet_fnn_reas_v1 would retune division gates; refusing")
    if LOCKED_GATES["max_sibling_alignment"] != 0.05:
        raise RuntimeError("learned_unet_orgdet_fnn_reas_v1 would retune alignment; refusing")
    if LOCKED_GATES["require_no_alternative_parent"]:
        raise RuntimeError("learned_unet_orgdet_fnn_reas_v1 must reassign the alternative parent")
    if not LOCKED_GATES["require_closer_than_current_parent"]:
        raise RuntimeError("learned_unet_orgdet_fnn_reas_v1 closer-parent gate disabled; refusing")
    if not LOCKED_GATES["require_current_parent_outdeg_1"]:
        raise RuntimeError("learned_unet_orgdet_fnn_reas_v1 would steal from a predicted fork")
    if not LOCKED_GATES["require_current_child_indeg_1"]:
        raise RuntimeError("learned_unet_orgdet_fnn_reas_v1 requires a unique current parent")
    if LOCKED_GATES["assignment"] != "reassign_claimed_fnn_child_when_division_v2_geometry":
        raise RuntimeError("learned_unet_orgdet_fnn_reas_v1 assignment drifted; refusing")

    max_parent = float(LOCKED_GATES["max_parent_child_um"])
    min_parent = float(LOCKED_GATES["min_parent_child_um"])
    min_sep = float(LOCKED_GATES["min_child_separation_um"])
    max_align = float(LOCKED_GATES["max_sibling_alignment"])
    min_disp = float(LOCKED_GATES["min_displacement_um"])
    max_children = int(LOCKED_GATES["max_children"])

    if graph.n_nodes == 0:
        out = graph.copy()
        out.metadata = dict(out.metadata)
        out.metadata["n_reassigned"] = 0
        out.metadata["n_division_proposals"] = 0
        out.metadata["n_division_rejected"] = {}
        return out

    index = graph.id_to_index()
    coords = graph.coords_zyx()
    out_map = graph.out_edges()
    incoming = graph.in_edges()
    by_t: dict[int, list[int]] = {}
    for nid, t in zip(graph.node_id, graph.t):
        by_t.setdefault(int(t), []).append(int(nid))

    rejected = {
        "predecessor": 0,
        "dt": 0,
        "parent_child": 0,
        "claimed_pool_empty": 0,
        "sibling_sep": 0,
        "alignment": 0,
        "closer": 0,
        "current_parent_outdeg": 0,
        "current_child_indeg": 0,
        "current_parent_time": 0,
        "unused_child": 0,
    }
    proposals: list[tuple[float, int, int, int]] = []
    parent_candidates = [int(nid) for nid, children in out_map.items() if len(children) == 1]
    for parent in parent_candidates:
        if LOCKED_GATES["require_parent_predecessor"] and len(incoming.get(parent, [])) == 0:
            rejected["predecessor"] += 1
            continue
        pidx = index[parent]
        t_parent = int(graph.t[pidx])
        child1 = int(out_map[parent][0])
        if child1 not in index:
            continue
        t_child = int(graph.t[index[child1]])
        if t_child != t_parent + 1:
            rejected["dt"] += 1
            continue
        parent_xyz = physical_coords(coords[pidx : pidx + 1], SCALE_ZYX)
        child1_xyz = physical_coords(coords[index[child1] : index[child1] + 1], SCALE_ZYX)
        d_parent_child1 = float(np.linalg.norm(child1_xyz[0] - parent_xyz[0]))
        if d_parent_child1 > max_parent or d_parent_child1 < min_parent:
            rejected["parent_child"] += 1
            continue
        claimed = []
        for nid in by_t.get(t_child, []):
            nid = int(nid)
            if nid == child1:
                continue
            parents = [int(p) for p in incoming.get(nid, [])]
            if len(parents) == 0:
                rejected["unused_child"] += 1
                continue
            if LOCKED_GATES["require_current_child_indeg_1"] and len(parents) != 1:
                rejected["current_child_indeg"] += 1
                continue
            current = parents[0]
            if current == parent:
                continue
            if current not in index:
                continue
            if int(graph.t[index[current]]) != t_parent:
                rejected["current_parent_time"] += 1
                continue
            if LOCKED_GATES["require_current_parent_outdeg_1"] and len(out_map.get(current, [])) != 1:
                rejected["current_parent_outdeg"] += 1
                continue
            claimed.append((nid, current))
        if not claimed:
            rejected["claimed_pool_empty"] += 1
            continue
        claimed_ids = [nid for nid, _current in claimed]
        claimed_idx = np.array([index[n] for n in claimed_ids], dtype=np.int64)
        claimed_xyz = physical_coords(coords[claimed_idx], SCALE_ZYX)
        d_parent = cdist(parent_xyz, claimed_xyz)[0]
        d_sib = cdist(child1_xyz, claimed_xyz)[0]
        for j, (nid, current) in enumerate(claimed):
            if d_parent[j] > max_parent or d_parent[j] < min_parent:
                rejected["parent_child"] += 1
                continue
            if d_sib[j] < min_sep:
                rejected["sibling_sep"] += 1
                continue
            cosine = sibling_alignment_cosine(
                parent_xyz[0],
                child1_xyz[0],
                claimed_xyz[j],
                min_displacement_um=min_disp,
            )
            if cosine is not None and cosine > max_align:
                rejected["alignment"] += 1
                continue
            current_xyz = physical_coords(coords[index[current] : index[current] + 1], SCALE_ZYX)
            d_current = float(np.linalg.norm(claimed_xyz[j] - current_xyz[0]))
            if LOCKED_GATES["require_closer_than_current_parent"] and not (float(d_parent[j]) < d_current):
                rejected["closer"] += 1
                continue
            proposals.append((float(d_parent[j]), parent, int(nid), int(current)))

    proposals.sort()
    claimed_parent: set[int] = set()
    claimed_child: set[int] = set()
    claimed_old: set[int] = set()
    swaps: list[tuple[int, int, int]] = []
    for _dist, parent, child, current in proposals:
        if parent in claimed_parent or child in claimed_child or current in claimed_old:
            continue
        if len(out_map.get(parent, [])) >= max_children:
            continue
        swaps.append((parent, child, current))
        claimed_parent.add(parent)
        claimed_child.add(child)
        claimed_old.add(current)

    remove = {(current, child) for _parent, child, current in swaps}
    extra = [(parent, child) for parent, child, _current in swaps]
    kept = [
        (int(src), int(dst))
        for src, dst in np.asarray(graph.edges).reshape(-1, 2)
        if (int(src), int(dst)) not in remove
    ]
    kept.extend(extra)
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
    out.metadata["overlay"] = "learned_unet_orgdet_fnn_reas_v1"
    out.metadata["config_hash"] = LOCKED_ORGDET_FNN_REAS_HASH
    out.metadata["parent_orgdet_fnn_hash"] = LOCKED_ORGDET_FNN_HASH
    out.metadata["rejected_fnn_div_hash"] = LOCKED_ORGDET_FNN_DIV_HASH
    out.metadata["division_config_hash"] = DIVISION_CONFIG_HASH
    out.metadata["n_reassigned"] = len(swaps)
    out.metadata["n_division_proposals"] = len(swaps)
    out.metadata["n_removed_edges"] = len(remove)
    out.metadata["n_added_edges"] = len(extra)
    out.metadata["n_division_rejected"] = rejected
    out.metadata["uses_intensity"] = False
    out.metadata["uses_transformer"] = False
    out.metadata["uses_ilp"] = False
    out.metadata["retunes_detection"] = False
    out.metadata["retunes_consecutive_links"] = False
    out.metadata["retunes_gap_gates"] = False
    out.metadata["retunes_division_gates"] = False
    out.metadata["is_12um_assignment"] = False
    out.metadata["is_fnn_div_retune"] = False
    return out


def apply_locked_fnn_reas(graph: TrackGraph) -> TrackGraph:
    """Reassign claimed FNN children when locked division_v2 geometry holds."""
    return reassign_claimed_fnn_children(graph)


def apply_orgdet_fnn_reas_overlay(graph: TrackGraph) -> TrackGraph:
    """Locked FNN, then claimed-child reassignment under locked geometry."""
    fnn = apply_orgdet_fnn_overlay(graph)
    return apply_locked_fnn_reas(fnn)


def overlay_summary(before: TrackGraph, after: TrackGraph) -> dict[str, Any]:
    return {
        "dataset": after.dataset,
        "n_nodes": after.n_nodes,
        "n_edges_before": before.n_edges,
        "n_edges_after": after.n_edges,
        "n_reassigned": int(after.metadata.get("n_reassigned") or 0),
        "n_removed_edges": int(after.metadata.get("n_removed_edges") or 0),
        "n_added_edges": int(after.metadata.get("n_added_edges") or 0),
        "n_forks_before": n_predicted_forks(before),
        "n_forks_after": n_predicted_forks(after),
        "n_division_proposals": int(after.metadata.get("n_division_proposals") or 0),
        "n_division_rejected": dict(after.metadata.get("n_division_rejected") or {}),
        "overlay": "learned_unet_orgdet_fnn_reas_v1",
        "config_hash": LOCKED_ORGDET_FNN_REAS_HASH,
    }


def refuse_existing_holdout_score(path: str | Path) -> None:
    dest = Path(path)
    if not dest.exists():
        return
    payload = json.loads(dest.read_text())
    if payload.get("status") == "holdout_scored" and payload.get("learned_orgdet_fnn_reas_combined") is not None:
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


def tuning_decision(learned_combined: float, n_reassigned: int) -> str:
    if float(learned_combined) < float(TUNE_KEEP_THRESHOLD):
        return "reject_learned_unet_orgdet_fnn_reas_v1"
    if int(n_reassigned) == 0:
        return "keep_as_noop_learned_unet_orgdet_fnn_reas_v1"
    return "keep_candidate_learned_unet_orgdet_fnn_reas_v1"


def holdout_decision(learned_combined: float) -> str:
    if float(learned_combined) <= float(ORGDET_FNN_LOCKED_HOLDOUT):
        return "reject_learned_unet_orgdet_fnn_reas_v1"
    return "keep_candidate_learned_unet_orgdet_fnn_reas_v1"


def runtime_estimate(
    n_volumes: int = 2,
    overlay_sec_per_volume: float = 36.0,
    reassign_overhead: float = 2.0,
    limit_hours: float = 12.0,
    peak_memory_mb: float = 900.0,
) -> dict[str, Any]:
    n = int(n_volumes)
    sec = float(n) * (float(overlay_sec_per_volume) + float(reassign_overhead))
    return {
        "n_volumes": n,
        "overlay_sec_per_volume": float(overlay_sec_per_volume),
        "reassign_overhead": float(reassign_overhead),
        "estimate_sec": sec,
        "estimate_hours": sec / 3600.0,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_budget": float(peak_memory_mb),
        "source": (
            "CPU locked orgdet+gap+prune+ELF+FNN then claimed-child "
            "reassignment under locked division_v2 geometry; no transformer; "
            "no ILP; no new UNet; no appearance Zarrs"
        ),
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
    }


__all__ = [
    "DIVISION_CONFIG_HASH",
    "HOLDOUT_STEMS",
    "INFER_STEMS",
    "LOCKED_GATES",
    "LOCKED_ORGDET_FNN_REAS_HASH",
    "ORGDET_FNN_LOCKED_HOLDOUT",
    "ORGDET_FNN_LOCKED_TUNE",
    "TUNE_KEEP_THRESHOLD",
    "TUNE_MARGIN",
    "apply_locked_fnn_reas",
    "apply_orgdet_fnn_reas_overlay",
    "ensure_orgdet_tuning_graph",
    "holdout_decision",
    "load_role_graphs",
    "n_predicted_forks",
    "orgdet_fnn_reas_config_hash",
    "orgdet_fnn_reas_config_payload",
    "overlay_summary",
    "reassign_claimed_fnn_children",
    "refuse_existing_holdout_score",
    "refuse_non_holdout_stems",
    "refuse_non_orgdet_node_count",
    "refuse_non_tuning_stems",
    "require_tuning_keep",
    "runtime_estimate",
    "tuning_decision",
]
