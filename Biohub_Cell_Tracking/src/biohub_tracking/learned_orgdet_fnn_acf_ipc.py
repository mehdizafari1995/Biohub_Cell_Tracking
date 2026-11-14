"""Keep the nearest already-chosen incoming parent after locked ACF.

Official 44b6 ACF residue is still association-dominated (160 conflict
FPs / 143 FNs; 0 predicted divisions). Global backward-NN among all
nodes at t−1 is closed: ``learned_unet_orgdet_fnn_acf_bnn_v1`` dropped
true parents as well as conflict FPs. Locked FNN already keeps only
source→forward-NN destinations. Incoming conflicts remain: several
already-chosen ACF parents can still claim one child. When a child has
two or more incoming dt=1 edges, keep only the nearest of those
already-chosen parents (ties kept). Unique incoming dt=1 edges stay.
This is not global t−1 nearest-neighbor. It does not reassign, does
not shrink the 12 µm linker/gap gates, and is not an ELF / FNN / ACF /
BNN / TCF / rank / division retune.

Never opens ``6bba`` during tuning. Does not write ``submission.csv``.
Not a competition submission. A local official score is not a
leaderboard number. Do not overwrite locked slugs. Do not retune locked
parent hashes. Do not use public 0.602 / 0.612 / 0.553.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .graphs import TrackGraph
from .learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from .learned_orgdet import LOCKED_ORGDET_HASH
from .learned_orgdet_elf import LOCKED_ORGDET_ELF_HASH
from .learned_orgdet_fnn import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LOCKED_ORGDET_FNN_HASH,
    NN_TIE_EPS,
    ensure_orgdet_tuning_graph,
    load_role_graphs,
    refuse_non_orgdet_node_count,
)
from .learned_orgdet_fnn_acf import (
    LEARNED_ORGDET_FNN_ACF_LOCKED_HOLDOUT,
    LEARNED_ORGDET_FNN_ACF_LOCKED_TUNE,
    LOCKED_ORGDET_FNN_ACF_HASH,
    MAX_REL_APPEARANCE,
    apply_orgdet_fnn_acf_overlay,
    resolve_sequence_zarr,
)
from .learned_orgdet_fnn_acf_bnn import LOCKED_ORGDET_FNN_ACF_BNN_HASH
from .learned_orgdet_fnn_div import LOCKED_ORGDET_FNN_DIV_HASH
from .learned_orgdet_fnn_rank import LOCKED_RANK_HASH as LOCKED_ORGDET_FNN_RANK_HASH
from .learned_orgdet_fnn_reas import LOCKED_ORGDET_FNN_REAS_HASH
from .learned_orgdet_gap import LOCKED_ORGDET_GAP_HASH
from .learned_orgdet_gap_prune import LOCKED_ORGDET_GAP_PRUNE_HASH
from .learned_orgop import (
    LOCKED_ORGOP_HASH as REJECTED_ORGOP_HASH,
    ORGANIZER_DET_THRESHOLD,
    ORGANIZER_POOL_KERNEL_UM,
)
from .learned_prune import drop_degree0
from .learned_split import HOLDOUT_SPECIMENS, TUNING_SPECIMEN, specimen_of
from .link import physical_coords

SCALE_ZYX = (1.625, 0.40625, 0.40625)
ORGDET_FNN_ACF_LOCKED_TUNE = LEARNED_ORGDET_FNN_ACF_LOCKED_TUNE
ORGDET_FNN_ACF_LOCKED_HOLDOUT = LEARNED_ORGDET_FNN_ACF_LOCKED_HOLDOUT
TUNE_MARGIN = 0.002000
TUNE_KEEP_THRESHOLD = ORGDET_FNN_ACF_LOCKED_TUNE + TUNE_MARGIN

LOCKED_GATES = {
    "keep_nearest_chosen_incoming": True,
    "drop_degree0_after": True,
    "nn_tie_eps": NN_TIE_EPS,
    "assignment": "unchanged_locked_orgdet_fnn_acf",
}


def orgdet_fnn_acf_ipc_config_payload() -> dict[str, Any]:
    """First incoming-parent conflict keep after locked ACF. Lock before 44b6."""
    return {
        "label": "learned_unet_orgdet_fnn_acf_ipc_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "orgdet_label": "learned_unet_orgdet_v1",
        "orgdet_gap_label": "learned_unet_orgdet_gap_v1",
        "orgdet_gap_prune_label": "learned_unet_orgdet_gap_prune_v1",
        "orgdet_elf_label": "learned_unet_orgdet_elf_v1",
        "orgdet_fnn_label": "learned_unet_orgdet_fnn_v1",
        "orgdet_fnn_acf_label": "learned_unet_orgdet_fnn_acf_v1",
        "rejected_fnn_rank_label": "learned_unet_orgdet_fnn_rank_v1",
        "rejected_fnn_div_label": "learned_unet_orgdet_fnn_div_v1",
        "rejected_fnn_reas_label": "learned_unet_orgdet_fnn_reas_v1",
        "rejected_fnn_acf_bnn_label": "learned_unet_orgdet_fnn_acf_bnn_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "orgdet_config_hash": LOCKED_ORGDET_HASH,
        "orgdet_gap_config_hash": LOCKED_ORGDET_GAP_HASH,
        "orgdet_gap_prune_config_hash": LOCKED_ORGDET_GAP_PRUNE_HASH,
        "orgdet_elf_config_hash": LOCKED_ORGDET_ELF_HASH,
        "orgdet_fnn_config_hash": LOCKED_ORGDET_FNN_HASH,
        "orgdet_fnn_acf_config_hash": LOCKED_ORGDET_FNN_ACF_HASH,
        "rejected_fnn_rank_hash": LOCKED_ORGDET_FNN_RANK_HASH,
        "rejected_fnn_div_hash": LOCKED_ORGDET_FNN_DIV_HASH,
        "rejected_fnn_reas_hash": LOCKED_ORGDET_FNN_REAS_HASH,
        "rejected_fnn_acf_bnn_hash": LOCKED_ORGDET_FNN_ACF_BNN_HASH,
        "source_graphs": "locked_learned_unet_orgdet_fnn_acf_graphs",
        "source_zarrs": "locked_tuning_zarr_export_two_44b6",
        "retunes_detection": False,
        "retunes_consecutive_links": False,
        "retunes_gap_gates": False,
        "retunes_max_dt1_edge_um": False,
        "retunes_nn_tie_eps": False,
        "retunes_max_rel_appearance": False,
        "uses_intensity": True,
        "uses_transformer": False,
        "uses_velocity": False,
        "uses_ilp": False,
        "linker": "orgdet_fnn_acf_then_keep_nearest_chosen_incoming_dt1",
        "keep_nearest_chosen_incoming": LOCKED_GATES["keep_nearest_chosen_incoming"],
        "drop_degree0_after": LOCKED_GATES["drop_degree0_after"],
        "nn_tie_eps": LOCKED_GATES["nn_tie_eps"],
        "assignment": LOCKED_GATES["assignment"],
        "tune_keep_threshold": TUNE_KEEP_THRESHOLD,
        "tune_margin": TUNE_MARGIN,
        "det_threshold": ORGANIZER_DET_THRESHOLD,
        "det_tta": True,
        "pool_kernel_um": ORGANIZER_POOL_KERNEL_UM,
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": False,
    }


def orgdet_fnn_acf_ipc_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(
        payload or orgdet_fnn_acf_ipc_config_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGDET_FNN_ACF_IPC_HASH = orgdet_fnn_acf_ipc_config_hash(orgdet_fnn_acf_ipc_config_payload())
assert LOCKED_ORGDET_FNN_ACF_IPC_HASH == "7302409595bee523"
assert LOCKED_ORGDET_FNN_ACF_HASH == "a66075085ac64400"
assert LOCKED_ORGDET_FNN_HASH == "95e9300eb84ba71c"
assert LOCKED_ORGDET_FNN_RANK_HASH == "2748c4a2b2823c0a"
assert LOCKED_ORGDET_FNN_DIV_HASH == "e184e6ce5bd90e46"
assert LOCKED_ORGDET_FNN_REAS_HASH == "39fb20d675e21d0e"
assert LOCKED_ORGDET_FNN_ACF_BNN_HASH == "bdb38cf13e226ced"
assert TUNE_KEEP_THRESHOLD == 0.7366197119237633
assert REJECTED_ORGOP_HASH == "250c5bfacd4a8198"
assert LOCKED_ORGDET_FNN_ACF_IPC_HASH != LOCKED_ORGDET_FNN_ACF_HASH
assert LOCKED_ORGDET_FNN_ACF_IPC_HASH != LOCKED_ORGDET_FNN_HASH
assert LOCKED_ORGDET_FNN_ACF_IPC_HASH != LOCKED_ORGDET_FNN_RANK_HASH
assert LOCKED_ORGDET_FNN_ACF_IPC_HASH != LOCKED_ORGDET_FNN_DIV_HASH
assert LOCKED_ORGDET_FNN_ACF_IPC_HASH != LOCKED_ORGDET_FNN_REAS_HASH
assert LOCKED_ORGDET_FNN_ACF_IPC_HASH != LOCKED_ORGDET_FNN_ACF_BNN_HASH
assert LEARNED_ORGDET_FNN_ACF_LOCKED_TUNE == 0.7346197119237633
assert LEARNED_ORGDET_FNN_ACF_LOCKED_HOLDOUT == 0.7555029702166214


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
            raise ValueError(f"learned_unet_orgdet_fnn_acf_ipc_v1 tuning refuses stem {stem}")


def refuse_non_holdout_stems(stems: Iterable[str]) -> None:
    for raw in stems:
        stem = _clean_stem(str(raw))
        if specimen_of(stem) in HOLDOUT_SPECIMENS or stem in HOLDOUT_STEMS:
            if stem not in HOLDOUT_STEMS:
                raise ValueError(f"learned_unet_orgdet_fnn_acf_ipc_v1 holdout refuses extra stem {stem}")
            continue
        raise ValueError(f"learned_unet_orgdet_fnn_acf_ipc_v1 holdout refuses non-holdout stem {stem}")


def _empty_ipc_graph(graph: TrackGraph, tie_eps: float) -> TrackGraph:
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
    out.metadata["n_dropped_conflict_incoming_edges"] = 0
    out.metadata["n_conflict_children"] = 0
    out.metadata["n_kept_unique_incoming_edges"] = 0
    out.metadata["n_kept_nearest_chosen_incoming"] = 0
    out.metadata["n_kept_non_dt1_edges"] = 0
    out.metadata["nn_tie_eps"] = float(tie_eps)
    return out


def _assemble_ipc_graph(
    graph: TrackGraph,
    kept: list[tuple[int, int]],
    dropped: int,
    n_conflict_children: int,
    kept_unique: int,
    kept_nearest: int,
    kept_non_dt1: int,
    tie_eps: float,
) -> TrackGraph:
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
    out.metadata["n_dropped_conflict_incoming_edges"] = int(dropped)
    out.metadata["n_conflict_children"] = int(n_conflict_children)
    out.metadata["n_kept_unique_incoming_edges"] = int(kept_unique)
    out.metadata["n_kept_nearest_chosen_incoming"] = int(kept_nearest)
    out.metadata["n_kept_non_dt1_edges"] = int(kept_non_dt1)
    out.metadata["n_nodes_before_ipc"] = int(graph.n_nodes)
    out.metadata["n_edges_before_ipc"] = int(graph.n_edges)
    out.metadata["nn_tie_eps"] = float(tie_eps)
    leftover = set(int(x) for x in out.node_id)
    if any(src not in leftover or dst not in leftover for src, dst in kept):
        raise RuntimeError("ipc kept a dangling edge; refusing")
    return out


def drop_non_nearest_chosen_incoming(graph: TrackGraph, tie_eps: float = NN_TIE_EPS) -> TrackGraph:
    """Keep unique incoming dt=1; on conflicts keep nearest already-chosen parents.

    Comparison is only among already-chosen incoming ACF parents of the
    same child. A closer unlinked node at t−1 is ignored. That is the
    difference from rejected global backward-NN. Do not retune
    ``nn_tie_eps``.
    """
    if float(tie_eps) != float(NN_TIE_EPS):
        raise RuntimeError("learned_unet_orgdet_fnn_acf_ipc_v1 would retune nn_tie_eps; refusing")
    nids = [int(x) for x in graph.node_id]
    if not nids:
        return _empty_ipc_graph(graph, tie_eps)
    xyz = physical_coords(graph.coords_zyx(), SCALE_ZYX)
    index = graph.id_to_index()
    incoming: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    non_dt1: list[tuple[int, int]] = []
    raw_edges = np.asarray(graph.edges).reshape(-1, 2) if graph.n_edges else np.zeros((0, 2), dtype=np.int64)
    for src, dst in raw_edges:
        s, d = int(src), int(dst)
        if s not in index or d not in index:
            raise RuntimeError("ipc saw a dangling edge; refusing")
        i, j = index[s], index[d]
        dt = int(graph.t[j]) - int(graph.t[i])
        if dt != 1:
            non_dt1.append((s, d))
            continue
        dest_dist = float(np.linalg.norm(xyz[j] - xyz[i]))
        incoming[d].append((s, d, dest_dist))
    keep: set[tuple[int, int]] = set(non_dt1)
    dropped = 0
    n_conflict_children = 0
    kept_unique = 0
    kept_nearest = 0
    for _dst, cands in incoming.items():
        if len(cands) <= 1:
            for s, d, _dist in cands:
                keep.add((s, d))
                kept_unique += 1
            continue
        n_conflict_children += 1
        min_dist = min(float(c[2]) for c in cands)
        for s, d, dist in cands:
            if float(dist) <= float(min_dist) + float(tie_eps):
                keep.add((s, d))
                kept_nearest += 1
            else:
                dropped += 1
    kept = [(int(s), int(d)) for s, d in raw_edges if (int(s), int(d)) in keep]
    return _assemble_ipc_graph(
        graph,
        kept,
        dropped,
        n_conflict_children,
        kept_unique,
        kept_nearest,
        len(non_dt1),
        tie_eps,
    )


def apply_orgdet_fnn_acf_ipc_overlay(
    graph: TrackGraph,
    appearance_by_id: Mapping[int, float] | None = None,
    zarr_path: str | Path | None = None,
) -> TrackGraph:
    """Locked ACF, then keep nearest already-chosen incoming parents and new isolates."""
    if orgdet_fnn_acf_ipc_config_hash() != LOCKED_ORGDET_FNN_ACF_IPC_HASH:
        raise RuntimeError("learned_unet_orgdet_fnn_acf_ipc_v1 hash drifted; refusing to retune")
    if LOCKED_GATES["assignment"] != "unchanged_locked_orgdet_fnn_acf":
        raise RuntimeError("learned_unet_orgdet_fnn_acf_ipc_v1 would retune assignment; refusing")
    if not LOCKED_GATES["keep_nearest_chosen_incoming"] or not LOCKED_GATES["drop_degree0_after"]:
        raise RuntimeError("learned_unet_orgdet_fnn_acf_ipc_v1 required filters disabled; refusing")
    if float(LOCKED_GATES["nn_tie_eps"]) != float(NN_TIE_EPS):
        raise RuntimeError("learned_unet_orgdet_fnn_acf_ipc_v1 would retune nn_tie_eps; refusing")
    acf = apply_orgdet_fnn_acf_overlay(graph, appearance_by_id=appearance_by_id, zarr_path=zarr_path)
    filtered = drop_non_nearest_chosen_incoming(acf, tie_eps=LOCKED_GATES["nn_tie_eps"])
    n_dropped_edges = int(filtered.metadata.get("n_dropped_conflict_incoming_edges") or 0)
    out = drop_degree0(filtered)
    out.metadata["overlay"] = "learned_unet_orgdet_fnn_acf_ipc_v1"
    out.metadata["config_hash"] = LOCKED_ORGDET_FNN_ACF_IPC_HASH
    out.metadata["parent_orgdet_hash"] = LOCKED_ORGDET_HASH
    out.metadata["parent_orgdet_gap_hash"] = LOCKED_ORGDET_GAP_HASH
    out.metadata["parent_orgdet_gap_prune_hash"] = LOCKED_ORGDET_GAP_PRUNE_HASH
    out.metadata["parent_orgdet_elf_hash"] = LOCKED_ORGDET_ELF_HASH
    out.metadata["parent_orgdet_fnn_hash"] = LOCKED_ORGDET_FNN_HASH
    out.metadata["parent_orgdet_fnn_acf_hash"] = LOCKED_ORGDET_FNN_ACF_HASH
    out.metadata["rejected_orgop_hash"] = REJECTED_ORGOP_HASH
    out.metadata["rejected_fnn_rank_hash"] = LOCKED_ORGDET_FNN_RANK_HASH
    out.metadata["rejected_fnn_div_hash"] = LOCKED_ORGDET_FNN_DIV_HASH
    out.metadata["rejected_fnn_reas_hash"] = LOCKED_ORGDET_FNN_REAS_HASH
    out.metadata["rejected_fnn_acf_bnn_hash"] = LOCKED_ORGDET_FNN_ACF_BNN_HASH
    out.metadata["n_dropped_conflict_incoming_edges"] = n_dropped_edges
    out.metadata["n_conflict_children"] = int(filtered.metadata.get("n_conflict_children") or 0)
    out.metadata["n_dropped_inconsistent_dt1_edges"] = int(
        acf.metadata.get("n_dropped_inconsistent_dt1_edges") or 0
    )
    out.metadata["n_dropped_non_forward_nn_edges"] = int(
        acf.metadata.get("n_dropped_non_forward_nn_edges") or 0
    )
    out.metadata["n_dropped_long_dt1_edges"] = int(acf.metadata.get("n_dropped_long_dt1_edges") or 0)
    out.metadata["max_rel_appearance"] = float(acf.metadata.get("max_rel_appearance") or MAX_REL_APPEARANCE)
    out.metadata["uses_intensity"] = True
    out.metadata["uses_transformer"] = False
    out.metadata["uses_ilp"] = False
    out.metadata["retunes_detection"] = False
    out.metadata["retunes_consecutive_links"] = False
    out.metadata["retunes_gap_gates"] = False
    out.metadata["retunes_max_dt1_edge_um"] = False
    out.metadata["retunes_nn_tie_eps"] = False
    out.metadata["retunes_max_rel_appearance"] = False
    out.metadata["is_12um_assignment"] = False
    out.metadata["is_global_backward_nn"] = False
    return out


def overlay_summary(before: TrackGraph, after: TrackGraph) -> dict[str, Any]:
    return {
        "dataset": after.dataset,
        "n_nodes_before": before.n_nodes,
        "n_nodes_after": after.n_nodes,
        "n_edges_before": before.n_edges,
        "n_edges_after": after.n_edges,
        "n_dropped_conflict_incoming_edges": int(after.metadata.get("n_dropped_conflict_incoming_edges") or 0),
        "n_conflict_children": int(after.metadata.get("n_conflict_children") or 0),
        "n_dropped_isolates": int(after.metadata.get("n_dropped_isolates") or 0),
        "n_dropped_inconsistent_dt1_edges": int(after.metadata.get("n_dropped_inconsistent_dt1_edges") or 0),
        "n_dropped_non_forward_nn_edges": int(after.metadata.get("n_dropped_non_forward_nn_edges") or 0),
        "n_dropped_long_dt1_edges": int(after.metadata.get("n_dropped_long_dt1_edges") or 0),
        "n_nodes_before_ipc": int(after.metadata.get("n_nodes_before_ipc") or after.n_nodes),
        "n_edges_before_ipc": int(after.metadata.get("n_edges_before_ipc") or after.n_edges),
        "nn_tie_eps": float(after.metadata.get("nn_tie_eps") or NN_TIE_EPS),
        "max_rel_appearance": float(after.metadata.get("max_rel_appearance") or MAX_REL_APPEARANCE),
        "overlay": "learned_unet_orgdet_fnn_acf_ipc_v1",
        "config_hash": LOCKED_ORGDET_FNN_ACF_IPC_HASH,
    }


def refuse_existing_holdout_score(path: str | Path) -> None:
    dest = Path(path)
    if not dest.exists():
        return
    payload = json.loads(dest.read_text())
    if payload.get("status") == "holdout_scored" and payload.get("learned_orgdet_fnn_acf_ipc_combined") is not None:
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
        return "reject_learned_unet_orgdet_fnn_acf_ipc_v1"
    if int(n_dropped_edges) == 0:
        return "keep_as_noop_learned_unet_orgdet_fnn_acf_ipc_v1"
    return "keep_candidate_learned_unet_orgdet_fnn_acf_ipc_v1"


def holdout_decision(learned_combined: float) -> str:
    if float(learned_combined) <= float(ORGDET_FNN_ACF_LOCKED_HOLDOUT):
        return "reject_learned_unet_orgdet_fnn_acf_ipc_v1"
    return "keep_candidate_learned_unet_orgdet_fnn_acf_ipc_v1"


def runtime_estimate(
    n_volumes: int = 2,
    overlay_sec_per_volume: float = 8.0,
    ipc_overhead: float = 2.0,
    limit_hours: float = 12.0,
    peak_memory_mb: float = 1100.0,
) -> dict[str, Any]:
    n = int(n_volumes)
    sec = float(n) * (float(overlay_sec_per_volume) + float(ipc_overhead))
    return {
        "n_volumes": n,
        "overlay_sec_per_volume": float(overlay_sec_per_volume),
        "ipc_overhead": float(ipc_overhead),
        "estimate_sec": sec,
        "estimate_hours": sec / 3600.0,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_budget": float(peak_memory_mb),
        "source": (
            "CPU locked orgdet+gap+prune+ELF+FNN+ACF then keep only the "
            "nearest already-chosen incoming dt=1 parent on conflicts and "
            "new degree-0 leftovers; two sequence Zarrs for the locked ACF "
            "parent; not global t-1 NN; no transformer; no ILP; no new UNet"
        ),
        "prefers_tuning_zarrs": True,
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
    }
