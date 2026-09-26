"""Drop non-forward nearest-neighbor dt=1 edges after locked ELF.

Official 44b6 ELF residue was association-dominated (203 conflict FPs /
140 FNs; 0 predicted divisions). Turning-angle, degree, short-fragment,
and another 8.0 µm dt=1 length filter stay closed. On the same ELF
graphs, 30 of 203 conflict FPs were not the source's nearest neighbor at
t+1, versus 10 of 910 TPs. Dropping those existing edges does not
reassign, does not shrink the 12 µm linker/gap gates, and is not a
short-fragment or dt=1 length retune.

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
from .learned_orgdet_elf import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LEARNED_ORGDET_ELF_LOCKED_HOLDOUT,
    LEARNED_ORGDET_ELF_LOCKED_TUNE,
    LOCKED_ORGDET_ELF_HASH,
    apply_orgdet_elf_overlay,
    ensure_orgdet_tuning_graph,
    load_role_graphs,
    refuse_non_orgdet_node_count,
)
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
ORGDET_ELF_LOCKED_TUNE = LEARNED_ORGDET_ELF_LOCKED_TUNE
ORGDET_ELF_LOCKED_HOLDOUT = LEARNED_ORGDET_ELF_LOCKED_HOLDOUT
# Official one-shot scores of this overlay. Do not retune. Do not rescore.
LEARNED_ORGDET_FNN_LOCKED_TUNE = 0.7246526336456757
LEARNED_ORGDET_FNN_LOCKED_HOLDOUT = 0.7520914330648746
TUNE_MARGIN = 0.002000
TUNE_KEEP_THRESHOLD = ORGDET_ELF_LOCKED_TUNE + TUNE_MARGIN
NN_TIE_EPS = 1e-9

LOCKED_GATES = {
    "keep_forward_nn_only": True,
    "drop_degree0_after": True,
    "nn_tie_eps": NN_TIE_EPS,
    "assignment": "unchanged_locked_orgdet_elf",
}


def orgdet_fnn_config_payload() -> dict[str, Any]:
    """First forward-NN keep after locked ELF. Lock before the 44b6 score."""
    return {
        "label": "learned_unet_orgdet_fnn_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "orgdet_label": "learned_unet_orgdet_v1",
        "orgdet_gap_label": "learned_unet_orgdet_gap_v1",
        "orgdet_gap_prune_label": "learned_unet_orgdet_gap_prune_v1",
        "orgdet_elf_label": "learned_unet_orgdet_elf_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "orgdet_config_hash": LOCKED_ORGDET_HASH,
        "orgdet_gap_config_hash": LOCKED_ORGDET_GAP_HASH,
        "orgdet_gap_prune_config_hash": LOCKED_ORGDET_GAP_PRUNE_HASH,
        "orgdet_elf_config_hash": LOCKED_ORGDET_ELF_HASH,
        "source_graphs": "locked_learned_unet_orgdet_graphs",
        "retunes_detection": False,
        "retunes_consecutive_links": False,
        "retunes_gap_gates": False,
        "retunes_max_dt1_edge_um": False,
        "uses_intensity": False,
        "uses_transformer": False,
        "uses_velocity": False,
        "uses_ilp": False,
        "linker": "orgdet_elf_then_keep_forward_nn_dt1",
        "keep_forward_nn_only": LOCKED_GATES["keep_forward_nn_only"],
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


def orgdet_fnn_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(
        payload or orgdet_fnn_config_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGDET_FNN_HASH = orgdet_fnn_config_hash(orgdet_fnn_config_payload())
assert LOCKED_ORGDET_ELF_HASH == "292c4f1f968f948d"
assert LOCKED_ORGDET_GAP_PRUNE_HASH == "862e3e7f7ac0330e"
assert REJECTED_ORGOP_HASH == "250c5bfacd4a8198"
assert TUNE_KEEP_THRESHOLD == 0.7118380747225755
assert LOCKED_ORGDET_FNN_HASH == "95e9300eb84ba71c"
assert LEARNED_ORGDET_FNN_LOCKED_TUNE == 0.7246526336456757
assert LEARNED_ORGDET_FNN_LOCKED_HOLDOUT == 0.7520914330648746


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
            raise ValueError(f"learned_unet_orgdet_fnn_v1 tuning refuses stem {stem}")


def refuse_non_holdout_stems(stems: Iterable[str]) -> None:
    for raw in stems:
        stem = _clean_stem(str(raw))
        if specimen_of(stem) in HOLDOUT_SPECIMENS or stem in HOLDOUT_STEMS:
            if stem not in HOLDOUT_STEMS:
                raise ValueError(f"learned_unet_orgdet_fnn_v1 holdout refuses extra stem {stem}")
            continue
        raise ValueError(f"learned_unet_orgdet_fnn_v1 holdout refuses non-holdout stem {stem}")


def drop_non_forward_nn_edges(graph: TrackGraph, tie_eps: float = NN_TIE_EPS) -> TrackGraph:
    """Keep dt=1 edges whose destination is a nearest neighbor of the source at t+1."""
    if float(tie_eps) != float(NN_TIE_EPS):
        raise RuntimeError("learned_unet_orgdet_fnn_v1 would retune nn_tie_eps; refusing")
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
        out.metadata["n_dropped_non_forward_nn_edges"] = 0
        out.metadata["n_kept_forward_nn_edges"] = 0
        out.metadata["n_kept_non_dt1_edges"] = 0
        out.metadata["nn_tie_eps"] = float(tie_eps)
        return out
    xyz = physical_coords(graph.coords_zyx(), SCALE_ZYX)
    index = graph.id_to_index()
    by_t: dict[int, list[int]] = defaultdict(list)
    for nid, t in zip(graph.node_id, graph.t):
        by_t[int(t)].append(int(nid))
    kept: list[tuple[int, int]] = []
    dropped = 0
    kept_fwd = 0
    kept_non_dt1 = 0
    for src, dst in np.asarray(graph.edges).reshape(-1, 2) if graph.n_edges else []:
        s, d = int(src), int(dst)
        if s not in index or d not in index:
            raise RuntimeError("fnn saw a dangling edge; refusing")
        i, j = index[s], index[d]
        dt = int(graph.t[j]) - int(graph.t[i])
        if dt != 1:
            kept.append((s, d))
            kept_non_dt1 += 1
            continue
        nxt = by_t.get(int(graph.t[i]) + 1)
        if not nxt:
            dropped += 1
            continue
        dest_xyz = xyz[j]
        src_xyz = xyz[i]
        dest_dist = float(np.linalg.norm(dest_xyz - src_xyz))
        min_dist = min(float(np.linalg.norm(xyz[index[n]] - src_xyz)) for n in nxt)
        if dest_dist <= min_dist + float(tie_eps):
            kept.append((s, d))
            kept_fwd += 1
        else:
            dropped += 1
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
    out.metadata["n_dropped_non_forward_nn_edges"] = int(dropped)
    out.metadata["n_kept_forward_nn_edges"] = int(kept_fwd)
    out.metadata["n_kept_non_dt1_edges"] = int(kept_non_dt1)
    out.metadata["n_nodes_before_fnn"] = int(graph.n_nodes)
    out.metadata["n_edges_before_fnn"] = int(graph.n_edges)
    out.metadata["nn_tie_eps"] = float(tie_eps)
    leftover = set(int(x) for x in out.node_id)
    if any(src not in leftover or dst not in leftover for src, dst in kept):
        raise RuntimeError("fnn kept a dangling edge; refusing")
    return out


def apply_orgdet_fnn_overlay(graph: TrackGraph) -> TrackGraph:
    """Locked ELF, then drop non-forward-NN dt=1 edges and new isolates."""
    if orgdet_fnn_config_hash() != LOCKED_ORGDET_FNN_HASH:
        raise RuntimeError("learned_unet_orgdet_fnn_v1 hash drifted; refusing to retune")
    if LOCKED_GATES["assignment"] != "unchanged_locked_orgdet_elf":
        raise RuntimeError("learned_unet_orgdet_fnn_v1 would retune assignment; refusing")
    if not LOCKED_GATES["keep_forward_nn_only"] or not LOCKED_GATES["drop_degree0_after"]:
        raise RuntimeError("learned_unet_orgdet_fnn_v1 required filters disabled; refusing")
    if float(LOCKED_GATES["nn_tie_eps"]) != float(NN_TIE_EPS):
        raise RuntimeError("learned_unet_orgdet_fnn_v1 would retune nn_tie_eps; refusing")
    elf = apply_orgdet_elf_overlay(graph)
    filtered = drop_non_forward_nn_edges(elf, tie_eps=LOCKED_GATES["nn_tie_eps"])
    n_dropped_edges = int(filtered.metadata.get("n_dropped_non_forward_nn_edges") or 0)
    out = drop_degree0(filtered)
    out.metadata["overlay"] = "learned_unet_orgdet_fnn_v1"
    out.metadata["config_hash"] = LOCKED_ORGDET_FNN_HASH
    out.metadata["parent_orgdet_hash"] = LOCKED_ORGDET_HASH
    out.metadata["parent_orgdet_gap_hash"] = LOCKED_ORGDET_GAP_HASH
    out.metadata["parent_orgdet_gap_prune_hash"] = LOCKED_ORGDET_GAP_PRUNE_HASH
    out.metadata["parent_orgdet_elf_hash"] = LOCKED_ORGDET_ELF_HASH
    out.metadata["rejected_orgop_hash"] = REJECTED_ORGOP_HASH
    out.metadata["n_dropped_non_forward_nn_edges"] = n_dropped_edges
    out.metadata["n_dropped_long_dt1_edges"] = int(elf.metadata.get("n_dropped_long_dt1_edges") or 0)
    out.metadata["uses_intensity"] = False
    out.metadata["uses_transformer"] = False
    out.metadata["uses_ilp"] = False
    out.metadata["retunes_detection"] = False
    out.metadata["retunes_consecutive_links"] = False
    out.metadata["retunes_gap_gates"] = False
    out.metadata["retunes_max_dt1_edge_um"] = False
    out.metadata["is_12um_assignment"] = False
    return out


def overlay_summary(before: TrackGraph, after: TrackGraph) -> dict[str, Any]:
    return {
        "dataset": after.dataset,
        "n_nodes_before": before.n_nodes,
        "n_nodes_after": after.n_nodes,
        "n_edges_before": before.n_edges,
        "n_edges_after": after.n_edges,
        "n_dropped_non_forward_nn_edges": int(after.metadata.get("n_dropped_non_forward_nn_edges") or 0),
        "n_dropped_isolates": int(after.metadata.get("n_dropped_isolates") or 0),
        "n_dropped_long_dt1_edges": int(after.metadata.get("n_dropped_long_dt1_edges") or 0),
        "n_nodes_before_fnn": int(after.metadata.get("n_nodes_before_fnn") or after.n_nodes),
        "n_edges_before_fnn": int(after.metadata.get("n_edges_before_fnn") or after.n_edges),
        "nn_tie_eps": float(after.metadata.get("nn_tie_eps") or NN_TIE_EPS),
        "overlay": "learned_unet_orgdet_fnn_v1",
        "config_hash": LOCKED_ORGDET_FNN_HASH,
    }


def refuse_existing_holdout_score(path: str | Path) -> None:
    dest = Path(path)
    if not dest.exists():
        return
    payload = json.loads(dest.read_text())
    if payload.get("status") == "holdout_scored" and payload.get("learned_orgdet_fnn_combined") is not None:
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
        return "reject_learned_unet_orgdet_fnn_v1"
    if int(n_dropped_edges) == 0:
        return "keep_as_noop_learned_unet_orgdet_fnn_v1"
    return "keep_candidate_learned_unet_orgdet_fnn_v1"


def holdout_decision(learned_combined: float) -> str:
    if float(learned_combined) <= float(ORGDET_ELF_LOCKED_HOLDOUT):
        return "reject_learned_unet_orgdet_fnn_v1"
    return "keep_candidate_learned_unet_orgdet_fnn_v1"


def runtime_estimate(
    n_volumes: int = 2,
    overlay_sec_per_volume: float = 6.0,
    fnn_overhead: float = 0.8,
    limit_hours: float = 12.0,
    peak_memory_mb: float = 900.0,
) -> dict[str, Any]:
    n = int(n_volumes)
    sec = float(n) * (float(overlay_sec_per_volume) + float(fnn_overhead))
    return {
        "n_volumes": n,
        "overlay_sec_per_volume": float(overlay_sec_per_volume),
        "fnn_overhead": float(fnn_overhead),
        "estimate_sec": sec,
        "estimate_hours": sec / 3600.0,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_budget": float(peak_memory_mb),
        "source": (
            "CPU locked orgdet+gap+prune+ELF then keep only forward nearest-neighbor "
            "dt=1 edges and new degree-0 leftovers; no transformer; no ILP; no new UNet"
        ),
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
    }
