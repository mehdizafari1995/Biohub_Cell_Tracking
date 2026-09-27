"""Drop appearance-inconsistent FNN dt=1 edges using the two tuning Zarrs.

Official 44b6 FNN residue is still association-dominated (174 conflict
FPs / 144 FNs; 0 predicted divisions). Rank / unused-child division /
claimed-child reassignment stay closed. On both frozen-tuning FNN
graphs, every official-matched TP dt=1 edge had 3x3x3-mean relative
appearance change <= 0.3756, while 13 conflict FPs and 1644 unmatched
leftover dt=1 edges exceeded 0.38. Dropping those already-chosen dt=1
edges, then new degree-0 leftovers, is an appearance filter. It does
not reassign, does not shrink the 12 µm linker/gap gates, and is not
an ELF / FNN / TCF / rank / division retune.

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

from .detect import sample_appearance
from .graphs import TrackGraph
from .io_zarr import open_volume
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

SCALE_ZYX = (1.625, 0.40625, 0.40625)
ORGDET_FNN_LOCKED_TUNE = LEARNED_ORGDET_FNN_LOCKED_TUNE
ORGDET_FNN_LOCKED_HOLDOUT = LEARNED_ORGDET_FNN_LOCKED_HOLDOUT
# Official one-shot scores of this overlay. Do not retune. Do not rescore.
LEARNED_ORGDET_FNN_ACF_LOCKED_TUNE = 0.7346197119237633
LEARNED_ORGDET_FNN_ACF_LOCKED_HOLDOUT = 0.7555029702166214
TUNE_MARGIN = 0.002000
TUNE_KEEP_THRESHOLD = ORGDET_FNN_LOCKED_TUNE + TUNE_MARGIN
# Locked from 44b6 FNN TP 3x3x3-mean relative change only (max 0.3756).
# Do not retune after the official score.
MAX_REL_APPEARANCE = 0.38
APPEARANCE_EPS = 1e-6
APPEARANCE_FEATURE = "patch_3x3x3_mean"

LOCKED_GATES = {
    "max_rel_appearance": MAX_REL_APPEARANCE,
    "appearance_eps": APPEARANCE_EPS,
    "appearance_feature": APPEARANCE_FEATURE,
    "drop_inconsistent_dt1": True,
    "drop_degree0_after": True,
    "assignment": "unchanged_locked_orgdet_fnn",
}


def orgdet_fnn_acf_config_payload() -> dict[str, Any]:
    """First FNN appearance-consistency filter. Lock before the 44b6 score."""
    return {
        "label": "learned_unet_orgdet_fnn_acf_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "orgdet_label": "learned_unet_orgdet_v1",
        "orgdet_gap_label": "learned_unet_orgdet_gap_v1",
        "orgdet_gap_prune_label": "learned_unet_orgdet_gap_prune_v1",
        "orgdet_elf_label": "learned_unet_orgdet_elf_v1",
        "orgdet_fnn_label": "learned_unet_orgdet_fnn_v1",
        "rejected_fnn_rank_label": "learned_unet_orgdet_fnn_rank_v1",
        "rejected_fnn_div_label": "learned_unet_orgdet_fnn_div_v1",
        "rejected_fnn_reas_label": "learned_unet_orgdet_fnn_reas_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "orgdet_config_hash": LOCKED_ORGDET_HASH,
        "orgdet_gap_config_hash": LOCKED_ORGDET_GAP_HASH,
        "orgdet_gap_prune_config_hash": LOCKED_ORGDET_GAP_PRUNE_HASH,
        "orgdet_elf_config_hash": LOCKED_ORGDET_ELF_HASH,
        "orgdet_fnn_config_hash": LOCKED_ORGDET_FNN_HASH,
        "rejected_fnn_rank_hash": LOCKED_ORGDET_FNN_RANK_HASH,
        "rejected_fnn_div_hash": LOCKED_ORGDET_FNN_DIV_HASH,
        "rejected_fnn_reas_hash": LOCKED_ORGDET_FNN_REAS_HASH,
        "source_graphs": "locked_learned_unet_orgdet_fnn_graphs",
        "source_zarrs": "locked_tuning_zarr_export_two_44b6",
        "retunes_detection": False,
        "retunes_consecutive_links": False,
        "retunes_gap_gates": False,
        "retunes_max_dt1_edge_um": False,
        "retunes_nn_tie_eps": False,
        "uses_intensity": True,
        "uses_transformer": False,
        "uses_velocity": False,
        "uses_ilp": False,
        "linker": "orgdet_fnn_then_drop_inconsistent_dt1_appearance",
        "max_rel_appearance": LOCKED_GATES["max_rel_appearance"],
        "appearance_eps": LOCKED_GATES["appearance_eps"],
        "appearance_feature": LOCKED_GATES["appearance_feature"],
        "drop_inconsistent_dt1": LOCKED_GATES["drop_inconsistent_dt1"],
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


def orgdet_fnn_acf_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(
        payload or orgdet_fnn_acf_config_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGDET_FNN_ACF_HASH = orgdet_fnn_acf_config_hash(orgdet_fnn_acf_config_payload())
assert LOCKED_ORGDET_FNN_ACF_HASH == "a66075085ac64400"
assert LOCKED_ORGDET_FNN_HASH == "95e9300eb84ba71c"
assert LOCKED_ORGDET_FNN_RANK_HASH == "2748c4a2b2823c0a"
assert LOCKED_ORGDET_FNN_DIV_HASH == "e184e6ce5bd90e46"
assert LOCKED_ORGDET_FNN_REAS_HASH == "39fb20d675e21d0e"
assert TUNE_KEEP_THRESHOLD == 0.7266526336456757
assert REJECTED_ORGOP_HASH == "250c5bfacd4a8198"
assert LOCKED_ORGDET_FNN_ACF_HASH != LOCKED_ORGDET_FNN_HASH
assert LOCKED_ORGDET_FNN_ACF_HASH != LOCKED_ORGDET_FNN_RANK_HASH
assert LOCKED_ORGDET_FNN_ACF_HASH != LOCKED_ORGDET_FNN_DIV_HASH
assert LOCKED_ORGDET_FNN_ACF_HASH != LOCKED_ORGDET_FNN_REAS_HASH
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
            raise ValueError(f"learned_unet_orgdet_fnn_acf_v1 tuning refuses stem {stem}")


def refuse_non_holdout_stems(stems: Iterable[str]) -> None:
    for raw in stems:
        stem = _clean_stem(str(raw))
        if specimen_of(stem) in HOLDOUT_SPECIMENS or stem in HOLDOUT_STEMS:
            if stem not in HOLDOUT_STEMS:
                raise ValueError(f"learned_unet_orgdet_fnn_acf_v1 holdout refuses extra stem {stem}")
            continue
        raise ValueError(f"learned_unet_orgdet_fnn_acf_v1 holdout refuses non-holdout stem {stem}")


def relative_appearance_change(src: float, dst: float, eps: float = APPEARANCE_EPS) -> float:
    if float(eps) != float(APPEARANCE_EPS):
        raise RuntimeError("learned_unet_orgdet_fnn_acf_v1 would retune appearance_eps; refusing")
    a = abs(float(src))
    b = abs(float(dst))
    return abs(float(src) - float(dst)) / max(a, b, float(eps))


def resolve_sequence_zarr(data_dir: str | Path, stem: str) -> Path:
    stem = _clean_stem(stem)
    data_dir = Path(data_dir)
    candidates = (
        data_dir / f"{stem}.zarr",
        data_dir / "train" / f"{stem}.zarr",
        data_dir.parent / "train" / f"{stem}.zarr",
    )
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(f"missing sequence Zarr for {stem}; appearance filter refuses to invent intensities")


def sample_graph_appearances(graph: TrackGraph, zarr_path: str | Path) -> dict[int, float]:
    """3x3x3-mean appearance at each node. One Zarr timepoint at a time."""
    zarr_path = Path(zarr_path)
    zarr_stem = _clean_stem(zarr_path.name)
    if zarr_stem != _clean_stem(graph.dataset):
        raise ValueError(f"zarr stem {zarr_stem} does not match graph {graph.dataset}")
    array, _meta = open_volume(zarr_path)
    by_t: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(graph.t):
        by_t[int(t)].append(int(i))
    appearance = np.zeros(graph.n_nodes, dtype=np.float64)
    coords = graph.coords_zyx()
    for t, idxs in by_t.items():
        frame = np.asarray(array[int(t)])
        _intensity, patch = sample_appearance(frame, coords[np.asarray(idxs, dtype=np.int64)])
        appearance[idxs] = patch
    return {int(nid): float(appearance[i]) for i, nid in enumerate(graph.node_id)}


def drop_inconsistent_dt1_edges(
    graph: TrackGraph,
    appearance_by_id: Mapping[int, float],
    max_rel: float = MAX_REL_APPEARANCE,
) -> TrackGraph:
    """Drop consecutive dt=1 edges whose relative 3x3x3 appearance exceeds ``max_rel``."""
    if float(max_rel) != float(MAX_REL_APPEARANCE):
        raise RuntimeError("learned_unet_orgdet_fnn_acf_v1 would retune max_rel_appearance; refusing")
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
        out.metadata["n_dropped_inconsistent_dt1_edges"] = 0
        out.metadata["n_kept_non_dt1_edges"] = 0
        out.metadata["max_rel_appearance"] = float(max_rel)
        return out
    index = graph.id_to_index()
    kept: list[tuple[int, int]] = []
    dropped = 0
    kept_non_dt1 = 0
    for src, dst in np.asarray(graph.edges).reshape(-1, 2) if graph.n_edges else []:
        s, d = int(src), int(dst)
        if s not in index or d not in index:
            raise RuntimeError("acf saw a dangling edge; refusing")
        if s not in appearance_by_id or d not in appearance_by_id:
            raise RuntimeError("acf missing appearance for an edge endpoint; refusing")
        i, j = index[s], index[d]
        dt = int(graph.t[j]) - int(graph.t[i])
        if dt != 1:
            kept.append((s, d))
            kept_non_dt1 += 1
            continue
        rel = relative_appearance_change(appearance_by_id[s], appearance_by_id[d])
        if rel > float(max_rel):
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
    out.metadata["n_dropped_inconsistent_dt1_edges"] = int(dropped)
    out.metadata["n_kept_non_dt1_edges"] = int(kept_non_dt1)
    out.metadata["n_nodes_before_acf"] = int(graph.n_nodes)
    out.metadata["n_edges_before_acf"] = int(graph.n_edges)
    out.metadata["max_rel_appearance"] = float(max_rel)
    leftover = set(int(x) for x in out.node_id)
    if any(src not in leftover or dst not in leftover for src, dst in kept):
        raise RuntimeError("acf kept a dangling edge; refusing")
    return out


def apply_orgdet_fnn_acf_overlay(
    graph: TrackGraph,
    appearance_by_id: Mapping[int, float] | None = None,
    zarr_path: str | Path | None = None,
) -> TrackGraph:
    """Locked FNN, then drop appearance-inconsistent dt=1 edges and new isolates."""
    if orgdet_fnn_acf_config_hash() != LOCKED_ORGDET_FNN_ACF_HASH:
        raise RuntimeError("learned_unet_orgdet_fnn_acf_v1 hash drifted; refusing to retune")
    if float(LOCKED_GATES["max_rel_appearance"]) != float(MAX_REL_APPEARANCE):
        raise RuntimeError("learned_unet_orgdet_fnn_acf_v1 would retune max_rel_appearance; refusing")
    if LOCKED_GATES["assignment"] != "unchanged_locked_orgdet_fnn":
        raise RuntimeError("learned_unet_orgdet_fnn_acf_v1 would retune assignment; refusing")
    if not LOCKED_GATES["drop_inconsistent_dt1"] or not LOCKED_GATES["drop_degree0_after"]:
        raise RuntimeError("learned_unet_orgdet_fnn_acf_v1 required filters disabled; refusing")
    if LOCKED_GATES["appearance_feature"] != APPEARANCE_FEATURE:
        raise RuntimeError("learned_unet_orgdet_fnn_acf_v1 would retune appearance_feature; refusing")
    fnn = apply_orgdet_fnn_overlay(graph)
    sampled = appearance_by_id
    if sampled is None:
        if zarr_path is None:
            raise RuntimeError("acf needs appearance_by_id or zarr_path; refusing")
        sampled = sample_graph_appearances(fnn, zarr_path)
    filtered = drop_inconsistent_dt1_edges(fnn, sampled, max_rel=LOCKED_GATES["max_rel_appearance"])
    n_dropped_edges = int(filtered.metadata.get("n_dropped_inconsistent_dt1_edges") or 0)
    out = drop_degree0(filtered)
    out.metadata["overlay"] = "learned_unet_orgdet_fnn_acf_v1"
    out.metadata["config_hash"] = LOCKED_ORGDET_FNN_ACF_HASH
    out.metadata["parent_orgdet_hash"] = LOCKED_ORGDET_HASH
    out.metadata["parent_orgdet_gap_hash"] = LOCKED_ORGDET_GAP_HASH
    out.metadata["parent_orgdet_gap_prune_hash"] = LOCKED_ORGDET_GAP_PRUNE_HASH
    out.metadata["parent_orgdet_elf_hash"] = LOCKED_ORGDET_ELF_HASH
    out.metadata["parent_orgdet_fnn_hash"] = LOCKED_ORGDET_FNN_HASH
    out.metadata["rejected_orgop_hash"] = REJECTED_ORGOP_HASH
    out.metadata["rejected_fnn_rank_hash"] = LOCKED_ORGDET_FNN_RANK_HASH
    out.metadata["rejected_fnn_div_hash"] = LOCKED_ORGDET_FNN_DIV_HASH
    out.metadata["rejected_fnn_reas_hash"] = LOCKED_ORGDET_FNN_REAS_HASH
    out.metadata["n_dropped_inconsistent_dt1_edges"] = n_dropped_edges
    out.metadata["n_dropped_non_forward_nn_edges"] = int(fnn.metadata.get("n_dropped_non_forward_nn_edges") or 0)
    out.metadata["n_dropped_long_dt1_edges"] = int(fnn.metadata.get("n_dropped_long_dt1_edges") or 0)
    out.metadata["uses_intensity"] = True
    out.metadata["uses_transformer"] = False
    out.metadata["uses_ilp"] = False
    out.metadata["retunes_detection"] = False
    out.metadata["retunes_consecutive_links"] = False
    out.metadata["retunes_gap_gates"] = False
    out.metadata["retunes_max_dt1_edge_um"] = False
    out.metadata["retunes_nn_tie_eps"] = False
    out.metadata["is_12um_assignment"] = False
    return out


def overlay_summary(before: TrackGraph, after: TrackGraph) -> dict[str, Any]:
    return {
        "dataset": after.dataset,
        "n_nodes_before": before.n_nodes,
        "n_nodes_after": after.n_nodes,
        "n_edges_before": before.n_edges,
        "n_edges_after": after.n_edges,
        "n_dropped_inconsistent_dt1_edges": int(after.metadata.get("n_dropped_inconsistent_dt1_edges") or 0),
        "n_dropped_isolates": int(after.metadata.get("n_dropped_isolates") or 0),
        "n_dropped_non_forward_nn_edges": int(after.metadata.get("n_dropped_non_forward_nn_edges") or 0),
        "n_dropped_long_dt1_edges": int(after.metadata.get("n_dropped_long_dt1_edges") or 0),
        "n_nodes_before_acf": int(after.metadata.get("n_nodes_before_acf") or after.n_nodes),
        "n_edges_before_acf": int(after.metadata.get("n_edges_before_acf") or after.n_edges),
        "max_rel_appearance": float(after.metadata.get("max_rel_appearance") or MAX_REL_APPEARANCE),
        "overlay": "learned_unet_orgdet_fnn_acf_v1",
        "config_hash": LOCKED_ORGDET_FNN_ACF_HASH,
    }


def refuse_existing_holdout_score(path: str | Path) -> None:
    dest = Path(path)
    if not dest.exists():
        return
    payload = json.loads(dest.read_text())
    if payload.get("status") == "holdout_scored" and payload.get("learned_orgdet_fnn_acf_combined") is not None:
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
        return "reject_learned_unet_orgdet_fnn_acf_v1"
    if int(n_dropped_edges) == 0:
        return "keep_as_noop_learned_unet_orgdet_fnn_acf_v1"
    return "keep_candidate_learned_unet_orgdet_fnn_acf_v1"


def holdout_decision(learned_combined: float) -> str:
    if float(learned_combined) <= float(ORGDET_FNN_LOCKED_HOLDOUT):
        return "reject_learned_unet_orgdet_fnn_acf_v1"
    return "keep_candidate_learned_unet_orgdet_fnn_acf_v1"


def runtime_estimate(
    n_volumes: int = 2,
    overlay_sec_per_volume: float = 8.0,
    acf_overhead: float = 8.0,
    limit_hours: float = 12.0,
    peak_memory_mb: float = 1100.0,
) -> dict[str, Any]:
    n = int(n_volumes)
    sec = float(n) * (float(overlay_sec_per_volume) + float(acf_overhead))
    return {
        "n_volumes": n,
        "overlay_sec_per_volume": float(overlay_sec_per_volume),
        "acf_overhead": float(acf_overhead),
        "estimate_sec": sec,
        "estimate_hours": sec / 3600.0,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_budget": float(peak_memory_mb),
        "source": (
            "CPU locked orgdet+gap+prune+ELF+FNN then drop dt=1 edges whose "
            f"3x3x3-mean relative appearance exceeds {MAX_REL_APPEARANCE} "
            "and new degree-0 leftovers; two sequence Zarrs; no transformer; "
            "no ILP; no new UNet"
        ),
        "prefers_tuning_zarrs": True,
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
    }
