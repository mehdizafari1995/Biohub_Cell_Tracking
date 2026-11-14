"""44b6-only official-scorer error decomposition of locked orgdet overlays.

Compares locked ``learned_unet_orgdet_v1``, ``learned_unet_orgdet_gap_v1``,
and ``learned_unet_orgdet_gap_prune_v1`` on frozen-tuning stems only.
Classifies remaining loss into detection misses, association misses, extra
``T_pred``, and short-track residue. Never opens ``6bba``. Does not retune
locked hashes. A local official score is not a leaderboard number.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import DEFAULT_MATCH_DISTANCE_UM, DEFAULT_SCALE_ZYX
from .errors import analyse_errors
from .evaluate import match_nodes
from .graphs import TrackGraph
from .learned_gap import graph_from_payload, load_graph_json
from .learned_orgdet import LOCKED_ORGDET_HASH, apply_orgdet_relink
from .learned_orgdet_gap import (
    INFER_STEMS,
    LOCKED_ORGDET_GAP_HASH,
    ORGDET_LOCKED_TUNE,
    apply_orgdet_gap_overlay,
    ensure_orgdet_tuning_graph,
)
from .learned_orgdet_gap_prune import (
    HOLDOUT_STEMS,
    LEARNED_ORGDET_GAP_PRUNE_LOCKED_TUNE,
    LOCKED_ORGDET_GAP_PRUNE_HASH,
    ORGDET_GAP_LOCKED_TUNE,
    apply_orgdet_gap_prune_overlay,
)
from .learned_split import HOLDOUT_SPECIMENS, TUNING_SPECIMEN, specimen_of
from .link import physical_coords

OVERLAY_LABELS = ("orgdet", "orgdet_gap", "orgdet_gap_prune")
LOCKED_COMBINED = {
    "orgdet": float(ORGDET_LOCKED_TUNE),
    "orgdet_gap": float(ORGDET_GAP_LOCKED_TUNE),
    "orgdet_gap_prune": float(LEARNED_ORGDET_GAP_PRUNE_LOCKED_TUNE),
}
LOCKED_HASHES = {
    "orgdet": LOCKED_ORGDET_HASH,
    "orgdet_gap": LOCKED_ORGDET_GAP_HASH,
    "orgdet_gap_prune": LOCKED_ORGDET_GAP_PRUNE_HASH,
}
# Sprint gate: new 44b6 combined must beat prune by this margin.
TUNE_MARGIN = 0.002000
PRUNE_LOCKED_TUNE = LEARNED_ORGDET_GAP_PRUNE_LOCKED_TUNE
TUNE_KEEP_THRESHOLD = PRUNE_LOCKED_TUNE + TUNE_MARGIN
SCALE_ZYX = DEFAULT_SCALE_ZYX
SUPPORT_GATE_UM = 12.0
SHORT_TRACK_MAX_NODES = 3
# Largest component size with 0 sparse-GT matches on both 44b6 prune graphs.
SAFE_SHORT_TRACK_MAX_NODES = 7
MIN_KEEP_COMPONENT_NODES = SAFE_SHORT_TRACK_MAX_NODES + 1


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
            raise ValueError(f"orgdet error decomp refuses non-tuning stem {stem}")
        if specimen_of(stem) in HOLDOUT_SPECIMENS or stem in HOLDOUT_STEMS:
            raise ValueError(f"orgdet error decomp refuses holdout stem {stem}")


def refuse_holdout_paths(paths: Iterable[str | Path]) -> None:
    for raw in paths:
        name = Path(raw).name
        stem = _clean_stem(name)
        if specimen_of(stem) in HOLDOUT_SPECIMENS or stem in HOLDOUT_STEMS or name.startswith("6bba"):
            raise ValueError(f"orgdet error decomp refuses holdout path {name}")


def build_locked_overlays(orgop: TrackGraph) -> dict[str, TrackGraph]:
    """Rebuild locked orgdet / gap / prune graphs from orgop detections."""
    refuse_non_tuning_stems([orgop.dataset])
    orgdet = ensure_orgdet_tuning_graph(orgop)
    if orgdet.n_edges == 0:
        orgdet = apply_orgdet_relink(orgop)
    return {
        "orgdet": orgdet,
        "orgdet_gap": apply_orgdet_gap_overlay(orgdet),
        "orgdet_gap_prune": apply_orgdet_gap_prune_overlay(orgdet),
    }


def _degrees(graph: TrackGraph) -> tuple[dict[int, int], dict[int, int]]:
    indeg: dict[int, int] = defaultdict(int)
    outdeg: dict[int, int] = defaultdict(int)
    for src, dst in np.asarray(graph.edges).reshape(-1, 2) if graph.n_edges else []:
        outdeg[int(src)] += 1
        indeg[int(dst)] += 1
    return indeg, outdeg


def component_sizes(graph: TrackGraph) -> dict[int, int]:
    """Undirected component size for every node (all edges, including gaps)."""
    parent = {int(nid): int(nid) for nid in graph.node_id}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for src, dst in np.asarray(graph.edges).reshape(-1, 2) if graph.n_edges else []:
        union(int(src), int(dst))
    sizes: dict[int, int] = defaultdict(int)
    for nid in graph.node_id:
        sizes[find(int(nid))] += 1
    return {int(nid): int(sizes[find(int(nid))]) for nid in graph.node_id}


def _has_temporal_support(graph: TrackGraph, gate_um: float = SUPPORT_GATE_UM) -> np.ndarray:
    """True when another detection exists within gate_um at t±1."""
    n = graph.n_nodes
    if n == 0:
        return np.zeros((0,), dtype=bool)
    xyz = physical_coords(graph.coords_zyx(), SCALE_ZYX)
    ts = np.asarray(graph.t, dtype=np.int64)
    by_t: dict[int, np.ndarray] = {}
    for t in np.unique(ts):
        by_t[int(t)] = np.where(ts == int(t))[0]
    supported = np.zeros(n, dtype=bool)
    gate2 = float(gate_um) ** 2
    for i in range(n):
        t = int(ts[i])
        for other_t in (t - 1, t + 1):
            idxs = by_t.get(other_t)
            if idxs is None or len(idxs) == 0:
                continue
            delta = xyz[idxs] - xyz[i]
            if np.any(np.sum(delta * delta, axis=1) <= gate2):
                supported[i] = True
                break
    return supported


def classify_sequence(
    pred: TrackGraph,
    gt: TrackGraph,
    official: Mapping[str, Any],
    scale_zyx: tuple[float, float, float] = SCALE_ZYX,
) -> dict[str, Any]:
    """Detection vs association breakdown. Unmatched preds are not node FPs."""
    analysis = analyse_errors(pred, gt, scale_zyx=scale_zyx)
    mapping = match_nodes(pred, gt, scale_zyx=scale_zyx, max_distance_um=DEFAULT_MATCH_DISTANCE_UM)
    matched_pred = set(mapping)
    matched_gt = set(mapping.values())
    gt_edges = {(int(s), int(d)) for s, d in gt.consecutive_edges()}
    pred_mapped = set()
    for src, dst in pred.consecutive_edges():
        pred_mapped.add((mapping.get(int(src)), mapping.get(int(dst))))

    detection_fn_edges = 0
    association_fn_edges = 0
    for src, dst in gt_edges:
        both = src in matched_gt and dst in matched_gt
        if (src, dst) in {(a, b) for a, b in pred_mapped if a is not None and b is not None}:
            continue
        if both:
            association_fn_edges += 1
        else:
            detection_fn_edges += 1

    indeg, outdeg = _degrees(pred)
    sizes = component_sizes(pred)
    support = _has_temporal_support(pred)
    id_to_idx = pred.id_to_index()
    degree0 = [int(nid) for nid in pred.node_id if indeg[int(nid)] == 0 and outdeg[int(nid)] == 0]
    degree0_matched = [nid for nid in degree0 if nid in matched_pred]
    unsupported = [int(nid) for nid, ok in zip(pred.node_id, support) if not ok]
    unsupported_matched = [nid for nid in unsupported if nid in matched_pred]
    short_nodes = [int(nid) for nid, sz in sizes.items() if sz <= SHORT_TRACK_MAX_NODES]
    short_matched = [nid for nid in short_nodes if nid in matched_pred]

    size_hist = Counter(sizes.values())
    frame_missed: dict[int, int] = defaultdict(int)
    gt_index = gt.id_to_index()
    for nid in analysis.missed_gt_ids:
        frame_missed[int(gt.t[gt_index[int(nid)]])] += 1

    pred_per_t = Counter(int(t) for t in pred.t)
    gt_per_t = Counter(int(t) for t in gt.t)
    dense_frames = []
    for t, n_pred in pred_per_t.items():
        n_gt = gt_per_t.get(int(t), 0)
        if n_gt and n_pred / max(n_gt, 1) >= 8.0:
            dense_frames.append(int(t))

    fn_total = detection_fn_edges + association_fn_edges
    detection_fn_share = detection_fn_edges / fn_total if fn_total else 0.0
    extra_pred = len(analysis.unmatched_pred_ids)
    t_pred_ratio = float(official.get("total_node_ratio") or 0.0)
    return {
        "dataset": pred.dataset,
        "n_pred_nodes": pred.n_nodes,
        "n_pred_edges": pred.n_edges,
        "n_gt_nodes": gt.n_nodes,
        "n_gt_edges": int(len(gt_edges)),
        "n_matched_gt": analysis.n_matched_gt,
        "missed_gt_nodes": len(analysis.missed_gt_ids),
        "unmatched_pred_nodes": extra_pred,
        "unmatched_pred_note": analysis.unmatched_pred_note,
        "official_tp_fp_fn": [
            int(official.get("edge_tp") or 0),
            int(official.get("edge_fp") or 0),
            int(official.get("edge_fn") or 0),
        ],
        "diagnostic_tp_fp_fn": [len(analysis.tp_edges), len(analysis.fp_edges), len(analysis.fn_edges)],
        "identity_switches": len(analysis.identity_switches),
        "gt_divisions": len(analysis.gt_divisions),
        "pred_divisions": len(analysis.pred_divisions),
        "recovered_divisions": len(analysis.recovered_divisions),
        "missed_divisions": len(analysis.missed_divisions),
        "detection_fn_edges": detection_fn_edges,
        "association_fn_edges": association_fn_edges,
        "detection_fn_share": detection_fn_share,
        "degree0_nodes": len(degree0),
        "degree0_matched_gt": len(degree0_matched),
        "unsupported_tpm1_nodes": len(unsupported),
        "unsupported_tpm1_matched_gt": len(unsupported_matched),
        "short_track_le3_nodes": len(short_nodes),
        "short_track_le3_matched_gt": len(short_matched),
        "component_size_hist": {str(k): int(v) for k, v in sorted(size_hist.items())},
        "missed_gt_by_frame": {str(k): int(v) for k, v in sorted(frame_missed.items())},
        "n_dense_pred_frames": len(dense_frames),
        "node_recall_official": float(official.get("node_recall") or 0.0),
        "total_node_ratio": t_pred_ratio,
        "adj_edge_jaccard": float(official.get("adj_edge_jaccard") or 0.0),
        "score": float(official.get("score") or 0.0),
    }


def aggregate_error_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "n_pred_nodes",
        "n_pred_edges",
        "n_gt_nodes",
        "n_gt_edges",
        "n_matched_gt",
        "missed_gt_nodes",
        "unmatched_pred_nodes",
        "detection_fn_edges",
        "association_fn_edges",
        "identity_switches",
        "degree0_nodes",
        "degree0_matched_gt",
        "unsupported_tpm1_nodes",
        "unsupported_tpm1_matched_gt",
        "short_track_le3_nodes",
        "short_track_le3_matched_gt",
        "gt_divisions",
        "pred_divisions",
        "recovered_divisions",
        "missed_divisions",
    )
    out = {k: int(sum(int(r.get(k) or 0) for r in rows)) for k in keys}
    fn = out["detection_fn_edges"] + out["association_fn_edges"]
    out["detection_fn_share"] = out["detection_fn_edges"] / fn if fn else 0.0
    out["official_tp"] = int(sum(int((r.get("official_tp_fp_fn") or [0])[0]) for r in rows))
    out["official_fp"] = int(sum(int((r.get("official_tp_fp_fn") or [0, 0])[1]) for r in rows))
    out["official_fn"] = int(sum(int((r.get("official_tp_fp_fn") or [0, 0, 0])[2]) for r in rows))
    return out


def dominant_error_mode(prune_aggregate: Mapping[str, Any]) -> dict[str, Any]:
    """Choose the remaining-loss bucket on locked prune graphs only."""
    det_fn = int(prune_aggregate.get("detection_fn_edges") or 0)
    assoc_fn = int(prune_aggregate.get("association_fn_edges") or 0)
    missed_gt = int(prune_aggregate.get("missed_gt_nodes") or 0)
    extra = int(prune_aggregate.get("unmatched_pred_nodes") or 0)
    unsupported = int(prune_aggregate.get("unsupported_tpm1_nodes") or 0)
    short = int(prune_aggregate.get("short_track_le3_nodes") or 0)
    official_fn = int(prune_aggregate.get("official_fn") or 0)
    official_fp = int(prune_aggregate.get("official_fp") or 0)
    fn = det_fn + assoc_fn
    detection_share = det_fn / fn if fn else 0.0
    if missed_gt > 0 and detection_share >= 0.45:
        mode = "detection"
        preferred = "temporal_consistency_detection_fusion"
        rationale = (
            "Official FN edges are mostly detection misses (unmatched GT endpoints). "
            "Prefer temporal-consistency detection calibration/fusion, not another 12 µm assignment."
        )
    elif extra >= 1000 and unsupported + short >= extra * 0.25:
        mode = "t_pred"
        preferred = "temporal_consistency_short_fragment_filter"
        rationale = (
            "Sparse-GT unmatched predictions inflate T_pred. "
            "Prefer dropping short temporally inconsistent fragments, not another 12 µm assignment."
        )
    elif assoc_fn > det_fn:
        mode = "association"
        preferred = "temporal_consistency_short_fragment_filter"
        rationale = (
            "Remaining FN edges have both GT endpoints matched, so assignment overlays "
            "stay closed. Unsupported t±1 leftovers are already 0 after prune. Short "
            f"fragments (≤{SAFE_SHORT_TRACK_MAX_NODES} nodes) still inflate T_pred and "
            "had 0 sparse-GT matches on both 44b6 prune graphs."
        )
    else:
        mode = "mixed"
        preferred = "temporal_consistency_short_fragment_filter"
        rationale = (
            "Mixed residue. Conservative short-fragment temporal-consistency filter is "
            "the non-rejected, non-12µm-assignment lever that can still cut T_pred."
        )
    return {
        "mode": mode,
        "preferred_hypothesis_family": preferred,
        "detection_fn_share": detection_share,
        "official_tp_fp_fn": [
            int(prune_aggregate.get("official_tp") or 0),
            official_fp,
            official_fn,
        ],
        "rationale": rationale,
        "tune_keep_threshold": TUNE_KEEP_THRESHOLD,
        "tune_margin": TUNE_MARGIN,
        "prune_locked_tune": PRUNE_LOCKED_TUNE,
    }


def recommend_hypothesis(mode_info: Mapping[str, Any]) -> dict[str, Any]:
    """Preregister exactly one non-rejected, non-12µm-assignment hypothesis."""
    family = str(mode_info.get("preferred_hypothesis_family") or "")
    if family == "temporal_consistency_detection_fusion":
        label = "learned_unet_orgdet_tcfuse_v1"
        linker = "orgdet_gap_prune_then_fuse_temporally_supported_leftovers"
        retunes_detection = True
        notes = (
            "Fuse leftover temporally supported detections onto locked orgdet+gap+prune. "
            "Not another 12 µm assignment. Frozen 6bba stays closed until 44b6 > "
            f"{TUNE_KEEP_THRESHOLD:.6f}."
        )
        extra = {"support_gate_um": SUPPORT_GATE_UM, "support_dt": 1}
    else:
        label = "learned_unet_orgdet_tcf_v1"
        linker = "orgdet_gap_prune_then_drop_short_fragments"
        retunes_detection = True
        notes = (
            "After locked orgdet+gap+prune, drop connected components with fewer than "
            f"{MIN_KEEP_COMPONENT_NODES} nodes. On both 44b6 prune graphs, fragments "
            f"of size ≤{SAFE_SHORT_TRACK_MAX_NODES} had 0 sparse-GT matches and still "
            "inflate T_pred. This is temporal-consistency detection calibration, not a "
            "new assignment and not a rejected overlay. Frozen 6bba stays closed until "
            f"official 44b6 exceeds {TUNE_KEEP_THRESHOLD:.6f}."
        )
        extra = {
            "min_component_nodes": MIN_KEEP_COMPONENT_NODES,
            "safe_short_track_max_nodes": SAFE_SHORT_TRACK_MAX_NODES,
        }
    payload = {
        "label": label,
        "parent_hash": LOCKED_ORGDET_GAP_PRUNE_HASH,
        "parent_label": "learned_unet_orgdet_gap_prune_v1",
        "linker": linker,
        "retunes_detection": retunes_detection,
        "retunes_consecutive_links": False,
        "retunes_gap_gates": False,
        "uses_intensity": False,
        "uses_transformer": False,
        "uses_velocity": False,
        "uses_ilp": False,
        "assignment": "unchanged_locked_orgdet_gap_prune",
        "tune_keep_threshold": TUNE_KEEP_THRESHOLD,
        "tune_margin": TUNE_MARGIN,
        "opened_holdout": False,
        "is_12um_assignment": False,
        "is_rejected_overlay": False,
        "notes": notes,
    }
    payload.update(extra)
    return payload


def load_orgop_graphs(graph_dir: str | Path, stems: Sequence[str] = INFER_STEMS) -> list[TrackGraph]:
    refuse_non_tuning_stems(stems)
    graphs: list[TrackGraph] = []
    for stem in stems:
        path = Path(graph_dir) / f"{stem}.json"
        if not path.exists():
            nested = Path(graph_dir) / "graphs" / f"{stem}.json"
            path = nested if nested.exists() else path
        if not path.exists():
            raise FileNotFoundError(f"missing orgop graph {path}; do not invent a score")
        raw = load_graph_json(path)
        if str(raw.get("dataset")) != stem:
            raise ValueError(f"{path} dataset {raw.get('dataset')!r} != {stem}")
        graphs.append(graph_from_payload(raw))
    return graphs
