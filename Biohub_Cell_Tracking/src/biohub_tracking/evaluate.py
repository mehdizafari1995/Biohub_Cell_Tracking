from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .config import DEFAULT_MATCH_DISTANCE_UM, DEFAULT_SCALE_ZYX
from .graphs import TrackGraph
from .link import physical_coords

ADJUSTMENT_ALPHA = 0.1
SCORE_DIVISION_WEIGHT = 0.1


@dataclass
class SampleMetrics:
    dataset: str
    edge_tp: int
    edge_fp: int
    edge_fn: int
    division_tp: int
    division_fp: int
    division_fn: int
    num_pred_nodes: int
    num_gt_nodes: int
    node_recall: float
    missed_detections: int
    identity_switches: int
    broken_tracks: int
    edge_jaccard: float
    adj_edge_jaccard: float
    division_jaccard: float
    score: float
    total_node_ratio: float
    estimated_n_nodes: float | None
    metric_label: str
    notes: str

    def as_dict(self) -> dict:
        return asdict(self)


def _jaccard(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return tp / denom if denom > 0 else float("nan")


def match_nodes(
    pred: TrackGraph,
    gt: TrackGraph,
    scale_zyx: tuple[float, float, float] = DEFAULT_SCALE_ZYX,
    max_distance_um: float = DEFAULT_MATCH_DISTANCE_UM,
) -> dict[int, int]:
    """Map predicted node_id -> GT node_id using per-timepoint Hungarian matching."""
    mapping: dict[int, int] = {}
    for t in np.unique(np.concatenate([pred.t, gt.t])):
        pred_idx = np.where(pred.t == t)[0]
        gt_idx = np.where(gt.t == t)[0]
        if len(pred_idx) == 0 or len(gt_idx) == 0:
            continue
        pred_xyz = physical_coords(pred.coords_zyx()[pred_idx], scale_zyx)
        gt_xyz = physical_coords(gt.coords_zyx()[gt_idx], scale_zyx)
        dists = cdist(pred_xyz, gt_xyz)
        cost = dists.copy()
        inf = max_distance_um * 1000.0 + 1.0
        cost[cost > max_distance_um] = inf
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if dists[r, c] <= max_distance_um:
                mapping[int(pred.node_id[pred_idx[r]])] = int(gt.node_id[gt_idx[c]])
    return mapping


def _edge_set(graph: TrackGraph) -> set[tuple[int, int]]:
    return {(int(s), int(d)) for s, d in graph.consecutive_edges()}


def _gt_partners(gt: TrackGraph) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    sources: dict[int, set[int]] = {}
    targets: dict[int, set[int]] = {}
    for src, dst in gt.consecutive_edges():
        sources.setdefault(int(dst), set()).add(int(src))
        targets.setdefault(int(src), set()).add(int(dst))
    return sources, targets


def evaluate_graph(
    pred: TrackGraph,
    gt: TrackGraph,
    scale_zyx: tuple[float, float, float] = DEFAULT_SCALE_ZYX,
    max_distance_um: float = DEFAULT_MATCH_DISTANCE_UM,
) -> SampleMetrics:
    """Local reimplementation of the published competition metric.

    Edge Jaccard and the node-count adjustment follow
    https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md
    Edge FP/FN use the sparse-GT rules from that document.

    Division Jaccard here is a *diagnostic proxy*: a predicted fork is a TP
    if the parent and both daughters match a GT fork within 7 µm. The official
    scorer uses a 5-generation local window. Labelled as such in `metric_label`.
    """
    pred_edges = pred.consecutive_edges()
    if pred.n_edges and len(pred_edges) != pred.n_edges:
        # cap out-degree at 2, keep first two by edge order
        keep = []
        seen = {}
        for src, dst in pred_edges:
            seen[int(src)] = seen.get(int(src), 0) + 1
            if seen[int(src)] <= 2:
                keep.append((int(src), int(dst)))
        pred_edges = np.asarray(keep, dtype=np.int64).reshape(-1, 2)

    mapping = match_nodes(pred, gt, scale_zyx, max_distance_um)
    gt_edge_set = _edge_set(gt)
    gt_sources, gt_targets = _gt_partners(gt)

    tp_edges: set[tuple[int, int]] = set()
    fp_edges: set[tuple[int, int]] = set()
    for src, dst in pred_edges:
        ms = mapping.get(int(src))
        md = mapping.get(int(dst))
        if ms is not None and md is not None and (ms, md) in gt_edge_set:
            tp_edges.add((ms, md))
            continue
        # Official sparse FP: predicted edge conflicts with a GT in/out partner.
        conflict = False
        if md is not None and gt_sources.get(md) and (ms is None or ms not in gt_sources[md]):
            conflict = True
        if ms is not None and gt_targets.get(ms) and (md is None or md not in gt_targets[ms]):
            conflict = True
        if conflict:
            fp_edges.add((int(src), int(dst)))

    edge_tp = len(tp_edges)
    edge_fn = len(gt_edge_set) - edge_tp
    edge_fp = len(fp_edges)
    edge_jaccard = _jaccard(edge_tp, edge_fp, edge_fn)

    n_true = gt.estimated_n_nodes
    if n_true is not None and n_true > 0:
        ratio = (pred.n_nodes - n_true) / n_true
        adj = max(0.0, edge_jaccard * (1.0 - ADJUSTMENT_ALPHA * ratio)) if edge_jaccard == edge_jaccard else float("nan")
    else:
        ratio = float("nan")
        adj = float("nan")

    matched_gt = set(mapping.values())
    node_recall = len(matched_gt) / gt.n_nodes if gt.n_nodes else float("nan")
    missed = gt.n_nodes - len(matched_gt)

    identity_switches = 0
    broken = 0
    pred_matched_edges = {(mapping.get(int(s)), mapping.get(int(d))) for s, d in pred_edges}
    for src, dst in gt.consecutive_edges():
        if (int(src), int(dst)) in tp_edges:
            continue
        broken += 1
        # switch: both nodes detected, but linked to the wrong partner
        if int(src) in matched_gt and int(dst) in matched_gt:
            if (int(src), int(dst)) not in pred_matched_edges:
                identity_switches += 1

    div = _division_proxy(pred, gt, mapping, pred_edges)
    division_jaccard = _jaccard(div["tp"], div["fp"], div["fn"])
    if adj == adj and division_jaccard == division_jaccard:
        score = adj + SCORE_DIVISION_WEIGHT * division_jaccard
    elif adj == adj:
        score = adj
    else:
        score = float("nan")

    notes = []
    if n_true is None:
        notes.append("adj_edge_jaccard is NaN because estimated_number_of_nodes is missing")
    notes.append("division_jaccard is a local parent-daughter proxy, not the official 5-generation window")

    return SampleMetrics(
        dataset=pred.dataset,
        edge_tp=edge_tp,
        edge_fp=edge_fp,
        edge_fn=edge_fn,
        division_tp=div["tp"],
        division_fp=div["fp"],
        division_fn=div["fn"],
        num_pred_nodes=pred.n_nodes,
        num_gt_nodes=gt.n_nodes,
        node_recall=float(node_recall),
        missed_detections=int(missed),
        identity_switches=int(identity_switches),
        broken_tracks=int(broken),
        edge_jaccard=float(edge_jaccard) if edge_jaccard == edge_jaccard else float("nan"),
        adj_edge_jaccard=float(adj) if adj == adj else float("nan"),
        division_jaccard=float(division_jaccard) if division_jaccard == division_jaccard else float("nan"),
        score=float(score) if score == score else float("nan"),
        total_node_ratio=float(ratio) if ratio == ratio else float("nan"),
        estimated_n_nodes=n_true,
        metric_label="local_reimplementation_of_official_edge_metric+division_proxy",
        notes="; ".join(notes),
    )


def _division_proxy(
    pred: TrackGraph,
    gt: TrackGraph,
    mapping: dict[int, int],
    pred_edges: np.ndarray,
) -> dict[str, int]:
    pred_out: dict[int, list[int]] = {}
    for src, dst in pred_edges:
        pred_out.setdefault(int(src), []).append(int(dst))
    gt_out: dict[int, list[int]] = {}
    for src, dst in gt.consecutive_edges():
        gt_out.setdefault(int(src), []).append(int(dst))
    pred_forks = [p for p, ch in pred_out.items() if len(ch) >= 2]
    gt_forks = [p for p, ch in gt_out.items() if len(ch) >= 2]

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    tp = 0
    inv_map = {}
    for pred_id, gt_id in mapping.items():
        inv_map.setdefault(gt_id, []).append(pred_id)

    for gparent in gt_forks:
        pred_parent_candidates = inv_map.get(int(gparent), [])
        matched_parent = next((p for p in pred_parent_candidates if p in pred_out and p not in used_pred), None)
        if matched_parent is None:
            continue
        gt_daughters = set(gt_out[gparent])
        pred_daughters = pred_out[matched_parent]
        mapped = {mapping.get(d) for d in pred_daughters}
        if len(mapped & gt_daughters) >= 2:
            tp += 1
            used_pred.add(matched_parent)
            used_gt.add(gparent)

    fn = len(gt_forks) - tp
    # Predicted forks with local GT evidence that are not TPs count as FP.
    fp = 0
    for p in pred_forks:
        if p in used_pred:
            continue
        mapped_parent = mapping.get(p)
        mapped_children = [mapping.get(c) for c in pred_out[p]]
        has_evidence = mapped_parent is not None or any(c is not None for c in mapped_children)
        if has_evidence:
            fp += 1
    return {"tp": tp, "fp": fp, "fn": fn}


def summarise(rows: list[SampleMetrics]) -> dict[str, float | int | str]:
    valid = [r for r in rows if r.edge_tp == r.edge_tp]
    if not valid:
        return {"n": 0, "score": float("nan"), "metric_label": "empty"}
    edge_tp = sum(r.edge_tp for r in valid)
    edge_fp = sum(r.edge_fp for r in valid)
    edge_fn = sum(r.edge_fn for r in valid)
    div_tp = sum(r.division_tp for r in valid)
    div_fp = sum(r.division_fp for r in valid)
    div_fn = sum(r.division_fn for r in valid)
    adj_rows = [r for r in valid if r.adj_edge_jaccard == r.adj_edge_jaccard]
    weights = [r.edge_tp + r.edge_fp + r.edge_fn for r in adj_rows]
    total_w = sum(weights)
    if total_w > 0:
        adj = sum(w * r.adj_edge_jaccard for w, r in zip(weights, adj_rows)) / total_w
    else:
        adj = float("nan")
    edge_j = _jaccard(edge_tp, edge_fp, edge_fn)
    div_j = _jaccard(div_tp, div_fp, div_fn)
    if adj == adj and div_j == div_j:
        score = adj + SCORE_DIVISION_WEIGHT * div_j
    else:
        score = adj
    return {
        "n": len(valid),
        "edge_tp": edge_tp,
        "edge_fp": edge_fp,
        "edge_fn": edge_fn,
        "division_tp": div_tp,
        "division_fp": div_fp,
        "division_fn": div_fn,
        "missed_detections": sum(r.missed_detections for r in valid),
        "identity_switches": sum(r.identity_switches for r in valid),
        "broken_tracks": sum(r.broken_tracks for r in valid),
        "edge_jaccard": edge_j,
        "adj_edge_jaccard": adj,
        "division_jaccard": div_j,
        "node_recall": sum(r.node_recall for r in valid) / len(valid),
        "score": score,
        "metric_label": valid[0].metric_label,
    }
