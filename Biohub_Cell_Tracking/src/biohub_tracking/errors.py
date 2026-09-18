from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .evaluate import match_nodes
from .graphs import TrackGraph


@dataclass
class ErrorAnalysis:
    dataset: str
    n_pred_nodes: int
    n_gt_nodes: int
    n_matched_gt: int
    missed_gt_ids: list[int]
    unmatched_pred_ids: list[int]
    unmatched_pred_note: str
    tp_edges: list[tuple[int, int]]
    fp_edges: list[tuple[int, int]]
    fn_edges: list[tuple[int, int]]
    identity_switches: list[tuple[int, int]]
    fragmented_tracks: list[tuple[int, int]]
    gt_divisions: list[int]
    pred_divisions: list[int]
    recovered_divisions: list[int]
    missed_divisions: list[int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "n_pred_nodes": self.n_pred_nodes,
            "n_gt_nodes": self.n_gt_nodes,
            "n_matched_gt": self.n_matched_gt,
            "missed_annotated": len(self.missed_gt_ids),
            "unmatched_predictions": len(self.unmatched_pred_ids),
            "unmatched_pred_note": self.unmatched_pred_note,
            "tp_edges": len(self.tp_edges),
            "fp_edges": len(self.fp_edges),
            "fn_edges": len(self.fn_edges),
            "identity_switches": len(self.identity_switches),
            "fragmented_tracks": len(self.fragmented_tracks),
            "gt_divisions": len(self.gt_divisions),
            "pred_divisions": len(self.pred_divisions),
            "recovered_divisions": len(self.recovered_divisions),
            "missed_divisions": len(self.missed_divisions),
        }


def analyse_errors(pred: TrackGraph, gt: TrackGraph, scale_zyx: tuple[float, float, float]) -> ErrorAnalysis:
    mapping = match_nodes(pred, gt, scale_zyx=scale_zyx)
    matched_gt = set(mapping.values())
    matched_pred = set(mapping)
    missed = [int(nid) for nid in gt.node_id if int(nid) not in matched_gt]
    unmatched_pred = [int(nid) for nid in pred.node_id if int(nid) not in matched_pred]

    gt_edge_set = {(int(s), int(d)) for s, d in gt.consecutive_edges()}
    gt_sources: dict[int, set[int]] = {}
    gt_targets: dict[int, set[int]] = {}
    for src, dst in gt.consecutive_edges():
        gt_sources.setdefault(int(dst), set()).add(int(src))
        gt_targets.setdefault(int(src), set()).add(int(dst))

    tp: set[tuple[int, int]] = set()
    fp: list[tuple[int, int]] = []
    pred_mapped_edges: set[tuple[int | None, int | None]] = set()
    for src, dst in pred.consecutive_edges():
        ms, md = mapping.get(int(src)), mapping.get(int(dst))
        pred_mapped_edges.add((ms, md))
        if ms is not None and md is not None and (ms, md) in gt_edge_set:
            tp.add((ms, md))
            continue
        conflict = False
        if md is not None and gt_sources.get(md) and (ms is None or ms not in gt_sources[md]):
            conflict = True
        if ms is not None and gt_targets.get(ms) and (md is None or md not in gt_targets[ms]):
            conflict = True
        if conflict:
            fp.append((int(src), int(dst)))

    fn = [e for e in gt_edge_set if e not in tp]
    switches = []
    for src, dst in fn:
        if src in matched_gt and dst in matched_gt and (src, dst) not in pred_mapped_edges:
            switches.append((src, dst))

    gt_out = gt.out_edges()
    pred_out = pred.out_edges()
    gt_divs = [nid for nid, ch in gt_out.items() if len(ch) >= 2]
    pred_divs = [nid for nid, ch in pred_out.items() if len(ch) >= 2]
    inv: dict[int, list[int]] = {}
    for p, g in mapping.items():
        inv.setdefault(g, []).append(p)
    recovered = []
    for gparent in gt_divs:
        candidates = inv.get(int(gparent), [])
        matched_parent = next((p for p in candidates if p in pred_out), None)
        if matched_parent is None:
            continue
        mapped_children = {mapping.get(c) for c in pred_out[matched_parent]}
        if len(mapped_children & set(gt_out[gparent])) >= 2:
            recovered.append(int(gparent))
    missed_divs = [d for d in gt_divs if d not in recovered]

    return ErrorAnalysis(
        dataset=pred.dataset,
        n_pred_nodes=pred.n_nodes,
        n_gt_nodes=gt.n_nodes,
        n_matched_gt=len(matched_gt),
        missed_gt_ids=missed,
        unmatched_pred_ids=unmatched_pred,
        unmatched_pred_note=(
            "Unmatched predictions are not automatic false positives. "
            "Ground truth is sparse; most unmatched detections are unlabeled cells or noise."
        ),
        tp_edges=sorted(tp),
        fp_edges=fp,
        fn_edges=fn,
        identity_switches=switches,
        fragmented_tracks=fn,
        gt_divisions=gt_divs,
        pred_divisions=pred_divs,
        recovered_divisions=recovered,
        missed_divisions=missed_divs,
    )
