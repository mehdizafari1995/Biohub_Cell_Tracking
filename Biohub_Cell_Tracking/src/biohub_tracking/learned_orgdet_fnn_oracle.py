"""44b6-only FNN detector oracle / feasibility audit.

Matches locked ``learned_unet_orgdet_fnn_v1`` detections to sparse GT
(7.0 µm official assignment). Measures how many GT edges have both
endpoints detected, how many of those sit in the locked time/space
candidate envelope, and how many remaining errors are
detection-unavoidable versus association-fixable. Quantifies formula
and official edge-Jaccard upper bounds plus the division term with
this fixed detector.

Never opens ``6bba``. Never writes ``submission.csv``. GT stays out of
inference and public-test code. Not a Kaggle run and not a leaderboard
score. Do not retune locked FNN / ELF / 12 µm / short-fragment levers.
Do not use public 0.602 / 0.612 / 0.553.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import DEFAULT_MATCH_DISTANCE_UM, DEFAULT_SCALE_ZYX
from .errors import analyse_errors
from .evaluate import match_nodes
from .graphs import TrackGraph
from .learned_orgdet_error_decomp import load_orgop_graphs, refuse_holdout_paths
from .learned_orgdet_fnn import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LEARNED_ORGDET_FNN_LOCKED_HOLDOUT,
    LEARNED_ORGDET_FNN_LOCKED_TUNE,
    LOCKED_ORGDET_FNN_HASH,
    apply_orgdet_fnn_overlay,
    ensure_orgdet_tuning_graph,
    refuse_non_tuning_stems,
)
from .learned_orgdet_gap import LOCKED_ORGDET_GAP_HASH
from .learned_split import HOLDOUT_SPECIMENS, TUNING_SPECIMEN, specimen_of
from .link import physical_coords

SCALE_ZYX = DEFAULT_SCALE_ZYX
MATCH_UM = DEFAULT_MATCH_DISTANCE_UM
# Locked association_v2 / orgdet / gap envelope. Diagnostic only; do not retune.
LINKER_GATE_UM = 12.0
MAX_GAP_FRAMES = 2
GAP_SCALE = "sqrt_dt"
# Locked ELF dt=1 length. Diagnostic only; do not retune.
ELF_DT1_UM = 8.0
TUNE_MARGIN = 0.002000
TUNE_KEEP_THRESHOLD = LEARNED_ORGDET_FNN_LOCKED_TUNE + TUNE_MARGIN
CANDIDATE_RECALL_GATE = 0.75
RANK_LABEL = "learned_unet_orgdet_fnn_rank_v1"
DET_LABEL = "learned_unet_orgdet_fnn_detrec_v1"


def _clean_stem(raw: str) -> str:
    stem = Path(raw).name
    for suffix in (".zarr", ".geff", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def refuse_oracle_holdout(paths: Iterable[str | Path] | None = None, stems: Iterable[str] | None = None) -> None:
    refuse_non_tuning_stems(INFER_STEMS)
    if stems is not None:
        refuse_non_tuning_stems(stems)
    if paths is not None:
        refuse_holdout_paths(paths)
        for raw in paths:
            name = Path(raw).name
            stem = _clean_stem(name)
            if specimen_of(stem) in HOLDOUT_SPECIMENS or stem in HOLDOUT_STEMS or name.startswith("6bba"):
                raise ValueError(f"fnn oracle refuses holdout path {name}")


def invert_mapping(pred_to_gt: Mapping[int, int]) -> dict[int, int]:
    inv: dict[int, int] = {}
    for pred_id, gt_id in pred_to_gt.items():
        gid = int(gt_id)
        if gid in inv:
            raise RuntimeError("fnn oracle saw a non-1-1 node match; refusing")
        inv[gid] = int(pred_id)
    return inv


def candidate_limit_um(dt: int, gate_um: float = LINKER_GATE_UM, gap_scale: str = GAP_SCALE) -> float:
    if int(dt) < 1:
        return 0.0
    if gap_scale != "sqrt_dt":
        raise RuntimeError("fnn oracle would invent a gap scale; refusing")
    return float(gate_um) * math.sqrt(float(dt))


def pair_in_candidate(
    dt: int,
    dist_um: float,
    gate_um: float = LINKER_GATE_UM,
    max_dt: int = MAX_GAP_FRAMES,
    gap_scale: str = GAP_SCALE,
) -> bool:
    if int(dt) < 1 or int(dt) > int(max_dt):
        return False
    return float(dist_um) <= candidate_limit_um(int(dt), gate_um=gate_um, gap_scale=gap_scale)


def _edge_jaccard(tp: int, fp: int, fn: int) -> float:
    den = int(tp) + int(fp) + int(fn)
    if den <= 0:
        return float("nan")
    return float(tp) / float(den)


def _pred_pair_geom(
    pred: TrackGraph,
    src_id: int,
    dst_id: int,
    scale_zyx: tuple[float, float, float] = SCALE_ZYX,
) -> tuple[int, float] | None:
    index = pred.id_to_index()
    if int(src_id) not in index or int(dst_id) not in index:
        return None
    i, j = index[int(src_id)], index[int(dst_id)]
    dt = int(pred.t[j]) - int(pred.t[i])
    xyz = physical_coords(pred.coords_zyx(), scale_zyx)
    dist = float(np.linalg.norm(xyz[j] - xyz[i]))
    return dt, dist


def classify_gt_edges(
    pred: TrackGraph,
    gt: TrackGraph,
    mapping: Mapping[int, int] | None = None,
    scale_zyx: tuple[float, float, float] = SCALE_ZYX,
) -> dict[str, Any]:
    """Per-GT-edge detection / candidate / association buckets. GT unused at inference."""
    if mapping is None:
        mapping = match_nodes(pred, gt, scale_zyx=scale_zyx, max_distance_um=MATCH_UM)
    gt_to_pred = invert_mapping(mapping)
    pred_edges = {(int(s), int(d)) for s, d in np.asarray(pred.edges).reshape(-1, 2)} if pred.n_edges else set()
    rows: list[dict[str, Any]] = []
    n_both = 0
    n_detect_fn = 0
    n_assoc_fn = 0
    n_both_in_candidate = 0
    n_both_in_dt1_12 = 0
    n_both_in_gap = 0
    n_both_in_elf = 0
    n_assoc_fn_in_candidate = 0
    n_current_tp = 0
    for src, dst in np.asarray(gt.edges).reshape(-1, 2) if gt.n_edges else []:
        gs, gd = int(src), int(dst)
        pred_src = gt_to_pred.get(gs)
        pred_dst = gt_to_pred.get(gd)
        both = pred_src is not None and pred_dst is not None
        linked = bool(both and (int(pred_src), int(pred_dst)) in pred_edges)
        dt = None
        dist = None
        in_cand = False
        in_dt1 = False
        in_gap = False
        in_elf = False
        if both:
            geom = _pred_pair_geom(pred, int(pred_src), int(pred_dst), scale_zyx=scale_zyx)
            if geom is None:
                raise RuntimeError("fnn oracle lost a matched pair; refusing")
            dt, dist = geom
            in_cand = pair_in_candidate(dt, dist)
            in_dt1 = int(dt) == 1 and float(dist) <= LINKER_GATE_UM
            in_gap = int(dt) == 2 and float(dist) <= candidate_limit_um(2)
            in_elf = int(dt) == 1 and float(dist) <= ELF_DT1_UM
            n_both += 1
            n_both_in_candidate += int(in_cand)
            n_both_in_dt1_12 += int(in_dt1)
            n_both_in_gap += int(in_gap)
            n_both_in_elf += int(in_elf)
            if linked:
                n_current_tp += 1
            else:
                n_assoc_fn += 1
                n_assoc_fn_in_candidate += int(in_cand)
        else:
            n_detect_fn += 1
        rows.append(
            {
                "gt_src": gs,
                "gt_dst": gd,
                "both_detected": both,
                "currently_linked": linked,
                "dt": dt,
                "dist_um": dist,
                "in_plausible_candidate": in_cand,
                "in_dt1_12um": in_dt1,
                "in_dt2_gap": in_gap,
                "in_elf_dt1_8um": in_elf,
            }
        )
    n_gt = len(rows)
    return {
        "dataset": pred.dataset,
        "n_gt_edges": n_gt,
        "n_both_detected": n_both,
        "n_detection_unavoidable": n_detect_fn,
        "n_association_fixable": n_assoc_fn,
        "n_association_fixable_in_candidate": n_assoc_fn_in_candidate,
        "n_currently_linked": n_current_tp,
        "n_both_detected_in_candidate": n_both_in_candidate,
        "n_both_detected_dt1_12um": n_both_in_dt1_12,
        "n_both_detected_dt2_gap": n_both_in_gap,
        "n_both_detected_elf_dt1": n_both_in_elf,
        "detection_fn_share": n_detect_fn / n_gt if n_gt else 0.0,
        "candidate_edge_recall": n_both_in_candidate / n_both if n_both else 0.0,
        "candidate_edge_recall_of_all_gt": n_both_in_candidate / n_gt if n_gt else 0.0,
        "edges": rows,
    }


def classify_divisions(
    pred: TrackGraph,
    gt: TrackGraph,
    mapping: Mapping[int, int] | None = None,
    scale_zyx: tuple[float, float, float] = SCALE_ZYX,
) -> dict[str, Any]:
    if mapping is None:
        mapping = match_nodes(pred, gt, scale_zyx=scale_zyx, max_distance_um=MATCH_UM)
    gt_to_pred = invert_mapping(mapping)
    gt_out = gt.out_edges()
    pred_out = pred.out_edges()
    gt_divs = [int(nid) for nid, kids in gt_out.items() if len(kids) >= 2]
    pred_divs = [int(nid) for nid, kids in pred_out.items() if len(kids) >= 2]
    recoverable = 0
    for parent in gt_divs:
        pred_parent = gt_to_pred.get(int(parent))
        matched_kids = [c for c in gt_out[int(parent)] if int(c) in gt_to_pred]
        if pred_parent is not None and len(matched_kids) >= 2:
            recoverable += 1
    return {
        "dataset": pred.dataset,
        "n_gt_divisions": len(gt_divs),
        "n_pred_divisions": len(pred_divs),
        "n_recoverable_divisions": recoverable,
        "n_detection_unavoidable_divisions": len(gt_divs) - recoverable,
    }


def build_oracle_graph(
    pred: TrackGraph,
    gt: TrackGraph,
    mapping: Mapping[int, int],
    *,
    require_candidate: bool,
    scale_zyx: tuple[float, float, float] = SCALE_ZYX,
) -> TrackGraph:
    """Same FNN nodes; keep only GT-true edges whose endpoints are detected.

    ``require_candidate=True`` further restricts to the locked 12 µm /
    ``sqrt(dt)`` / max_gap=2 envelope. This is a feasibility graph, not an
    inference path. GT is not used outside this audit.
    """
    gt_to_pred = invert_mapping(mapping)
    edges: list[tuple[int, int]] = []
    for src, dst in np.asarray(gt.edges).reshape(-1, 2) if gt.n_edges else []:
        pred_src = gt_to_pred.get(int(src))
        pred_dst = gt_to_pred.get(int(dst))
        if pred_src is None or pred_dst is None:
            continue
        if require_candidate:
            geom = _pred_pair_geom(pred, int(pred_src), int(pred_dst), scale_zyx=scale_zyx)
            if geom is None or not pair_in_candidate(*geom):
                continue
        edges.append((int(pred_src), int(pred_dst)))
    out = TrackGraph(
        dataset=pred.dataset,
        node_id=pred.node_id,
        t=pred.t,
        z=pred.z,
        y=pred.y,
        x=pred.x,
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2) if edges else np.zeros((0, 2), dtype=np.int64),
        estimated_n_nodes=pred.estimated_n_nodes,
    )
    out.metadata = dict(pred.metadata)
    out.metadata["oracle"] = "detection" if not require_candidate else "candidate"
    out.metadata["n_oracle_edges"] = int(out.n_edges)
    out.metadata["uses_gt_at_inference"] = False
    return out


def formula_bounds(edge_info: Mapping[str, Any]) -> dict[str, Any]:
    n_gt = int(edge_info.get("n_gt_edges") or 0)
    n_both = int(edge_info.get("n_both_detected") or 0)
    n_cand = int(edge_info.get("n_both_detected_in_candidate") or 0)
    return {
        "detection_oracle_tp": n_both,
        "detection_oracle_fp": 0,
        "detection_oracle_fn": n_gt - n_both,
        "detection_oracle_edge_jaccard": _edge_jaccard(n_both, 0, n_gt - n_both),
        "candidate_oracle_tp": n_cand,
        "candidate_oracle_fp": 0,
        "candidate_oracle_fn": n_gt - n_cand,
        "candidate_oracle_edge_jaccard": _edge_jaccard(n_cand, 0, n_gt - n_cand),
    }


def audit_sequence(
    pred: TrackGraph,
    gt: TrackGraph,
    official: Mapping[str, Any] | None = None,
    scale_zyx: tuple[float, float, float] = SCALE_ZYX,
) -> dict[str, Any]:
    refuse_oracle_holdout(stems=[pred.dataset, gt.dataset])
    mapping = match_nodes(pred, gt, scale_zyx=scale_zyx, max_distance_um=MATCH_UM)
    analysis = analyse_errors(pred, gt, scale_zyx=scale_zyx)
    edges = classify_gt_edges(pred, gt, mapping=mapping, scale_zyx=scale_zyx)
    divisions = classify_divisions(pred, gt, mapping=mapping, scale_zyx=scale_zyx)
    bounds = formula_bounds(edges)
    det_oracle = build_oracle_graph(pred, gt, mapping, require_candidate=False, scale_zyx=scale_zyx)
    cand_oracle = build_oracle_graph(pred, gt, mapping, require_candidate=True, scale_zyx=scale_zyx)
    return {
        "dataset": pred.dataset,
        "n_pred_nodes": pred.n_nodes,
        "n_pred_edges": pred.n_edges,
        "n_gt_nodes": gt.n_nodes,
        "n_matched_gt": analysis.n_matched_gt,
        "missed_gt_nodes": len(analysis.missed_gt_ids),
        "unmatched_pred_nodes": len(analysis.unmatched_pred_ids),
        "unmatched_pred_note": analysis.unmatched_pred_note,
        "official_tp_fp_fn": [
            int((official or {}).get("edge_tp") or 0),
            int((official or {}).get("edge_fp") or 0),
            int((official or {}).get("edge_fn") or 0),
        ],
        "official_score": float((official or {}).get("score") or 0.0),
        "official_adj_edge_jaccard": float((official or {}).get("adj_edge_jaccard") or 0.0),
        "official_division_tp_fp_fn": [
            int((official or {}).get("division_tp") or 0),
            int((official or {}).get("division_fp") or 0),
            int((official or {}).get("division_fn") or 0),
        ],
        "edges": {k: v for k, v in edges.items() if k != "edges"},
        "edge_rows": edges["edges"],
        "divisions": divisions,
        "formula_bounds": bounds,
        "detection_oracle_n_edges": det_oracle.n_edges,
        "candidate_oracle_n_edges": cand_oracle.n_edges,
        "oracles": {"detection": det_oracle, "candidate": cand_oracle},
    }


def aggregate_audits(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "n_pred_nodes",
        "n_pred_edges",
        "n_gt_nodes",
        "n_matched_gt",
        "missed_gt_nodes",
        "unmatched_pred_nodes",
        "detection_oracle_n_edges",
        "candidate_oracle_n_edges",
    )
    out = {k: int(sum(int(r.get(k) or 0) for r in rows)) for k in keys}
    edge_keys = (
        "n_gt_edges",
        "n_both_detected",
        "n_detection_unavoidable",
        "n_association_fixable",
        "n_association_fixable_in_candidate",
        "n_currently_linked",
        "n_both_detected_in_candidate",
        "n_both_detected_dt1_12um",
        "n_both_detected_dt2_gap",
        "n_both_detected_elf_dt1",
    )
    for key in edge_keys:
        out[key] = int(sum(int((r.get("edges") or {}).get(key) or 0) for r in rows))
    div_keys = (
        "n_gt_divisions",
        "n_pred_divisions",
        "n_recoverable_divisions",
        "n_detection_unavoidable_divisions",
    )
    for key in div_keys:
        out[key] = int(sum(int((r.get("divisions") or {}).get(key) or 0) for r in rows))
    n_gt = out["n_gt_edges"]
    n_both = out["n_both_detected"]
    n_cand = out["n_both_detected_in_candidate"]
    out["detection_fn_share"] = out["n_detection_unavoidable"] / n_gt if n_gt else 0.0
    out["candidate_edge_recall"] = n_cand / n_both if n_both else 0.0
    out["candidate_edge_recall_of_all_gt"] = n_cand / n_gt if n_gt else 0.0
    out["formula_bounds"] = formula_bounds(
        {
            "n_gt_edges": n_gt,
            "n_both_detected": n_both,
            "n_both_detected_in_candidate": n_cand,
        }
    )
    out["official_tp"] = int(sum(int((r.get("official_tp_fp_fn") or [0])[0]) for r in rows))
    out["official_fp"] = int(sum(int((r.get("official_tp_fp_fn") or [0, 0])[1]) for r in rows))
    out["official_fn"] = int(sum(int((r.get("official_tp_fp_fn") or [0, 0, 0])[2]) for r in rows))
    return out


def rank_config_payload() -> dict[str, Any]:
    """Preregistered learned ranking / constrained selection. Lock before any score."""
    return {
        "label": RANK_LABEL,
        "parent_label": "learned_unet_orgdet_fnn_v1",
        "parent_hash": LOCKED_ORGDET_FNN_HASH,
        "parent_orgdet_gap_hash": LOCKED_ORGDET_GAP_HASH,
        "source_graphs": "locked_learned_unet_orgdet_fnn_graphs",
        "linker": "fnn_detections_then_learned_rank_inside_locked_candidate_envelope",
        "candidate_gate_um": LINKER_GATE_UM,
        "candidate_max_dt": MAX_GAP_FRAMES,
        "candidate_gap_scale": GAP_SCALE,
        "hard_negatives": True,
        "train_specimen": TUNING_SPECIMEN,
        "verify_nonzero_link_head": True,
        "retunes_detection": False,
        "retunes_consecutive_links": False,
        "retunes_gap_gates": False,
        "retunes_max_dt1_edge_um": False,
        "retunes_nn_tie_eps": False,
        "uses_intensity": False,
        "uses_transformer": True,
        "uses_velocity": False,
        "uses_ilp": False,
        "assignment": "unchanged_locked_candidate_envelope",
        "tune_keep_threshold": TUNE_KEEP_THRESHOLD,
        "tune_margin": TUNE_MARGIN,
        "opened_holdout": False,
        "is_12um_assignment": False,
        "is_rejected_overlay": False,
        "writes_submission_csv": False,
        "notes": (
            "Rank pairwise candidates already inside the locked 12 µm / "
            "sqrt(dt) / max_gap=2 envelope, then constrained 1-1 selection. "
            "44b6 train only. Hard negatives. Fail closed unless the link "
            "head shows nonzero learning. Not another 12 µm assignment, "
            "short-fragment retune, or 8.0 µm dt=1 retune. Frozen 6bba "
            f"stays closed until official 44b6 exceeds {TUNE_KEEP_THRESHOLD:.6f}."
        ),
    }


def detector_config_payload() -> dict[str, Any]:
    """Preregistered detector intervention if candidate recall cannot gain."""
    return {
        "label": DET_LABEL,
        "parent_label": "learned_unet_orgdet_fnn_v1",
        "parent_hash": LOCKED_ORGDET_FNN_HASH,
        "source_graphs": "locked_learned_unet_orgop_graphs",
        "linker": "fnn_then_recall_oriented_detection_repass",
        "retunes_detection": True,
        "retunes_consecutive_links": False,
        "retunes_gap_gates": False,
        "retunes_max_dt1_edge_um": False,
        "uses_intensity": False,
        "uses_transformer": False,
        "uses_velocity": False,
        "uses_ilp": False,
        "assignment": "unchanged_locked_orgdet_fnn",
        "tune_keep_threshold": TUNE_KEEP_THRESHOLD,
        "tune_margin": TUNE_MARGIN,
        "opened_holdout": False,
        "is_12um_assignment": False,
        "is_rejected_overlay": False,
        "writes_submission_csv": False,
        "notes": (
            "Candidate-set ranking cannot beat FNN by the +0.002000 gate. "
            "Next is a 44b6-only recall-oriented detection repass on locked "
            "orgop detections that does not retune 12 µm, ELF, FNN, or TCF. "
            f"Frozen 6bba stays closed until official 44b6 exceeds {TUNE_KEEP_THRESHOLD:.6f}."
        ),
    }


def config_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_RANK_HASH = config_hash(rank_config_payload())
LOCKED_DET_HASH = config_hash(detector_config_payload())
assert LOCKED_ORGDET_FNN_HASH == "95e9300eb84ba71c"
assert LOCKED_RANK_HASH == "2748c4a2b2823c0a"
assert abs(TUNE_KEEP_THRESHOLD - 0.7266526336456757) < 1e-12


def recommend_hypothesis(
    aggregate: Mapping[str, Any],
    *,
    candidate_official_score: float | None = None,
    detection_official_score: float | None = None,
    fnn_official_score: float = LEARNED_ORGDET_FNN_LOCKED_TUNE,
) -> dict[str, Any]:
    """Choose ranking vs detector from candidate-edge recall and official room."""
    bounds = aggregate.get("formula_bounds") or {}
    cand_formula = float(bounds.get("candidate_oracle_edge_jaccard") or 0.0)
    det_formula = float(bounds.get("detection_oracle_edge_jaccard") or 0.0)
    cand_score = float(candidate_official_score) if candidate_official_score is not None else cand_formula
    det_score = float(detection_official_score) if detection_official_score is not None else det_formula
    recall = float(aggregate.get("candidate_edge_recall") or 0.0)
    detect_share = float(aggregate.get("detection_fn_share") or 0.0)
    assoc_in_cand = int(aggregate.get("n_association_fixable_in_candidate") or 0)
    fnn = float(fnn_official_score)
    material_candidate = cand_score >= float(TUNE_KEEP_THRESHOLD) and (cand_score - fnn) >= float(TUNE_MARGIN)
    high_recall = recall >= float(CANDIDATE_RECALL_GATE)
    if material_candidate and high_recall and assoc_in_cand >= 0:
        family = "learned_edge_ranking_constrained_selection"
        payload = rank_config_payload()
        rationale = (
            "Both-endpoint-detected GT edges mostly sit inside the locked "
            "12 µm / sqrt(dt) candidate envelope, and a perfect selection "
            f"among those candidates has room above FNN {fnn:.6f} by "
            f">= {TUNE_MARGIN:.6f}. Preregister learned ranking with hard "
            "negatives and a nonzero link-head canary. Do not retune gates."
        )
    else:
        family = "detector_intervention"
        payload = detector_config_payload()
        rationale = (
            "Candidate-edge recall or the candidate-oracle upper bound does "
            f"not support a +{TUNE_MARGIN:.6f} official 44b6 gain versus FNN "
            f"{fnn:.6f}. Association ranking inside the locked envelope is "
            "closed. Prefer a 44b6-only detector intervention that is not "
            "TCF, ELF, FNN, or another 12 µm assignment."
        )
    hashed = config_hash(payload)
    return {
        "family": family,
        "label": payload["label"],
        "config_hash": hashed,
        "payload": payload,
        "rationale": rationale,
        "candidate_official_or_formula": cand_score,
        "detection_official_or_formula": det_score,
        "fnn_official": fnn,
        "candidate_edge_recall": recall,
        "detection_fn_share": detect_share,
        "material_candidate_gain": material_candidate,
        "high_candidate_recall": high_recall,
        "tune_keep_threshold": TUNE_KEEP_THRESHOLD,
        "tune_margin": TUNE_MARGIN,
        "opened_holdout": False,
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
    }


def build_fnn_from_orgop(orgop: TrackGraph) -> TrackGraph:
    refuse_oracle_holdout(stems=[orgop.dataset])
    orgdet = ensure_orgdet_tuning_graph(orgop)
    return apply_orgdet_fnn_overlay(orgdet)


def load_fnn_tuning_graphs(graph_dir: str | Path, stems: Sequence[str] = INFER_STEMS) -> list[TrackGraph]:
    refuse_oracle_holdout(stems=stems, paths=[graph_dir])
    graphs = []
    for orgop in load_orgop_graphs(graph_dir, stems):
        graphs.append(build_fnn_from_orgop(orgop))
    return graphs
