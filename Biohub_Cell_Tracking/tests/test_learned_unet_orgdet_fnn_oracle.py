from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biohub_tracking.graphs import TrackGraph
from biohub_tracking.learned_orgdet_fnn import (
    LEARNED_ORGDET_FNN_LOCKED_TUNE,
    LOCKED_ORGDET_FNN_HASH,
)
from biohub_tracking.learned_orgdet_fnn_oracle import (
    CANDIDATE_RECALL_GATE,
    DET_LABEL,
    INFER_STEMS,
    LINKER_GATE_UM,
    LOCKED_DET_HASH,
    LOCKED_RANK_HASH,
    MATCH_UM,
    RANK_LABEL,
    TUNE_KEEP_THRESHOLD,
    TUNE_MARGIN,
    aggregate_audits,
    audit_sequence,
    build_oracle_graph,
    classify_divisions,
    classify_gt_edges,
    config_hash,
    detector_config_payload,
    invert_mapping,
    pair_in_candidate,
    rank_config_payload,
    recommend_hypothesis,
    refuse_oracle_holdout,
)

ROOT = Path(__file__).resolve().parents[1]


def _graph(node_id, t, z, y, x, edges, dataset="44b6_12dfb391") -> TrackGraph:
    return TrackGraph(
        dataset=dataset,
        node_id=node_id,
        t=t,
        z=z,
        y=y,
        x=x,
        edges=edges,
    )


def test_refuses_holdout_stems_and_paths():
    with pytest.raises(ValueError, match="refuses stem"):
        refuse_oracle_holdout(stems=["6bba_05db0fb1"])
    with pytest.raises(ValueError, match="holdout"):
        refuse_oracle_holdout(paths=[ROOT / "data" / "train" / "6bba_05db0fb1.geff"])
    refuse_oracle_holdout(stems=INFER_STEMS)


def test_pair_in_candidate_uses_locked_envelope():
    assert pair_in_candidate(1, 11.9) is True
    assert pair_in_candidate(1, 12.1) is False
    assert pair_in_candidate(2, LINKER_GATE_UM * (2**0.5) - 0.01) is True
    assert pair_in_candidate(2, LINKER_GATE_UM * (2**0.5) + 0.01) is False
    assert pair_in_candidate(3, 1.0) is False
    assert pair_in_candidate(0, 1.0) is False


def test_classify_detection_versus_association_and_candidate():
    # 1 voxel in x = 0.40625 µm. dx=10 → 4.0625 µm (inside 8 and 12).
    # dx=40 → 16.25 µm (outside 12 µm).
    gt = _graph(
        [1, 2, 3, 4],
        [0, 1, 2, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 10, 20, 200],
        [(1, 2), (2, 3), (1, 4)],
    )
    pred = _graph(
        [10, 20, 30],
        [0, 1, 2],
        [0, 0, 0],
        [0, 0, 0],
        [0, 10, 20],
        [(10, 20)],
    )
    mapping = {10: 1, 20: 2, 30: 3}
    assert invert_mapping(mapping) == {1: 10, 2: 20, 3: 30}
    info = classify_gt_edges(pred, gt, mapping=mapping)
    assert info["n_gt_edges"] == 3
    assert info["n_both_detected"] == 2
    assert info["n_detection_unavoidable"] == 1
    assert info["n_currently_linked"] == 1
    assert info["n_association_fixable"] == 1
    assert info["n_association_fixable_in_candidate"] == 1
    assert info["n_both_detected_in_candidate"] == 2
    assert info["candidate_edge_recall"] == pytest.approx(1.0)
    assert MATCH_UM == 7.0


def test_oracle_graph_keeps_only_detected_true_edges():
    gt = _graph(
        [1, 2, 3],
        [0, 1, 2],
        [0, 0, 0],
        [0, 0, 0],
        [0, 10, 20],
        [(1, 2), (2, 3)],
    )
    pred = _graph(
        [10, 20, 30, 99],
        [0, 1, 2, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 10, 20, 80],
        [(10, 99)],
    )
    mapping = {10: 1, 20: 2, 30: 3}
    det = build_oracle_graph(pred, gt, mapping, require_candidate=False)
    assert det.n_nodes == pred.n_nodes
    assert {tuple(map(int, e)) for e in det.edges} == {(10, 20), (20, 30)}
    cand = build_oracle_graph(pred, gt, mapping, require_candidate=True)
    assert {tuple(map(int, e)) for e in cand.edges} == {(10, 20), (20, 30)}


def test_far_true_edge_is_detection_ok_but_outside_candidate():
    gt = _graph(
        [1, 2],
        [0, 1],
        [0, 0],
        [0, 0],
        [0, 80],
        [(1, 2)],
    )
    pred = _graph(
        [10, 20],
        [0, 1],
        [0, 0],
        [0, 0],
        [0, 80],
        [],
    )
    mapping = {10: 1, 20: 2}
    info = classify_gt_edges(pred, gt, mapping=mapping)
    assert info["n_both_detected"] == 1
    assert info["n_both_detected_in_candidate"] == 0
    assert info["n_association_fixable"] == 1
    assert info["n_association_fixable_in_candidate"] == 0
    cand = build_oracle_graph(pred, gt, mapping, require_candidate=True)
    assert cand.n_edges == 0
    det = build_oracle_graph(pred, gt, mapping, require_candidate=False)
    assert det.n_edges == 1


def test_division_recoverable_requires_parent_and_two_children():
    gt = _graph(
        [1, 2, 3],
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 10, 20],
        [(1, 2), (1, 3)],
    )
    pred_ok = _graph(
        [10, 20, 30],
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 10, 20],
        [],
    )
    pred_miss = _graph(
        [10, 20],
        [0, 1],
        [0, 0],
        [0, 0],
        [0, 10],
        [],
    )
    ok = classify_divisions(pred_ok, gt, mapping={10: 1, 20: 2, 30: 3})
    miss = classify_divisions(pred_miss, gt, mapping={10: 1, 20: 2})
    assert ok["n_gt_divisions"] == 1
    assert ok["n_recoverable_divisions"] == 1
    assert miss["n_recoverable_divisions"] == 0
    assert miss["n_detection_unavoidable_divisions"] == 1


def test_audit_sequence_and_aggregate_do_not_open_holdout():
    gt = _graph(
        [1, 2],
        [0, 1],
        [0, 0],
        [0, 0],
        [0, 10],
        [(1, 2)],
        dataset="44b6_12dfb391",
    )
    pred = _graph(
        [10, 20],
        [0, 1],
        [0, 0],
        [0, 0],
        [0, 10],
        [(10, 20)],
        dataset="44b6_12dfb391",
    )
    row = audit_sequence(pred, gt, official={"edge_tp": 1, "edge_fp": 0, "edge_fn": 0, "score": 0.7})
    assert row["dataset"] == "44b6_12dfb391"
    assert row["edges"]["n_both_detected"] == 1
    assert "oracles" in row
    holdout = _graph([1], [0], [0], [0], [0], [], dataset="6bba_05db0fb1")
    with pytest.raises(ValueError, match="refuses stem"):
        audit_sequence(holdout, holdout)
    agg = aggregate_audits([row])
    assert agg["n_gt_edges"] == 1
    assert agg["official_tp"] == 1
    assert agg["formula_bounds"]["candidate_oracle_fp"] == 0


def test_recommend_ranking_when_candidate_room_exists():
    hyp = recommend_hypothesis(
        {
            "candidate_edge_recall": 0.95,
            "detection_fn_share": 0.1,
            "n_association_fixable_in_candidate": 12,
            "formula_bounds": {
                "candidate_oracle_edge_jaccard": 0.80,
                "detection_oracle_edge_jaccard": 0.81,
            },
        },
        candidate_official_score=0.80,
        fnn_official_score=LEARNED_ORGDET_FNN_LOCKED_TUNE,
    )
    assert hyp["family"] == "learned_edge_ranking_constrained_selection"
    assert hyp["label"] == RANK_LABEL
    assert hyp["config_hash"] == LOCKED_RANK_HASH
    assert hyp["payload"]["hard_negatives"] is True
    assert hyp["payload"]["verify_nonzero_link_head"] is True
    assert hyp["payload"]["is_12um_assignment"] is False
    assert hyp["payload"]["opened_holdout"] is False
    assert hyp["tune_keep_threshold"] == pytest.approx(TUNE_KEEP_THRESHOLD)
    assert TUNE_KEEP_THRESHOLD == pytest.approx(LEARNED_ORGDET_FNN_LOCKED_TUNE + TUNE_MARGIN)
    assert CANDIDATE_RECALL_GATE == pytest.approx(0.75)


def test_recommend_detector_when_candidate_cannot_gain():
    hyp = recommend_hypothesis(
        {
            "candidate_edge_recall": 0.2,
            "detection_fn_share": 0.7,
            "n_association_fixable_in_candidate": 0,
            "formula_bounds": {
                "candidate_oracle_edge_jaccard": 0.70,
                "detection_oracle_edge_jaccard": 0.71,
            },
        },
        candidate_official_score=0.70,
        fnn_official_score=LEARNED_ORGDET_FNN_LOCKED_TUNE,
    )
    assert hyp["family"] == "detector_intervention"
    assert hyp["label"] == DET_LABEL
    assert hyp["config_hash"] == LOCKED_DET_HASH
    assert hyp["payload"]["is_rejected_overlay"] is False
    assert "12 µm" in hyp["rationale"] or "12" in hyp["rationale"]


def test_preregistered_hashes_are_stable():
    assert LOCKED_ORGDET_FNN_HASH == "95e9300eb84ba71c"
    assert LOCKED_RANK_HASH == "2748c4a2b2823c0a"
    assert config_hash(rank_config_payload()) == LOCKED_RANK_HASH
    assert config_hash(detector_config_payload()) == LOCKED_DET_HASH
    assert LOCKED_RANK_HASH != LOCKED_DET_HASH
    assert len(LOCKED_RANK_HASH) == 16
