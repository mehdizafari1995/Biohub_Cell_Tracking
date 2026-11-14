from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biohub_tracking.graphs import TrackGraph
from biohub_tracking.learned_orgdet_error_decomp import (
    INFER_STEMS,
    TUNE_KEEP_THRESHOLD,
    TUNE_MARGIN,
    aggregate_error_rows,
    classify_sequence,
    component_sizes,
    dominant_error_mode,
    recommend_hypothesis,
    refuse_holdout_paths,
    refuse_non_tuning_stems,
)
from biohub_tracking.learned_orgdet_gap_prune import LEARNED_ORGDET_GAP_PRUNE_LOCKED_TUNE

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
    with pytest.raises(ValueError, match="non-tuning"):
        refuse_non_tuning_stems(["6bba_05db0fb1"])
    with pytest.raises(ValueError, match="holdout"):
        refuse_holdout_paths([ROOT / "data" / "train" / "6bba_05db0fb1.geff"])
    refuse_non_tuning_stems(INFER_STEMS)


def test_classify_detection_versus_association_fn():
    # GT: 1-2-3. Pred matches 1 and 3 but misses node 2 (detection) and
    # also fails to link 1-2 even after a substitute (association on 4-5).
    gt = _graph(
        [1, 2, 3],
        [0, 1, 2],
        [0, 0, 0],
        [0, 0, 0],
        [0, 10, 20],
        [(1, 2), (2, 3)],
    )
    pred = _graph(
        [10, 20, 30],
        [0, 1, 2],
        [0, 0, 0],
        [0, 0, 0],
        [0, 80, 20],
        [(10, 20), (20, 30)],
    )
    official = {
        "edge_tp": 0,
        "edge_fp": 0,
        "edge_fn": 2,
        "total_node_ratio": 0.2,
        "node_recall": 0.67,
        "adj_edge_jaccard": 0.5,
        "score": 0.5,
    }
    row = classify_sequence(pred, gt, official)
    assert row["missed_gt_nodes"] == 1
    assert row["detection_fn_edges"] >= 1
    assert row["unmatched_pred_nodes"] >= 1
    assert "not automatic false positives" in row["unmatched_pred_note"]


def test_component_sizes_and_dominant_detection_mode():
    graph = _graph(
        [1, 2, 3, 4],
        [0, 1, 2, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 10, 20, 80],
        [(1, 2), (2, 3)],
    )
    sizes = component_sizes(graph)
    assert sizes[1] == 3
    assert sizes[4] == 1
    mode = dominant_error_mode(
        {
            "detection_fn_edges": 80,
            "association_fn_edges": 20,
            "missed_gt_nodes": 12,
            "unmatched_pred_nodes": 400,
            "unsupported_tpm1_nodes": 50,
            "short_track_le3_nodes": 40,
            "official_tp": 908,
            "official_fp": 229,
            "official_fn": 142,
        }
    )
    assert mode["mode"] == "detection"
    assert mode["preferred_hypothesis_family"] == "temporal_consistency_detection_fusion"
    hyp = recommend_hypothesis(mode)
    assert hyp["label"] == "learned_unet_orgdet_tcfuse_v1"
    assert hyp["is_12um_assignment"] is False
    assert hyp["opened_holdout"] is False
    assert hyp["tune_keep_threshold"] == pytest.approx(LEARNED_ORGDET_GAP_PRUNE_LOCKED_TUNE + TUNE_MARGIN)
    assert TUNE_KEEP_THRESHOLD == pytest.approx(0.6954371804770478)


def test_dominant_association_recommends_short_fragment_filter():
    mode = dominant_error_mode(
        {
            "detection_fn_edges": 27,
            "association_fn_edges": 115,
            "missed_gt_nodes": 16,
            "unmatched_pred_nodes": 95698,
            "unsupported_tpm1_nodes": 0,
            "short_track_le3_nodes": 462,
            "official_tp": 908,
            "official_fp": 229,
            "official_fn": 142,
        }
    )
    assert mode["mode"] == "association"
    assert mode["preferred_hypothesis_family"] == "temporal_consistency_short_fragment_filter"
    hyp = recommend_hypothesis(mode)
    assert hyp["label"] == "learned_unet_orgdet_tcf_v1"
    assert hyp["is_12um_assignment"] is False
    assert hyp["min_component_nodes"] == 8


def test_dominant_t_pred_recommends_filter():
    mode = dominant_error_mode(
        {
            "detection_fn_edges": 10,
            "association_fn_edges": 20,
            "missed_gt_nodes": 0,
            "unmatched_pred_nodes": 5000,
            "unsupported_tpm1_nodes": 2000,
            "short_track_le3_nodes": 800,
            "official_tp": 908,
            "official_fp": 229,
            "official_fn": 142,
        }
    )
    assert mode["mode"] == "t_pred"
    hyp = recommend_hypothesis(mode)
    assert hyp["label"] == "learned_unet_orgdet_tcf_v1"
    assert hyp["linker"] == "orgdet_gap_prune_then_drop_short_fragments"
    assert hyp["min_component_nodes"] == 8
    assert hyp["retunes_consecutive_links"] is False


def test_aggregate_error_rows():
    rows = [
        {
            "n_pred_nodes": 10,
            "n_pred_edges": 8,
            "n_gt_nodes": 4,
            "n_gt_edges": 3,
            "n_matched_gt": 3,
            "missed_gt_nodes": 1,
            "unmatched_pred_nodes": 7,
            "detection_fn_edges": 2,
            "association_fn_edges": 1,
            "identity_switches": 0,
            "degree0_nodes": 1,
            "degree0_matched_gt": 0,
            "unsupported_tpm1_nodes": 2,
            "unsupported_tpm1_matched_gt": 0,
            "short_track_le3_nodes": 3,
            "short_track_le3_matched_gt": 0,
            "gt_divisions": 1,
            "pred_divisions": 0,
            "recovered_divisions": 0,
            "missed_divisions": 1,
            "official_tp_fp_fn": [5, 2, 3],
        },
        {
            "n_pred_nodes": 5,
            "n_pred_edges": 4,
            "n_gt_nodes": 2,
            "n_gt_edges": 1,
            "n_matched_gt": 2,
            "missed_gt_nodes": 0,
            "unmatched_pred_nodes": 3,
            "detection_fn_edges": 0,
            "association_fn_edges": 1,
            "identity_switches": 1,
            "degree0_nodes": 0,
            "degree0_matched_gt": 0,
            "unsupported_tpm1_nodes": 0,
            "unsupported_tpm1_matched_gt": 0,
            "short_track_le3_nodes": 0,
            "short_track_le3_matched_gt": 0,
            "gt_divisions": 1,
            "pred_divisions": 0,
            "recovered_divisions": 0,
            "missed_divisions": 1,
            "official_tp_fp_fn": [2, 1, 1],
        },
    ]
    agg = aggregate_error_rows(rows)
    assert agg["n_pred_nodes"] == 15
    assert agg["detection_fn_edges"] == 2
    assert agg["association_fn_edges"] == 2
    assert agg["official_tp"] == 7
    assert agg["missed_divisions"] == 2
    assert agg["detection_fn_share"] == pytest.approx(0.5)
