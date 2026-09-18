from __future__ import annotations

import math

import pytest

from biohub_tracking.graphs import TrackGraph
from biohub_tracking.official_eval import (
    OFFICIAL_COMMIT,
    compare_official_and_diagnostic,
    evaluate_official,
    official_scorer_available,
    summarise_official,
)


pytestmark = pytest.mark.skipif(not official_scorer_available(), reason="official scorer source missing")


def _graph(name, nodes, edges, estimated=10.0):
    node_id, t, z, y, x = zip(*nodes)
    return TrackGraph(
        dataset=name,
        node_id=node_id,
        t=t,
        z=z,
        y=y,
        x=x,
        edges=edges,
        estimated_n_nodes=estimated,
    )


def test_official_perfect_match_is_one():
    gt = _graph("gt", [(1, 0, 10, 20, 30), (2, 1, 10, 20, 30)], [(1, 2)], estimated=2)
    pred = gt.copy()
    pred.dataset = "pred"
    result = evaluate_official(pred, gt)
    assert result.edge_tp == 1
    assert result.edge_fp == 0
    assert result.edge_fn == 0
    assert result.edge_jaccard == pytest.approx(1.0)
    assert result.adj_edge_jaccard == pytest.approx(1.0)
    assert result.official_commit == OFFICIAL_COMMIT
    assert result.scorer.startswith("royerlab")


def test_combined_score_uses_adjusted_not_raw_edge_jaccard():
    """The reported combined score is adj_edge_jaccard, not raw edge Jaccard.

    Over-predicting nodes lowers the adjusted term while leaving raw Jaccard
    unchanged when extra edges are among unmatched cells.
    """
    gt = _graph("gt", [(1, 0, 10, 10, 10), (2, 1, 10, 10, 10)], [(1, 2)], estimated=2)
    pred = _graph(
        "pred",
        [
            (1, 0, 10, 10, 10),
            (2, 1, 10, 10, 10),
            (3, 0, 50, 50, 50),
            (4, 1, 50, 50, 50),
        ],
        [(1, 2), (3, 4)],
        estimated=2,
    )
    result = evaluate_official(pred, gt)
    assert result.edge_jaccard == pytest.approx(1.0)
    assert result.edge_fp == 0
    assert result.num_pred_nodes == 4
    expected_adj = max(0.0, 1.0 * (1.0 - 0.1 * (4 - 2) / 2))
    assert result.adj_edge_jaccard == pytest.approx(expected_adj)
    assert result.score == pytest.approx(expected_adj)
    assert result.score < result.edge_jaccard


def test_unmatched_predictions_are_not_edge_false_positives():
    gt = _graph("gt", [(1, 0, 10, 10, 10), (2, 1, 10, 10, 10)], [(1, 2)], estimated=100)
    pred = _graph(
        "pred",
        [
            (1, 0, 10, 10, 10),
            (2, 1, 10, 10, 10),
            (3, 0, 50, 50, 50),
            (4, 1, 50, 50, 50),
        ],
        [(1, 2), (3, 4)],
        estimated=100,
    )
    result = evaluate_official(pred, gt)
    assert result.edge_tp == 1
    assert result.edge_fp == 0
    assert result.edge_fn == 0


def test_extra_edge_beyond_track_end_not_penalized():
    gt = _graph(
        "gt",
        [(1, 0, 0, 0, 0), (2, 1, 0, 0, 1), (3, 2, 0, 0, 2)],
        [(1, 2), (2, 3)],
        estimated=3,
    )
    pred = _graph(
        "pred",
        [(1, 0, 0, 0, 0), (2, 1, 0, 0, 1), (3, 2, 0, 0, 2), (4, 3, 0, 0, 3)],
        [(1, 2), (2, 3), (3, 4)],
        estimated=3,
    )
    result = evaluate_official(pred, gt, max_distance_um=0.5)
    assert result.edge_jaccard == pytest.approx(1.0)


def test_nonconsecutive_time_edge_is_dropped():
    gt = _graph(
        "gt",
        [(1, 0, 0, 0, 0), (2, 1, 0, 0, 0), (3, 2, 0, 0, 0)],
        [(1, 2), (2, 3)],
        estimated=3,
    )
    pred = _graph(
        "pred",
        [(1, 0, 0, 0, 0), (2, 1, 0, 0, 0), (3, 2, 0, 0, 0)],
        [(1, 3)],
        estimated=3,
    )
    result = evaluate_official(pred, gt, max_distance_um=1.0)
    assert result.edge_tp == 0
    assert result.edge_jaccard == pytest.approx(0.0)


def test_scale_is_zyx_physical_microns():
    gt = _graph("gt", [(1, 0, 0, 0, 0), (2, 1, 0, 0, 0)], [(1, 2)], estimated=2)
    pred = _graph("pred", [(1, 0, 4, 0, 0), (2, 1, 4, 0, 0)], [(1, 2)], estimated=2)
    isotropic = evaluate_official(pred, gt, scale_zyx=(1.0, 1.0, 1.0), max_distance_um=15.0)
    assert isotropic.edge_jaccard == pytest.approx(1.0)
    scaled = evaluate_official(pred, gt, scale_zyx=(4.0, 1.0, 1.0), max_distance_um=15.0)
    assert scaled.edge_jaccard == pytest.approx(0.0)


def test_matching_uses_integer_time_index_not_physical_time():
    gt = _graph("gt", [(1, 0, 0, 0, 0), (2, 1, 0, 0, 0)], [(1, 2)], estimated=2)
    pred = _graph("pred", [(1, 5, 0, 0, 0), (2, 6, 0, 0, 0)], [(1, 2)], estimated=2)
    result = evaluate_official(pred, gt, max_distance_um=1.0)
    assert result.edge_jaccard == pytest.approx(0.0)


def test_official_and_diagnostic_agree_on_simple_edges():
    gt = _graph(
        "gt",
        [(1, 0, 0, 0, 0), (2, 0, 0, 40, 0), (3, 1, 0, 0, 0), (4, 1, 0, 40, 0)],
        [(1, 3), (2, 4)],
        estimated=4,
    )
    pred = gt.copy()
    pred.dataset = "pred"
    comparison = compare_official_and_diagnostic(pred, gt)
    assert comparison["official"]["edge_tp"] == comparison["diagnostic"]["edge_tp"] == 2
    assert comparison["official"]["edge_jaccard"] == pytest.approx(comparison["diagnostic"]["edge_jaccard"])


def test_summarise_drops_division_term_when_none_present():
    gt = _graph("gt", [(1, 0, 0, 0, 0), (2, 1, 0, 0, 0)], [(1, 2)], estimated=2)
    pred = gt.copy()
    pred.dataset = "pred"
    row = evaluate_official(pred, gt)
    summary = summarise_official([row])
    assert math.isnan(summary["division_jaccard"])
    assert summary["score"] == pytest.approx(summary["adj_edge_jaccard"])
