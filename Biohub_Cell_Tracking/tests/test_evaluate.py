import numpy as np

from biohub_tracking.evaluate import evaluate_graph
from biohub_tracking.graphs import TrackGraph


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


def test_perfect_linear_track():
    gt = _graph(
        "gt",
        [(1, 0, 10, 20, 30), (2, 1, 10, 20, 30)],
        [(1, 2)],
        estimated=2,
    )
    pred = gt.copy()
    pred.dataset = "pred"
    metrics = evaluate_graph(pred, gt)
    assert metrics.edge_tp == 1
    assert metrics.edge_fp == 0
    assert metrics.edge_fn == 0
    assert metrics.edge_jaccard == 1.0
    assert metrics.adj_edge_jaccard == 1.0
    assert metrics.missed_detections == 0


def test_identity_switch_is_fn_and_fp():
    gt = _graph(
        "gt",
        [
            (1, 0, 0, 0, 0),
            (2, 0, 0, 40, 0),
            (3, 1, 0, 0, 0),
            (4, 1, 0, 40, 0),
        ],
        [(1, 3), (2, 4)],
        estimated=4,
    )
    pred = _graph(
        "pred",
        [
            (1, 0, 0, 0, 0),
            (2, 0, 0, 40, 0),
            (3, 1, 0, 0, 0),
            (4, 1, 0, 40, 0),
        ],
        [(1, 4), (2, 3)],
        estimated=4,
    )
    metrics = evaluate_graph(pred, gt)
    assert metrics.edge_tp == 0
    assert metrics.edge_fn == 2
    assert metrics.identity_switches == 2


def test_unmatched_extra_nodes_are_not_edge_fp():
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
    metrics = evaluate_graph(pred, gt)
    assert metrics.edge_tp == 1
    assert metrics.edge_fp == 0
    assert metrics.edge_fn == 0


def test_filter_timepoints_scales_estimated_count():
    gt = _graph(
        "gt",
        [(1, 0, 10, 10, 10), (2, 1, 10, 10, 10), (3, 2, 10, 10, 10)],
        [(1, 2), (2, 3)],
        estimated=300,
    )
    sliced = gt.filter_timepoints([0, 1])
    assert sliced.n_nodes == 2
    assert sliced.n_edges == 1
    assert sliced.estimated_n_nodes == 200


def test_division_proxy_true_positive():
    gt = _graph(
        "gt",
        [
            (1, 0, 10, 10, 10),
            (2, 1, 10, 8, 10),
            (3, 1, 10, 12, 10),
        ],
        [(1, 2), (1, 3)],
        estimated=3,
    )
    pred = gt.copy()
    pred.dataset = "pred"
    metrics = evaluate_graph(pred, gt)
    assert metrics.division_tp == 1
    assert metrics.division_fn == 0
