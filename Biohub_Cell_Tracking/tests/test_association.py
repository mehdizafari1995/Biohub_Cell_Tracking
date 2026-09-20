from __future__ import annotations

import numpy as np

from biohub_tracking.association import FrameDetections, track_from_detections
from biohub_tracking.config import LinkingConfig
from biohub_tracking.division import propose_divisions


SCALE = (1.625, 0.40625, 0.40625)
SHAPE = (64, 256, 256)


def _frames_linear() -> list[FrameDetections]:
    frames = []
    nid = 1
    for t, y in enumerate([10.0, 12.0, 14.0, 16.0]):
        coords = np.array([[8.0, y, 20.0]], dtype=float)
        frames.append(
            FrameDetections(
                t=t,
                coords=coords,
                intensity=np.array([10.0]),
                appearance=np.array([10.0]),
                node_ids=np.array([nid], dtype=np.int64),
            )
        )
        nid += 1
    return frames


def test_constant_velocity_prefers_predicted_cell():
    frames = []
    # track A moving +y, distractor sitting still
    coords = [
        np.array([[8.0, 10.0, 20.0], [8.0, 40.0, 20.0]], dtype=float),
        np.array([[8.0, 14.0, 20.0], [8.0, 40.0, 20.0]], dtype=float),
        np.array([[8.0, 18.0, 20.0], [8.0, 16.0, 20.0]], dtype=float),
    ]
    nid = 1
    for t, c in enumerate(coords):
        n = len(c)
        frames.append(
            FrameDetections(
                t=t,
                coords=c,
                intensity=np.ones(n),
                appearance=np.ones(n),
                node_ids=np.arange(nid, nid + n, dtype=np.int64),
            )
        )
        nid += n
    cfg = LinkingConfig(method="association_v2", motion_model="constant_velocity", assignment="greedy", max_distance_um=20.0)
    graph = track_from_detections("toy", frames, cfg, SCALE, SHAPE)
    out = graph.out_edges()
    # node 1 (t0 cell A) should continue to node 3 then node 5
    assert 3 in out.get(1, [])
    assert 5 in out.get(3, [])


def test_gap_closing_inserts_interpolated_node():
    frames = [
        FrameDetections(0, np.array([[8.0, 10.0, 20.0]]), np.array([5.0]), np.array([5.0]), np.array([1])),
        FrameDetections(2, np.array([[8.0, 14.0, 20.0]]), np.array([5.0]), np.array([5.0]), np.array([2])),
    ]
    cfg = LinkingConfig(method="association_v2", max_gap_frames=1, interpolate_gaps=True, max_distance_um=20.0)
    graph = track_from_detections("gap", frames, cfg, SCALE, SHAPE)
    assert graph.n_nodes == 3
    assert graph.n_edges == 2
    times = sorted(int(t) for t in graph.t)
    assert times == [0, 1, 2]


def test_gated_hungarian_does_not_assign_beyond_gate():
    frames = [
        FrameDetections(0, np.array([[8.0, 10.0, 20.0]]), np.array([5.0]), np.array([5.0]), np.array([1])),
        FrameDetections(1, np.array([[8.0, 200.0, 200.0]]), np.array([5.0]), np.array([5.0]), np.array([2])),
    ]
    cfg = LinkingConfig(
        method="association_v2",
        assignment="gated_hungarian",
        max_distance_um=15.0,
    )
    graph = track_from_detections("far", frames, cfg, SCALE, SHAPE)
    assert graph.n_edges == 0


def test_division_adds_second_child_and_not_a_third():
    # parent 1 at t=0, children 2 and 3 at t=1, extra 4 should not attach after cap
    from biohub_tracking.graphs import TrackGraph

    graph = TrackGraph(
        dataset="div",
        node_id=[1, 2, 3, 4],
        t=[0, 1, 1, 1],
        z=[8, 8, 8, 8],
        y=[20, 22, 18, 80],
        x=[20, 20, 20, 20],
        edges=[(1, 2)],
    )
    cfg = LinkingConfig(
        allow_divisions=True,
        division_gap_um=12.0,
        max_parent_child_um=12.0,
        min_child_separation_um=1.0,
        max_children=2,
    )
    out = propose_divisions(graph, None, cfg, SCALE)
    children = out.out_edges()[1]
    assert len(children) == 2
    assert 4 not in children
    # same child cannot be claimed by two parents
    graph2 = TrackGraph(
        dataset="div2",
        node_id=[1, 2, 3, 4],
        t=[0, 0, 1, 1],
        z=[8, 8, 8, 8],
        y=[20, 21, 20, 22],
        x=[20, 20, 20, 20],
        edges=[(1, 3), (2, 3)],
    )
    # 3 already has two incoming conceptually; propose should not steal for a second parent
    # rebuild as one-child parents competing for free node 4
    graph3 = TrackGraph(
        dataset="div3",
        node_id=[1, 2, 3, 4],
        t=[0, 0, 1, 1],
        z=[8, 8, 8, 8],
        y=[20, 24, 20, 24],
        x=[20, 20, 20, 20],
        edges=[(1, 3)],
    )
    out3 = propose_divisions(graph3, None, cfg, SCALE)
    claimed = []
    for parent, ch in out3.out_edges().items():
        claimed.extend(ch)
    assert len(claimed) == len(set(claimed))


def test_oracle_gt_centroids_accept_association_v2():
    from biohub_tracking.config import load_config
    from biohub_tracking.graphs import TrackGraph
    from biohub_tracking.oracle import link_gt_centroids
    from pathlib import Path

    gt = TrackGraph(
        dataset="oracle_gap",
        node_id=[1, 2],
        t=[0, 2],
        z=[8, 8],
        y=[10, 14],
        x=[20, 20],
        edges=[(1, 2)],
    )
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "association_v2.yaml")
    oracle = link_gt_centroids(gt, cfg.linking, SCALE)
    assert oracle.n_nodes >= 2
    assert oracle.metadata.get("oracle") is True
    times = sorted(int(t) for t in oracle.t)
    assert 0 in times and 2 in times

