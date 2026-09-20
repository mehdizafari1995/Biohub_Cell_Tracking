from __future__ import annotations

import json

import numpy as np
import yaml

from biohub_tracking.config import LinkingConfig, load_config
from biohub_tracking.division import (
    alternative_parent_exists,
    propose_divisions,
    sibling_alignment_cosine,
)
from biohub_tracking.graphs import TrackGraph

SCALE = (1.625, 0.40625, 0.40625)
ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _v2(**overrides) -> LinkingConfig:
    cfg = dict(
        allow_divisions=True,
        max_parent_child_um=8.5,
        min_parent_child_um=0.0,
        min_child_separation_um=8.7,
        max_children=2,
        require_parent_predecessor=True,
        require_no_alternative_parent=True,
        alternative_parent_gate_um=12.0,
        max_sibling_alignment=0.05,
        min_displacement_um=1.0,
        gate_um=12.0,
    )
    cfg.update(overrides)
    return LinkingConfig(**cfg)


def _graph(node_id, t, z, y, x, edges, dataset="toy") -> TrackGraph:
    return TrackGraph(
        dataset=dataset,
        node_id=node_id,
        t=t,
        z=z,
        y=y,
        x=x,
        edges=edges,
    )


def test_v2_config_loads_conservative_flags():
    pipe = load_config(ROOT / "configs" / "division_v2.yaml")
    link = pipe.linking
    assert link.allow_divisions is True
    assert link.require_parent_predecessor is True
    assert link.require_no_alternative_parent is True
    assert link.max_sibling_alignment == 0.05
    assert link.min_child_separation_um == 8.7
    assert 8.5 <= float(link.max_parent_child_um) <= 10.0
    assert link.method == "association_v2"
    dumped = pipe.to_dict()["linking"]
    assert dumped["require_parent_predecessor"] is True
    assert dumped["alternative_parent_gate_um"] == 12.0
    lock = json.loads((ROOT / "docs" / "division_priors.json").read_text())["locked_operating_point"]
    raw = yaml.safe_load((ROOT / "configs" / "division_v2.yaml").read_text())["linking"]
    for key in (
        "max_parent_child_um",
        "min_child_separation_um",
        "max_sibling_alignment",
        "require_parent_predecessor",
        "require_no_alternative_parent",
        "alternative_parent_gate_um",
    ):
        assert raw[key] == lock[key]


def test_sibling_alignment_cosine_opposite_and_parallel():
    parent = np.array([0.0, 0.0, 0.0])
    left = np.array([0.0, -5.0, 0.0])
    right = np.array([0.0, 5.0, 0.0])
    same = np.array([0.0, 6.0, 0.0])
    assert sibling_alignment_cosine(parent, left, right) == -1.0
    assert sibling_alignment_cosine(parent, right, same) > 0.9
    assert sibling_alignment_cosine(parent, np.array([0.0, 0.2, 0.0]), right, min_displacement_um=1.0) is None


def test_v2_rejects_parent_without_predecessor():
    # Same geometry as the v1 happy path, but no incoming edge on the parent.
    graph = _graph(
        [1, 2, 3],
        [0, 1, 1],
        [8, 8, 8],
        [20, 8, 32],
        [20, 20, 20],
        [(1, 2)],
    )
    out = propose_divisions(graph, None, _v2(), SCALE)
    assert out.out_edges()[1] == [2]
    assert out.metadata["n_division_proposals"] == 0
    assert out.metadata["n_division_rejected"]["predecessor"] >= 1


def test_v2_accepts_anti_aligned_orphan_with_predecessor():
    # grandparent 1 -> parent 2 at t=1; children 3 (linked) and 4 (orphan) at t=2.
    # 18 voxels * 0.40625 µm = 7.31 µm parent–child; sibling 36 voxels = 14.6 µm.
    graph = _graph(
        [1, 2, 3, 4],
        [0, 1, 2, 2],
        [8, 8, 8, 8],
        [40, 40, 22, 58],
        [40, 40, 40, 40],
        [(1, 2), (2, 3)],
    )
    out = propose_divisions(graph, None, _v2(), SCALE)
    children = sorted(out.out_edges()[2])
    assert children == [3, 4]
    assert out.metadata["n_division_proposals"] == 1


def test_v2_rejects_aligned_same_side_children():
    graph = _graph(
        [1, 2, 3, 4],
        [0, 1, 2, 2],
        [8, 8, 8, 8],
        [40, 40, 50, 56],
        [40, 40, 40, 40],
        [(1, 2), (2, 3)],
    )
    out = propose_divisions(graph, None, _v2(min_child_separation_um=1.0), SCALE)
    assert 4 not in out.out_edges()[2]
    assert out.metadata["n_division_rejected"]["alignment"] >= 1


def test_v2_rejects_when_another_parent_is_within_gate():
    # Node 5 at t=1 sits next to orphan 4, so 4 is a plausible 1-1 continuation.
    graph = _graph(
        [1, 2, 5, 3, 4],
        [0, 1, 1, 2, 2],
        [8, 8, 8, 8, 8],
        [40, 40, 58, 22, 58],
        [40, 40, 40, 40, 40],
        [(1, 2), (2, 3)],
    )
    out = propose_divisions(graph, None, _v2(), SCALE)
    assert 4 not in out.out_edges().get(2, [])
    assert out.metadata["n_division_rejected"]["alternative_parent"] >= 1


def test_v2_rejects_close_siblings():
    graph = _graph(
        [1, 2, 3, 4],
        [0, 1, 2, 2],
        [8, 8, 8, 8],
        [40, 40, 38, 42],
        [40, 40, 40, 40],
        [(1, 2), (2, 3)],
    )
    out = propose_divisions(graph, None, _v2(), SCALE)
    assert 4 not in out.out_edges()[2]
    assert out.metadata["n_division_rejected"]["sibling_sep"] >= 1


def test_alternative_parent_helper():
    coords = np.array([[8.0, 40.0, 40.0], [8.0, 61.0, 40.0]], dtype=float)
    index = {2: 0, 5: 1}
    child = np.array([8.0, 61.0, 40.0])
    assert alternative_parent_exists(child, 2, [2, 5], index, coords, SCALE, 12.0) is True
    assert alternative_parent_exists(child, 2, [2], index, coords, SCALE, 12.0) is False


def test_v1_defaults_still_attach_without_predecessor():
    graph = _graph(
        [1, 2, 3, 4],
        [0, 1, 1, 1],
        [8, 8, 8, 8],
        [20, 22, 18, 80],
        [20, 20, 20, 20],
        [(1, 2)],
    )
    cfg = LinkingConfig(
        allow_divisions=True,
        division_gap_um=12.0,
        max_parent_child_um=12.0,
        min_child_separation_um=1.0,
        max_children=2,
    )
    out = propose_divisions(graph, None, cfg, SCALE)
    assert len(out.out_edges()[1]) == 2
    assert 4 not in out.out_edges()[1]
