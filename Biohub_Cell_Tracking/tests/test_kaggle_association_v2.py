from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

from kaggle_association_v2 import (  # noqa: E402
    COLUMNS,
    GATE_UM,
    INTENSITY_WEIGHT,
    MAX_GAP_FRAMES,
    MAX_LINK_UM,
    _Track,
    appearance_cost,
    associate_pairs,
    audit_submission,
    config_hash,
    config_payload,
    link_detections,
)
from biohub_tracking.association import FrameDetections, track_from_detections
from biohub_tracking.config import load_config
from biohub_tracking.graphs import TrackGraph
from biohub_tracking.submit import graphs_to_submission

_MAKE = importlib.util.spec_from_file_location("make_kaggle_kernel", ROOT / "scripts" / "make_kaggle_kernel.py")
assert _MAKE and _MAKE.loader
_make_mod = importlib.util.module_from_spec(_MAKE)
_MAKE.loader.exec_module(_make_mod)
KERNELS = _make_mod.KERNELS
build_kernel = _make_mod.build_kernel
metadata_payload = _make_mod.metadata_payload


SCALE = (1.625, 0.40625, 0.40625)
SHAPE = (64, 256, 256)


def _valid_df() -> pd.DataFrame:
    graph = TrackGraph(
        dataset="44b6_0113de3b",
        node_id=[1, 2, 3],
        t=[0, 1, 2],
        z=[10, 11, 12],
        y=[20, 21, 22],
        x=[30, 31, 32],
        edges=[(1, 2), (2, 3)],
    )
    return graphs_to_submission([graph])


def test_association_v2_kernel_matches_locked_yaml():
    payload = config_payload()
    locked = yaml.safe_load((ROOT / "configs" / "association_v2.yaml").read_text())
    assert payload["label"] == "association_v2"
    assert payload["linking"] == "association_v2"
    assert payload["anisotropic"] is True
    assert payload["suppress_duplicates"] is True
    assert payload["min_distance_um"] == 4.0
    assert payload["allow_divisions"] is False
    assert payload["interpolate_gaps"] is True
    assert payload["assignment"] == "greedy"
    assert payload["motion_model"] == "none"
    assert payload["max_children"] == 1
    assert MAX_LINK_UM == float(locked["linking"]["max_distance_um"])
    assert GATE_UM == float(locked["linking"]["gate_um"])
    assert MAX_GAP_FRAMES == int(locked["linking"]["max_gap_frames"])
    assert INTENSITY_WEIGHT == float(locked["linking"]["intensity_weight"])
    assert config_hash(payload) == "2c6966b5204ba542"
    assert config_hash(payload) == config_hash(config_payload())
    cfg = load_config(ROOT / "configs" / "association_v2.yaml")
    assert cfg.linking.method == "association_v2"
    assert cfg.linking.gate_um == GATE_UM
    assert cfg.linking.max_gap_frames == MAX_GAP_FRAMES


def test_kernel_gap_close_inserts_interpolated_consecutive_nodes():
    frames = [
        (0, np.array([[8.0, 10.0, 20.0]]), np.array([5.0]), np.array([5.0]), np.array([1])),
        (2, np.array([[8.0, 14.0, 20.0]]), np.array([5.0]), np.array([5.0]), np.array([2])),
    ]
    nodes, edges, next_id, n_interp = link_detections(frames, next_id=3)
    assert n_interp == 1
    assert len(nodes) == 3
    assert len(edges) == 2
    times = sorted(int(n["t"]) for n in nodes)
    assert times == [0, 1, 2]
    assert next_id == 4
    t_by_id = {int(n["node_id"]): int(n["t"]) for n in nodes}
    for edge in edges:
        assert t_by_id[int(edge["target_id"])] == t_by_id[int(edge["source_id"])] + 1


def test_kernel_matches_library_on_gap_and_rejects_far_pairs():
    lib_frames = [
        FrameDetections(0, np.array([[8.0, 10.0, 20.0]]), np.array([5.0]), np.array([5.0]), np.array([1])),
        FrameDetections(2, np.array([[8.0, 14.0, 20.0]]), np.array([5.0]), np.array([5.0]), np.array([2])),
    ]
    cfg = load_config(ROOT / "configs" / "association_v2.yaml")
    graph = track_from_detections("gap", lib_frames, cfg.linking, SCALE, SHAPE)
    nodes, edges, _, n_interp = link_detections(
        [
            (0, np.array([[8.0, 10.0, 20.0]]), np.array([5.0]), np.array([5.0]), np.array([1])),
            (2, np.array([[8.0, 14.0, 20.0]]), np.array([5.0]), np.array([5.0]), np.array([2])),
        ],
        next_id=3,
    )
    assert graph.n_nodes == len(nodes) == 3
    assert graph.n_edges == len(edges) == 2
    assert n_interp == 1

    far = [
        (0, np.array([[8.0, 10.0, 20.0]]), np.array([5.0]), np.array([5.0]), np.array([1])),
        (1, np.array([[8.0, 200.0, 200.0]]), np.array([5.0]), np.array([5.0]), np.array([2])),
    ]
    nodes_far, edges_far, _, n_interp_far = link_detections(far, next_id=3)
    assert len(nodes_far) == 2
    assert edges_far == []
    assert n_interp_far == 0


def test_intensity_cost_prefers_similar_appearance():
    close = appearance_cost(10.0, 10.5, INTENSITY_WEIGHT, GATE_UM)
    far = appearance_cost(10.0, 20.0, INTENSITY_WEIGHT, GATE_UM)
    assert 0 < close < far
    tr = _Track()
    tr.times = [0]
    tr.positions = [np.array([8.0, 10.0, 20.0], dtype=np.float64)]
    tr.intensity = [10.0]
    tr.appearance = [10.0]
    # Identical positions; the similar-intensity candidate must win.
    coords = np.array([[8.0, 11.0, 20.0], [8.0, 11.0, 20.0]], dtype=np.float64)
    appearance = np.array([10.2, 20.0])
    intensity = np.array([10.2, 20.0])
    pairs = associate_pairs([tr], 1, coords, appearance, intensity)
    assert pairs
    assert pairs[0][1] == 0
    assert pairs[0][2] < far + 1.0


def test_audit_accepts_valid_submission():
    df = _valid_df()
    errors = audit_submission(
        df,
        discovered=["44b6_0113de3b"],
        shapes={"44b6_0113de3b": (100, 64, 256, 256)},
        sample_names=["44b6_0113de3b", "44b6_0b24845f"],
    )
    assert errors == []
    assert list(df.columns) == COLUMNS


def test_packaged_kernel_is_private_internet_off(tmp_path, monkeypatch):
    spec = KERNELS["association_v2"]
    meta = metadata_payload(spec)
    assert meta["id"] == "mahdimzafari/biohub-association-v2"
    assert meta["id"] != "mahdimzafari/biohub-baseline-v1"
    assert meta["is_private"] is True
    assert meta["enable_internet"] is False
    assert meta["enable_gpu"] is False
    assert meta["competition_sources"] == ["biohub-cell-tracking-during-development"]

    written = build_kernel("association_v2")
    kernel_meta = json.loads((ROOT / spec["kernel_dir"] / "kernel-metadata.json").read_text())
    assert kernel_meta["is_private"] is True
    assert kernel_meta["enable_internet"] is False
    notebook = json.loads((ROOT / spec["notebook"]).read_text())
    assert notebook["metadata"]["kaggle"]["isInternetEnabled"] is False
    source = (ROOT / spec["source"]).read_text()
    assert "biohub_tracking" not in source
    assert "competitions submit" not in source.lower()
    assert written
