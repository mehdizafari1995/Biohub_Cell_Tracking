from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from biohub_tracking.graphs import TrackGraph
from biohub_tracking.learned_gap import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LEARNED_LOCKED_HOLDOUT,
    LEARNED_LOCKED_TUNE,
    LOCKED_GAP_HASH,
    LOCKED_GATES,
    apply_locked_overlay,
    gap_config_hash,
    gap_config_payload,
    graph_from_payload,
    holdout_decision,
    overlay_summary,
    refuse_existing_holdout_score,
    refuse_non_holdout_stems,
    refuse_non_tuning_stems,
    require_tuning_keep,
    tuning_decision,
)
from biohub_tracking.learned_holdout import LOCKED_HOLDOUT_HASH
from biohub_tracking.learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH

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


def test_payload_matches_yaml_and_locked_hash():
    payload = gap_config_payload()
    assert gap_config_hash(payload) == LOCKED_GAP_HASH
    assert LOCKED_GAP_HASH == "f4d888d190c45a4a"
    assert payload["train_config_hash"] == TRAIN_CONFIG_HASH
    assert payload["infer_config_hash"] == LOCKED_INFER_HASH
    assert payload["holdout_config_hash"] == LOCKED_HOLDOUT_HASH
    assert payload["retunes_detection"] is False
    assert payload["retunes_consecutive_links"] is False
    assert payload["uses_intensity"] is False
    assert payload["writes_submission_csv"] is False
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_gap_v1.yaml").read_text())
    for key in (
        "gate_um",
        "max_gap_frames",
        "gap_scale",
        "assignment",
        "interpolate_gaps",
        "max_parents_per_node",
        "max_children_per_node",
        "intensity_weight",
        "linker",
    ):
        assert yaml_cfg[key] == payload[key] == LOCKED_GATES.get(key, payload[key])


def test_locked_gates_match_association_v2():
    raw = yaml.safe_load((ROOT / "configs" / "association_v2.yaml").read_text())["linking"]
    assert raw["gate_um"] == LOCKED_GATES["gate_um"] == 12.0
    assert raw["max_gap_frames"] == LOCKED_GATES["max_gap_frames"] == 2
    assert raw["interpolate_gaps"] is True
    assert raw["assignment"] == "greedy"
    assert yaml.safe_load((ROOT / "configs" / "learned_unet_infer_v1.yaml").read_text())[
        "max_gap_frames"
    ] == 0


def test_overlay_interpolates_one_missing_frame():
    graph = _graph(
        [1, 2],
        [0, 2],
        [8, 8],
        [40, 42],
        [40, 40],
        [],
    )
    out = apply_locked_overlay(graph)
    assert out.n_nodes == 3
    assert out.n_edges == 2
    assert sorted(int(t) for t in out.t) == [0, 1, 2]
    assert out.metadata["n_gap_links"] == 1
    assert out.metadata["n_interpolated"] == 1
    summary = overlay_summary(graph, out)
    assert summary["n_extra_edges"] == 2
    assert summary["n_extra_nodes"] == 1


def test_overlay_keeps_existing_consecutive_edge():
    graph = _graph(
        [1, 2, 3],
        [0, 1, 2],
        [8, 8, 8],
        [40, 41, 80],
        [40, 40, 40],
        [(1, 2)],
    )
    out = apply_locked_overlay(graph)
    assert list(map(tuple, out.edges.tolist())) == [(1, 2)]
    assert out.n_nodes == 3
    assert out.metadata["n_gap_links"] == 0
    assert out.metadata["retunes_consecutive_links"] is False


def test_overlay_rejects_beyond_scaled_gate():
    # 80 y-voxels * 0.40625 µm = 32.5 µm > 12 * sqrt(2) ≈ 16.97 µm
    graph = _graph(
        [1, 2],
        [0, 2],
        [8, 8],
        [10, 90],
        [10, 10],
        [],
    )
    out = apply_locked_overlay(graph)
    assert out.n_edges == 0
    assert out.n_nodes == 2
    assert out.metadata["n_gap_links"] == 0


def test_overlay_is_one_to_one():
    graph = _graph(
        [1, 2, 3],
        [0, 2, 2],
        [8, 8, 8],
        [40, 41, 42],
        [40, 40, 40],
        [],
    )
    out = apply_locked_overlay(graph)
    assert out.metadata["n_gap_links"] == 1
    children = [dst for _src, dst in out.edges.tolist() if int(dst) in {2, 3}]
    assert len(children) == 1


def test_graph_from_payload_roundtrip():
    payload = {
        "dataset": "44b6_12dfb391",
        "node_id": [1, 2],
        "t": [0, 1],
        "z": [1, 1],
        "y": [2, 3],
        "x": [4, 5],
        "edges": [[1, 2]],
    }
    graph = graph_from_payload(payload)
    assert graph.n_nodes == 2
    assert graph.n_edges == 1
    assert graph.dataset == "44b6_12dfb391"


def test_role_stem_guards():
    refuse_non_tuning_stems(INFER_STEMS)
    refuse_non_holdout_stems(HOLDOUT_STEMS)
    with pytest.raises(ValueError, match="tuning refuses"):
        refuse_non_tuning_stems(["6bba_05db0fb1"])
    with pytest.raises(ValueError, match="holdout refuses"):
        refuse_non_holdout_stems(["44b6_12dfb391"])


def test_tuning_and_holdout_decisions():
    assert tuning_decision(LEARNED_LOCKED_TUNE - 0.01, 1) == "reject_learned_unet_gap_v1"
    assert tuning_decision(LEARNED_LOCKED_TUNE, 0) == "keep_as_noop_learned_unet_gap_v1"
    assert tuning_decision(LEARNED_LOCKED_TUNE + 0.01, 2) == "keep_candidate_learned_unet_gap_v1"
    assert holdout_decision(LEARNED_LOCKED_HOLDOUT - 0.001) == "reject_learned_unet_gap_v1"
    assert holdout_decision(LEARNED_LOCKED_HOLDOUT) == "keep_candidate_learned_unet_gap_v1"


def test_holdout_requires_keep_and_refuses_second_score(tmp_path: Path):
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit, match="tuning report missing"):
        require_tuning_keep(missing)
    reject = tmp_path / "reject.json"
    reject.write_text(json.dumps({"decision": "reject_learned_unet_gap_v1"}))
    with pytest.raises(SystemExit, match="refusing holdout"):
        require_tuning_keep(reject)
    keep = tmp_path / "keep.json"
    keep.write_text(json.dumps({"decision": "keep_as_noop_learned_unet_gap_v1"}))
    assert require_tuning_keep(keep)["decision"].startswith("keep_")
    scored = tmp_path / "holdout.json"
    scored.write_text(json.dumps({"status": "holdout_scored", "learned_gap_combined": 0.7, "decision": "x"}))
    with pytest.raises(SystemExit, match="score frozen holdout twice"):
        refuse_existing_holdout_score(scored)


def test_run_script_imports_without_kaggle():
    spec = importlib.util.spec_from_file_location(
        "run_learned_unet_gap",
        ROOT / "scripts" / "run_learned_unet_gap.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.TUNING_REPORT.name == "learned_unet_gap_v1_tune.json"
    assert mod.HOLDOUT_REPORT.name == "learned_unet_gap_v1_holdout.json"
    assert callable(mod.extract_selected_geffs)
    assert np.isfinite(LEARNED_LOCKED_TUNE)
