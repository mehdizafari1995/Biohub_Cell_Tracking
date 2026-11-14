from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from biohub_tracking.graphs import TrackGraph
from biohub_tracking.learned_gap import LOCKED_GAP_HASH, LOCKED_GATES as GAP_GATES
from biohub_tracking.learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from biohub_tracking.learned_orgdet import LOCKED_ORGDET_HASH
from biohub_tracking.learned_orgdet_gap import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LOCKED_GATES,
    LOCKED_ORGDET_GAP_HASH,
    LOCKED_ORGDET_HOLDOUT_NODE_COUNTS,
    LOCKED_ORGDET_NODE_COUNTS,
    ORGDET_LOCKED_HOLDOUT,
    ORGDET_LOCKED_TUNE,
    apply_orgdet_gap_overlay,
    ensure_orgdet_tuning_graph,
    holdout_decision,
    orgdet_gap_config_hash,
    orgdet_gap_config_payload,
    overlay_summary,
    refuse_existing_holdout_score,
    refuse_non_holdout_stems,
    refuse_non_orgdet_node_count,
    refuse_non_tuning_stems,
    require_tuning_keep,
    runtime_estimate,
    tuning_decision,
)
from biohub_tracking.learned_orgop import LOCKED_ORGOP_HASH as REJECTED_ORGOP_HASH

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
    payload = orgdet_gap_config_payload()
    assert orgdet_gap_config_hash(payload) == LOCKED_ORGDET_GAP_HASH
    assert LOCKED_ORGDET_GAP_HASH == "ee31f36151325f9a"
    assert payload["train_config_hash"] == TRAIN_CONFIG_HASH
    assert payload["infer_config_hash"] == LOCKED_INFER_HASH
    assert payload["orgdet_config_hash"] == LOCKED_ORGDET_HASH
    assert payload["gap_config_hash"] == LOCKED_GAP_HASH
    assert payload["retunes_detection"] is False
    assert payload["retunes_consecutive_links"] is False
    assert payload["retunes_gap_gates"] is False
    assert payload["uses_intensity"] is False
    assert payload["uses_transformer"] is False
    assert payload["uses_ilp"] is False
    assert payload["writes_submission_csv"] is False
    assert LOCKED_ORGDET_HASH == "80b20589198148b7"
    assert LOCKED_GAP_HASH == "f4d888d190c45a4a"
    assert REJECTED_ORGOP_HASH == "250c5bfacd4a8198"
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_orgdet_gap_v1.yaml").read_text())
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
        "orgdet_config_hash",
        "gap_config_hash",
        "det_threshold",
        "pool_kernel_um",
    ):
        assert yaml_cfg[key] == payload[key]


def test_locked_gates_match_association_v2_and_gap():
    raw = yaml.safe_load((ROOT / "configs" / "association_v2.yaml").read_text())["linking"]
    assert raw["gate_um"] == LOCKED_GATES["gate_um"] == GAP_GATES["gate_um"] == 12.0
    assert raw["max_gap_frames"] == LOCKED_GATES["max_gap_frames"] == 2
    assert raw["interpolate_gaps"] is True
    assert yaml.safe_load((ROOT / "configs" / "learned_unet_orgdet_v1.yaml").read_text())[
        "max_gap_frames"
    ] == 0


def test_overlay_interpolates_one_missing_frame_on_orgdet_leftovers():
    graph = _graph(
        [1, 2, 3, 4],
        [0, 1, 0, 2],
        [8, 8, 8, 8],
        [40, 41, 80, 82],
        [40, 40, 40, 40],
        [(1, 2)],
    )
    out = apply_orgdet_gap_overlay(graph)
    assert (1, 2) in {tuple(row) for row in out.edges.tolist()}
    assert out.metadata["n_gap_links"] == 1
    assert out.metadata["n_interpolated"] == 1
    assert out.metadata["overlay"] == "learned_unet_orgdet_gap_v1"
    assert out.metadata["config_hash"] == LOCKED_ORGDET_GAP_HASH
    assert out.metadata["parent_orgdet_hash"] == LOCKED_ORGDET_HASH
    assert out.metadata["retunes_consecutive_links"] is False
    summary = overlay_summary(graph, out)
    assert summary["n_extra_edges"] == 2
    assert summary["n_extra_nodes"] == 1


def test_overlay_keeps_existing_consecutive_orgdet_edge():
    graph = _graph(
        [1, 2, 3],
        [0, 1, 2],
        [8, 8, 8],
        [40, 41, 80],
        [40, 40, 40],
        [(1, 2)],
    )
    out = apply_orgdet_gap_overlay(graph)
    assert list(map(tuple, out.edges.tolist())) == [(1, 2)]
    assert out.n_nodes == 3
    assert out.metadata["n_gap_links"] == 0


def test_overlay_rejects_beyond_scaled_gate():
    graph = _graph(
        [1, 2],
        [0, 2],
        [8, 8],
        [10, 90],
        [10, 10],
        [],
    )
    out = apply_orgdet_gap_overlay(graph)
    assert out.n_edges == 0
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
    out = apply_orgdet_gap_overlay(graph)
    assert out.metadata["n_gap_links"] == 1
    children = [dst for _src, dst in out.edges.tolist() if int(dst) in {2, 3}]
    assert len(children) == 1


def test_ensure_orgdet_tuning_graph_relinks_empty_orgop():
    graph = _graph(
        [1, 2],
        [0, 1],
        [8, 8],
        [40, 41],
        [40, 40],
        [],
    )
    out = ensure_orgdet_tuning_graph(graph)
    assert out.n_edges == 1
    assert tuple(out.edges[0]) == (1, 2)


def test_ensure_orgdet_refuses_holdout_stem():
    graph = _graph([1], [0], [1], [1], [1], [], dataset="6bba_05db0fb1")
    with pytest.raises(ValueError, match="tuning refuses"):
        ensure_orgdet_tuning_graph(graph)


def test_role_stem_and_node_count_guards():
    refuse_non_tuning_stems(INFER_STEMS)
    refuse_non_holdout_stems(HOLDOUT_STEMS)
    with pytest.raises(ValueError, match="tuning refuses"):
        refuse_non_tuning_stems(["6bba_05db0fb1"])
    with pytest.raises(ValueError, match="holdout refuses"):
        refuse_non_holdout_stems(["44b6_12dfb391"])
    refuse_non_orgdet_node_count(
        {"dataset": "44b6_12dfb391", "node_id": list(range(LOCKED_ORGDET_NODE_COUNTS["44b6_12dfb391"]))},
        "tuning",
    )
    with pytest.raises(ValueError, match="nodes != locked orgdet"):
        refuse_non_orgdet_node_count({"dataset": "44b6_12dfb391", "node_id": [1]}, "tuning")
    assert LOCKED_ORGDET_HOLDOUT_NODE_COUNTS["6bba_05db0fb1"] == 84328


def test_tuning_and_holdout_decisions():
    assert tuning_decision(ORGDET_LOCKED_TUNE, 10) == "reject_learned_unet_orgdet_gap_v1"
    assert tuning_decision(ORGDET_LOCKED_TUNE + 0.01, 0) == "keep_as_noop_learned_unet_orgdet_gap_v1"
    assert tuning_decision(ORGDET_LOCKED_TUNE + 0.01, 2) == "keep_candidate_learned_unet_orgdet_gap_v1"
    assert holdout_decision(ORGDET_LOCKED_HOLDOUT) == "reject_learned_unet_orgdet_gap_v1"
    assert holdout_decision(ORGDET_LOCKED_HOLDOUT + 0.001) == "keep_candidate_learned_unet_orgdet_gap_v1"
    assert ORGDET_LOCKED_HOLDOUT == 0.7052260455024029
    assert ORGDET_LOCKED_TUNE == 0.6795779484295005


def test_holdout_requires_keep_and_refuses_second_score(tmp_path: Path):
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit, match="tuning report missing"):
        require_tuning_keep(missing)
    reject = tmp_path / "reject.json"
    reject.write_text(json.dumps({"decision": "reject_learned_unet_orgdet_gap_v1"}))
    with pytest.raises(SystemExit, match="refusing holdout"):
        require_tuning_keep(reject)
    keep = tmp_path / "keep.json"
    keep.write_text(json.dumps({"decision": "keep_candidate_learned_unet_orgdet_gap_v1"}))
    assert require_tuning_keep(keep)["decision"].startswith("keep_")
    scored = tmp_path / "holdout.json"
    scored.write_text(
        json.dumps({"status": "holdout_scored", "learned_orgdet_gap_combined": 0.7, "decision": "x"})
    )
    with pytest.raises(SystemExit, match="score frozen holdout twice"):
        refuse_existing_holdout_score(scored)


def test_runtime_estimate_under_limit():
    estimate = runtime_estimate()
    assert estimate["under_limit"] is True
    assert estimate["estimate_hours"] < 1.0


def test_run_script_imports_without_kaggle():
    spec = importlib.util.spec_from_file_location(
        "run_learned_unet_orgdet_gap",
        ROOT / "scripts" / "run_learned_unet_orgdet_gap.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.TUNING_REPORT.name == "learned_unet_orgdet_gap_v1_tune.json"
    assert mod.HOLDOUT_REPORT.name == "learned_unet_orgdet_gap_v1_holdout.json"
    assert callable(mod.extract_selected_geffs)
    assert np.isfinite(ORGDET_LOCKED_TUNE)


def test_committed_holdout_report_is_keep_and_refuses_rescore():
    path = ROOT / "reports" / "learned_unet_orgdet_gap_v1_holdout.json"
    payload = json.loads(path.read_text())
    assert payload["status"] == "holdout_scored"
    assert payload["decision"] == "keep_candidate_learned_unet_orgdet_gap_v1"
    assert payload["config_hash"] == LOCKED_ORGDET_GAP_HASH
    assert payload["learned_orgdet_gap_combined"] == 0.7228052416579369
    assert payload["orgdet_locked_reference"] == ORGDET_LOCKED_HOLDOUT
    assert payload["n_extra_edges"] == 10427
    assert payload["official_summary"]["division_tp"] == 0
    assert payload["official_summary"]["division_fp"] == 0
    assert payload["official_summary"]["division_fn"] == 5
    assert payload["is_kaggle_run"] is False
    assert payload["is_leaderboard_score"] is False
    with pytest.raises(SystemExit, match="score frozen holdout twice"):
        refuse_existing_holdout_score(path)
