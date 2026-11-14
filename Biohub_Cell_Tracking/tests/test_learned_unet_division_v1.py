from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from biohub_tracking.graphs import TrackGraph
from biohub_tracking.learned_division import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LEARNED_LOCKED_HOLDOUT,
    LEARNED_LOCKED_TUNE,
    LOCKED_DIVISION_HASH,
    LOCKED_GATES,
    apply_locked_overlay,
    division_config_hash,
    division_config_payload,
    graph_from_payload,
    holdout_decision,
    n_predicted_forks,
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
SCALE = (1.625, 0.40625, 0.40625)


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
    payload = division_config_payload()
    assert division_config_hash(payload) == LOCKED_DIVISION_HASH
    assert LOCKED_DIVISION_HASH == "8761fb66cc2254f3"
    assert payload["train_config_hash"] == TRAIN_CONFIG_HASH
    assert payload["infer_config_hash"] == LOCKED_INFER_HASH
    assert payload["holdout_config_hash"] == LOCKED_HOLDOUT_HASH
    assert payload["division_config_hash"] == "0fa756963e5d42d1"
    assert payload["retunes_detection"] is False
    assert payload["retunes_division_gates"] is False
    assert payload["uses_intensity"] is False
    assert payload["writes_submission_csv"] is False
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_division_v1.yaml").read_text())
    for key in (
        "max_parent_child_um",
        "min_child_separation_um",
        "max_sibling_alignment",
        "require_parent_predecessor",
        "require_no_alternative_parent",
        "alternative_parent_gate_um",
        "min_displacement_um",
        "max_children",
        "gate_um",
    ):
        assert yaml_cfg[key] == payload[key] == LOCKED_GATES[key]


def test_locked_gates_match_division_v2_priors():
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
        assert raw[key] == lock[key] == LOCKED_GATES[key]


def test_overlay_accepts_anti_aligned_orphan():
    graph = _graph(
        [1, 2, 3, 4],
        [0, 1, 2, 2],
        [8, 8, 8, 8],
        [40, 40, 22, 58],
        [40, 40, 40, 40],
        [(1, 2), (2, 3)],
    )
    out = apply_locked_overlay(graph)
    assert sorted(out.out_edges()[2]) == [3, 4]
    assert out.metadata["n_division_proposals"] == 1
    assert n_predicted_forks(out) == 1
    summary = overlay_summary(graph, out)
    assert summary["n_extra_edges"] == 1


def test_overlay_rejects_parent_without_predecessor():
    graph = _graph(
        [1, 2, 3],
        [0, 1, 1],
        [8, 8, 8],
        [20, 8, 32],
        [20, 20, 20],
        [(1, 2)],
    )
    out = apply_locked_overlay(graph)
    assert out.out_edges()[1] == [2]
    assert out.metadata["n_division_proposals"] == 0
    assert n_predicted_forks(out) == 0


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
    assert tuning_decision(LEARNED_LOCKED_TUNE - 0.01, 1) == "reject_learned_unet_division_v1"
    assert tuning_decision(LEARNED_LOCKED_TUNE, 0) == "keep_as_noop_learned_unet_division_v1"
    assert tuning_decision(LEARNED_LOCKED_TUNE + 0.01, 2) == "keep_candidate_learned_unet_division_v1"
    assert holdout_decision(LEARNED_LOCKED_HOLDOUT - 0.001) == "reject_learned_unet_division_v1"
    assert holdout_decision(LEARNED_LOCKED_HOLDOUT) == "keep_candidate_learned_unet_division_v1"


def test_holdout_requires_keep_and_refuses_second_score(tmp_path: Path):
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit, match="tuning report missing"):
        require_tuning_keep(missing)
    reject = tmp_path / "reject.json"
    reject.write_text(json.dumps({"decision": "reject_learned_unet_division_v1"}))
    with pytest.raises(SystemExit, match="refusing holdout"):
        require_tuning_keep(reject)
    keep = tmp_path / "keep.json"
    keep.write_text(json.dumps({"decision": "keep_as_noop_learned_unet_division_v1"}))
    assert require_tuning_keep(keep)["decision"].startswith("keep_")
    scored = tmp_path / "holdout.json"
    scored.write_text(json.dumps({"status": "holdout_scored", "learned_division_combined": 0.7, "decision": "x"}))
    with pytest.raises(SystemExit, match="score frozen holdout twice"):
        refuse_existing_holdout_score(scored)


def test_run_script_imports_without_kaggle():
    spec = importlib.util.spec_from_file_location(
        "run_learned_unet_division",
        ROOT / "scripts" / "run_learned_unet_division.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.TUNING_REPORT.name == "learned_unet_division_v1_tune.json"
    assert mod.HOLDOUT_REPORT.name == "learned_unet_division_v1_holdout.json"
    assert callable(mod.extract_selected_geffs)
