from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from biohub_tracking.graphs import TrackGraph
from biohub_tracking.learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from biohub_tracking.learned_orgdet import LOCKED_ORGDET_HASH
from biohub_tracking.learned_orgdet_elf import LOCKED_ORGDET_ELF_HASH
from biohub_tracking.learned_orgdet_fnn import LOCKED_ORGDET_FNN_HASH
from biohub_tracking.learned_orgdet_fnn_div import (
    LOCKED_ORGDET_FNN_DIV_HASH,
    apply_locked_fnn_div,
)
from biohub_tracking.learned_orgdet_fnn_reas import (
    DIVISION_CONFIG_HASH,
    HOLDOUT_STEMS,
    INFER_STEMS,
    LOCKED_GATES,
    LOCKED_ORGDET_FNN_REAS_HASH,
    ORGDET_FNN_LOCKED_HOLDOUT,
    ORGDET_FNN_LOCKED_TUNE,
    TUNE_KEEP_THRESHOLD,
    apply_locked_fnn_reas,
    holdout_decision,
    n_predicted_forks,
    orgdet_fnn_reas_config_hash,
    orgdet_fnn_reas_config_payload,
    overlay_summary,
    refuse_existing_holdout_score,
    refuse_non_holdout_stems,
    refuse_non_tuning_stems,
    require_tuning_keep,
    runtime_estimate,
    tuning_decision,
)
from biohub_tracking.learned_orgdet_gap import LOCKED_ORGDET_GAP_HASH
from biohub_tracking.learned_orgdet_gap_prune import LOCKED_ORGDET_GAP_PRUNE_HASH

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


def _claimed_geometry() -> TrackGraph:
    # P at (z=8,y=40,x=40), C1 at y=20, C2 at y=60: parent-child 8.125 µm,
    # sibling sep 16.25 µm, anti-aligned cosine -1.0. Q at y=82 is farther
    # from C2 (8.9375 µm) than P, so the closer-parent gate passes.
    return _graph(
        [1, 2, 3, 4, 5],
        [0, 1, 2, 2, 1],
        [8, 8, 8, 8, 8],
        [40, 40, 20, 60, 82],
        [40, 40, 40, 40, 40],
        [(1, 2), (2, 3), (5, 4)],
    )


def test_payload_matches_yaml_and_locked_hash():
    payload = orgdet_fnn_reas_config_payload()
    assert orgdet_fnn_reas_config_hash(payload) == LOCKED_ORGDET_FNN_REAS_HASH
    assert LOCKED_ORGDET_FNN_REAS_HASH == "39fb20d675e21d0e"
    assert LOCKED_ORGDET_FNN_REAS_HASH != LOCKED_ORGDET_FNN_HASH
    assert LOCKED_ORGDET_FNN_REAS_HASH != LOCKED_ORGDET_FNN_DIV_HASH
    assert payload["train_config_hash"] == TRAIN_CONFIG_HASH
    assert payload["infer_config_hash"] == LOCKED_INFER_HASH
    assert payload["orgdet_config_hash"] == LOCKED_ORGDET_HASH == "80b20589198148b7"
    assert payload["orgdet_gap_config_hash"] == LOCKED_ORGDET_GAP_HASH == "ee31f36151325f9a"
    assert payload["orgdet_gap_prune_config_hash"] == LOCKED_ORGDET_GAP_PRUNE_HASH == "862e3e7f7ac0330e"
    assert payload["orgdet_elf_config_hash"] == LOCKED_ORGDET_ELF_HASH == "292c4f1f968f948d"
    assert payload["orgdet_fnn_config_hash"] == LOCKED_ORGDET_FNN_HASH == "95e9300eb84ba71c"
    assert payload["rejected_fnn_div_hash"] == LOCKED_ORGDET_FNN_DIV_HASH == "e184e6ce5bd90e46"
    assert payload["division_config_hash"] == DIVISION_CONFIG_HASH == "0fa756963e5d42d1"
    assert payload["retunes_detection"] is False
    assert payload["retunes_consecutive_links"] is False
    assert payload["retunes_gap_gates"] is False
    assert payload["retunes_division_gates"] is False
    assert payload["require_no_alternative_parent"] is False
    assert payload["require_closer_than_current_parent"] is True
    assert payload["uses_intensity"] is False
    assert payload["uses_transformer"] is False
    assert payload["uses_ilp"] is False
    assert payload["writes_submission_csv"] is False
    assert payload["assignment"] == "reassign_claimed_fnn_child_when_division_v2_geometry"
    assert payload["tune_keep_threshold"] == pytest.approx(ORGDET_FNN_LOCKED_TUNE + 0.002)
    assert TUNE_KEEP_THRESHOLD == pytest.approx(0.7266526336456757)
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_orgdet_fnn_reas_v1.yaml").read_text())
    for key, value in payload.items():
        assert yaml_cfg[key] == value, key


def test_locked_geometry_matches_division_v2_priors():
    lock = json.loads((ROOT / "docs" / "division_priors.json").read_text())["locked_operating_point"]
    raw = yaml.safe_load((ROOT / "configs" / "division_v2.yaml").read_text())["linking"]
    for key in (
        "max_parent_child_um",
        "min_child_separation_um",
        "max_sibling_alignment",
        "require_parent_predecessor",
        "alternative_parent_gate_um",
    ):
        assert raw[key] == lock[key] == LOCKED_GATES[key]
    assert lock["require_no_alternative_parent"] is True
    assert LOCKED_GATES["require_no_alternative_parent"] is False


def test_overlay_reassigns_closer_claimed_child():
    graph = _claimed_geometry()
    out = apply_locked_fnn_reas(graph)
    assert sorted(out.out_edges()[2]) == [3, 4]
    assert 4 not in out.out_edges().get(5, [])
    assert out.metadata["n_reassigned"] == 1
    assert n_predicted_forks(out) == 1
    summary = overlay_summary(graph, out)
    assert summary["n_reassigned"] == 1
    assert summary["n_edges_before"] == summary["n_edges_after"] == 3
    assert summary["config_hash"] == "39fb20d675e21d0e"
    leftover = apply_locked_fnn_div(graph)
    assert leftover.metadata["n_division_proposals"] == 0
    assert leftover.out_edges()[2] == [3]


def test_overlay_rejects_farther_new_parent():
    graph = _graph(
        [1, 2, 3, 4, 5],
        [0, 1, 2, 2, 1],
        [8, 8, 8, 8, 8],
        [40, 40, 20, 60, 58],
        [40, 40, 40, 40, 40],
        [(1, 2), (2, 3), (5, 4)],
    )
    out = apply_locked_fnn_reas(graph)
    assert out.out_edges()[2] == [3]
    assert out.out_edges()[5] == [4]
    assert out.metadata["n_reassigned"] == 0
    assert out.metadata["n_division_rejected"]["closer"] >= 1


def test_overlay_does_not_attach_unused_child():
    graph = _graph(
        [1, 2, 3, 4],
        [0, 1, 2, 2],
        [8, 8, 8, 8],
        [40, 40, 20, 60],
        [40, 40, 40, 40],
        [(1, 2), (2, 3)],
    )
    out = apply_locked_fnn_reas(graph)
    assert out.out_edges()[2] == [3]
    assert out.metadata["n_reassigned"] == 0
    assert 4 not in {int(dst) for _src, dst in np.asarray(out.edges).reshape(-1, 2)}
    orphan = apply_locked_fnn_div(graph)
    assert sorted(orphan.out_edges()[2]) == [3, 4]


def test_overlay_rejects_parent_without_predecessor():
    graph = _graph(
        [1, 2, 3],
        [0, 1, 1],
        [8, 8, 8],
        [20, 8, 32],
        [20, 20, 20],
        [(1, 2)],
    )
    out = apply_locked_fnn_reas(graph)
    assert out.out_edges()[1] == [2]
    assert out.metadata["n_reassigned"] == 0
    assert n_predicted_forks(out) == 0


def test_role_stem_guards():
    refuse_non_tuning_stems(INFER_STEMS)
    refuse_non_holdout_stems(HOLDOUT_STEMS)
    with pytest.raises(ValueError, match="tuning refuses"):
        refuse_non_tuning_stems(["6bba_05db0fb1"])
    with pytest.raises(ValueError, match="holdout refuses"):
        refuse_non_holdout_stems(["44b6_12dfb391"])


def test_tuning_and_holdout_decisions():
    assert tuning_decision(0.7246526336456757, 2) == "reject_learned_unet_orgdet_fnn_reas_v1"
    assert tuning_decision(0.72600, 2) == "reject_learned_unet_orgdet_fnn_reas_v1"
    assert tuning_decision(TUNE_KEEP_THRESHOLD, 0) == "keep_as_noop_learned_unet_orgdet_fnn_reas_v1"
    assert tuning_decision(TUNE_KEEP_THRESHOLD, 2) == "keep_candidate_learned_unet_orgdet_fnn_reas_v1"
    assert holdout_decision(ORGDET_FNN_LOCKED_HOLDOUT) == "reject_learned_unet_orgdet_fnn_reas_v1"
    assert holdout_decision(ORGDET_FNN_LOCKED_HOLDOUT + 0.001) == "keep_candidate_learned_unet_orgdet_fnn_reas_v1"


def test_holdout_requires_keep_and_refuses_second_score(tmp_path: Path):
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit, match="tuning report missing"):
        require_tuning_keep(missing)
    reject = tmp_path / "reject.json"
    reject.write_text(json.dumps({"decision": "reject_learned_unet_orgdet_fnn_reas_v1"}))
    with pytest.raises(SystemExit, match="refusing holdout"):
        require_tuning_keep(reject)
    keep = tmp_path / "keep.json"
    keep.write_text(json.dumps({"decision": "keep_candidate_learned_unet_orgdet_fnn_reas_v1"}))
    assert require_tuning_keep(keep)["decision"].startswith("keep_")
    scored = tmp_path / "holdout.json"
    scored.write_text(
        json.dumps(
            {
                "status": "holdout_scored",
                "learned_orgdet_fnn_reas_combined": 0.75,
                "decision": "x",
            }
        )
    )
    with pytest.raises(SystemExit, match="score frozen holdout twice"):
        refuse_existing_holdout_score(scored)


def test_runtime_estimate_under_limit():
    est = runtime_estimate()
    assert est["under_limit"] is True
    assert est["is_kaggle_run"] is False
    assert est["is_leaderboard_score"] is False


def test_no_kernel_and_run_script_imports():
    spec = importlib.util.spec_from_file_location(
        "run_learned_unet_orgdet_fnn_reas",
        ROOT / "scripts" / "run_learned_unet_orgdet_fnn_reas.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.TUNING_REPORT.name == "learned_unet_orgdet_fnn_reas_v1_tune.json"
    assert mod.HOLDOUT_REPORT.name == "learned_unet_orgdet_fnn_reas_v1_holdout.json"
    assert callable(mod.extract_selected_geffs)
    make_spec = importlib.util.spec_from_file_location(
        "make_kaggle_kernel_fnn_reas",
        ROOT / "scripts" / "make_kaggle_kernel.py",
    )
    assert make_spec and make_spec.loader
    make = importlib.util.module_from_spec(make_spec)
    make_spec.loader.exec_module(make)
    assert "learned_unet_orgdet_fnn_reas_v1" not in make.KERNELS
    assert "learned_unet_orgdet_fnn_div_v1" not in make.KERNELS
