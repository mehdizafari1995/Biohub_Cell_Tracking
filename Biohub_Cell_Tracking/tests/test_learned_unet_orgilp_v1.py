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
from biohub_tracking.learned_orgdet_gap import LOCKED_ORGDET_GAP_HASH
from biohub_tracking.learned_orgilp import (
    INFER_STEMS,
    LOCKED_GATES,
    LOCKED_ORGILP_HASH,
    MAX_LINK_UM,
    ORGDET_GAP_LOCKED_TUNE,
    ORGILP_KERNEL_SLUG,
    PreflightError,
    apply_orgilp,
    apply_orgilp_payload,
    holdout_decision,
    orgilp_config_hash,
    orgilp_config_payload,
    overlay_summary,
    preflight_orgilp,
    refuse_existing_holdout_score,
    refuse_non_tuning_stems,
    refuse_public_lb_tune_targets,
    require_tuning_keep,
    runtime_estimate,
    spatial_gate_candidates,
    spatial_prob,
    tuning_decision,
)
from biohub_tracking.learned_orgop import LOCKED_ORGOP_HASH as REJECTED_ORGOP_HASH
from biohub_tracking.learned_split import HoldoutLeakError

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
    payload = orgilp_config_payload()
    assert orgilp_config_hash(payload) == LOCKED_ORGILP_HASH
    assert LOCKED_ORGILP_HASH == "dbc6134cddf2048d"
    assert payload["train_config_hash"] == TRAIN_CONFIG_HASH == "4c38035d6657f2ad"
    assert payload["infer_config_hash"] == LOCKED_INFER_HASH == "d7578b5f64ddac08"
    assert payload["rejected_orgop_hash"] == REJECTED_ORGOP_HASH == "250c5bfacd4a8198"
    assert payload["orgdet_config_hash"] == LOCKED_ORGDET_HASH == "80b20589198148b7"
    assert payload["orgdet_gap_config_hash"] == LOCKED_ORGDET_GAP_HASH == "ee31f36151325f9a"
    assert payload["uses_transformer"] is False
    assert payload["uses_ilp"] is True
    assert payload["retunes_detection"] is False
    assert payload["writes_submission_csv"] is False
    assert payload["candidate_source"] == "spatial_gate_12um"
    assert payload["edge_threshold"] == 0.0
    assert payload["linker"] == "tracksdata_ilp"
    assert payload["rejection_threshold"] == ORGDET_GAP_LOCKED_TUNE == 0.6872212709981039
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_orgilp_v1.yaml").read_text())
    for key, value in payload.items():
        assert yaml_cfg[key] == value, key


def test_locked_gates_are_spatial_not_softmax():
    assert LOCKED_GATES["gate_um"] == MAX_LINK_UM == 12.0
    assert LOCKED_GATES["max_gap_frames"] == 0
    assert LOCKED_GATES["edge_threshold"] == 0.0
    assert LOCKED_GATES["uses_transformer"] is False
    assert LOCKED_GATES["candidate_source"] == "spatial_gate_12um"
    assert "softmax" not in orgilp_config_payload()["edge_activation"]


def test_spatial_candidates_keep_in_gate_and_drop_beyond():
    cands = spatial_gate_candidates(
        [1, 2, 3],
        [0, 1, 1],
        [8, 8, 8],
        [40, 41, 80],
        [40, 40, 40],
    )
    pairs = {(int(s), int(d)) for s, d, _p, _dist in cands}
    assert (1, 2) in pairs
    assert (1, 3) not in pairs
    close = [row for row in cands if row[0] == 1 and row[1] == 2][0]
    assert close[3] < 12.0
    assert close[2] == spatial_prob(close[3])
    assert close[2] > 0.5


def test_preflight_fails_closed_on_zero_detections_and_candidates():
    with pytest.raises(PreflightError, match="zero_detections"):
        preflight_orgilp({"dataset": "44b6_12dfb391", "node_id": [], "t": [], "z": [], "y": [], "x": []})
    far = {
        "dataset": "44b6_12dfb391",
        "node_id": [1, 2],
        "t": [0, 1],
        "z": [8, 8],
        "y": [10, 200],
        "x": [10, 200],
    }
    with pytest.raises(PreflightError, match="zero_candidate_edges"):
        preflight_orgilp(far)
    with pytest.raises(PreflightError, match="holdout_stem"):
        preflight_orgilp(
            {
                "dataset": "6bba_05db0fb1",
                "node_id": [1, 2],
                "t": [0, 1],
                "z": [8, 8],
                "y": [40, 41],
                "x": [40, 40],
            }
        )


def test_preflight_fails_closed_on_skipped_ilp():
    payload = {
        "dataset": "44b6_12dfb391",
        "node_id": [1, 2],
        "t": [0, 1],
        "z": [8, 8],
        "y": [40, 41],
        "x": [40, 40],
    }
    cands = spatial_gate_candidates(
        payload["node_id"], payload["t"], payload["z"], payload["y"], payload["x"]
    )
    with pytest.raises(PreflightError, match="ilp_skipped_no_candidates"):
        preflight_orgilp(
            payload,
            candidates=cands,
            solved={"solver_status": "ilp_skipped_no_candidates", "edges": []},
        )


def test_orgilp_keeps_close_spatial_edge_and_refuses_holdout():
    pytest.importorskip("tracksdata")
    payload = {
        "dataset": "44b6_12dfb391",
        "node_id": [1, 2, 3],
        "t": [0, 1, 1],
        "z": [8, 8, 8],
        "y": [40, 41, 80],
        "x": [40, 40, 40],
        "edges": [],
    }
    solved = apply_orgilp_payload(payload)
    assert solved["solver_status"] == "tracksdata_ilp"
    assert solved["n_candidate_edges"] >= 1
    assert [1, 2] in solved["edges"]
    assert [1, 3] not in solved["edges"]
    graph = apply_orgilp(
        _graph([1, 2, 3], [0, 1, 1], [8, 8, 8], [40, 41, 80], [40, 40, 40], [])
    )
    assert graph.n_nodes == 3
    assert graph.n_edges >= 1
    assert tuple(graph.edges[0]) == (1, 2)
    assert graph.metadata["config_hash"] == LOCKED_ORGILP_HASH
    assert graph.metadata["uses_transformer"] is False
    summary = overlay_summary(
        _graph([1, 2, 3], [0, 1, 1], [8, 8, 8], [40, 41, 80], [40, 40, 40], []),
        graph,
    )
    assert summary["detection_counts_match"] is True
    with pytest.raises(HoldoutLeakError):
        apply_orgilp_payload({**payload, "dataset": "6bba_05db0fb1"})


def test_role_and_lb_guards():
    refuse_non_tuning_stems(INFER_STEMS)
    with pytest.raises(HoldoutLeakError, match="tuning refuses"):
        refuse_non_tuning_stems(["6bba_05db0fb1"])
    with pytest.raises(ValueError, match="0.553"):
        refuse_public_lb_tune_targets(0.553)
    with pytest.raises(ValueError, match="0.612"):
        refuse_public_lb_tune_targets("0.612")
    refuse_public_lb_tune_targets(ORGDET_GAP_LOCKED_TUNE)


def test_preregistered_rejection_threshold():
    assert tuning_decision(ORGDET_GAP_LOCKED_TUNE) == "reject_learned_unet_orgilp_v1"
    assert tuning_decision(ORGDET_GAP_LOCKED_TUNE + 1e-9) == "keep_candidate_learned_unet_orgilp_v1"
    assert holdout_decision(0.722805) == "reject_learned_unet_orgilp_v1"
    assert holdout_decision(0.722806) == "keep_candidate_learned_unet_orgilp_v1"


def test_holdout_requires_keep_and_refuses_second_score(tmp_path: Path):
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit, match="tuning report missing"):
        require_tuning_keep(missing)
    reject = tmp_path / "reject.json"
    reject.write_text(json.dumps({"decision": "reject_learned_unet_orgilp_v1"}))
    with pytest.raises(SystemExit, match="refusing holdout"):
        require_tuning_keep(reject)
    keep = tmp_path / "keep.json"
    keep.write_text(json.dumps({"decision": "keep_candidate_learned_unet_orgilp_v1"}))
    assert require_tuning_keep(keep)["decision"].startswith("keep_")
    scored = tmp_path / "holdout.json"
    scored.write_text(json.dumps({"status": "holdout_scored", "learned_combined": 0.7, "decision": "x"}))
    with pytest.raises(SystemExit, match="twice"):
        refuse_existing_holdout_score(scored)


def test_tuning_report_is_reject_and_not_a_leaderboard_score():
    report = json.loads((ROOT / "reports" / "learned_unet_orgilp_v1_tune.json").read_text())
    assert report["decision"] == "reject_learned_unet_orgilp_v1"
    assert report["config_hash"] == LOCKED_ORGILP_HASH == "dbc6134cddf2048d"
    assert report["is_kaggle_run"] is False
    assert report["is_leaderboard_score"] is False
    assert report["learned_combined"] == 0.6768108254815931
    assert report["orgdet_gap_locked_tune"] == ORGDET_GAP_LOCKED_TUNE
    assert report["learned_combined"] <= ORGDET_GAP_LOCKED_TUNE
    assert report["overlays"][0]["n_candidate_edges"] == 333502
    assert report["overlays"][1]["n_candidate_edges"] == 133546
    assert report["overlays"][0]["n_ilp_edges"] == 58805
    assert report["overlays"][1]["n_ilp_edges"] == 26045
    assert all(row["solver_status"] == "tracksdata_ilp" for row in report["overlays"])
    assert report["official_summary"]["division_tp"] == 0
    assert report["official_summary"]["division_fn"] == 2


def test_kernel_slug_is_new_and_does_not_overwrite_locked():
    spec = importlib.util.spec_from_file_location(
        "push_kaggle_kernel", ROOT / "scripts" / "push_kaggle_kernel.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert ORGILP_KERNEL_SLUG == "mahdimzafari/biohub-learned-unet-orgilp-v1"
    locked = {
        "mahdimzafari/biohub-learned-unet-orgop-v1",
        "mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1",
        "mahdimzafari/biohub-learned-unet-orgdet-test-v1",
        "mahdimzafari/biohub-learned-unet-v1",
    }
    assert locked <= mod.FORBIDDEN_SLUGS
    assert ORGILP_KERNEL_SLUG not in locked
    estimate = runtime_estimate()
    assert estimate["under_limit"] is True
    assert estimate["n_volumes"] == 2


def test_orgop_ilp_wrapper_still_refuses_holdout_by_default():
    from biohub_tracking.learned_orgop import apply_tracksdata_ilp

    payload = {
        "dataset": "6bba_05db0fb1",
        "node_id": [1, 2],
        "t": [0, 1],
        "z": [0, 0],
        "y": [0, 0],
        "x": [0, 1],
        "candidate_edges": [[1, 2, 0.9, 1.0]],
    }
    with pytest.raises(HoldoutLeakError):
        apply_tracksdata_ilp(payload)
