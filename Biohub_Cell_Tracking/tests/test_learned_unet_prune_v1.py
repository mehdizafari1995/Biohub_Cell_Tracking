from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from biohub_tracking.graphs import TrackGraph
from biohub_tracking.learned_gap import (
    LEARNED_GAP_LOCKED_HOLDOUT,
    LOCKED_GAP_HASH,
    apply_locked_overlay as apply_gap_overlay,
)
from biohub_tracking.learned_holdout import LOCKED_HOLDOUT_HASH
from biohub_tracking.learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from biohub_tracking.learned_prune import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LEARNED_GAP_LOCKED_TUNE,
    LEARNED_PRUNE_LOCKED_HOLDOUT,
    LEARNED_PRUNE_LOCKED_TUNE,
    LOCKED_GATES,
    LOCKED_PRUNE_HASH,
    apply_locked_overlay,
    drop_degree0,
    graph_from_payload,
    holdout_decision,
    overlay_summary,
    prune_config_hash,
    prune_config_payload,
    refuse_existing_holdout_score,
    refuse_non_holdout_stems,
    refuse_non_tuning_stems,
    require_tuning_keep,
    runtime_estimate,
    tuning_decision,
)

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
    payload = prune_config_payload()
    assert prune_config_hash(payload) == LOCKED_PRUNE_HASH
    assert LOCKED_PRUNE_HASH == "115753ea16aaf021"
    assert LOCKED_PRUNE_HASH != LOCKED_GAP_HASH
    assert payload["train_config_hash"] == TRAIN_CONFIG_HASH
    assert payload["infer_config_hash"] == LOCKED_INFER_HASH
    assert payload["holdout_config_hash"] == LOCKED_HOLDOUT_HASH
    assert payload["gap_config_hash"] == LOCKED_GAP_HASH == "f4d888d190c45a4a"
    assert payload["retunes_detection"] is False
    assert payload["retunes_consecutive_links"] is False
    assert payload["retunes_gap_gates"] is False
    assert payload["uses_intensity"] is False
    assert payload["uses_transformer"] is False
    assert payload["drop_degree0"] is True
    assert payload["writes_submission_csv"] is False
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_prune_v1.yaml").read_text())
    for key in (
        "gate_um",
        "max_gap_frames",
        "gap_scale",
        "assignment",
        "interpolate_gaps",
        "drop_degree0",
        "max_parents_per_node",
        "max_children_per_node",
        "intensity_weight",
        "linker",
    ):
        assert yaml_cfg[key] == payload[key] == LOCKED_GATES.get(key, payload[key])


def test_locked_gates_match_gap_overlay():
    raw = yaml.safe_load((ROOT / "configs" / "learned_unet_gap_v1.yaml").read_text())
    assert raw["gate_um"] == LOCKED_GATES["gate_um"] == 12.0
    assert raw["max_gap_frames"] == LOCKED_GATES["max_gap_frames"] == 2
    assert raw["interpolate_gaps"] is True
    assert raw["assignment"] == "greedy"
    assert raw["gap_scale"] == "sqrt_dt"


def test_drop_degree0_keeps_linked_nodes_and_edges():
    graph = _graph(
        [1, 2, 3],
        [0, 1, 1],
        [8, 8, 8],
        [40, 41, 90],
        [40, 40, 40],
        [(1, 2)],
    )
    out = drop_degree0(graph)
    assert out.n_nodes == 2
    assert set(int(x) for x in out.node_id) == {1, 2}
    assert list(map(tuple, out.edges.tolist())) == [(1, 2)]
    assert out.metadata["n_dropped_isolates"] == 1


def test_overlay_drops_isolate_after_gap():
    # Node 3 is far from the gap pair, so it stays unmatched and is dropped.
    graph = _graph(
        [1, 2, 3],
        [0, 2, 1],
        [8, 8, 8],
        [40, 42, 200],
        [40, 40, 200],
        [],
    )
    gapped = apply_gap_overlay(graph)
    assert gapped.n_nodes == 4  # interpolated midpoint for 1->2
    assert gapped.n_edges == 2
    out = apply_locked_overlay(graph)
    assert 3 not in set(int(x) for x in out.node_id)
    assert out.n_edges == gapped.n_edges == 2
    assert out.metadata["n_dropped_isolates"] == 1
    assert out.metadata["n_gap_links"] == 1
    assert out.metadata["parent_gap_hash"] == LOCKED_GAP_HASH
    assert out.metadata["retunes_gap_gates"] is False
    summary = overlay_summary(graph, out)
    assert summary["n_dropped_isolates"] == 1
    assert summary["n_extra_edges"] == 2


def test_overlay_keeps_gap_linked_former_isolate():
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
    assert out.metadata["n_dropped_isolates"] == 0
    assert out.metadata["n_interpolated"] == 1


def test_overlay_keeps_existing_consecutive_edge():
    graph = _graph(
        [1, 2, 3],
        [0, 1, 1],
        [8, 8, 8],
        [40, 41, 200],
        [40, 40, 200],
        [(1, 2)],
    )
    out = apply_locked_overlay(graph)
    assert list(map(tuple, out.edges.tolist())) == [(1, 2)]
    assert set(int(x) for x in out.node_id) == {1, 2}
    assert out.metadata["n_dropped_isolates"] == 1
    assert out.metadata["retunes_consecutive_links"] is False


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
    assert LEARNED_GAP_LOCKED_TUNE == 0.6584515375141551
    assert LEARNED_GAP_LOCKED_HOLDOUT == 0.7010255203349978
    assert tuning_decision(LEARNED_GAP_LOCKED_TUNE - 0.01, 1) == "reject_learned_unet_prune_v1"
    assert tuning_decision(LEARNED_GAP_LOCKED_TUNE, 0) == "keep_as_noop_learned_unet_prune_v1"
    assert tuning_decision(LEARNED_GAP_LOCKED_TUNE + 0.01, 2) == "keep_candidate_learned_unet_prune_v1"
    assert holdout_decision(LEARNED_GAP_LOCKED_HOLDOUT - 0.001) == "reject_learned_unet_prune_v1"
    assert holdout_decision(LEARNED_GAP_LOCKED_HOLDOUT) == "keep_candidate_learned_unet_prune_v1"
    assert LEARNED_PRUNE_LOCKED_TUNE == 0.6620813548547098
    assert LEARNED_PRUNE_LOCKED_HOLDOUT == 0.7013778258203534
    assert LEARNED_PRUNE_LOCKED_TUNE > LEARNED_GAP_LOCKED_TUNE
    assert LEARNED_PRUNE_LOCKED_HOLDOUT > LEARNED_GAP_LOCKED_HOLDOUT


def test_committed_holdout_report_is_keep_and_refuses_rescore():
    path = ROOT / "reports" / "learned_unet_prune_v1_holdout.json"
    payload = json.loads(path.read_text())
    assert payload["status"] == "holdout_scored"
    assert payload["decision"] == "keep_candidate_learned_unet_prune_v1"
    assert payload["config_hash"] == LOCKED_PRUNE_HASH
    assert payload["learned_prune_combined"] == LEARNED_PRUNE_LOCKED_HOLDOUT
    assert payload["n_dropped_isolates"] == 491
    assert payload["official_summary"]["division_tp"] == 0
    assert payload["official_summary"]["division_fp"] == 0
    assert payload["official_summary"]["division_fn"] == 5
    assert payload["official_summary"]["edge_jaccard"] == 0.7433450527373179
    assert payload["is_kaggle_run"] is False
    assert payload["is_leaderboard_score"] is False
    with pytest.raises(SystemExit, match="score frozen holdout twice"):
        refuse_existing_holdout_score(path)


def test_holdout_requires_keep_and_refuses_second_score(tmp_path: Path):
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit, match="tuning report missing"):
        require_tuning_keep(missing)
    reject = tmp_path / "reject.json"
    reject.write_text(json.dumps({"decision": "reject_learned_unet_prune_v1"}))
    with pytest.raises(SystemExit, match="refusing holdout"):
        require_tuning_keep(reject)
    keep = tmp_path / "keep.json"
    keep.write_text(json.dumps({"decision": "keep_as_noop_learned_unet_prune_v1"}))
    assert require_tuning_keep(keep)["decision"].startswith("keep_")
    scored = tmp_path / "holdout.json"
    scored.write_text(json.dumps({"status": "holdout_scored", "learned_prune_combined": 0.7, "decision": "x"}))
    with pytest.raises(SystemExit, match="score frozen holdout twice"):
        refuse_existing_holdout_score(scored)


def test_runtime_estimate_under_limit():
    est = runtime_estimate(n_volumes=2)
    assert est["under_limit"] is True
    assert est["estimate_hours"] < 12.0
    assert est["is_kaggle_run"] is False
    assert est["is_leaderboard_score"] is False
    hidden = runtime_estimate(n_volumes=199)
    assert hidden["under_limit"] is True
    assert hidden["estimate_hours"] < 1.0


def test_run_script_imports_without_kaggle():
    spec = importlib.util.spec_from_file_location(
        "run_learned_unet_prune",
        ROOT / "scripts" / "run_learned_unet_prune.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.TUNING_REPORT.name == "learned_unet_prune_v1_tune.json"
    assert mod.HOLDOUT_REPORT.name == "learned_unet_prune_v1_holdout.json"
    assert callable(mod.extract_selected_geffs)
    assert np.isfinite(LEARNED_GAP_LOCKED_TUNE)
