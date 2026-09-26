from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from biohub_tracking.graphs import TrackGraph
from biohub_tracking.learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from biohub_tracking.learned_orgdet import LOCKED_ORGDET_HASH
from biohub_tracking.learned_orgdet_error_decomp import TUNE_KEEP_THRESHOLD, TUNE_MARGIN
from biohub_tracking.learned_orgdet_gap import LOCKED_ORGDET_GAP_HASH
from biohub_tracking.learned_orgdet_gap_prune import LOCKED_ORGDET_GAP_PRUNE_HASH
from biohub_tracking.learned_orgdet_tcf import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LOCKED_GATES,
    LOCKED_ORGDET_TCF_HASH,
    ORGDET_GAP_PRUNE_LOCKED_HOLDOUT,
    ORGDET_GAP_PRUNE_LOCKED_TUNE,
    apply_orgdet_tcf_overlay,
    drop_short_components,
    holdout_decision,
    orgdet_tcf_config_hash,
    orgdet_tcf_config_payload,
    overlay_summary,
    refuse_existing_holdout_score,
    refuse_non_holdout_stems,
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
    payload = orgdet_tcf_config_payload()
    assert orgdet_tcf_config_hash(payload) == LOCKED_ORGDET_TCF_HASH
    assert LOCKED_ORGDET_TCF_HASH == "a6b46dfcf410478f"
    assert LOCKED_ORGDET_TCF_HASH != LOCKED_ORGDET_GAP_PRUNE_HASH
    assert payload["train_config_hash"] == TRAIN_CONFIG_HASH
    assert payload["infer_config_hash"] == LOCKED_INFER_HASH
    assert payload["orgdet_config_hash"] == LOCKED_ORGDET_HASH == "80b20589198148b7"
    assert payload["orgdet_gap_config_hash"] == LOCKED_ORGDET_GAP_HASH == "ee31f36151325f9a"
    assert payload["orgdet_gap_prune_config_hash"] == LOCKED_ORGDET_GAP_PRUNE_HASH == "862e3e7f7ac0330e"
    assert payload["retunes_detection"] is True
    assert payload["retunes_consecutive_links"] is False
    assert payload["retunes_gap_gates"] is False
    assert payload["uses_intensity"] is False
    assert payload["uses_transformer"] is False
    assert payload["uses_ilp"] is False
    assert payload["min_component_nodes"] == 8
    assert payload["assignment"] == "unchanged_locked_orgdet_gap_prune"
    assert payload["writes_submission_csv"] is False
    assert payload["tune_keep_threshold"] == pytest.approx(ORGDET_GAP_PRUNE_LOCKED_TUNE + TUNE_MARGIN)
    assert TUNE_KEEP_THRESHOLD == pytest.approx(0.6954371804770478)
    assert REJECTED_ORGOP_HASH == "250c5bfacd4a8198"
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_orgdet_tcf_v1.yaml").read_text())
    for key in (
        "min_component_nodes",
        "safe_short_track_max_nodes",
        "drop_short_fragments",
        "assignment",
        "linker",
        "orgdet_gap_prune_config_hash",
        "tune_margin",
    ):
        assert yaml_cfg[key] == payload[key]


def test_refuses_holdout_during_tuning():
    with pytest.raises(ValueError, match="tuning refuses"):
        refuse_non_tuning_stems(["6bba_05db0fb1"])
    refuse_non_tuning_stems(INFER_STEMS)
    refuse_non_holdout_stems(HOLDOUT_STEMS)
    with pytest.raises(ValueError, match="non-holdout"):
        refuse_non_holdout_stems(["44b6_12dfb391"])


def test_drop_short_components_keeps_long_tracks():
    # 7-node chain should drop; 8-node chain should stay.
    short_ids = list(range(1, 8))
    long_ids = list(range(10, 18))
    node_id = short_ids + long_ids
    t = list(range(7)) + list(range(8))
    z = [0] * 15
    y = [0] * 15
    x = list(range(7)) + list(range(8))
    edges = [(i, i + 1) for i in short_ids[:-1]] + [(i, i + 1) for i in long_ids[:-1]]
    graph = _graph(node_id, t, z, y, x, edges)
    out = drop_short_components(graph)
    assert set(int(x) for x in out.node_id) == set(long_ids)
    assert out.n_edges == 7
    assert out.metadata["n_dropped_short_nodes"] == 7
    assert out.metadata["n_dropped_short_edges"] == 6


def test_drop_short_refuses_retuned_min_size():
    graph = _graph([1, 2], [0, 1], [0, 0], [0, 0], [0, 1], [(1, 2)])
    with pytest.raises(RuntimeError, match="retune min_component_nodes"):
        drop_short_components(graph, min_nodes=3)


def test_tuning_and_holdout_gates():
    assert tuning_decision(0.6954371804770477, 10) == "reject_learned_unet_orgdet_tcf_v1"
    assert tuning_decision(0.6954371804770478, 10) == "keep_candidate_learned_unet_orgdet_tcf_v1"
    assert tuning_decision(0.70, 0) == "keep_as_noop_learned_unet_orgdet_tcf_v1"
    assert holdout_decision(ORGDET_GAP_PRUNE_LOCKED_HOLDOUT) == "reject_learned_unet_orgdet_tcf_v1"
    assert holdout_decision(ORGDET_GAP_PRUNE_LOCKED_HOLDOUT + 1e-6) == "keep_candidate_learned_unet_orgdet_tcf_v1"


def test_require_tuning_keep_and_refuse_second_holdout(tmp_path: Path):
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit, match="tuning report missing"):
        require_tuning_keep(missing)
    reject = tmp_path / "reject.json"
    reject.write_text(json.dumps({"decision": "reject_learned_unet_orgdet_tcf_v1"}))
    with pytest.raises(SystemExit, match="tuning decision"):
        require_tuning_keep(reject)
    hold = tmp_path / "hold.json"
    hold.write_text(json.dumps({"status": "holdout_scored", "learned_orgdet_tcf_combined": 0.7}))
    with pytest.raises(SystemExit, match="score frozen holdout twice"):
        refuse_existing_holdout_score(hold)


def test_runtime_estimate_under_kaggle_limit():
    est = runtime_estimate()
    assert est["under_limit"] is True
    assert est["is_kaggle_run"] is False
    assert est["is_leaderboard_score"] is False


def test_overlay_summary_counts():
    before = _graph([1, 2, 3], [0, 1, 2], [0, 0, 0], [0, 0, 0], [0, 1, 2], [(1, 2)])
    after = drop_short_components(
        _graph(
            list(range(1, 9)),
            list(range(8)),
            [0] * 8,
            [0] * 8,
            list(range(8)),
            [(i, i + 1) for i in range(1, 8)],
        )
    )
    summary = overlay_summary(before, after)
    assert summary["overlay"] == "learned_unet_orgdet_tcf_v1"
    assert summary["config_hash"] == LOCKED_ORGDET_TCF_HASH
    assert summary["n_nodes_after"] == 8
