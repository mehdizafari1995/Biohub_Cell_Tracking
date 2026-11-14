from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from biohub_tracking.graphs import TrackGraph
from biohub_tracking.learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from biohub_tracking.learned_orgdet import LOCKED_ORGDET_HASH
from biohub_tracking.learned_orgdet_elf import LEARNED_ORGDET_ELF_LOCKED_HOLDOUT, LEARNED_ORGDET_ELF_LOCKED_TUNE, LOCKED_ORGDET_ELF_HASH
from biohub_tracking.learned_orgdet_gap import LOCKED_ORGDET_GAP_HASH
from biohub_tracking.learned_orgdet_gap_prune import LOCKED_ORGDET_GAP_PRUNE_HASH
from biohub_tracking.learned_orgdet_fnn import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LEARNED_ORGDET_FNN_LOCKED_HOLDOUT,
    LEARNED_ORGDET_FNN_LOCKED_TUNE,
    LOCKED_GATES,
    LOCKED_ORGDET_FNN_HASH,
    NN_TIE_EPS,
    ORGDET_ELF_LOCKED_HOLDOUT,
    ORGDET_ELF_LOCKED_TUNE,
    TUNE_KEEP_THRESHOLD,
    TUNE_MARGIN,
    apply_orgdet_fnn_overlay,
    drop_non_forward_nn_edges,
    drop_non_forward_nn_edges_naive,
    holdout_decision,
    orgdet_fnn_config_hash,
    orgdet_fnn_config_payload,
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
    payload = orgdet_fnn_config_payload()
    assert orgdet_fnn_config_hash(payload) == LOCKED_ORGDET_FNN_HASH
    assert LOCKED_ORGDET_FNN_HASH == "95e9300eb84ba71c"
    assert LOCKED_ORGDET_FNN_HASH != LOCKED_ORGDET_ELF_HASH
    assert ORGDET_ELF_LOCKED_TUNE == LEARNED_ORGDET_ELF_LOCKED_TUNE == 0.7098380747225755
    assert ORGDET_ELF_LOCKED_HOLDOUT == LEARNED_ORGDET_ELF_LOCKED_HOLDOUT == 0.7377873461563521
    assert LEARNED_ORGDET_FNN_LOCKED_TUNE == 0.7246526336456757
    assert LEARNED_ORGDET_FNN_LOCKED_HOLDOUT == 0.7520914330648746
    assert payload["train_config_hash"] == TRAIN_CONFIG_HASH
    assert payload["infer_config_hash"] == LOCKED_INFER_HASH
    assert payload["orgdet_config_hash"] == LOCKED_ORGDET_HASH == "80b20589198148b7"
    assert payload["orgdet_gap_config_hash"] == LOCKED_ORGDET_GAP_HASH == "ee31f36151325f9a"
    assert payload["orgdet_gap_prune_config_hash"] == LOCKED_ORGDET_GAP_PRUNE_HASH == "862e3e7f7ac0330e"
    assert payload["orgdet_elf_config_hash"] == LOCKED_ORGDET_ELF_HASH == "292c4f1f968f948d"
    assert payload["retunes_detection"] is False
    assert payload["retunes_consecutive_links"] is False
    assert payload["retunes_gap_gates"] is False
    assert payload["retunes_max_dt1_edge_um"] is False
    assert payload["uses_intensity"] is False
    assert payload["uses_transformer"] is False
    assert payload["uses_ilp"] is False
    assert payload["keep_forward_nn_only"] is True
    assert payload["assignment"] == "unchanged_locked_orgdet_elf"
    assert payload["writes_submission_csv"] is False
    assert payload["tune_keep_threshold"] == pytest.approx(ORGDET_ELF_LOCKED_TUNE + TUNE_MARGIN)
    assert TUNE_KEEP_THRESHOLD == pytest.approx(0.7118380747225755)
    assert REJECTED_ORGOP_HASH == "250c5bfacd4a8198"
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_orgdet_fnn_v1.yaml").read_text())
    for key in (
        "keep_forward_nn_only",
        "drop_degree0_after",
        "assignment",
        "linker",
        "orgdet_elf_config_hash",
        "tune_margin",
        "retunes_max_dt1_edge_um",
    ):
        assert yaml_cfg[key] == payload[key]
    assert yaml_cfg["nn_tie_eps"] == pytest.approx(payload["nn_tie_eps"])
    assert yaml_cfg["tune_keep_threshold"] == pytest.approx(payload["tune_keep_threshold"])


def test_refuses_holdout_during_tuning():
    with pytest.raises(ValueError, match="tuning refuses"):
        refuse_non_tuning_stems(["6bba_05db0fb1"])
    refuse_non_tuning_stems(INFER_STEMS)
    refuse_non_holdout_stems(HOLDOUT_STEMS)
    with pytest.raises(ValueError, match="non-holdout"):
        refuse_non_holdout_stems(["44b6_12dfb391"])


def test_drop_non_forward_keeps_nearest_and_drops_farther():
    # x-scale 0.40625 µm. A->B is 1 voxel; A->C is 10 voxels.
    graph = _graph(
        [1, 2, 3],
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 10],
        [(1, 2), (1, 3)],
    )
    out = drop_non_forward_nn_edges(graph)
    assert set(map(tuple, out.edges)) == {(1, 2)}
    assert out.metadata["n_dropped_non_forward_nn_edges"] == 1
    assert out.metadata["n_kept_forward_nn_edges"] == 1
    assert out.n_nodes == 3


def test_drop_non_forward_keeps_tied_nearest():
    graph = _graph(
        [1, 2, 3],
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 4, -4],
        [(1, 2), (1, 3)],
    )
    out = drop_non_forward_nn_edges(graph)
    assert set(map(tuple, out.edges)) == {(1, 2), (1, 3)}
    assert out.metadata["n_dropped_non_forward_nn_edges"] == 0


def test_drop_non_forward_keeps_non_dt1_edges():
    graph = _graph(
        [1, 2],
        [0, 2],
        [0, 20],
        [0, 0],
        [0, 0],
        [(1, 2)],
    )
    out = drop_non_forward_nn_edges(graph)
    assert set(map(tuple, out.edges)) == {(1, 2)}
    assert out.metadata["n_kept_non_dt1_edges"] == 1
    assert out.metadata["n_dropped_non_forward_nn_edges"] == 0


def test_vectorized_fnn_matches_naive_keep_drop():
    graph = _graph(
        [1, 2, 3, 4, 5, 6],
        [0, 0, 1, 1, 1, 2],
        [0, 1, 0, 0, 2, 0],
        [0, 0, 0, 4, 0, 0],
        [0, 8, 1, 10, 20, 1],
        [(1, 3), (1, 4), (1, 5), (2, 3), (2, 5), (3, 6), (4, 6)],
    )
    fast = drop_non_forward_nn_edges(graph)
    naive = drop_non_forward_nn_edges_naive(graph)
    assert set(map(tuple, fast.edges)) == set(map(tuple, naive.edges))
    assert fast.metadata["n_dropped_non_forward_nn_edges"] == naive.metadata["n_dropped_non_forward_nn_edges"]
    assert fast.metadata["n_kept_forward_nn_edges"] == naive.metadata["n_kept_forward_nn_edges"]
    assert fast.metadata["n_kept_non_dt1_edges"] == naive.metadata["n_kept_non_dt1_edges"]


def test_drop_non_forward_refuses_retuned_eps():
    graph = _graph([1, 2], [0, 1], [0, 0], [0, 0], [0, 1], [(1, 2)])
    with pytest.raises(RuntimeError, match="retune nn_tie_eps"):
        drop_non_forward_nn_edges(graph, tie_eps=1e-3)


def test_tuning_and_holdout_gates():
    assert tuning_decision(0.7118380747225754, 10) == "reject_learned_unet_orgdet_fnn_v1"
    assert tuning_decision(0.7118380747225755, 10) == "keep_candidate_learned_unet_orgdet_fnn_v1"
    assert tuning_decision(0.72, 0) == "keep_as_noop_learned_unet_orgdet_fnn_v1"
    assert holdout_decision(ORGDET_ELF_LOCKED_HOLDOUT) == "reject_learned_unet_orgdet_fnn_v1"
    assert holdout_decision(ORGDET_ELF_LOCKED_HOLDOUT + 1e-6) == "keep_candidate_learned_unet_orgdet_fnn_v1"


def test_require_tuning_keep_and_refuse_second_holdout(tmp_path: Path):
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit, match="tuning report missing"):
        require_tuning_keep(missing)
    reject = tmp_path / "reject.json"
    reject.write_text(json.dumps({"decision": "reject_learned_unet_orgdet_fnn_v1"}))
    with pytest.raises(SystemExit, match="tuning decision"):
        require_tuning_keep(reject)
    hold = tmp_path / "hold.json"
    hold.write_text(json.dumps({"status": "holdout_scored", "learned_orgdet_fnn_combined": 0.7}))
    with pytest.raises(SystemExit, match="score frozen holdout twice"):
        refuse_existing_holdout_score(hold)


def test_runtime_estimate_under_kaggle_limit():
    est = runtime_estimate()
    assert est["under_limit"] is True
    assert est["is_kaggle_run"] is False
    assert est["is_leaderboard_score"] is False


def test_overlay_summary_counts():
    before = _graph([1, 2, 3], [0, 1, 1], [0, 0, 0], [0, 0, 0], [0, 1, 10], [(1, 2), (1, 3)])
    after = drop_non_forward_nn_edges(before)
    after.metadata["overlay"] = "learned_unet_orgdet_fnn_v1"
    after.metadata["config_hash"] = LOCKED_ORGDET_FNN_HASH
    summary = overlay_summary(before, after)
    assert summary["overlay"] == "learned_unet_orgdet_fnn_v1"
    assert summary["config_hash"] == LOCKED_ORGDET_FNN_HASH
    assert summary["n_dropped_non_forward_nn_edges"] == 1
    assert summary["nn_tie_eps"] == NN_TIE_EPS


def test_apply_overlay_refuses_hash_drift(monkeypatch):
    graph = _graph([1, 2], [0, 1], [0, 1], [0, 0], [0, 0], [(1, 2)])
    monkeypatch.setattr(
        "biohub_tracking.learned_orgdet_fnn.orgdet_fnn_config_hash",
        lambda payload=None: "deadbeefdeadbeef",
    )
    with pytest.raises(RuntimeError, match="hash drifted"):
        apply_orgdet_fnn_overlay(graph)


def test_locked_gates_are_not_assignment():
    assert LOCKED_GATES["assignment"] == "unchanged_locked_orgdet_elf"
    assert LOCKED_GATES["keep_forward_nn_only"] is True
    assert orgdet_fnn_config_payload()["assignment"] == "unchanged_locked_orgdet_elf"
    assert orgdet_fnn_config_payload()["retunes_max_dt1_edge_um"] is False
