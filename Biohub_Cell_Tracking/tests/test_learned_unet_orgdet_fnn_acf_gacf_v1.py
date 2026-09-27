from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from biohub_tracking.graphs import TrackGraph
from biohub_tracking.learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from biohub_tracking.learned_orgdet import LOCKED_ORGDET_HASH
from biohub_tracking.learned_orgdet_elf import LOCKED_ORGDET_ELF_HASH
from biohub_tracking.learned_orgdet_fnn import LOCKED_ORGDET_FNN_HASH
from biohub_tracking.learned_orgdet_fnn_acf import (
    LEARNED_ORGDET_FNN_ACF_LOCKED_HOLDOUT,
    LEARNED_ORGDET_FNN_ACF_LOCKED_TUNE,
    LOCKED_ORGDET_FNN_ACF_HASH,
    MAX_REL_APPEARANCE,
)
from biohub_tracking.learned_orgdet_fnn_acf_bnn import LOCKED_ORGDET_FNN_ACF_BNN_HASH
from biohub_tracking.learned_orgdet_fnn_acf_gacf import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LOCKED_GATES,
    LOCKED_ORGDET_FNN_ACF_GACF_HASH,
    MAX_GAP_DT,
    ORGDET_FNN_ACF_LOCKED_HOLDOUT,
    ORGDET_FNN_ACF_LOCKED_TUNE,
    TUNE_KEEP_THRESHOLD,
    TUNE_MARGIN,
    drop_inconsistent_gap_edges,
    holdout_decision,
    orgdet_fnn_acf_gacf_config_hash,
    orgdet_fnn_acf_gacf_config_payload,
    overlay_summary,
    refuse_existing_holdout_score,
    refuse_non_holdout_stems,
    refuse_non_tuning_stems,
    require_tuning_keep,
    runtime_estimate,
    tuning_decision,
)
from biohub_tracking.learned_orgdet_fnn_acf_ipc import LOCKED_ORGDET_FNN_ACF_IPC_HASH
from biohub_tracking.learned_orgdet_fnn_div import LOCKED_ORGDET_FNN_DIV_HASH
from biohub_tracking.learned_orgdet_fnn_rank import LOCKED_RANK_HASH
from biohub_tracking.learned_orgdet_fnn_reas import LOCKED_ORGDET_FNN_REAS_HASH
from biohub_tracking.learned_orgdet_gap import LOCKED_ORGDET_GAP_HASH
from biohub_tracking.learned_orgdet_gap_prune import LOCKED_ORGDET_GAP_PRUNE_HASH
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
    payload = orgdet_fnn_acf_gacf_config_payload()
    assert orgdet_fnn_acf_gacf_config_hash(payload) == LOCKED_ORGDET_FNN_ACF_GACF_HASH
    assert LOCKED_ORGDET_FNN_ACF_GACF_HASH == "5ce49a6d175c8032"
    assert LOCKED_ORGDET_FNN_ACF_GACF_HASH != LOCKED_ORGDET_FNN_ACF_HASH
    assert LOCKED_ORGDET_FNN_ACF_GACF_HASH != LOCKED_ORGDET_FNN_HASH
    assert LOCKED_ORGDET_FNN_ACF_GACF_HASH != LOCKED_RANK_HASH
    assert LOCKED_ORGDET_FNN_ACF_GACF_HASH != LOCKED_ORGDET_FNN_DIV_HASH
    assert LOCKED_ORGDET_FNN_ACF_GACF_HASH != LOCKED_ORGDET_FNN_REAS_HASH
    assert LOCKED_ORGDET_FNN_ACF_GACF_HASH != LOCKED_ORGDET_FNN_ACF_BNN_HASH
    assert LOCKED_ORGDET_FNN_ACF_GACF_HASH != LOCKED_ORGDET_FNN_ACF_IPC_HASH
    assert ORGDET_FNN_ACF_LOCKED_TUNE == LEARNED_ORGDET_FNN_ACF_LOCKED_TUNE == 0.7346197119237633
    assert ORGDET_FNN_ACF_LOCKED_HOLDOUT == LEARNED_ORGDET_FNN_ACF_LOCKED_HOLDOUT == 0.7555029702166214
    assert payload["train_config_hash"] == TRAIN_CONFIG_HASH
    assert payload["infer_config_hash"] == LOCKED_INFER_HASH
    assert payload["orgdet_config_hash"] == LOCKED_ORGDET_HASH == "80b20589198148b7"
    assert payload["orgdet_gap_config_hash"] == LOCKED_ORGDET_GAP_HASH == "ee31f36151325f9a"
    assert payload["orgdet_gap_prune_config_hash"] == LOCKED_ORGDET_GAP_PRUNE_HASH == "862e3e7f7ac0330e"
    assert payload["orgdet_elf_config_hash"] == LOCKED_ORGDET_ELF_HASH == "292c4f1f968f948d"
    assert payload["orgdet_fnn_config_hash"] == LOCKED_ORGDET_FNN_HASH == "95e9300eb84ba71c"
    assert payload["orgdet_fnn_acf_config_hash"] == LOCKED_ORGDET_FNN_ACF_HASH == "a66075085ac64400"
    assert payload["rejected_fnn_rank_hash"] == LOCKED_RANK_HASH == "2748c4a2b2823c0a"
    assert payload["rejected_fnn_div_hash"] == LOCKED_ORGDET_FNN_DIV_HASH == "e184e6ce5bd90e46"
    assert payload["rejected_fnn_reas_hash"] == LOCKED_ORGDET_FNN_REAS_HASH == "39fb20d675e21d0e"
    assert payload["rejected_fnn_acf_bnn_hash"] == LOCKED_ORGDET_FNN_ACF_BNN_HASH == "bdb38cf13e226ced"
    assert payload["rejected_fnn_acf_ipc_hash"] == LOCKED_ORGDET_FNN_ACF_IPC_HASH == "7302409595bee523"
    assert payload["retunes_detection"] is False
    assert payload["retunes_consecutive_links"] is False
    assert payload["retunes_gap_gates"] is False
    assert payload["retunes_max_dt1_edge_um"] is False
    assert payload["retunes_nn_tie_eps"] is False
    assert payload["retunes_max_rel_appearance"] is False
    assert payload["uses_intensity"] is True
    assert payload["uses_transformer"] is False
    assert payload["uses_ilp"] is False
    assert payload["drop_inconsistent_gap"] is True
    assert payload["assignment"] == "unchanged_locked_orgdet_fnn_acf"
    assert payload["writes_submission_csv"] is False
    assert payload["source_zarrs"] == "locked_tuning_zarr_export_two_44b6"
    assert payload["tune_keep_threshold"] == pytest.approx(ORGDET_FNN_ACF_LOCKED_TUNE + TUNE_MARGIN)
    assert TUNE_KEEP_THRESHOLD == pytest.approx(0.7366197119237633)
    assert MAX_REL_APPEARANCE == 0.38
    assert MAX_GAP_DT == 3
    assert REJECTED_ORGOP_HASH == "250c5bfacd4a8198"
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_orgdet_fnn_acf_gacf_v1.yaml").read_text())
    for key in (
        "drop_inconsistent_gap",
        "drop_degree0_after",
        "assignment",
        "linker",
        "orgdet_fnn_acf_config_hash",
        "rejected_fnn_acf_bnn_hash",
        "rejected_fnn_acf_ipc_hash",
        "tune_margin",
        "retunes_max_dt1_edge_um",
        "retunes_nn_tie_eps",
        "retunes_max_rel_appearance",
        "appearance_feature",
        "max_gap_dt",
        "source_zarrs",
    ):
        assert yaml_cfg[key] == payload[key]
    assert yaml_cfg["max_rel_appearance"] == pytest.approx(payload["max_rel_appearance"])
    assert yaml_cfg["appearance_eps"] == pytest.approx(payload["appearance_eps"])
    assert yaml_cfg["tune_keep_threshold"] == pytest.approx(payload["tune_keep_threshold"])


def test_refuses_holdout_during_tuning():
    with pytest.raises(ValueError, match="tuning refuses"):
        refuse_non_tuning_stems(["6bba_05db0fb1"])
    refuse_non_tuning_stems(INFER_STEMS)
    refuse_non_holdout_stems(HOLDOUT_STEMS)
    with pytest.raises(ValueError, match="non-holdout"):
        refuse_non_holdout_stems(["44b6_12dfb391"])


def test_drop_inconsistent_interpolated_gap_chain():
    # Real leftovers 1 (t=0) and 2 (t=2); interpolant 3 is not a source id.
    graph = _graph(
        [1, 2, 3],
        [0, 2, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [(1, 3), (3, 2)],
    )
    appearance = {1: 100.0, 2: 200.0, 3: 130.0}
    out = drop_inconsistent_gap_edges(graph, appearance, source_ids={1, 2})
    assert set(map(tuple, out.edges)) == set()
    assert out.metadata["n_dropped_inconsistent_gap_edges"] == 2
    assert out.metadata["n_dropped_inconsistent_gap_chains"] == 1
    assert out.metadata["n_checked_gap_chains"] == 1
    assert out.n_nodes == 3
    assert out.metadata["max_rel_appearance"] == MAX_REL_APPEARANCE == 0.38


def test_drop_keeps_consistent_gap_chain():
    graph = _graph(
        [1, 2, 3],
        [0, 2, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [(1, 3), (3, 2)],
    )
    appearance = {1: 100.0, 2: 110.0, 3: 105.0}
    out = drop_inconsistent_gap_edges(graph, appearance, source_ids={1, 2})
    assert set(map(tuple, out.edges)) == {(1, 3), (3, 2)}
    assert out.metadata["n_dropped_inconsistent_gap_edges"] == 0
    assert out.metadata["n_kept_gap_chains"] == 1


def test_drop_inconsistent_dt2_skip_without_interpolant():
    graph = _graph(
        [1, 2],
        [0, 2],
        [0, 20],
        [0, 0],
        [0, 0],
        [(1, 2)],
    )
    out = drop_inconsistent_gap_edges(graph, {1: 10.0, 2: 1000.0}, source_ids={1, 2})
    assert set(map(tuple, out.edges)) == set()
    assert out.metadata["n_dropped_inconsistent_gap_edges"] == 1
    assert out.metadata["n_dropped_inconsistent_gap_chains"] == 1


def test_drop_keeps_dt1_even_when_inconsistent():
    # dt=1 is locked ACF's job. GACF must not retune consecutive links.
    graph = _graph(
        [1, 2],
        [0, 1],
        [0, 0],
        [0, 0],
        [0, 1],
        [(1, 2)],
    )
    out = drop_inconsistent_gap_edges(graph, {1: 10.0, 2: 1000.0}, source_ids={1, 2})
    assert set(map(tuple, out.edges)) == {(1, 2)}
    assert out.metadata["n_dropped_inconsistent_gap_edges"] == 0
    assert out.metadata["n_kept_dt1_edges"] == 1


def test_drop_refuses_retuned_threshold_or_missing_appearance():
    graph = _graph([1, 2], [0, 2], [0, 0], [0, 0], [0, 1], [(1, 2)])
    with pytest.raises(RuntimeError, match="retune max_rel_appearance"):
        drop_inconsistent_gap_edges(graph, {1: 1.0, 2: 1.0}, source_ids={1, 2}, max_rel=0.10)
    with pytest.raises(RuntimeError, match="retune max_gap_dt"):
        drop_inconsistent_gap_edges(graph, {1: 1.0, 2: 1.0}, source_ids={1, 2}, max_gap_dt=2)
    with pytest.raises(RuntimeError, match="missing appearance"):
        drop_inconsistent_gap_edges(graph, {1: 1.0}, source_ids={1, 2})


def test_incomplete_interpolant_fragment_is_not_dropped():
    # ACF already broke the chain; GACF must not invent the missing hop.
    graph = _graph(
        [1, 3],
        [0, 1],
        [0, 0],
        [0, 0],
        [0, 0],
        [(1, 3)],
    )
    out = drop_inconsistent_gap_edges(graph, {1: 10.0, 3: 1000.0}, source_ids={1})
    assert set(map(tuple, out.edges)) == {(1, 3)}
    assert out.metadata["n_dropped_inconsistent_gap_edges"] == 0
    assert out.metadata["n_checked_gap_chains"] == 0


def test_tuning_and_holdout_gates():
    assert tuning_decision(0.7366197119237632, 10) == "reject_learned_unet_orgdet_fnn_acf_gacf_v1"
    assert tuning_decision(0.7366197119237633, 10) == "keep_candidate_learned_unet_orgdet_fnn_acf_gacf_v1"
    assert tuning_decision(0.80, 0) == "keep_as_noop_learned_unet_orgdet_fnn_acf_gacf_v1"
    assert holdout_decision(ORGDET_FNN_ACF_LOCKED_HOLDOUT) == "reject_learned_unet_orgdet_fnn_acf_gacf_v1"
    assert holdout_decision(ORGDET_FNN_ACF_LOCKED_HOLDOUT + 1e-6) == "keep_candidate_learned_unet_orgdet_fnn_acf_gacf_v1"


def test_require_tuning_keep_and_refuse_second_holdout(tmp_path: Path):
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit, match="tuning report missing"):
        require_tuning_keep(missing)
    reject = tmp_path / "reject.json"
    reject.write_text(json.dumps({"decision": "reject_learned_unet_orgdet_fnn_acf_gacf_v1"}))
    with pytest.raises(SystemExit, match="tuning decision"):
        require_tuning_keep(reject)
    hold = tmp_path / "hold.json"
    hold.write_text(json.dumps({"status": "holdout_scored", "learned_orgdet_fnn_acf_gacf_combined": 0.7}))
    with pytest.raises(SystemExit, match="score frozen holdout twice"):
        refuse_existing_holdout_score(hold)


def test_runtime_estimate_under_kaggle_limit():
    est = runtime_estimate()
    assert est["under_limit"] is True
    assert est["is_kaggle_run"] is False
    assert est["is_leaderboard_score"] is False
    assert est["prefers_tuning_zarrs"] is True


def test_rejected_tune_report_missed_gate():
    report = json.loads((ROOT / "reports" / "learned_unet_orgdet_fnn_acf_gacf_v1_tune.json").read_text())
    assert report["decision"] == "reject_learned_unet_orgdet_fnn_acf_gacf_v1"
    assert report["config_hash"] == LOCKED_ORGDET_FNN_ACF_GACF_HASH == "5ce49a6d175c8032"
    assert report["learned_orgdet_fnn_acf_gacf_combined"] < TUNE_KEEP_THRESHOLD
    assert report["n_dropped_inconsistent_gap_edges"] == 160
    assert report["opened_holdout"] is False
    assert report["is_leaderboard_score"] is False
    assert report["official_summary"]["edge_jaccard"] == pytest.approx(0.7495867768595041)


def test_overlay_summary_counts():
    before = _graph([1, 2, 3], [0, 2, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [(1, 3), (3, 2)])
    after = drop_inconsistent_gap_edges(before, {1: 100.0, 2: 200.0, 3: 130.0}, source_ids={1, 2})
    after.metadata["overlay"] = "learned_unet_orgdet_fnn_acf_gacf_v1"
    after.metadata["config_hash"] = LOCKED_ORGDET_FNN_ACF_GACF_HASH
    summary = overlay_summary(before, after)
    assert summary["overlay"] == "learned_unet_orgdet_fnn_acf_gacf_v1"
    assert summary["config_hash"] == LOCKED_ORGDET_FNN_ACF_GACF_HASH
    assert summary["n_nodes_after"] == 3
    assert summary["n_dropped_inconsistent_gap_edges"] == 2
    assert summary["n_dropped_inconsistent_gap_chains"] == 1
    assert LOCKED_GATES["assignment"] == "unchanged_locked_orgdet_fnn_acf"
    assert LOCKED_GATES["drop_inconsistent_gap"] is True
