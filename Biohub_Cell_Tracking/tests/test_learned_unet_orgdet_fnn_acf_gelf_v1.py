from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from biohub_tracking.graphs import TrackGraph
from biohub_tracking.learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from biohub_tracking.learned_orgdet import LOCKED_ORGDET_HASH
from biohub_tracking.learned_orgdet_elf import LOCKED_ORGDET_ELF_HASH, MAX_DT1_EDGE_UM
from biohub_tracking.learned_orgdet_fnn import LOCKED_ORGDET_FNN_HASH
from biohub_tracking.learned_orgdet_fnn_acf import (
    LEARNED_ORGDET_FNN_ACF_LOCKED_HOLDOUT,
    LEARNED_ORGDET_FNN_ACF_LOCKED_TUNE,
    LOCKED_ORGDET_FNN_ACF_HASH,
)
from biohub_tracking.learned_orgdet_fnn_acf_bnn import LOCKED_ORGDET_FNN_ACF_BNN_HASH
from biohub_tracking.learned_orgdet_fnn_acf_gacf import LOCKED_ORGDET_FNN_ACF_GACF_HASH
from biohub_tracking.learned_orgdet_fnn_acf_gelf import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LOCKED_GATES,
    LOCKED_ORGDET_FNN_ACF_GELF_HASH,
    MAX_GAP_DT,
    MAX_GAP_EDGE_UM,
    ORGDET_FNN_ACF_LOCKED_HOLDOUT,
    ORGDET_FNN_ACF_LOCKED_TUNE,
    TUNE_KEEP_THRESHOLD,
    TUNE_MARGIN,
    apply_orgdet_fnn_acf_gelf_overlay,
    chain_class_from_hops,
    classify_gap_chains,
    diagnostic_decision,
    drop_long_gap_edges,
    extract_gap_chains,
    holdout_decision,
    hop_class,
    orgdet_fnn_acf_gelf_config_hash,
    orgdet_fnn_acf_gelf_config_payload,
    overlay_summary,
    propose_gap_length_gate,
    refuse_existing_holdout_score,
    refuse_non_holdout_stems,
    refuse_non_tuning_stems,
    require_supported_gate,
    require_tuning_keep,
    round_up_tenth_um,
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


def test_payload_parent_hashes_and_unlocked_gate():
    payload = orgdet_fnn_acf_gelf_config_payload()
    assert orgdet_fnn_acf_gelf_config_hash(payload) == LOCKED_ORGDET_FNN_ACF_GELF_HASH
    assert LOCKED_ORGDET_FNN_ACF_GELF_HASH == "59812f90385ed4d2"
    assert len(LOCKED_ORGDET_FNN_ACF_GELF_HASH) == 16
    assert LOCKED_ORGDET_FNN_ACF_GELF_HASH != LOCKED_ORGDET_FNN_ACF_HASH
    assert LOCKED_ORGDET_FNN_ACF_GELF_HASH != LOCKED_ORGDET_FNN_ACF_GACF_HASH
    assert LOCKED_ORGDET_FNN_ACF_GELF_HASH != LOCKED_ORGDET_FNN_HASH
    assert LOCKED_ORGDET_FNN_ACF_GELF_HASH != LOCKED_RANK_HASH
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
    assert payload["rejected_fnn_acf_gacf_hash"] == LOCKED_ORGDET_FNN_ACF_GACF_HASH == "5ce49a6d175c8032"
    assert payload["retunes_detection"] is False
    assert payload["retunes_consecutive_links"] is False
    assert payload["retunes_gap_gates"] is False
    assert payload["retunes_max_dt1_edge_um"] is False
    assert payload["retunes_nn_tie_eps"] is False
    assert payload["retunes_max_rel_appearance"] is False
    assert payload["uses_intensity"] is False
    assert payload["uses_transformer"] is False
    assert payload["uses_ilp"] is False
    assert payload["drop_long_gap"] is True
    assert payload["assignment"] == "unchanged_locked_orgdet_fnn_acf"
    assert payload["writes_submission_csv"] is False
    assert payload["source_zarrs"] == "locked_tuning_zarr_export_two_44b6"
    assert payload["tune_keep_threshold"] == pytest.approx(ORGDET_FNN_ACF_LOCKED_TUNE + TUNE_MARGIN)
    assert TUNE_KEEP_THRESHOLD == pytest.approx(0.7366197119237633)
    assert MAX_GAP_DT == 3
    assert MAX_DT1_EDGE_UM == 8.0
    assert REJECTED_ORGOP_HASH == "250c5bfacd4a8198"
    assert LOCKED_GATES["assignment"] == "unchanged_locked_orgdet_fnn_acf"
    assert payload["max_gap_edge_um"] is None
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_orgdet_fnn_acf_gelf_v1.yaml").read_text())
    for key in (
        "drop_long_gap",
        "drop_degree0_after",
        "assignment",
        "linker",
        "orgdet_fnn_acf_config_hash",
        "rejected_fnn_acf_bnn_hash",
        "rejected_fnn_acf_ipc_hash",
        "rejected_fnn_acf_gacf_hash",
        "tune_margin",
        "retunes_max_dt1_edge_um",
        "retunes_nn_tie_eps",
        "retunes_max_rel_appearance",
        "max_gap_dt",
        "source_zarrs",
        "max_gap_edge_um",
        "uses_intensity",
    ):
        assert yaml_cfg[key] == payload[key]
    assert yaml_cfg["tune_keep_threshold"] == pytest.approx(payload["tune_keep_threshold"])


def test_refuses_holdout_during_tuning():
    with pytest.raises(ValueError, match="tuning refuses"):
        refuse_non_tuning_stems(["6bba_05db0fb1"])
    refuse_non_tuning_stems(INFER_STEMS)
    refuse_non_holdout_stems(HOLDOUT_STEMS)
    with pytest.raises(ValueError, match="non-holdout"):
        refuse_non_holdout_stems(["44b6_12dfb391"])


def test_round_up_tenth_is_strictly_above():
    assert round_up_tenth_um(7.6219) == pytest.approx(7.7)
    assert round_up_tenth_um(8.0) == pytest.approx(8.1)
    assert round_up_tenth_um(0.0) == pytest.approx(0.1)


def test_extract_interpolated_and_dt2_skip():
    graph = _graph(
        [1, 2, 3],
        [0, 2, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 24, 12],
        [(1, 3), (3, 2)],
    )
    chains = extract_gap_chains(graph, source_ids={1, 2})
    assert len(chains) == 1
    assert chains[0]["dt"] == 2
    assert chains[0]["interpolated"] is True
    assert chains[0]["n_hops"] == 2
    assert chains[0]["length_um"] == pytest.approx(24 * 0.40625)

    skip = _graph([1, 2], [0, 2], [0, 0], [0, 0], [0, 10], [(1, 2)])
    skip_chains = extract_gap_chains(skip, source_ids={1, 2})
    assert len(skip_chains) == 1
    assert skip_chains[0]["interpolated"] is False
    assert skip_chains[0]["n_hops"] == 1


def test_extract_ignores_dt1():
    graph = _graph([1, 2], [0, 1], [0, 0], [0, 0], [0, 40], [(1, 2)])
    assert extract_gap_chains(graph, source_ids={1, 2}) == []


def test_drop_long_interpolated_gap_chain():
    graph = _graph(
        [1, 2, 3],
        [0, 2, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 40, 20],
        [(1, 3), (3, 2)],
    )
    max_um = 8.0 if MAX_GAP_EDGE_UM is None else float(MAX_GAP_EDGE_UM)
    # 40 px * 0.40625 = 16.25 µm, longer than 8.0.
    out = drop_long_gap_edges(graph, source_ids={1, 2}, max_um=max_um)
    if 16.25 > max_um:
        assert set(map(tuple, out.edges)) == set()
        assert out.metadata["n_dropped_long_gap_edges"] == 2
        assert out.metadata["n_dropped_long_gap_chains"] == 1
    else:
        assert set(map(tuple, out.edges)) == {(1, 3), (3, 2)}
    assert out.n_nodes == 3
    assert out.metadata["n_kept_dt1_edges"] in {0, 2}


def test_drop_keeps_short_gap_and_dt1():
    short = _graph(
        [1, 2, 3],
        [0, 2, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 4, 2],
        [(1, 3), (3, 2)],
    )
    max_um = 8.0 if MAX_GAP_EDGE_UM is None else float(MAX_GAP_EDGE_UM)
    out = drop_long_gap_edges(short, source_ids={1, 2}, max_um=max_um)
    # 4 px * 0.40625 = 1.625 µm, kept under any 0.1-round gate above TP.
    if 1.625 <= max_um:
        assert set(map(tuple, out.edges)) == {(1, 3), (3, 2)}
        assert out.metadata["n_dropped_long_gap_chains"] == 0
        assert out.metadata["n_kept_gap_chains"] == 1

    dt1 = _graph([1, 2], [0, 1], [0, 0], [0, 0], [0, 80], [(1, 2)])
    kept = drop_long_gap_edges(dt1, source_ids={1, 2}, max_um=max_um)
    assert set(map(tuple, kept.edges)) == {(1, 2)}
    assert kept.metadata["n_dropped_long_gap_edges"] == 0
    assert kept.metadata["n_kept_dt1_edges"] == 1


def test_drop_refuses_retuned_gap_dt_or_locked_um():
    graph = _graph([1, 2], [0, 2], [0, 0], [0, 0], [0, 1], [(1, 2)])
    with pytest.raises(RuntimeError, match="retune max_gap_dt"):
        drop_long_gap_edges(graph, source_ids={1, 2}, max_um=8.0, max_gap_dt=2)
    if MAX_GAP_EDGE_UM is not None:
        with pytest.raises(RuntimeError, match="retune max_gap_edge_um"):
            drop_long_gap_edges(graph, source_ids={1, 2}, max_um=float(MAX_GAP_EDGE_UM) + 1.0)


def test_incomplete_interpolant_fragment_is_not_dropped():
    graph = _graph([1, 3], [0, 1], [0, 0], [0, 0], [0, 80], [(1, 3)])
    max_um = 8.0 if MAX_GAP_EDGE_UM is None else float(MAX_GAP_EDGE_UM)
    out = drop_long_gap_edges(graph, source_ids={1}, max_um=max_um)
    assert set(map(tuple, out.edges)) == {(1, 3)}
    assert out.metadata["n_checked_gap_chains"] == 0


def test_hop_and_chain_classes():
    assert hop_class(1, 2, {1: 10, 2: 20}, {(10, 20)}, {20: {10}}, {10: {20}}) == "tp"
    assert hop_class(1, 2, {1: 10, 2: 20}, set(), {20: {99}}, {10: {99}}) == "conflict_fp"
    assert hop_class(1, 2, {}, set(), {}, {}) == "leftover"
    assert chain_class_from_hops(["leftover", "tp"]) == "tp"
    assert chain_class_from_hops(["leftover", "conflict_fp"]) == "conflict_fp"
    assert chain_class_from_hops(["leftover"]) == "leftover"


def test_classify_and_propose_gate_separates_conflict_fp():
    pred = _graph(
        [1, 2, 3, 4],
        [0, 2, 0, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 4, 0, 40],
        [(1, 2), (3, 4)],
    )
    gt = _graph(
        [10, 20, 30, 40],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 4, 0, 1],
        [(10, 20), (30, 40)],
        dataset="gt",
    )
    chains = extract_gap_chains(pred, source_ids={1, 2, 3, 4})
    classified = classify_gap_chains(pred, gt, chains)
    assert {c["start"] for c in classified} == {1, 3}
    proposal = propose_gap_length_gate(
        [
            {"length_um": 7.62, "chain_class": "tp", "dt": 2},
            {"length_um": 9.4, "chain_class": "conflict_fp", "dt": 2},
            {"length_um": 20.0, "chain_class": "leftover", "dt": 3},
        ]
    )
    assert proposal["supported"] is True
    assert proposal["max_gap_edge_um"] == pytest.approx(7.7)
    assert proposal["n_conflict_fp_above_gate"] == 1
    assert proposal["n_leftover_above_gate"] == 1
    assert proposal["leftover_not_negatives"] is True
    assert diagnostic_decision(proposal) == "lock_learned_unet_orgdet_fnn_acf_gelf_v1"


def test_propose_rejects_when_leftover_only_or_no_tp():
    leftover_only = propose_gap_length_gate(
        [{"length_um": 20.0, "chain_class": "leftover", "dt": 2}]
    )
    assert leftover_only["supported"] is False
    assert leftover_only["reason"] == "no_tp_gap_chains"
    assert diagnostic_decision(leftover_only) == "reject_learned_unet_orgdet_fnn_acf_gelf_v1"

    mixed = propose_gap_length_gate(
        [
            {"length_um": 9.4, "chain_class": "tp", "dt": 2},
            {"length_um": 9.1, "chain_class": "conflict_fp", "dt": 2},
        ]
    )
    assert mixed["supported"] is False
    assert mixed["reason"] == "no_conflict_fp_above_gate"


def test_unlocked_overlay_refuses_score():
    if MAX_GAP_EDGE_UM is None:
        with pytest.raises(SystemExit, match="did not lock a gate"):
            require_supported_gate()
        graph = _graph([1, 2], [0, 2], [0, 0], [0, 0], [0, 1], [(1, 2)])
        with pytest.raises(RuntimeError, match="unlocked"):
            apply_orgdet_fnn_acf_gelf_overlay(graph)


def test_tuning_and_holdout_gates():
    assert tuning_decision(0.7366197119237632, 10) == "reject_learned_unet_orgdet_fnn_acf_gelf_v1"
    assert tuning_decision(0.7366197119237633, 10) == "keep_candidate_learned_unet_orgdet_fnn_acf_gelf_v1"
    assert tuning_decision(0.80, 0) == "keep_as_noop_learned_unet_orgdet_fnn_acf_gelf_v1"
    assert holdout_decision(ORGDET_FNN_ACF_LOCKED_HOLDOUT) == "reject_learned_unet_orgdet_fnn_acf_gelf_v1"
    assert holdout_decision(ORGDET_FNN_ACF_LOCKED_HOLDOUT + 1e-6) == "keep_candidate_learned_unet_orgdet_fnn_acf_gelf_v1"


def test_require_tuning_keep_and_refuse_second_holdout(tmp_path: Path):
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit, match="tuning report missing"):
        require_tuning_keep(missing)
    reject = tmp_path / "reject.json"
    reject.write_text(json.dumps({"decision": "reject_learned_unet_orgdet_fnn_acf_gelf_v1"}))
    with pytest.raises(SystemExit, match="tuning decision"):
        require_tuning_keep(reject)
    hold = tmp_path / "hold.json"
    hold.write_text(json.dumps({"status": "holdout_scored", "learned_orgdet_fnn_acf_gelf_combined": 0.7}))
    with pytest.raises(SystemExit, match="score frozen holdout twice"):
        refuse_existing_holdout_score(hold)


def test_runtime_estimate_under_kaggle_limit():
    est = runtime_estimate()
    assert est["under_limit"] is True
    assert est["is_kaggle_run"] is False
    assert est["is_leaderboard_score"] is False
    assert est["prefers_tuning_zarrs"] is True


def test_rejected_diag_report_did_not_score_overlay():
    report = json.loads((ROOT / "reports" / "learned_unet_orgdet_fnn_acf_gelf_v1_diag.json").read_text())
    assert report["decision"] == "reject_learned_unet_orgdet_fnn_acf_gelf_v1"
    assert report["config_hash"] == LOCKED_ORGDET_FNN_ACF_GELF_HASH == "59812f90385ed4d2"
    assert report["proposal"]["reason"] == "no_conflict_fp_above_gate"
    assert report["proposal"]["supported"] is False
    assert report["proposal"]["n_tp"] == 14
    assert report["proposal"]["n_conflict_fp"] == 4
    assert report["proposal"]["n_leftover"] == 2235
    assert report["proposal"]["n_conflict_fp_above_gate"] == 0
    assert report["proposal"]["leftover_not_negatives"] is True
    assert report["official_overlay_scored"] is False
    assert report["opened_holdout"] is False
    assert report["is_leaderboard_score"] is False
    assert MAX_GAP_EDGE_UM is None


def test_overlay_summary_counts():
    before = _graph([1, 2, 3], [0, 2, 1], [0, 0, 0], [0, 0, 0], [0, 40, 20], [(1, 3), (3, 2)])
    max_um = 8.0 if MAX_GAP_EDGE_UM is None else float(MAX_GAP_EDGE_UM)
    after = drop_long_gap_edges(before, source_ids={1, 2}, max_um=max_um)
    after.metadata["overlay"] = "learned_unet_orgdet_fnn_acf_gelf_v1"
    after.metadata["config_hash"] = LOCKED_ORGDET_FNN_ACF_GELF_HASH
    summary = overlay_summary(before, after)
    assert summary["overlay"] == "learned_unet_orgdet_fnn_acf_gelf_v1"
    assert summary["config_hash"] == LOCKED_ORGDET_FNN_ACF_GELF_HASH
    assert summary["n_nodes_after"] == 3
