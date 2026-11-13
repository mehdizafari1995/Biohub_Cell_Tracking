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
from biohub_tracking.learned_orgdet_fnn_acf_gelf import LOCKED_ORGDET_FNN_ACF_GELF_HASH
from biohub_tracking.learned_orgdet_fnn_acf_ipc import LOCKED_ORGDET_FNN_ACF_IPC_HASH
from biohub_tracking.learned_orgdet_fnn_acf_tac import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LOCKED_GATES,
    LOCKED_ORGDET_FNN_ACF_TAC_HASH,
    MAX_GAP_DT,
    MAX_TURN_DEG,
    ORGDET_FNN_ACF_LOCKED_HOLDOUT,
    ORGDET_FNN_ACF_LOCKED_TUNE,
    TUNE_KEEP_THRESHOLD,
    TUNE_MARGIN,
    apply_orgdet_fnn_acf_tac_overlay,
    classify_dt1_turns,
    diagnostic_decision,
    drop_sharp_dt1_turns,
    extract_dt1_turns,
    holdout_decision,
    orgdet_fnn_acf_tac_config_hash,
    orgdet_fnn_acf_tac_config_payload,
    overlay_summary,
    propose_turning_angle_gate,
    refuse_existing_holdout_score,
    refuse_non_holdout_stems,
    refuse_non_tuning_stems,
    require_supported_gate,
    require_tuning_keep,
    round_up_degree,
    runtime_estimate,
    turning_angle_deg,
    tuning_decision,
)
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
    payload = orgdet_fnn_acf_tac_config_payload()
    assert orgdet_fnn_acf_tac_config_hash(payload) == LOCKED_ORGDET_FNN_ACF_TAC_HASH
    assert LOCKED_ORGDET_FNN_ACF_TAC_HASH == "4dfc5514c65496e5"
    assert len(LOCKED_ORGDET_FNN_ACF_TAC_HASH) == 16
    assert LOCKED_ORGDET_FNN_ACF_TAC_HASH != LOCKED_ORGDET_FNN_ACF_HASH
    assert LOCKED_ORGDET_FNN_ACF_TAC_HASH != LOCKED_ORGDET_FNN_ACF_GELF_HASH
    assert LOCKED_ORGDET_FNN_ACF_TAC_HASH != LOCKED_ORGDET_FNN_ACF_GACF_HASH
    assert LOCKED_ORGDET_FNN_ACF_TAC_HASH != LOCKED_ORGDET_FNN_HASH
    assert LOCKED_ORGDET_FNN_ACF_TAC_HASH != LOCKED_RANK_HASH
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
    assert payload["rejected_fnn_acf_gelf_hash"] == LOCKED_ORGDET_FNN_ACF_GELF_HASH == "59812f90385ed4d2"
    assert payload["retunes_detection"] is False
    assert payload["retunes_consecutive_links"] is False
    assert payload["retunes_gap_gates"] is False
    assert payload["retunes_max_dt1_edge_um"] is False
    assert payload["retunes_nn_tie_eps"] is False
    assert payload["retunes_max_rel_appearance"] is False
    assert payload["uses_intensity"] is False
    assert payload["uses_transformer"] is False
    assert payload["uses_ilp"] is False
    assert payload["uses_velocity"] is False
    assert payload["drop_sharp_dt1_turns"] is True
    assert payload["assignment"] == "unchanged_locked_orgdet_fnn_acf"
    assert payload["writes_submission_csv"] is False
    assert payload["source_zarrs"] == "locked_tuning_zarr_export_two_44b6"
    assert payload["tune_keep_threshold"] == pytest.approx(ORGDET_FNN_ACF_LOCKED_TUNE + TUNE_MARGIN)
    assert TUNE_KEEP_THRESHOLD == pytest.approx(0.7366197119237633)
    assert MAX_GAP_DT == 3
    assert MAX_DT1_EDGE_UM == 8.0
    assert REJECTED_ORGOP_HASH == "250c5bfacd4a8198"
    assert LOCKED_GATES["assignment"] == "unchanged_locked_orgdet_fnn_acf"
    assert payload["max_turn_deg"] is None
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_orgdet_fnn_acf_tac_v1.yaml").read_text())
    for key in (
        "drop_sharp_dt1_turns",
        "drop_degree0_after",
        "assignment",
        "linker",
        "orgdet_fnn_acf_config_hash",
        "rejected_fnn_acf_bnn_hash",
        "rejected_fnn_acf_ipc_hash",
        "rejected_fnn_acf_gacf_hash",
        "rejected_fnn_acf_gelf_hash",
        "tune_margin",
        "retunes_max_dt1_edge_um",
        "retunes_nn_tie_eps",
        "retunes_max_rel_appearance",
        "max_gap_dt",
        "source_zarrs",
        "max_turn_deg",
        "uses_intensity",
        "uses_velocity",
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


def test_round_up_degree_is_strictly_above():
    assert round_up_degree(47.0) == pytest.approx(48.0)
    assert round_up_degree(47.1) == pytest.approx(48.0)
    assert round_up_degree(0.0) == pytest.approx(1.0)


def test_turning_angle_collinear_and_right_angle():
    cosine, angle = turning_angle_deg((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    assert cosine == pytest.approx(1.0)
    assert angle == pytest.approx(0.0)
    cosine90, angle90 = turning_angle_deg((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0))
    assert cosine90 == pytest.approx(0.0)
    assert angle90 == pytest.approx(90.0)
    cosine180, angle180 = turning_angle_deg((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert cosine180 == pytest.approx(-1.0)
    assert angle180 == pytest.approx(180.0)
    degen_c, degen_a = turning_angle_deg((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert degen_c is None and degen_a is None


def test_extract_requires_unique_dt1_predecessor():
    collinear = _graph(
        [1, 2, 3],
        [0, 1, 2],
        [0, 0, 0],
        [0, 0, 0],
        [0, 10, 20],
        [(1, 2), (2, 3)],
    )
    turns = extract_dt1_turns(collinear)
    assert len(turns) == 1
    assert turns[0]["pred"] == 1
    assert turns[0]["src"] == 2
    assert turns[0]["dst"] == 3
    assert turns[0]["turning_angle_deg"] == pytest.approx(0.0)

    no_pred = _graph([1, 2], [0, 1], [0, 0], [0, 0], [0, 10], [(1, 2)])
    assert extract_dt1_turns(no_pred) == []

    gap_pred = _graph(
        [1, 2, 3],
        [0, 2, 3],
        [0, 0, 0],
        [0, 0, 0],
        [0, 10, 20],
        [(1, 2), (2, 3)],
    )
    assert extract_dt1_turns(gap_pred) == []


def test_drop_sharp_right_angle_keeps_collinear_and_dt1_without_pred():
    graph = _graph(
        [1, 2, 3, 4, 5],
        [0, 1, 2, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 10, 10, 0, 80],
        [(1, 2), (2, 3), (4, 5)],
    )
    # 1->2->3 is a 90-degree turn (x then y). 4->5 has no predecessor.
    # 10 px * 0.40625 = 4.0625 µm along x, then 0 along x / 10 along y... wait
    # node 3 is x=10, same as node 2, y still 0. That's a zero-length hop.
    # Rebuild with a y displacement for the turn.
    graph = _graph(
        [1, 2, 3, 4, 5],
        [0, 1, 2, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 10, 0, 0],
        [0, 10, 10, 0, 80],
        [(1, 2), (2, 3), (4, 5)],
    )
    max_deg = 45.0 if MAX_TURN_DEG is None else float(MAX_TURN_DEG)
    out = drop_sharp_dt1_turns(graph, max_deg=max_deg)
    kept = set(map(tuple, out.edges))
    if 90.0 > max_deg:
        assert (2, 3) not in kept
        assert out.metadata["n_dropped_sharp_dt1_turns"] == 1
    assert (1, 2) in kept
    assert (4, 5) in kept
    assert out.n_nodes == 5


def test_drop_keeps_collinear_under_any_positive_gate():
    collinear = _graph(
        [1, 2, 3],
        [0, 1, 2],
        [0, 0, 0],
        [0, 0, 0],
        [0, 10, 20],
        [(1, 2), (2, 3)],
    )
    max_deg = 1.0 if MAX_TURN_DEG is None else float(MAX_TURN_DEG)
    out = drop_sharp_dt1_turns(collinear, max_deg=max_deg)
    assert set(map(tuple, out.edges)) == {(1, 2), (2, 3)}
    assert out.metadata["n_dropped_sharp_dt1_turns"] == 0
    assert out.metadata["n_kept_dt1_turns"] == 1


def test_drop_refuses_retuned_locked_deg():
    graph = _graph([1, 2, 3], [0, 1, 2], [0, 0, 0], [0, 0, 0], [0, 1, 2], [(1, 2), (2, 3)])
    if MAX_TURN_DEG is not None:
        with pytest.raises(RuntimeError, match="retune max_turn_deg"):
            drop_sharp_dt1_turns(graph, max_deg=float(MAX_TURN_DEG) + 1.0)


def test_classify_and_propose_gate_separates_conflict_fp():
    proposal = propose_turning_angle_gate(
        [
            {"turning_angle_deg": 12.4, "cosine": 0.97, "edge_class": "tp"},
            {"turning_angle_deg": 88.0, "cosine": 0.03, "edge_class": "conflict_fp"},
            {"turning_angle_deg": 170.0, "cosine": -0.98, "edge_class": "leftover"},
        ]
    )
    assert proposal["supported"] is True
    assert proposal["max_turn_deg"] == pytest.approx(13.0)
    assert proposal["n_conflict_fp_above_gate"] == 1
    assert proposal["n_leftover_above_gate"] == 1
    assert proposal["leftover_not_negatives"] is True
    assert diagnostic_decision(proposal) == "lock_learned_unet_orgdet_fnn_acf_tac_v1"

    pred = _graph(
        [1, 2, 3],
        [0, 1, 2],
        [0, 0, 0],
        [0, 0, 0],
        [0, 10, 20],
        [(1, 2), (2, 3)],
    )
    gt = _graph(
        [10, 20, 30],
        [0, 1, 2],
        [0, 0, 0],
        [0, 0, 0],
        [0, 10, 20],
        [(10, 20), (20, 30)],
        dataset="gt",
    )
    classified = classify_dt1_turns(pred, gt, extract_dt1_turns(pred))
    assert classified[0]["edge_class"] == "tp"


def test_propose_rejects_when_leftover_only_or_no_separation():
    leftover_only = propose_turning_angle_gate(
        [{"turning_angle_deg": 90.0, "cosine": 0.0, "edge_class": "leftover"}]
    )
    assert leftover_only["supported"] is False
    assert leftover_only["reason"] == "no_tp_dt1_turns"
    assert diagnostic_decision(leftover_only) == "reject_learned_unet_orgdet_fnn_acf_tac_v1"

    mixed = propose_turning_angle_gate(
        [
            {"turning_angle_deg": 90.0, "cosine": 0.0, "edge_class": "tp"},
            {"turning_angle_deg": 45.0, "cosine": 0.7, "edge_class": "conflict_fp"},
        ]
    )
    assert mixed["supported"] is False
    assert mixed["reason"] == "no_conflict_fp_above_gate"

    u_turn = propose_turning_angle_gate(
        [
            {"turning_angle_deg": 180.0, "cosine": -1.0, "edge_class": "tp"},
            {"turning_angle_deg": 179.0, "cosine": -0.999, "edge_class": "conflict_fp"},
        ]
    )
    assert u_turn["supported"] is False
    assert u_turn["reason"] == "tp_covers_full_angle"


def test_unlocked_overlay_refuses_score():
    if MAX_TURN_DEG is None:
        with pytest.raises(SystemExit, match="did not lock a gate"):
            require_supported_gate()
        graph = _graph([1, 2, 3], [0, 1, 2], [0, 0, 0], [0, 0, 0], [0, 1, 2], [(1, 2), (2, 3)])
        with pytest.raises(RuntimeError, match="unlocked"):
            apply_orgdet_fnn_acf_tac_overlay(graph)


def test_tuning_and_holdout_gates():
    assert tuning_decision(0.7366197119237632, 10) == "reject_learned_unet_orgdet_fnn_acf_tac_v1"
    assert tuning_decision(0.7366197119237633, 10) == "keep_candidate_learned_unet_orgdet_fnn_acf_tac_v1"
    assert tuning_decision(0.80, 0) == "keep_as_noop_learned_unet_orgdet_fnn_acf_tac_v1"
    assert holdout_decision(ORGDET_FNN_ACF_LOCKED_HOLDOUT) == "reject_learned_unet_orgdet_fnn_acf_tac_v1"
    assert holdout_decision(ORGDET_FNN_ACF_LOCKED_HOLDOUT + 1e-6) == "keep_candidate_learned_unet_orgdet_fnn_acf_tac_v1"


def test_require_tuning_keep_and_refuse_second_holdout(tmp_path: Path):
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit, match="tuning report missing"):
        require_tuning_keep(missing)
    reject = tmp_path / "reject.json"
    reject.write_text(json.dumps({"decision": "reject_learned_unet_orgdet_fnn_acf_tac_v1"}))
    with pytest.raises(SystemExit, match="tuning decision"):
        require_tuning_keep(reject)
    hold = tmp_path / "hold.json"
    hold.write_text(json.dumps({"status": "holdout_scored", "learned_orgdet_fnn_acf_tac_combined": 0.7}))
    with pytest.raises(SystemExit, match="score frozen holdout twice"):
        refuse_existing_holdout_score(hold)


def test_runtime_estimate_under_kaggle_limit():
    est = runtime_estimate()
    assert est["under_limit"] is True
    assert est["is_kaggle_run"] is False
    assert est["is_leaderboard_score"] is False
    assert est["prefers_tuning_zarrs"] is True


def test_rejected_diag_report_did_not_score_overlay():
    report = json.loads((ROOT / "reports" / "learned_unet_orgdet_fnn_acf_tac_v1_diag.json").read_text())
    assert report["decision"] == "reject_learned_unet_orgdet_fnn_acf_tac_v1"
    assert report["config_hash"] == LOCKED_ORGDET_FNN_ACF_TAC_HASH == "4dfc5514c65496e5"
    assert report["proposal"]["reason"] == "tp_covers_full_angle"
    assert report["proposal"]["supported"] is False
    assert report["proposal"]["n_tp"] == 563
    assert report["proposal"]["n_conflict_fp"] == 112
    assert report["proposal"]["n_leftover"] == 53564
    assert report["proposal"]["tp_max_deg"] == pytest.approx(180.0)
    assert report["proposal"]["tp_min_cosine"] == pytest.approx(-1.0)
    assert report["proposal"]["leftover_not_negatives"] is True
    assert report["official_overlay_scored"] is False
    assert report["opened_holdout"] is False
    assert report["is_leaderboard_score"] is False
    assert MAX_TURN_DEG is None


def test_overlay_summary_counts():
    before = _graph(
        [1, 2, 3],
        [0, 1, 2],
        [0, 0, 0],
        [0, 0, 10],
        [0, 10, 10],
        [(1, 2), (2, 3)],
    )
    max_deg = 45.0 if MAX_TURN_DEG is None else float(MAX_TURN_DEG)
    after = drop_sharp_dt1_turns(before, max_deg=max_deg)
    after.metadata["overlay"] = "learned_unet_orgdet_fnn_acf_tac_v1"
    after.metadata["config_hash"] = LOCKED_ORGDET_FNN_ACF_TAC_HASH
    summary = overlay_summary(before, after)
    assert summary["overlay"] == "learned_unet_orgdet_fnn_acf_tac_v1"
    assert summary["config_hash"] == LOCKED_ORGDET_FNN_ACF_TAC_HASH
    assert summary["n_nodes_after"] == 3
