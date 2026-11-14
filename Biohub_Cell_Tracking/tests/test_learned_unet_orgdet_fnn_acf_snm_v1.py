from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from biohub_tracking.graphs import TrackGraph
from biohub_tracking.learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from biohub_tracking.learned_orgdet import LOCKED_ORGDET_HASH
from biohub_tracking.learned_orgdet_elf import LOCKED_ORGDET_ELF_HASH, MAX_DT1_EDGE_UM
from biohub_tracking.learned_orgdet_fnn import LOCKED_ORGDET_FNN_HASH, NN_TIE_EPS
from biohub_tracking.learned_orgdet_fnn_acf import (
    LEARNED_ORGDET_FNN_ACF_LOCKED_HOLDOUT,
    LEARNED_ORGDET_FNN_ACF_LOCKED_TUNE,
    LOCKED_ORGDET_FNN_ACF_HASH,
)
from biohub_tracking.learned_orgdet_fnn_acf_bnn import LOCKED_ORGDET_FNN_ACF_BNN_HASH
from biohub_tracking.learned_orgdet_fnn_acf_gacf import LOCKED_ORGDET_FNN_ACF_GACF_HASH
from biohub_tracking.learned_orgdet_fnn_acf_gelf import LOCKED_ORGDET_FNN_ACF_GELF_HASH
from biohub_tracking.learned_orgdet_fnn_acf_ipc import LOCKED_ORGDET_FNN_ACF_IPC_HASH
from biohub_tracking.learned_orgdet_fnn_acf_scf import LOCKED_ORGDET_FNN_ACF_SCF_HASH
from biohub_tracking.learned_orgdet_fnn_acf_snm import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LOCKED_GATES,
    LOCKED_ORGDET_FNN_ACF_SNM_HASH,
    MAX_GAP_DT,
    MIN_SECOND_NN_MARGIN_UM,
    ORGDET_FNN_ACF_LOCKED_HOLDOUT,
    ORGDET_FNN_ACF_LOCKED_TUNE,
    TUNE_KEEP_THRESHOLD,
    TUNE_MARGIN,
    apply_orgdet_fnn_acf_snm_overlay,
    classify_dt1_second_nn_margins,
    diagnostic_decision,
    drop_ambiguous_dt1_second_nn_edges,
    extract_dt1_second_nn_margins,
    holdout_decision,
    hop_second_nn_margin,
    orgdet_fnn_acf_snm_config_hash,
    orgdet_fnn_acf_snm_config_payload,
    overlay_summary,
    propose_second_nn_margin_gate,
    refuse_existing_holdout_score,
    refuse_non_holdout_stems,
    refuse_non_tuning_stems,
    require_supported_gate,
    require_tuning_keep,
    round_down_tenth_um,
    runtime_estimate,
    tuning_decision,
)
from biohub_tracking.learned_orgdet_fnn_acf_tac import LOCKED_ORGDET_FNN_ACF_TAC_HASH
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
    payload = orgdet_fnn_acf_snm_config_payload()
    assert orgdet_fnn_acf_snm_config_hash(payload) == LOCKED_ORGDET_FNN_ACF_SNM_HASH
    assert LOCKED_ORGDET_FNN_ACF_SNM_HASH == "87f46b2bfee49fe8"
    assert len(LOCKED_ORGDET_FNN_ACF_SNM_HASH) == 16
    assert LOCKED_ORGDET_FNN_ACF_SNM_HASH != LOCKED_ORGDET_FNN_ACF_HASH
    assert LOCKED_ORGDET_FNN_ACF_SNM_HASH != LOCKED_ORGDET_FNN_ACF_SCF_HASH
    assert LOCKED_ORGDET_FNN_ACF_SNM_HASH != LOCKED_ORGDET_FNN_ACF_TAC_HASH
    assert LOCKED_ORGDET_FNN_ACF_SNM_HASH != LOCKED_ORGDET_FNN_ACF_GELF_HASH
    assert LOCKED_ORGDET_FNN_ACF_SNM_HASH != LOCKED_ORGDET_FNN_HASH
    assert LOCKED_ORGDET_FNN_ACF_SNM_HASH != LOCKED_RANK_HASH
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
    assert payload["rejected_fnn_acf_tac_hash"] == LOCKED_ORGDET_FNN_ACF_TAC_HASH == "4dfc5514c65496e5"
    assert payload["rejected_fnn_acf_scf_hash"] == LOCKED_ORGDET_FNN_ACF_SCF_HASH == "a259d184c9f99046"
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
    assert payload["drop_ambiguous_dt1_second_nn"] is True
    assert payload["assignment"] == "unchanged_locked_orgdet_fnn_acf"
    assert payload["writes_submission_csv"] is False
    assert payload["source_zarrs"] == "locked_tuning_zarr_export_two_44b6"
    assert payload["tune_keep_threshold"] == pytest.approx(ORGDET_FNN_ACF_LOCKED_TUNE + TUNE_MARGIN)
    assert TUNE_KEEP_THRESHOLD == pytest.approx(0.7366197119237633)
    assert MAX_GAP_DT == 3
    assert MAX_DT1_EDGE_UM == 8.0
    assert NN_TIE_EPS == 1e-9
    assert REJECTED_ORGOP_HASH == "250c5bfacd4a8198"
    assert LOCKED_GATES["assignment"] == "unchanged_locked_orgdet_fnn_acf"
    assert payload["min_second_nn_margin_um"] is None
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_orgdet_fnn_acf_snm_v1.yaml").read_text())
    for key in (
        "drop_ambiguous_dt1_second_nn",
        "drop_degree0_after",
        "assignment",
        "linker",
        "orgdet_fnn_acf_config_hash",
        "rejected_fnn_acf_bnn_hash",
        "rejected_fnn_acf_ipc_hash",
        "rejected_fnn_acf_gacf_hash",
        "rejected_fnn_acf_gelf_hash",
        "rejected_fnn_acf_tac_hash",
        "rejected_fnn_acf_scf_hash",
        "tune_margin",
        "retunes_max_dt1_edge_um",
        "retunes_nn_tie_eps",
        "retunes_max_rel_appearance",
        "max_gap_dt",
        "source_zarrs",
        "min_second_nn_margin_um",
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


def test_round_down_and_second_nn_margin():
    assert round_down_tenth_um(0.37) == pytest.approx(0.3)
    assert round_down_tenth_um(0.30) == pytest.approx(0.2)
    assert round_down_tenth_um(0.0) == pytest.approx(-0.1)
    d1, d2, margin, rel = hop_second_nn_margin((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), [(4.0, 0.0, 0.0)])
    assert d1 == pytest.approx(1.0)
    assert d2 == pytest.approx(4.0)
    assert margin == pytest.approx(3.0)
    assert rel == pytest.approx(3.0)
    d1b, d2b, marginb, relb = hop_second_nn_margin((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), [])
    assert d1b == pytest.approx(1.0)
    assert d2b is None
    assert marginb is None
    assert relb is None


def test_extract_requires_second_t1_neighbor():
    unique = _graph(
        [1, 2, 3],
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 10, 40],
        [(1, 2)],
    )
    hops = extract_dt1_second_nn_margins(unique)
    assert len(hops) == 1
    assert hops[0]["src"] == 1
    assert hops[0]["dst"] == 2
    assert hops[0]["has_second"] is True
    assert hops[0]["margin_um"] == pytest.approx(30 * 0.40625)

    lone = _graph([1, 2], [0, 1], [0, 0], [0, 0], [0, 10], [(1, 2)])
    assert extract_dt1_second_nn_margins(lone) == []

    gap = _graph(
        [1, 2, 3],
        [0, 2, 2],
        [0, 0, 0],
        [0, 0, 0],
        [0, 10, 40],
        [(1, 2)],
    )
    assert extract_dt1_second_nn_margins(gap) == []


def test_drop_keeps_large_margin_and_no_second():
    graph = _graph(
        [1, 2, 3, 4, 5],
        [0, 1, 1, 5, 6],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 10, 11, 0, 80],
        [(1, 2), (4, 5)],
    )
    # 1->2 vs node 3: 10 vs 11 voxels = 0.40625 µm margin.
    # 4->5 is on a later pair of frames with no second t+1 neighbor.
    min_um = 0.5 if MIN_SECOND_NN_MARGIN_UM is None else float(MIN_SECOND_NN_MARGIN_UM)
    out = drop_ambiguous_dt1_second_nn_edges(graph, min_um=min_um)
    kept = set(map(tuple, out.edges))
    if 0.40625 < min_um:
        assert (1, 2) not in kept
        assert out.metadata["n_dropped_ambiguous_dt1_second_nn"] == 1
    assert (4, 5) in kept
    assert out.n_nodes == 5


def test_drop_keeps_unique_large_margin_under_any_positive_gate():
    unique = _graph(
        [1, 2, 3],
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 10, 80],
        [(1, 2)],
    )
    min_um = 0.1 if MIN_SECOND_NN_MARGIN_UM is None else float(MIN_SECOND_NN_MARGIN_UM)
    out = drop_ambiguous_dt1_second_nn_edges(unique, min_um=min_um)
    assert set(map(tuple, out.edges)) == {(1, 2)}
    assert out.metadata["n_dropped_ambiguous_dt1_second_nn"] == 0
    assert out.metadata["n_kept_dt1_second_nn"] == 1


def test_drop_refuses_retuned_locked_um():
    graph = _graph([1, 2, 3], [0, 1, 1], [0, 0, 0], [0, 0, 0], [0, 1, 8], [(1, 2)])
    if MIN_SECOND_NN_MARGIN_UM is not None:
        with pytest.raises(RuntimeError, match="retune min_second_nn_margin_um"):
            drop_ambiguous_dt1_second_nn_edges(graph, min_um=float(MIN_SECOND_NN_MARGIN_UM) + 0.1)


def test_classify_and_propose_gate_separates_conflict_fp():
    proposal = propose_second_nn_margin_gate(
        [
            {"margin_um": 0.37, "rel_margin": 0.2, "edge_class": "tp"},
            {"margin_um": 0.05, "rel_margin": 0.05, "edge_class": "conflict_fp"},
            {"margin_um": 0.01, "rel_margin": 0.01, "edge_class": "leftover"},
        ]
    )
    assert proposal["supported"] is True
    assert proposal["min_second_nn_margin_um"] == pytest.approx(0.3)
    assert proposal["n_conflict_fp_below_gate"] == 1
    assert proposal["n_leftover_below_gate"] == 1
    assert proposal["leftover_not_negatives"] is True
    assert proposal["relative_not_gated"] is True
    assert diagnostic_decision(proposal) == "lock_learned_unet_orgdet_fnn_acf_snm_v1"

    pred = _graph(
        [1, 2, 3],
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 10, 40],
        [(1, 2)],
    )
    gt = _graph(
        [10, 20, 30],
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 10, 40],
        [(10, 20)],
        dataset="gt",
    )
    classified = classify_dt1_second_nn_margins(pred, gt, extract_dt1_second_nn_margins(pred))
    assert classified[0]["edge_class"] == "tp"


def test_propose_rejects_when_leftover_only_or_no_separation():
    leftover_only = propose_second_nn_margin_gate(
        [{"margin_um": 0.4, "rel_margin": 0.5, "edge_class": "leftover"}]
    )
    assert leftover_only["supported"] is False
    assert leftover_only["reason"] == "no_tp_dt1_margins"
    assert diagnostic_decision(leftover_only) == "reject_learned_unet_orgdet_fnn_acf_snm_v1"

    mixed = propose_second_nn_margin_gate(
        [
            {"margin_um": 0.05, "rel_margin": 0.1, "edge_class": "tp"},
            {"margin_um": 0.40, "rel_margin": 0.4, "edge_class": "conflict_fp"},
        ]
    )
    assert mixed["supported"] is False
    assert mixed["reason"] == "tp_covers_full_margin"

    no_sep = propose_second_nn_margin_gate(
        [
            {"margin_um": 0.37, "rel_margin": 0.2, "edge_class": "tp"},
            {"margin_um": 0.40, "rel_margin": 0.3, "edge_class": "conflict_fp"},
        ]
    )
    assert no_sep["supported"] is False
    assert no_sep["reason"] == "no_conflict_fp_below_gate"


def test_unlocked_overlay_refuses_score():
    if MIN_SECOND_NN_MARGIN_UM is None:
        with pytest.raises(SystemExit, match="did not lock a gate"):
            require_supported_gate()
        graph = _graph([1, 2, 3], [0, 1, 1], [0, 0, 0], [0, 0, 0], [0, 1, 8], [(1, 2)])
        with pytest.raises(RuntimeError, match="unlocked"):
            apply_orgdet_fnn_acf_snm_overlay(graph)


def test_tuning_and_holdout_gates():
    assert tuning_decision(0.7366197119237632, 10) == "reject_learned_unet_orgdet_fnn_acf_snm_v1"
    assert tuning_decision(0.7366197119237633, 10) == "keep_candidate_learned_unet_orgdet_fnn_acf_snm_v1"
    assert tuning_decision(0.80, 0) == "keep_as_noop_learned_unet_orgdet_fnn_acf_snm_v1"
    assert holdout_decision(ORGDET_FNN_ACF_LOCKED_HOLDOUT) == "reject_learned_unet_orgdet_fnn_acf_snm_v1"
    assert holdout_decision(ORGDET_FNN_ACF_LOCKED_HOLDOUT + 1e-6) == "keep_candidate_learned_unet_orgdet_fnn_acf_snm_v1"


def test_require_tuning_keep_and_refuse_second_holdout(tmp_path: Path):
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit, match="tuning report missing"):
        require_tuning_keep(missing)
    reject = tmp_path / "reject.json"
    reject.write_text(json.dumps({"decision": "reject_learned_unet_orgdet_fnn_acf_snm_v1"}))
    with pytest.raises(SystemExit, match="tuning decision"):
        require_tuning_keep(reject)
    hold = tmp_path / "hold.json"
    hold.write_text(json.dumps({"status": "holdout_scored", "learned_orgdet_fnn_acf_snm_combined": 0.7}))
    with pytest.raises(SystemExit, match="score frozen holdout twice"):
        refuse_existing_holdout_score(hold)


def test_runtime_estimate_under_kaggle_limit():
    est = runtime_estimate()
    assert est["under_limit"] is True
    assert est["is_kaggle_run"] is False
    assert est["is_leaderboard_score"] is False
    assert est["prefers_tuning_zarrs"] is True


def test_overlay_summary_counts():
    before = _graph(
        [1, 2, 3],
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 10, 11],
        [(1, 2)],
    )
    min_um = 0.5 if MIN_SECOND_NN_MARGIN_UM is None else float(MIN_SECOND_NN_MARGIN_UM)
    after = drop_ambiguous_dt1_second_nn_edges(before, min_um=min_um)
    after.metadata["overlay"] = "learned_unet_orgdet_fnn_acf_snm_v1"
    after.metadata["config_hash"] = LOCKED_ORGDET_FNN_ACF_SNM_HASH
    summary = overlay_summary(before, after)
    assert summary["overlay"] == "learned_unet_orgdet_fnn_acf_snm_v1"
    assert summary["config_hash"] == LOCKED_ORGDET_FNN_ACF_SNM_HASH
    assert summary["n_nodes_after"] == 3
