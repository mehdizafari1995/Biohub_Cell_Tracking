from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from biohub_tracking.learned_orgdet_fnn_oracle import (
    LINKER_GATE_UM,
    LOCKED_RANK_HASH,
    pair_in_candidate,
)
from biohub_tracking.learned_orgdet_fnn_rank_canary import (
    LOCKED_CANARY_HASH,
    build_envelope_hard_negative_pairs,
    canary_config_hash,
    canary_config_payload,
    evaluate_pair_scores,
    pair_batches,
    public_lb_not_a_tune_target,
    refuse_holdout_design,
)
from biohub_tracking.learned_split import HoldoutLeakError, TRAIN_CONFIG_HASH

ROOT = Path(__file__).resolve().parents[1]


def test_canary_hash_is_immutable_and_matches_yaml():
    payload = canary_config_payload()
    cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_orgdet_fnn_rank_canary_v1.yaml").read_text())
    assert canary_config_hash() == "f7dc3d3292ce813d"
    assert LOCKED_CANARY_HASH == "f7dc3d3292ce813d"
    assert payload["parent_rank_hash"] == LOCKED_RANK_HASH == "2748c4a2b2823c0a"
    assert payload["parent_fnn_hash"] == "95e9300eb84ba71c"
    assert payload["train_config_hash"] == TRAIN_CONFIG_HASH == "4c38035d6657f2ad"
    assert payload["kernel_push"] is False
    assert payload["opens_6bba"] is False
    assert payload["writes_submission_csv"] is False
    assert payload["scores_official_44b6"] is False
    assert payload["launches_ranker_helper"] is False
    assert payload["hard_negatives"] is True
    assert payload["candidate_gate_um"] == LINKER_GATE_UM == 12.0
    assert payload["candidate_max_dt"] == 2
    assert payload["candidate_gap_scale"] == "sqrt_dt"
    for key, value in payload.items():
        assert cfg[key] == value, key


def test_canary_refuses_holdout_and_public_lb():
    with pytest.raises(HoldoutLeakError):
        refuse_holdout_design(["6bba_05db0fb1"])
    refuse_holdout_design(["44b6_12dfb391"])
    for banned in ("0.553", "0.602", "0.612"):
        with pytest.raises(ValueError, match=banned):
            public_lb_not_a_tune_target(banned)
    public_lb_not_a_tune_target(0.7246526336456757)


def test_hard_negatives_stay_inside_locked_envelope():
    pairs = build_envelope_hard_negative_pairs()
    assert pairs["dataset"] == "44b6_canary"
    assert pairs["opened_holdout"] is False
    assert pairs["n_hard"] >= 1
    assert pairs["n_true"] > pairs["n_hard"]
    assert pairs["n_excluded"] >= 1
    assert pairs["allows_two_children"] is True
    for row in pairs["candidates"]:
        assert pair_in_candidate(row["dt"], row["dist_um"]) is True
        assert row["in_candidate"] is True
        assert row["label"] in (0, 1)
    for row in pairs["excluded"]:
        assert pair_in_candidate(row["dt"], row["dist_um"]) is False
        assert row["label"] is None
    dts = {int(r["dt"]) for r in pairs["candidates"]}
    assert dts == {1, 2}
    two_child = next(r for r in pairs["ranking"] if len(r["true"]) >= 2)
    assert two_child["dt"] == 1
    assert len(two_child["hard"]) >= 1


def test_pair_batches_mask_only_candidates():
    pairs = build_envelope_hard_negative_pairs()
    batches = pair_batches(pairs)
    assert {int(b["dt"]) for b in batches} == {1, 2}
    n_pairs = sum(len(b["pair_index"]) for b in batches)
    assert n_pairs == pairs["n_true"] + pairs["n_hard"]
    n_rank = sum(len(b["ranking_index"]) for b in batches)
    assert n_rank >= pairs["n_hard"]
    with pytest.raises(HoldoutLeakError):
        pair_batches({"dataset": "6bba_05db0fb1", "src_zyx_vox": [], "tgt_zyx_vox": [], "candidates": [], "ranking": []})


def test_evaluate_pair_scores_requires_true_above_hard():
    batches = [
        {
            "dt": 1,
            "ranking_index": [[0, 0, 1], [0, 2, 1]],
        }
    ]
    good = evaluate_pair_scores({1: [[3.0, 0.0, 2.5]]}, batches, min_margin=0.5, min_ranking_acc=1.0)
    assert good["noncollapsed"] is True
    assert good["ranking_acc"] == 1.0
    bad = evaluate_pair_scores({1: [[0.0, 3.0, -1.0]]}, batches, min_margin=0.5, min_ranking_acc=1.0)
    assert bad["noncollapsed"] is False


def test_canary_is_not_a_kernel_and_does_not_overwrite_rank_hash():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "push_kaggle_kernel", ROOT / "scripts" / "push_kaggle_kernel.py"
    )
    assert spec and spec.loader
    push = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(push)
    spec2 = importlib.util.spec_from_file_location(
        "make_kaggle_kernel", ROOT / "scripts" / "make_kaggle_kernel.py"
    )
    assert spec2 and spec2.loader
    make = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(make)
    assert "learned_unet_orgdet_fnn_rank_canary_v1" not in make.KERNELS
    assert "learned_unet_orgdet_fnn_rank_v1" not in make.KERNELS
    assert "mahdimzafari/biohub-learned-unet-orgdet-fnn-fast-test-v1" in push.FORBIDDEN_SLUGS
    assert LOCKED_RANK_HASH == "2748c4a2b2823c0a"


def test_canary_report_is_44b6_only_when_present():
    path = ROOT / "reports" / "learned_unet_orgdet_fnn_rank_canary_v1.json"
    if not path.exists():
        pytest.skip("canary report not written yet")
    report = json.loads(path.read_text())
    assert report["status"] == "canary_ok"
    assert report["decision"] == "canary_ok"
    assert report["config_hash"] == "f7dc3d3292ce813d"
    assert report["parent_rank_hash"] == "2748c4a2b2823c0a"
    assert report["result"]["opened_holdout"] is False
    assert report["result"]["used_199_video_weights"] is False
    assert report["result"]["kernel_pushed"] is False
    assert report["result"]["scored_official_44b6"] is False
    assert report["result"]["launched_ranker_helper"] is False
    assert report["result"]["checkpoint_reload_exact"] is True
    assert report["result"]["nonzero_edge_grad"] is True
    assert report["result"]["noncollapsed_pair_scores"] is True
    assert report["result"]["scores"]["ranking_acc"] == 1.0
    assert report["result"]["pair_graph"]["allows_two_children"] is True
    assert report["is_kaggle_run"] is False
    assert report["is_leaderboard_score"] is False
