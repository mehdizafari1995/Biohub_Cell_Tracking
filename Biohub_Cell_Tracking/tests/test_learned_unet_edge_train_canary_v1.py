from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from biohub_tracking.learned_checkpoint_audit import (
    dense_softmax_rarely_exceeds_half,
    graph_candidate_counts,
    refuse_silent_partial_transformer_load,
    refuse_zero_candidate_edges,
    softmax_max_over_sources,
    transformer_keys,
)
from biohub_tracking.learned_edge_canary import (
    LOCKED_CANARY_HASH,
    canary_config_hash,
    canary_config_payload,
    public_lb_not_a_tune_target,
    refuse_holdout_design,
    runtime_projection,
)
from biohub_tracking.learned_orgop import LOCKED_ORGOP_HASH, ORGOP_KERNEL_SLUG
from biohub_tracking.learned_split import HoldoutLeakError, TRAIN_CONFIG_HASH

ROOT = Path(__file__).resolve().parents[1]


def test_canary_hash_is_immutable_and_matches_yaml():
    payload = canary_config_payload()
    cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_edge_train_canary_v1.yaml").read_text())
    assert canary_config_hash() == "1a3d524e4d922701"
    assert LOCKED_CANARY_HASH == "1a3d524e4d922701"
    assert payload["parent_orgop_hash"] == LOCKED_ORGOP_HASH == "250c5bfacd4a8198"
    assert payload["train_config_hash"] == TRAIN_CONFIG_HASH == "4c38035d6657f2ad"
    assert payload["kernel_push"] is False
    assert payload["opens_6bba"] is False
    assert payload["writes_submission_csv"] is False
    for key, value in payload.items():
        assert cfg[key] == value, key


def test_canary_refuses_holdout_and_public_lb():
    with pytest.raises(HoldoutLeakError):
        refuse_holdout_design(["6bba_05db0fb1"])
    refuse_holdout_design(["44b6_0113de3b"])
    with pytest.raises(ValueError, match="0.553"):
        public_lb_not_a_tune_target("0.553")


def test_silent_partial_transformer_load_is_refused():
    refuse_silent_partial_transformer_load([], [])
    with pytest.raises(RuntimeError, match="transformer"):
        refuse_silent_partial_transformer_load(
            ["transformer.pair_mlp.5.weight", "unet.head.weight"]
        )
    keys = [
        "unet.head.weight",
        "detect_head.weight",
        "transformer.proj.weight",
        "transformer.pair_mlp.5.weight",
    ]
    assert transformer_keys(keys) == [
        "transformer.proj.weight",
        "transformer.pair_mlp.5.weight",
    ]


def test_zero_candidate_edges_would_have_caught_orgop():
    with pytest.raises(RuntimeError, match="0 transformer candidate"):
        refuse_zero_candidate_edges([0, 0], n_nodes=[62480, 28672])
    refuse_zero_candidate_edges([12, 8], n_nodes=[100, 80])
    report = json.loads((ROOT / "reports" / "learned_unet_orgop_v1_tune.json").read_text())
    assert report["decision"] == "reject_learned_unet_orgop_v1"
    assert report["n_candidate_edges"] == [0, 0]
    with pytest.raises(RuntimeError, match="0 transformer candidate"):
        refuse_zero_candidate_edges(report["n_candidate_edges"], n_nodes=report["n_nodes"])


def test_orgop_graphs_counts_refuse_holdout_payloads():
    with pytest.raises(HoldoutLeakError):
        graph_candidate_counts([{"dataset": "6bba_05db0fb1", "node_id": [1], "candidate_edges": []}])
    counts = graph_candidate_counts(
        [
            {
                "dataset": "44b6_12dfb391",
                "node_id": [1, 2],
                "candidate_edges": [],
                "edges": [],
                "n_candidate_edges": 0,
            }
        ]
    )
    assert counts["n_candidate_edges"] == [0]
    assert counts["opened_holdout"] is False


def test_dense_softmax_threshold_explains_zero_edges():
    assert dense_softmax_rarely_exceeds_half(400, 80, peak=3.0) is True
    logits = np.zeros((4, 4), dtype=np.float64)
    logits[0, 0] = 20.0
    assert softmax_max_over_sources(logits) > 0.5
    proj = runtime_projection()
    assert proj["rows"][0]["epochs"] == 3
    assert proj["rows"][1]["under_12h"] is True
    assert proj["organizer_50_epoch_exceeds_12h"] is True
    assert proj["kernel_push"] is False


def test_canary_and_diagnosis_reports_are_44b6_only():
    canary = json.loads((ROOT / "reports" / "learned_unet_edge_train_canary_v1.json").read_text())
    assert canary["status"] == "canary_ok"
    assert canary["config_hash"] == "1a3d524e4d922701"
    assert canary["result"]["opened_holdout"] is False
    assert canary["result"]["used_199_video_weights"] is False
    assert canary["result"]["kernel_pushed"] is False
    assert canary["result"]["final_n_gt_half"] >= 1
    assert canary["result"]["checkpoint_reload_exact"] is True
    diag = json.loads((ROOT / "reports" / "learned_unet_orgop_zero_edge_diagnosis.json").read_text())
    assert diag["decision"] == "no_implementation_defect_transformer_near_init"
    assert diag["orgop_graphs"]["n_candidate_edges"] == [0, 0]
    assert diag["opened_holdout"] is False
    assert "learned_unet_orgop_fix_v1" not in diag["decision"]
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
    assert ORGOP_KERNEL_SLUG in push.FORBIDDEN_SLUGS
    assert "learned_unet_edge_train_canary_v1" not in make.KERNELS
    assert "learned_unet_orgop_fix_v1" not in make.KERNELS
