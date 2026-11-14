from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from biohub_tracking.graphs import TrackGraph
from biohub_tracking.learned_orgdet_fnn import LOCKED_ORGDET_FNN_HASH
from biohub_tracking.learned_orgdet_fnn_oracle import (
    LOCKED_RANK_HASH,
    TUNE_KEEP_THRESHOLD,
    pair_in_candidate,
    rank_config_payload,
)
from biohub_tracking.learned_orgdet_fnn_rank import (
    FEAT_DIM,
    KERNEL_SLUG,
    MAX_IN,
    MAX_OUT,
    apply_rank_overlay_with_scores,
    constrained_select,
    distance_fallback_scores,
    enumerate_candidates,
    helper_config_payload,
    holdout_decision,
    label_candidates,
    node_coord_features,
    public_lb_not_a_tune_target,
    refuse_holdout_design,
    runtime_estimate,
    tuning_decision,
)
from biohub_tracking.learned_split import HoldoutLeakError

ROOT = Path(__file__).resolve().parents[1]


def _graph(dataset: str, nodes: list[tuple[int, int, int, int, int]], edges: list[tuple[int, int]]) -> TrackGraph:
    nids = [n[0] for n in nodes]
    t = [n[1] for n in nodes]
    z = [n[2] for n in nodes]
    y = [n[3] for n in nodes]
    x = [n[4] for n in nodes]
    return TrackGraph(
        dataset=dataset,
        node_id=np.asarray(nids, dtype=np.int64),
        t=np.asarray(t, dtype=np.int64),
        z=np.asarray(z, dtype=np.int64),
        y=np.asarray(y, dtype=np.int64),
        x=np.asarray(x, dtype=np.int64),
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2) if edges else np.zeros((0, 2), dtype=np.int64),
    )


def test_locked_rank_hash_is_immutable_and_matches_yaml():
    payload = rank_config_payload()
    cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_orgdet_fnn_rank_v1.yaml").read_text())
    assert LOCKED_RANK_HASH == "2748c4a2b2823c0a"
    assert payload["parent_hash"] == LOCKED_ORGDET_FNN_HASH == "95e9300eb84ba71c"
    assert payload["writes_submission_csv"] is False
    assert payload["opened_holdout"] is False
    assert payload["uses_transformer"] is True
    assert payload["hard_negatives"] is True
    assert abs(float(payload["tune_keep_threshold"]) - 0.7266526336456757) < 1e-12
    for key, value in payload.items():
        if key == "notes":
            continue
        assert cfg[key] == value, key


def test_helper_does_not_retune_parent_or_open_holdout():
    helper = helper_config_payload()
    assert helper["config_hash"] == "2748c4a2b2823c0a"
    assert helper["kernel_slug"] == KERNEL_SLUG
    assert helper["opens_6bba"] is False
    assert helper["writes_submission_csv"] is False
    assert helper["uses_69_volume_train"] is False
    assert helper["kernel_push_default"] is False
    assert helper["max_in"] == MAX_IN == 1
    assert helper["max_out"] == MAX_OUT == 2


def test_refuses_holdout_and_public_lb():
    with pytest.raises(HoldoutLeakError):
        refuse_holdout_design(["6bba_05db0fb1"])
    refuse_holdout_design(["44b6_12dfb391"])
    for banned in ("0.553", "0.602", "0.612"):
        with pytest.raises(ValueError, match=banned):
            public_lb_not_a_tune_target(banned)
    public_lb_not_a_tune_target(0.7246526336456757)


def test_enumerate_candidates_stays_in_locked_envelope_and_allows_dt2():
    # 8 voxels in x at 0.40625 µm = 3.25 µm (inside 12). 40 voxels = 16.25 µm (outside).
    pred = _graph(
        "44b6_12dfb391",
        [
            (1, 0, 8, 8, 8),
            (2, 1, 8, 8, 16),
            (3, 1, 8, 8, 48),
            (4, 2, 8, 8, 20),
        ],
        [],
    )
    cands = enumerate_candidates(pred)
    assert cands
    dts = {int(r["dt"]) for r in cands}
    assert 1 in dts
    assert 2 in dts
    for row in cands:
        assert pair_in_candidate(row["dt"], row["dist_um"]) is True
    pairs = {(int(r["src"]), int(r["tgt"]), int(r["dt"])) for r in cands}
    assert (1, 2, 1) in pairs
    assert (1, 3, 1) not in pairs
    with pytest.raises(HoldoutLeakError):
        enumerate_candidates(_graph("6bba_05db0fb1", [(1, 0, 1, 1, 1)], []))


def test_label_candidates_marks_true_and_hard_inside_envelope():
    pred = _graph(
        "44b6_12dfb391",
        [
            (10, 0, 8, 8, 8),
            (11, 1, 8, 8, 16),
            (12, 1, 8, 8, 20),
        ],
        [],
    )
    gt = _graph(
        "44b6_12dfb391",
        [
            (100, 0, 8, 8, 8),
            (101, 1, 8, 8, 16),
        ],
        [(100, 101)],
    )
    labeled = label_candidates(pred, gt, enumerate_candidates(pred))
    by_pair = {(int(r["src"]), int(r["tgt"])): int(r["label"]) for r in labeled}
    assert by_pair[(10, 11)] == 1
    assert by_pair[(10, 12)] == 0
    assert sum(r["label"] == 0 for r in labeled) >= 1


def test_constrained_select_is_1_in_2_out_deterministic_and_unique():
    cands = [
        {"src": 1, "tgt": 10, "dt": 1, "dist_um": 1.0},
        {"src": 1, "tgt": 11, "dt": 1, "dist_um": 1.1},
        {"src": 1, "tgt": 12, "dt": 1, "dist_um": 1.2},
        {"src": 2, "tgt": 10, "dt": 1, "dist_um": 1.3},
        {"src": 2, "tgt": 11, "dt": 1, "dist_um": 9.0},
    ]
    scores = [5.0, 4.0, 3.0, 2.0, 1.0]
    kept = constrained_select(cands, scores)
    edges = [(cands[i]["src"], cands[i]["tgt"]) for i in kept]
    assert (1, 10) in edges
    assert (1, 11) in edges
    assert (1, 12) not in edges
    assert (2, 10) not in edges
    assert (2, 11) not in edges
    assert len(edges) == len(set(edges))
    ins = [t for _, t in edges]
    assert ins.count(10) == 1
    assert ins.count(11) == 1
    outs = [s for s, _ in edges]
    assert outs.count(1) == 2
    again = constrained_select(cands, scores)
    assert again == kept
    low = constrained_select(cands, [-1.0] * len(cands), min_logit=0.0)
    assert low == []
    ranked_only = constrained_select(cands, [-1.0] * len(cands))
    assert ranked_only 


def test_apply_overlay_drops_isolates_and_writes_no_duplicate_edges():
    pred = _graph(
        "44b6_267148e4",
        [
            (1, 0, 8, 8, 8),
            (2, 1, 8, 8, 16),
            (3, 1, 8, 8, 20),
            (99, 5, 8, 8, 8),
        ],
        [(1, 2)],
    )
    cands = enumerate_candidates(pred)
    scores = distance_fallback_scores(cands)
    after = apply_rank_overlay_with_scores(pred, scores, cands)
    assert after.dataset == "44b6_267148e4"
    assert 99 not in set(int(x) for x in after.node_id)
    edge_set = {(int(s), int(d)) for s, d in after.edges}
    assert len(edge_set) == after.n_edges
    assert after.metadata["config_hash"] == "2748c4a2b2823c0a"
    assert after.metadata["is_12um_assignment"] is False


def test_coord_features_are_deterministic_and_finite():
    z = np.array([8.0, 9.0])
    y = np.array([10.0, 11.0])
    x = np.array([12.0, 13.0])
    t = np.array([0.0, 1.0])
    a = node_coord_features(z, y, x, t, feat_dim=FEAT_DIM)
    b = node_coord_features(z, y, x, t, feat_dim=FEAT_DIM)
    assert a.shape == (2, 64)
    assert np.allclose(a, b)
    assert np.isfinite(a).all()


def test_gate_requires_plus_0_002_and_refuses_holdout_without_keep():
    assert abs(TUNE_KEEP_THRESHOLD - 0.7266526336456757) < 1e-12
    assert tuning_decision(0.7246526336456757) == "reject_learned_unet_orgdet_fnn_rank_v1"
    assert tuning_decision(0.72600) == "reject_learned_unet_orgdet_fnn_rank_v1"
    assert tuning_decision(TUNE_KEEP_THRESHOLD).startswith("keep_")
    assert holdout_decision(0.7520914330648746) == "reject_learned_unet_orgdet_fnn_rank_v1"
    assert holdout_decision(0.7520914330648747).startswith("keep_")


def test_runtime_estimate_prefers_two_tuning_volumes():
    est = runtime_estimate()
    assert est["n_volumes"] == 2
    assert est["uses_69_volume_train"] is False
    assert est["prefers_tuning_zarrs"] is True
    assert est["under_limit"] is True
    assert est["is_kaggle_run"] is False


def test_ranker_is_packaged_as_new_slug_and_does_not_overwrite_locked():
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
    spec3 = importlib.util.spec_from_file_location(
        "fetch_kernel_output", ROOT / "scripts" / "fetch_kernel_output.py"
    )
    assert spec3 and spec3.loader
    fetch = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(fetch)
    assert "learned_unet_orgdet_fnn_rank_v1" in make.KERNELS
    assert make.KERNELS["learned_unet_orgdet_fnn_rank_v1"]["kernel_id"] == KERNEL_SLUG
    assert KERNEL_SLUG not in push.FORBIDDEN_SLUGS
    assert "mahdimzafari/biohub-learned-unet-orgdet-fnn-fast-test-v1" in push.FORBIDDEN_SLUGS
    assert KERNEL_SLUG in fetch.ALLOWED_SLUGS
    assert make.KERNELS["learned_unet_orgdet_fnn_rank_v1"].get("enable_gpu") is True
    sources = make.KERNELS["learned_unet_orgdet_fnn_rank_v1"]["kernel_sources"]
    assert "mahdimzafari/biohub-learned-unet-orgop-v1" in sources
    assert "mahdimzafari/biohub-geff-export" in sources
    assert "mahdimzafari/biohub-tuning-zarr-export" in sources
    assert "mahdimzafari/biohub-learned-unet-orgdet-fnn-fast-test-v1" not in sources
