from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from biohub_tracking.learned_link import (  # noqa: E402
    INFER_KERNEL_SLUG,
    INFER_LOCKED_TUNE,
    INFER_STEMS,
    LINK_KERNEL_SLUG,
    LOCKED_INFER_HASH,
    LOCKED_LINK_HASH,
    TRAIN_CONFIG_HASH,
    TRAIN_KERNEL_SLUG,
    assign_transformer_links,
    extract_pos_features,
    find_weights,
    graph_from_payload,
    graph_payload,
    link_config_hash,
    link_config_payload,
    refuse_non_infer_stems,
    runtime_estimate,
    verify_train_manifest,
    write_graph_json,
    assert_tuning_only,
)
from biohub_tracking.learned_split import HoldoutLeakError  # noqa: E402
from kaggle_learned_unet_link_v1 import (  # noqa: E402
    CONFIG_LABEL,
    assert_no_submission_csv,
    link_config_hash as nb_hash,
    link_config_payload as nb_payload,
    main as link_main,
)

_MAKE = importlib.util.spec_from_file_location("make_kaggle_kernel", ROOT / "scripts" / "make_kaggle_kernel.py")
assert _MAKE and _MAKE.loader
_make_mod = importlib.util.module_from_spec(_MAKE)
_MAKE.loader.exec_module(_make_mod)


def test_link_payload_matches_notebook_yaml_and_locked_hash():
    payload = link_config_payload()
    assert payload == nb_payload()
    assert link_config_hash(payload) == nb_hash(payload)
    assert link_config_hash(payload) == LOCKED_LINK_HASH
    assert LOCKED_LINK_HASH == "3a7423c6858ace41"
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_link_v1.yaml").read_text())
    assert yaml_cfg["label"] == CONFIG_LABEL
    assert yaml_cfg["infer_stems"] == list(INFER_STEMS)
    assert yaml_cfg["writes_submission_csv"] is False
    assert yaml_cfg["kernel_slug"] == LINK_KERNEL_SLUG
    assert yaml_cfg["train_kernel_slug"] == TRAIN_KERNEL_SLUG
    assert yaml_cfg["infer_kernel_slug"] == INFER_KERNEL_SLUG
    assert yaml_cfg["train_config_hash"] == TRAIN_CONFIG_HASH
    assert yaml_cfg["infer_config_hash"] == LOCKED_INFER_HASH
    assert yaml_cfg["linker"] == "transformer_softmax_1to1"
    assert yaml_cfg["edge_activation"] == "softmax"
    assert yaml_cfg["edge_threshold"] == 0.5
    assert yaml_cfg["normalize"] == "window_minmax"
    assert INFER_LOCKED_TUNE == 0.6557507105877639


def test_link_refuses_holdout_and_non_tuning_stems():
    refuse_non_infer_stems(INFER_STEMS)
    try:
        refuse_non_infer_stems(["6bba_05db0fb1"])
        raise AssertionError("expected HoldoutLeakError")
    except HoldoutLeakError:
        pass
    try:
        refuse_non_infer_stems(["44b6_aaa"])
        raise AssertionError("expected HoldoutLeakError")
    except HoldoutLeakError:
        pass
    try:
        assert_tuning_only("validation")
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert "holdout" in str(exc)
    assert_tuning_only("tuning")


def test_transformer_assignment_is_greedy_one_to_one(tmp_path: Path):
    probs = np.array(
        [
            [0.9, 0.2, 0.1],
            [0.3, 0.8, 0.05],
            [0.01, 0.01, 0.02],
        ]
    )
    links = assign_transformer_links(probs, threshold=0.5)
    assert links[0][:2] == (0, 0)
    assert links[1][:2] == (1, 1)
    assert all(p > 0.5 for _i, _j, p in links)
    sources = [i for i, _j, _p in links]
    targets = [j for _i, j, _p in links]
    assert len(sources) == len(set(sources))
    assert len(targets) == len(set(targets))
    empty = assign_transformer_links(np.zeros((0, 0)), threshold=0.5)
    assert empty == []
    feats = extract_pos_features(np.array([[0.0, 1.0, 2.0, 3.0]]), (10, 8, 8, 8))
    assert feats.shape == (1, 32)
    payload = graph_payload("44b6_12dfb391", np.array([[0, 1, 2, 3]]), [(1, 1)])
    path = write_graph_json(tmp_path / "44b6_12dfb391.json", payload)
    graph = graph_from_payload(json.loads(path.read_text()))
    assert graph.dataset == "44b6_12dfb391"
    try:
        write_graph_json(tmp_path / "bad.json", graph_payload("6bba_05db0fb1", np.array([[0, 1, 2, 3]]), []))
        raise AssertionError("expected HoldoutLeakError")
    except HoldoutLeakError:
        pass


def test_softmax_threshold_half_is_incompatible_with_dense_sources():
    """Document the v1 failure: softmax over hundreds of sources rarely exceeds 0.5."""
    n_src, n_tgt = 400, 80
    logits = np.zeros((n_src, n_tgt), dtype=np.float64)
    logits[0] = 3.0
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(np.clip(shifted, -40.0, 40.0))
    probs = exp / np.clip(exp.sum(axis=0, keepdims=True), 1e-8, None)
    assert float(probs.max()) < 0.5
    assert assign_transformer_links(probs, threshold=0.5) == []
    winners = assign_transformer_links(probs, threshold=0.0)
    assert winners
    assert winners[0][0] == 0
    assert len({j for _i, j, _p in winners}) == len(winners)


def test_rejected_link_tune_report_stays_44b6_and_zero_edges():
    report = json.loads((ROOT / "reports" / "learned_unet_link_v1_tune.json").read_text())
    assert report["decision"] == "reject_learned_unet_link_v1"
    assert report["config_hash"] == LOCKED_LINK_HASH
    assert report["learned_combined"] == 0.0
    assert report["is_leaderboard_score"] is False
    stems = [row["dataset"] for row in report["per_sequence"]]
    assert stems == list(INFER_STEMS)
    assert all(row["score"] == 0.0 for row in report["per_sequence"])
    assert all(row["num_pred_nodes"] > 0 for row in report["per_sequence"])
    error = json.loads((ROOT / "reports" / "learned_unet_link_v1_error.json").read_text())
    assert error["n_edges"] == [0, 0]
    assert error["detection_counts_match_infer"] is True
    assert error["opened_holdout"] is False


def test_runtime_estimate_tuning_and_hidden_under_12h():
    tune = runtime_estimate(2)
    assert tune["under_limit"] is True
    assert tune["estimate_hours"] < 0.2
    hidden = runtime_estimate(199)
    assert hidden["under_limit"] is True
    assert hidden["estimate_hours"] < 12.0


def test_kernel_is_helper_not_submission_and_packages_gpu(tmp_path: Path, monkeypatch):
    source = (ROOT / "notebooks" / "kaggle_learned_unet_link_v1.py").read_text()
    assert "to_csv(" not in source
    assert "graphs_to_submission" not in source
    assert "writes_submission_csv" in source
    assert "transformer_softmax_1to1" in source
    assert "index_features" in source
    spec = _make_mod.KERNELS["learned_unet_link_v1"]
    assert spec["kernel_id"] == LINK_KERNEL_SLUG
    assert spec.get("enable_gpu") is True
    assert TRAIN_KERNEL_SLUG in spec.get("kernel_sources", [])
    meta = _make_mod.metadata_payload(spec)
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is False
    assert meta["is_private"] is True
    assert meta["id"] == LINK_KERNEL_SLUG
    assert TRAIN_KERNEL_SLUG in meta["kernel_sources"]
    nb = _make_mod.build_notebook(source, spec["markdown"], enable_gpu=True)
    assert nb["metadata"]["kaggle"]["accelerator"] == "gpu"
    assert nb["metadata"]["kaggle"]["isInternetEnabled"] is False

    train = tmp_path / "data" / "train"
    train.mkdir(parents=True)
    for name in INFER_STEMS:
        (train / f"{name}.zarr").mkdir()
    monkeypatch.setenv("BIOHUB_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("BIOHUB_LEARNED_DRY_RUN", "1")
    monkeypatch.chdir(tmp_path)
    result = link_main()
    assert result["status"] == "dry_run"
    assert result["config_hash"] == LOCKED_LINK_HASH
    assert result["is_competition_submission"] is False
    assert not (tmp_path / "submission.csv").exists()
    assert_no_submission_csv(tmp_path)


def test_blocked_no_weights_records_search_and_writes_no_submission(tmp_path: Path, monkeypatch):
    train = tmp_path / "data" / "train"
    train.mkdir(parents=True)
    for name in INFER_STEMS:
        (train / f"{name}.zarr").mkdir()
    monkeypatch.setenv("BIOHUB_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("BIOHUB_LEARNED_DRY_RUN", raising=False)
    monkeypatch.delenv("BIOHUB_LEARNED_WEIGHTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    result = link_main()
    assert result["status"] == "blocked_no_weights"
    assert result["config_hash"] == LOCKED_LINK_HASH
    assert "weight_search" in result
    assert result["is_competition_submission"] is False
    assert not (tmp_path / "submission.csv").exists()
    assert_no_submission_csv(tmp_path)


def test_push_forbids_locked_allows_link_and_fetch_allows_link():
    spec = importlib.util.spec_from_file_location(
        "push_kaggle_kernel",
        ROOT / "scripts" / "push_kaggle_kernel.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert TRAIN_KERNEL_SLUG in mod.FORBIDDEN_SLUGS
    assert INFER_KERNEL_SLUG in mod.FORBIDDEN_SLUGS
    assert "mahdimzafari/biohub-learned-unet-gap-test-v1" in mod.FORBIDDEN_SLUGS
    assert LINK_KERNEL_SLUG in mod.FORBIDDEN_SLUGS
    fetch = importlib.util.spec_from_file_location(
        "fetch_kernel_output",
        ROOT / "scripts" / "fetch_kernel_output.py",
    )
    assert fetch and fetch.loader
    fetch_mod = importlib.util.module_from_spec(fetch)
    fetch.loader.exec_module(fetch_mod)
    assert LINK_KERNEL_SLUG in fetch_mod.ALLOWED_SLUGS
    assert find_weights is not None
    assert verify_train_manifest is not None
