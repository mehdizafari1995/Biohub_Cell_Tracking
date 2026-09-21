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

from biohub_tracking.learned_orgdet import LOCKED_ORGDET_HASH  # noqa: E402
from biohub_tracking.learned_orgop import (  # noqa: E402
    BANNED_PUBLIC_WEIGHT_MARKERS,
    LOCKED_ORGOP_HASH as REJECTED_ORGOP_HASH,
    PRUNE_LOCKED_HOLDOUT,
)
from biohub_tracking.learned_orgdet_holdout import (  # noqa: E402
    HOLDOUT_STEMS,
    INFER_KERNEL_SLUG,
    LOCKED_ORGDET_HOLDOUT_HASH,
    ORGDET_HOLDOUT_KERNEL_SLUG,
    ORGDET_KERNEL_SLUG,
    ORGOP_KERNEL_SLUG,
    TRAIN_CONFIG_HASH,
    TRAIN_KERNEL_SLUG,
    TUNING_INFER_STEMS,
    assert_holdout_only,
    assert_no_submission_csv,
    graph_from_payload,
    graph_payload,
    holdout_decision,
    orgdet_holdout_config_hash,
    orgdet_holdout_config_payload,
    public_lb_not_a_tune_target,
    refuse_existing_holdout_score,
    refuse_non_holdout_stems,
    refuse_public_orgdet_holdout_weights,
    refuse_tuning_stems,
    require_orgdet_tuning_keep,
    runtime_estimate,
    write_graph_json,
)
from biohub_tracking.learned_split import HoldoutLeakError  # noqa: E402
from kaggle_learned_unet_orgdet_holdout_v1 import (  # noqa: E402
    CONFIG_LABEL,
    LOCKED_ORGDET_HOLDOUT_HASH as NB_HASH,
    HoldoutLeakError as NbHoldoutLeakError,
    orgdet_holdout_config_hash as nb_hash,
    orgdet_holdout_config_payload as nb_payload,
    main as holdout_main,
    refuse_public_weight_sources as nb_refuse_public,
)

_MAKE = importlib.util.spec_from_file_location("make_kaggle_kernel", ROOT / "scripts" / "make_kaggle_kernel.py")
assert _MAKE and _MAKE.loader
_make_mod = importlib.util.module_from_spec(_MAKE)
_MAKE.loader.exec_module(_make_mod)


def test_orgdet_holdout_payload_matches_notebook_yaml_and_locked_hash():
    payload = orgdet_holdout_config_payload()
    assert payload == nb_payload()
    assert orgdet_holdout_config_hash(payload) == nb_hash(payload)
    assert orgdet_holdout_config_hash(payload) == LOCKED_ORGDET_HOLDOUT_HASH
    assert LOCKED_ORGDET_HOLDOUT_HASH == "912f0c19b572a054"
    assert LOCKED_ORGDET_HOLDOUT_HASH == NB_HASH
    assert payload["orgdet_config_hash"] == LOCKED_ORGDET_HASH == "80b20589198148b7"
    assert payload["rejected_orgop_hash"] == REJECTED_ORGOP_HASH == "250c5bfacd4a8198"
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_orgdet_holdout_v1.yaml").read_text())
    assert yaml_cfg["label"] == CONFIG_LABEL
    assert yaml_cfg["infer_stems"] == list(HOLDOUT_STEMS)
    assert yaml_cfg["writes_submission_csv"] is False
    assert yaml_cfg["one_shot_holdout"] is True
    assert yaml_cfg["kernel_slug"] == ORGDET_HOLDOUT_KERNEL_SLUG
    assert yaml_cfg["train_kernel_slug"] == TRAIN_KERNEL_SLUG
    assert yaml_cfg["infer_kernel_slug"] == INFER_KERNEL_SLUG
    assert yaml_cfg["orgdet_kernel_slug"] == ORGDET_KERNEL_SLUG
    assert yaml_cfg["orgop_kernel_slug"] == ORGOP_KERNEL_SLUG
    assert yaml_cfg["linker"] == "greedy_gate_12um"
    assert yaml_cfg["motion_model"] == "last_position"
    assert yaml_cfg["assignment"] == "greedy"
    assert yaml_cfg["gate_um"] == 12.0
    assert yaml_cfg["det_threshold"] == 0.99
    assert yaml_cfg["pool_kernel_um"] == 3.0
    assert yaml_cfg["uses_transformer"] is False
    assert yaml_cfg["uses_ilp"] is False
    assert yaml_cfg["retunes_detection"] is False
    assert yaml_cfg["train_config_hash"] == TRAIN_CONFIG_HASH
    assert ORGDET_HOLDOUT_KERNEL_SLUG != ORGOP_KERNEL_SLUG
    assert ORGDET_HOLDOUT_KERNEL_SLUG != INFER_KERNEL_SLUG
    assert "holdout-v1" in ORGDET_HOLDOUT_KERNEL_SLUG
    assert PRUNE_LOCKED_HOLDOUT == 0.7013778258203534


def test_orgdet_holdout_refuses_tuning_and_non_holdout_stems():
    refuse_non_holdout_stems(HOLDOUT_STEMS)
    try:
        refuse_non_holdout_stems(["44b6_12dfb391"])
        raise AssertionError("expected HoldoutLeakError")
    except HoldoutLeakError:
        pass
    try:
        refuse_tuning_stems(TUNING_INFER_STEMS)
        raise AssertionError("expected HoldoutLeakError")
    except HoldoutLeakError:
        pass
    try:
        refuse_non_holdout_stems(["6bba_aaa"])
        raise AssertionError("expected HoldoutLeakError")
    except HoldoutLeakError:
        pass
    try:
        assert_holdout_only("tuning")
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert "validation" in str(exc)
    assert_holdout_only("validation")


def test_orgdet_holdout_graph_write_refuses_tuning(tmp_path: Path):
    coords = np.array([[0, 1, 2, 3]], dtype=np.int16)
    payload = graph_payload("6bba_05db0fb1", coords, [])
    path = write_graph_json(tmp_path / "6bba_05db0fb1.json", payload)
    graph = graph_from_payload(json.loads(path.read_text()))
    assert graph.dataset == "6bba_05db0fb1"
    assert graph.n_nodes == 1
    try:
        write_graph_json(tmp_path / "bad.json", graph_payload("44b6_12dfb391", coords, []))
        raise AssertionError("expected HoldoutLeakError")
    except HoldoutLeakError:
        pass


def test_refuse_existing_holdout_score_and_require_keep(tmp_path: Path):
    empty = tmp_path / "missing.json"
    refuse_existing_holdout_score(empty)
    pending = tmp_path / "pending.json"
    pending.write_text(json.dumps({"status": "packaged"}))
    refuse_existing_holdout_score(pending)
    scored = tmp_path / "scored.json"
    scored.write_text(json.dumps({"status": "holdout_scored", "learned_combined": 0.5, "decision": "x"}))
    try:
        refuse_existing_holdout_score(scored)
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert "twice" in str(exc)
    try:
        require_orgdet_tuning_keep(tmp_path / "absent.json")
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert "missing" in str(exc)
    rejected = tmp_path / "reject.json"
    rejected.write_text(json.dumps({"decision": "reject_learned_unet_orgdet_v1", "config_hash": LOCKED_ORGDET_HASH}))
    try:
        require_orgdet_tuning_keep(rejected)
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert "reject" in str(exc)
    keep = tmp_path / "keep.json"
    keep.write_text(
        json.dumps(
            {
                "decision": "keep_candidate_learned_unet_orgdet_v1",
                "config_hash": LOCKED_ORGDET_HASH,
            }
        )
    )
    assert require_orgdet_tuning_keep(keep)["decision"].startswith("keep_")
    assert holdout_decision(PRUNE_LOCKED_HOLDOUT) == "reject_learned_unet_orgdet_holdout_v1"
    assert holdout_decision(PRUNE_LOCKED_HOLDOUT + 1e-9) == "keep_candidate_learned_unet_orgdet_holdout_v1"


def test_refuses_public_weights_and_public_lb_tune_target():
    try:
        refuse_public_orgdet_holdout_weights(["pilkwang/biohub-deepcenter-unet3d-center-prior"])
        raise AssertionError("expected HoldoutLeakError")
    except HoldoutLeakError:
        pass
    try:
        nb_refuse_public(["giorgosi/biohub-divnet"])
        raise AssertionError("expected HoldoutLeakError")
    except (HoldoutLeakError, NbHoldoutLeakError):
        pass
    refuse_public_orgdet_holdout_weights(["mahdimzafari/biohub-learned-unet-v1"])
    try:
        public_lb_not_a_tune_target("0.553")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    public_lb_not_a_tune_target(0.701378)
    assert "pilkwang/biohub-deepcenter" in BANNED_PUBLIC_WEIGHT_MARKERS


def test_runtime_estimate_holdout_and_hidden_under_12h():
    holdout = runtime_estimate(2)
    assert holdout["under_limit"] is True
    assert holdout["estimate_hours"] < 0.2
    hidden = runtime_estimate(199)
    assert hidden["under_limit"] is True
    assert hidden["estimate_hours"] < 12.0
    assert hidden["estimate_hours"] < 6.0


def test_kernel_is_helper_not_submission_and_packages_gpu(tmp_path: Path, monkeypatch):
    source = (ROOT / "notebooks" / "kaggle_learned_unet_orgdet_holdout_v1.py").read_text()
    assert "to_csv(" not in source
    assert "graphs_to_submission" not in source
    assert "writes_submission_csv" in source
    assert "one_shot_holdout" in source
    assert "44b6_12dfb391" in source
    assert "refuse_tuning_stems" in source
    assert "0.99" in source
    assert "pool_kernel_um" in source
    spec = _make_mod.KERNELS["learned_unet_orgdet_holdout_v1"]
    assert spec["kernel_id"] == ORGDET_HOLDOUT_KERNEL_SLUG
    assert spec["kernel_id"] != ORGOP_KERNEL_SLUG
    assert spec.get("enable_gpu") is True
    assert TRAIN_KERNEL_SLUG in spec.get("kernel_sources", [])
    meta = _make_mod.metadata_payload(spec)
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is False
    assert meta["is_private"] is True
    assert meta["id"] == ORGDET_HOLDOUT_KERNEL_SLUG
    assert TRAIN_KERNEL_SLUG in meta["kernel_sources"]
    nb = _make_mod.build_notebook(source, spec["markdown"], enable_gpu=True)
    assert nb["metadata"]["kaggle"]["accelerator"] == "gpu"
    assert nb["metadata"]["kaggle"]["isInternetEnabled"] is False

    train = tmp_path / "data" / "train"
    train.mkdir(parents=True)
    for name in HOLDOUT_STEMS:
        (train / f"{name}.zarr").mkdir()
    monkeypatch.setenv("BIOHUB_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("BIOHUB_LEARNED_DRY_RUN", "1")
    monkeypatch.setenv("BIOHUB_WORKING", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    result = holdout_main()
    assert result["status"] == "dry_run"
    assert result["config_hash"] == LOCKED_ORGDET_HOLDOUT_HASH
    assert result["is_competition_submission"] is False
    assert not (tmp_path / "submission.csv").exists()
    assert_no_submission_csv(tmp_path)


def test_blocked_no_weights_records_search_and_writes_no_submission(tmp_path: Path, monkeypatch):
    train = tmp_path / "data" / "train"
    train.mkdir(parents=True)
    for name in HOLDOUT_STEMS:
        (train / f"{name}.zarr").mkdir()
    monkeypatch.setenv("BIOHUB_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("BIOHUB_WORKING", str(tmp_path))
    monkeypatch.delenv("BIOHUB_LEARNED_DRY_RUN", raising=False)
    monkeypatch.delenv("BIOHUB_LEARNED_WEIGHTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    result = holdout_main()
    assert result["status"] == "blocked_no_weights"
    assert result["config_hash"] == LOCKED_ORGDET_HOLDOUT_HASH
    assert "weight_search" in result
    assert result["is_competition_submission"] is False
    assert not (tmp_path / "submission.csv").exists()


def test_push_forbids_locked_slugs_and_fetch_allows_orgdet_holdout():
    spec = importlib.util.spec_from_file_location(
        "push_kaggle_kernel",
        ROOT / "scripts" / "push_kaggle_kernel.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert TRAIN_KERNEL_SLUG in mod.FORBIDDEN_SLUGS
    assert INFER_KERNEL_SLUG in mod.FORBIDDEN_SLUGS
    assert ORGOP_KERNEL_SLUG in mod.FORBIDDEN_SLUGS
    assert "mahdimzafari/biohub-learned-unet-holdout-v1" in mod.FORBIDDEN_SLUGS
    assert "mahdimzafari/biohub-learned-unet-prune-test-v1" in mod.FORBIDDEN_SLUGS
    assert ORGDET_HOLDOUT_KERNEL_SLUG in mod.FORBIDDEN_SLUGS
    fetch = importlib.util.spec_from_file_location(
        "fetch_kernel_output",
        ROOT / "scripts" / "fetch_kernel_output.py",
    )
    assert fetch and fetch.loader
    fetch_mod = importlib.util.module_from_spec(fetch)
    fetch.loader.exec_module(fetch_mod)
    assert ORGDET_HOLDOUT_KERNEL_SLUG in fetch_mod.ALLOWED_SLUGS
    assert ORGOP_KERNEL_SLUG in fetch_mod.ALLOWED_SLUGS
    assert ORGDET_KERNEL_SLUG in fetch_mod.ALLOWED_SLUGS


def test_keep_orgdet_holdout_report_stays_6bba_and_beats_prune():
    report = json.loads((ROOT / "reports" / "learned_unet_orgdet_holdout_v1_holdout.json").read_text())
    assert report["decision"] == "keep_candidate_learned_unet_orgdet_holdout_v1"
    assert report["config_hash"] == LOCKED_ORGDET_HOLDOUT_HASH
    assert report["learned_combined"] > PRUNE_LOCKED_HOLDOUT
    assert report["is_leaderboard_score"] is False
    stems = [row["dataset"] for row in report["per_sequence"]]
    assert stems == list(HOLDOUT_STEMS)
    assert all(row["num_pred_nodes"] > 0 for row in report["per_sequence"])
    error = json.loads((ROOT / "reports" / "learned_unet_orgdet_holdout_v1_error.json").read_text())
    assert error["opened_holdout"] is True
    assert error["one_shot"] is True
    assert error["n_pred_nodes"] == [84328, 37482]
