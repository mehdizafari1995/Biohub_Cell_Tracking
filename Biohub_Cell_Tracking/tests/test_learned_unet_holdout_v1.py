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

from biohub_tracking.learned_holdout import (  # noqa: E402
    ASSOC_LOCKED_HOLDOUT,
    HOLDOUT_KERNEL_SLUG,
    HOLDOUT_STEMS,
    INFER_KERNEL_SLUG,
    LOCKED_HOLDOUT_HASH,
    TRAIN_CONFIG_HASH,
    TRAIN_KERNEL_SLUG,
    TUNING_INFER_STEMS,
    assert_holdout_only,
    assert_no_submission_csv,
    graph_from_payload,
    graph_payload,
    holdout_config_hash,
    holdout_config_payload,
    refuse_existing_holdout_score,
    refuse_non_holdout_stems,
    refuse_tuning_stems,
    write_graph_json,
)
from biohub_tracking.learned_infer import LOCKED_INFER_HASH  # noqa: E402
from biohub_tracking.learned_split import HoldoutLeakError  # noqa: E402
from kaggle_learned_unet_holdout_v1 import (  # noqa: E402
    CONFIG_LABEL,
    infer_config_hash as nb_hash,
    infer_config_payload as nb_payload,
    main as holdout_main,
)

_MAKE = importlib.util.spec_from_file_location("make_kaggle_kernel", ROOT / "scripts" / "make_kaggle_kernel.py")
assert _MAKE and _MAKE.loader
_make_mod = importlib.util.module_from_spec(_MAKE)
_MAKE.loader.exec_module(_make_mod)


def test_holdout_payload_matches_notebook_yaml_and_locked_hash():
    payload = holdout_config_payload()
    assert payload == nb_payload()
    assert holdout_config_hash(payload) == nb_hash(payload)
    assert holdout_config_hash(payload) == LOCKED_HOLDOUT_HASH
    assert payload["infer_config_hash"] == LOCKED_INFER_HASH
    assert payload["train_config_hash"] == TRAIN_CONFIG_HASH
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_holdout_v1.yaml").read_text())
    assert yaml_cfg["label"] == CONFIG_LABEL
    assert yaml_cfg["infer_stems"] == list(HOLDOUT_STEMS)
    assert yaml_cfg["writes_submission_csv"] is False
    assert yaml_cfg["one_shot_holdout"] is True
    assert yaml_cfg["kernel_slug"] == HOLDOUT_KERNEL_SLUG
    assert yaml_cfg["train_kernel_slug"] == TRAIN_KERNEL_SLUG
    assert yaml_cfg["infer_kernel_slug"] == INFER_KERNEL_SLUG
    assert yaml_cfg["linker"] == "greedy_gate_12um"
    assert yaml_cfg["gate_um"] == 12.0
    assert yaml_cfg["normalize"] == "window_minmax"
    assert ASSOC_LOCKED_HOLDOUT == 0.5915697467955445


def test_holdout_refuses_tuning_and_non_holdout_stems():
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


def test_holdout_graph_write_refuses_tuning(tmp_path: Path):
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


def test_refuse_existing_holdout_score(tmp_path: Path):
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


def test_kernel_is_helper_not_submission_and_packages_gpu(tmp_path: Path, monkeypatch):
    source = (ROOT / "notebooks" / "kaggle_learned_unet_holdout_v1.py").read_text()
    assert "to_csv(" not in source
    assert "graphs_to_submission" not in source
    assert "writes_submission_csv" in source
    assert "one_shot_holdout" in source
    assert "44b6_12dfb391" in source
    assert "refuse_tuning_stems" in source
    spec = _make_mod.KERNELS["learned_unet_holdout_v1"]
    assert spec["kernel_id"] == HOLDOUT_KERNEL_SLUG
    assert spec["kernel_id"] != INFER_KERNEL_SLUG
    assert spec.get("enable_gpu") is True
    assert TRAIN_KERNEL_SLUG in spec.get("kernel_sources", [])
    meta = _make_mod.metadata_payload(spec)
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is False
    assert meta["is_private"] is True
    assert meta["id"] == HOLDOUT_KERNEL_SLUG
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
    monkeypatch.chdir(tmp_path)
    result = holdout_main()
    assert result["status"] == "dry_run"
    assert result["config_hash"] == LOCKED_HOLDOUT_HASH
    assert result["is_competition_submission"] is False
    assert not (tmp_path / "submission.csv").exists()
    assert_no_submission_csv(tmp_path)


def test_blocked_no_weights_records_search_and_writes_no_submission(tmp_path: Path, monkeypatch):
    train = tmp_path / "data" / "train"
    train.mkdir(parents=True)
    for name in HOLDOUT_STEMS:
        (train / f"{name}.zarr").mkdir()
    monkeypatch.setenv("BIOHUB_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("BIOHUB_LEARNED_DRY_RUN", raising=False)
    monkeypatch.delenv("BIOHUB_LEARNED_WEIGHTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    result = holdout_main()
    assert result["status"] == "blocked_no_weights"
    assert result["config_hash"] == LOCKED_HOLDOUT_HASH
    assert "weight_search" in result
    assert result["is_competition_submission"] is False
    assert not (tmp_path / "submission.csv").exists()


def test_push_locks_train_infer_and_completed_holdout_slug():
    spec = importlib.util.spec_from_file_location(
        "push_kaggle_kernel",
        ROOT / "scripts" / "push_kaggle_kernel.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert TRAIN_KERNEL_SLUG in mod.FORBIDDEN_SLUGS
    assert INFER_KERNEL_SLUG in mod.FORBIDDEN_SLUGS
    assert HOLDOUT_KERNEL_SLUG in mod.FORBIDDEN_SLUGS
    fetch = importlib.util.spec_from_file_location(
        "fetch_kernel_output",
        ROOT / "scripts" / "fetch_kernel_output.py",
    )
    assert fetch and fetch.loader
    fetch_mod = importlib.util.module_from_spec(fetch)
    fetch.loader.exec_module(fetch_mod)
    assert HOLDOUT_KERNEL_SLUG in fetch_mod.ALLOWED_SLUGS
    assert INFER_KERNEL_SLUG in fetch_mod.ALLOWED_SLUGS
    assert TRAIN_KERNEL_SLUG in fetch_mod.ALLOWED_SLUGS
