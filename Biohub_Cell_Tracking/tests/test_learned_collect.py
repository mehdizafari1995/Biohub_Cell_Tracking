from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

from biohub_tracking.learned_collect import (
    GAP_TEST_CONFIG_HASH,
    GAP_TEST_SLUG,
    HOLDOUT_CONFIG_HASH,
    HOLDOUT_SLUG,
    INFER_SLUG,
    LINK_SLUG,
    LINK_V2_CONFIG_HASH,
    LINK_V2_SLUG,
    RETRAIN_CONFIG_HASH,
    RETRAIN_SLUG,
    TEST_CONFIG_HASH,
    TEST_SLUG,
    TRAIN_SLUG,
    assert_helper_dest,
    collect_decision,
    extract_learned_holdout_output,
    extract_learned_infer_output,
    extract_learned_retrain_output,
    extract_learned_test_output,
    extract_learned_train_output,
    refuse_submission_outputs,
    safe_kernel_status,
    summarize_collected_retrain,
    summarize_collected_train,
)
from biohub_tracking.learned_retrain import retrain_config_payload
from biohub_tracking.learned_split import (
    TRAIN_CONFIG_HASH,
    HoldoutLeakError,
    KERNEL_SLUG,
)

ROOT = Path(__file__).resolve().parents[1]


def _zip(path: Path, members: dict[str, bytes | str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in members.items():
            data = payload.encode() if isinstance(payload, str) else payload
            zf.writestr(name, data)
    return path


def test_collect_decision_and_safe_status():
    running = safe_kernel_status(
        {
            "status": "running",
            "hasFailureMessage": False,
            "lastRunTime": "2026-09-20T08:44:11Z",
        },
        TRAIN_SLUG,
    )
    assert running["is_running"] is True
    assert running["is_complete"] is False
    assert running["is_competition_submission"] is False
    assert running["lastRunTime"] == "2026-09-20T08:44:11Z"
    assert collect_decision(running) == "wait_running"
    complete = safe_kernel_status({"status": "complete", "hasFailureMessage": False}, TRAIN_SLUG)
    assert collect_decision(complete) == "download_complete_helper"
    errored = safe_kernel_status(
        {"status": "error", "hasFailureMessage": True, "failureMessage": "train_error"},
        TRAIN_SLUG,
    )
    assert collect_decision(errored) == "record_train_error"
    assert errored["failureMessage"] == "train_error"


def test_extract_train_zip_keeps_weights_and_refuses_submission(tmp_path: Path):
    dest = tmp_path / "artifacts" / "kaggle" / "learned_unet_v1"
    dest.mkdir(parents=True)
    assert_helper_dest(dest)
    archive = _zip(
        tmp_path / "ok.zip",
        {
            "train_manifest.json": json.dumps(
                {
                    "status": "trained",
                    "config_hash": TRAIN_CONFIG_HASH,
                    "splits": {"train": ["44b6_aaa"], "test": ["44b6_12dfb391", "44b6_267148e4"]},
                    "is_kaggle_run": True,
                }
            ),
            "nested/learned_unet_v1_best.pt": b"fake-weights",
            "__notebook__.ipynb": "{}",
            "custom.css": "x",
        },
    )
    written = extract_learned_train_output(archive, dest)
    assert set(written) == {"train_manifest.json", "learned_unet_v1_best.pt"}
    summary = summarize_collected_train(dest)
    assert summary["wrote_best"] is True
    assert summary["has_submission_csv"] is False
    assert summary["is_leaderboard_score"] is False

    banned = _zip(tmp_path / "banned.zip", {"submission.csv": "id,dataset\n"})
    with pytest.raises(ValueError, match="submission.csv"):
        extract_learned_train_output(banned, dest / "bad")
    with pytest.raises(ValueError, match="submission.csv"):
        refuse_submission_outputs(["nested/submission.csv"])


def test_extract_retrain_zip_keeps_weights_and_refuses_submission(tmp_path: Path):
    dest = tmp_path / "artifacts" / "kaggle" / "learned_unet_retrain_v1"
    dest.mkdir(parents=True)
    assert_helper_dest(dest)
    history = [
        {
            "epoch": i,
            "train_loss": 0.3 - 0.01 * i,
            "det_loss": 0.02,
            "edge_loss": 0.25,
            "transformer_grad_norm": 2.0,
        }
        for i in range(3)
    ]
    archive = _zip(
        tmp_path / "retrain_ok.zip",
        {
            "retrain_manifest.json": json.dumps(
                {
                    "status": "trained",
                    "config_hash": RETRAIN_CONFIG_HASH,
                    "config": retrain_config_payload(),
                    "splits": {"train": ["44b6_aaa"], "test": ["44b6_12dfb391", "44b6_267148e4"]},
                    "history": history,
                    "checkpoint_reload_exact": True,
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "wrote_best": True,
                    "wrote_last": True,
                    "resumed_from": "/kaggle/input/notebooks/mahdimzafari/biohub-learned-unet-v1/learned_unet_v1_last.pt",
                    "is_kaggle_run": True,
                    "is_competition_submission": False,
                    "best_train_loss": 0.28,
                }
            ),
            "learned_unet_retrain_v1_best.pt": b"fake-best",
            "learned_unet_retrain_v1_last.pt": b"fake-last",
            "dataset_splits.json": json.dumps(
                {"train": ["44b6_aaa"], "test": ["44b6_12dfb391", "44b6_267148e4"]}
            ),
        },
    )
    written = extract_learned_retrain_output(archive, dest)
    assert "retrain_manifest.json" in written
    assert "learned_unet_retrain_v1_best.pt" in written
    summary = summarize_collected_retrain(dest)
    assert summary["status"] == "trained"
    assert summary["config_hash"] == RETRAIN_CONFIG_HASH
    assert summary["n_epochs"] == 3
    assert summary["has_submission_csv"] is False
    assert RETRAIN_SLUG.endswith("biohub-learned-unet-retrain-v1")
    banned = _zip(tmp_path / "retrain_banned.zip", {"submission.csv": "id,dataset\n"})
    with pytest.raises(ValueError, match="submission.csv"):
        extract_learned_retrain_output(banned, dest / "bad")


def test_extract_train_zip_refuses_holdout_manifest_and_volumes(tmp_path: Path):
    dest = tmp_path / "out"
    leaky = _zip(
        tmp_path / "leaky.zip",
        {
            "train_manifest.json": json.dumps(
                {"splits": {"train": ["6bba_05db0fb1"], "test": []}}
            )
        },
    )
    with pytest.raises(HoldoutLeakError):
        extract_learned_train_output(leaky, dest)
    volumes = _zip(tmp_path / "vol.zip", {"44b6_12dfb391.zarr": b"nope"})
    with pytest.raises(ValueError, match="volume/GEFF"):
        extract_learned_train_output(volumes, dest)
    unexpected = _zip(tmp_path / "extra.zip", {"weights.bin": b"nope"})
    with pytest.raises(ValueError, match="unexpected"):
        extract_learned_train_output(unexpected, dest)


def test_extract_infer_zip_keeps_tuning_graphs_only(tmp_path: Path):
    dest = tmp_path / "artifacts" / "kaggle" / "learned_unet_infer_v1"
    dest.mkdir(parents=True)
    archive = _zip(
        tmp_path / "infer.zip",
        {
            "infer_manifest.json": json.dumps({"status": "inferred"}),
            "graphs/44b6_12dfb391.json": json.dumps({"dataset": "44b6_12dfb391"}),
            "graphs/44b6_267148e4.json": json.dumps({"dataset": "44b6_267148e4"}),
        },
    )
    written = extract_learned_infer_output(archive, dest)
    assert (dest / "graphs" / "44b6_12dfb391.json").exists()
    assert set(written) == {
        "infer_manifest.json",
        "44b6_12dfb391.json",
        "44b6_267148e4.json",
    }
    holdout = _zip(
        tmp_path / "holdout.zip",
        {"graphs/6bba_05db0fb1.json": json.dumps({"dataset": "6bba_05db0fb1"})},
    )
    with pytest.raises(HoldoutLeakError):
        extract_learned_infer_output(holdout, dest / "bad")


def test_extract_holdout_zip_keeps_6bba_graphs_only(tmp_path: Path):
    dest = tmp_path / "artifacts" / "kaggle" / "learned_unet_holdout_v1"
    dest.mkdir(parents=True)
    archive = _zip(
        tmp_path / "holdout.zip",
        {
            "infer_manifest.json": json.dumps({"status": "inferred"}),
            "graphs/6bba_05db0fb1.json": json.dumps({"dataset": "6bba_05db0fb1"}),
            "graphs/6bba_c328f2fd.json": json.dumps({"dataset": "6bba_c328f2fd"}),
        },
    )
    written = extract_learned_holdout_output(archive, dest)
    assert (dest / "graphs" / "6bba_05db0fb1.json").exists()
    assert set(written) == {
        "infer_manifest.json",
        "6bba_05db0fb1.json",
        "6bba_c328f2fd.json",
    }
    tuning = _zip(
        tmp_path / "tuning.zip",
        {"graphs/44b6_12dfb391.json": json.dumps({"dataset": "44b6_12dfb391"})},
    )
    with pytest.raises(HoldoutLeakError):
        extract_learned_holdout_output(tuning, dest / "bad")
    banned = _zip(tmp_path / "sub.zip", {"submission.csv": "id,dataset\n"})
    with pytest.raises(ValueError, match="submission.csv"):
        extract_learned_holdout_output(banned, dest / "sub")


def test_extract_test_zip_keeps_submission_and_refuses_weights(tmp_path: Path):
    dest = tmp_path / "artifacts" / "kaggle" / "learned_unet_test_v1"
    dest.mkdir(parents=True)
    archive = _zip(
        tmp_path / "test.zip",
        {
            "run_manifest.json": json.dumps({"status": "inferred", "config_hash": TEST_CONFIG_HASH}),
            "submission.csv": "id,dataset,row_type\n0,demo,node\n",
            "__notebook__.ipynb": "{}",
        },
    )
    written = extract_learned_test_output(archive, dest)
    assert set(written) == {"run_manifest.json", "submission.csv"}
    assert (dest / "submission.csv").exists()
    weights = _zip(tmp_path / "w.zip", {"learned_unet_v1_last.pt": b"nope"})
    with pytest.raises(ValueError, match="weights/volume"):
        extract_learned_test_output(weights, dest / "bad")
    extra = _zip(tmp_path / "extra.zip", {"graphs.json": "{}"})
    with pytest.raises(ValueError, match="unexpected"):
        extract_learned_test_output(extra, dest / "extra")


def test_assert_helper_dest_stays_under_artifacts_kaggle(tmp_path: Path):
    with pytest.raises(ValueError, match="artifacts/kaggle"):
        assert_helper_dest(tmp_path / "not_ignored")


def test_collect_modules_stay_stdlib_only():
    heavy = ("import numpy", "import scipy", "import torch", "from numpy", "from scipy")
    for rel in (
        "src/biohub_tracking/learned_collect.py",
        "src/biohub_tracking/learned_split.py",
        "src/biohub_tracking/kaggle_api.py",
        "src/biohub_tracking/experiments.py",
        "scripts/collect_learned_unet_v1.py",
        "scripts/collect_learned_unet_infer_v1.py",
        "scripts/collect_learned_unet_holdout_v1.py",
        "scripts/collect_learned_unet_test_v1.py",
        "scripts/collect_learned_unet_gap_test_v1.py",
        "scripts/collect_learned_unet_link_v1.py",
        "scripts/collect_learned_unet_link_v2.py",
        "src/biohub_tracking/learned_test.py",
    ):
        text = (ROOT / rel).read_text()
        for needle in heavy:
            assert needle not in text, f"{rel} must not {needle}"


def test_package_import_is_lazy():
    code = (
        "import biohub_tracking, sys\n"
        "assert 'biohub_tracking.track' not in sys.modules\n"
        "assert 'scipy' not in sys.modules\n"
        "from biohub_tracking.kaggle_api import kernel_status\n"
        "from biohub_tracking.learned_collect import collect_decision\n"
        "assert 'scipy' not in sys.modules\n"
        "assert 'numpy' not in sys.modules\n"
    )
    proc = __import__("subprocess").run(
        [__import__("sys").executable, "-c", code],
        cwd=str(ROOT),
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_collect_script_locks_train_slug_and_allows_status_only():
    spec = importlib.util.spec_from_file_location(
        "collect_learned_unet_v1",
        ROOT / "scripts" / "collect_learned_unet_v1.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.TRAIN_SLUG == KERNEL_SLUG
    assert INFER_SLUG != mod.TRAIN_SLUG
    infer_spec = importlib.util.spec_from_file_location(
        "collect_learned_unet_infer_v1",
        ROOT / "scripts" / "collect_learned_unet_infer_v1.py",
    )
    assert infer_spec and infer_spec.loader
    infer_mod = importlib.util.module_from_spec(infer_spec)
    infer_spec.loader.exec_module(infer_mod)
    assert infer_mod.INFER_SLUG == INFER_SLUG
    assert infer_mod.TRAIN_SLUG == KERNEL_SLUG
    fetch = importlib.util.spec_from_file_location(
        "fetch_kernel_output",
        ROOT / "scripts" / "fetch_kernel_output.py",
    )
    assert fetch and fetch.loader
    fetch_mod = importlib.util.module_from_spec(fetch)
    fetch.loader.exec_module(fetch_mod)
    assert KERNEL_SLUG in fetch_mod.ALLOWED_SLUGS
    assert INFER_SLUG in fetch_mod.ALLOWED_SLUGS
    holdout_spec = importlib.util.spec_from_file_location(
        "collect_learned_unet_holdout_v1",
        ROOT / "scripts" / "collect_learned_unet_holdout_v1.py",
    )
    assert holdout_spec and holdout_spec.loader
    holdout_mod = importlib.util.module_from_spec(holdout_spec)
    holdout_spec.loader.exec_module(holdout_mod)
    assert holdout_mod.HOLDOUT_SLUG == HOLDOUT_SLUG
    assert holdout_mod.HOLDOUT_CONFIG_HASH == HOLDOUT_CONFIG_HASH
    assert HOLDOUT_SLUG in fetch_mod.ALLOWED_SLUGS
    assert HOLDOUT_SLUG != INFER_SLUG
    assert HOLDOUT_SLUG != KERNEL_SLUG
    test_spec = importlib.util.spec_from_file_location(
        "collect_learned_unet_test_v1",
        ROOT / "scripts" / "collect_learned_unet_test_v1.py",
    )
    assert test_spec and test_spec.loader
    test_mod = importlib.util.module_from_spec(test_spec)
    test_spec.loader.exec_module(test_mod)
    assert test_mod.TEST_SLUG == TEST_SLUG
    assert test_mod.TEST_CONFIG_HASH == TEST_CONFIG_HASH
    assert TEST_SLUG in fetch_mod.ALLOWED_SLUGS
    assert TEST_SLUG != HOLDOUT_SLUG
    gap_spec = importlib.util.spec_from_file_location(
        "collect_learned_unet_gap_test_v1",
        ROOT / "scripts" / "collect_learned_unet_gap_test_v1.py",
    )
    assert gap_spec and gap_spec.loader
    gap_mod = importlib.util.module_from_spec(gap_spec)
    gap_spec.loader.exec_module(gap_mod)
    assert gap_mod.GAP_TEST_SLUG == GAP_TEST_SLUG
    assert gap_mod.GAP_TEST_CONFIG_HASH == GAP_TEST_CONFIG_HASH
    assert GAP_TEST_SLUG in fetch_mod.ALLOWED_SLUGS
    assert GAP_TEST_SLUG != TEST_SLUG
    link_spec = importlib.util.spec_from_file_location(
        "collect_learned_unet_link_v1",
        ROOT / "scripts" / "collect_learned_unet_link_v1.py",
    )
    assert link_spec and link_spec.loader
    link_mod = importlib.util.module_from_spec(link_spec)
    link_spec.loader.exec_module(link_mod)
    assert link_mod.LINK_SLUG == LINK_SLUG
    assert LINK_SLUG in fetch_mod.ALLOWED_SLUGS
    assert LINK_SLUG != INFER_SLUG
    assert LINK_SLUG != GAP_TEST_SLUG
    link_v2_spec = importlib.util.spec_from_file_location(
        "collect_learned_unet_link_v2",
        ROOT / "scripts" / "collect_learned_unet_link_v2.py",
    )
    assert link_v2_spec and link_v2_spec.loader
    link_v2_mod = importlib.util.module_from_spec(link_v2_spec)
    link_v2_spec.loader.exec_module(link_v2_mod)
    assert link_v2_mod.LINK_V2_SLUG == LINK_V2_SLUG
    assert LINK_V2_CONFIG_HASH == "1d4015dc499bd604"
    assert LINK_V2_SLUG in fetch_mod.ALLOWED_SLUGS
    assert LINK_V2_SLUG != LINK_SLUG
    assert LINK_V2_SLUG != INFER_SLUG
    from biohub_tracking.learned_collect import (
        ORGDET_GAP_TEST_CONFIG_HASH,
        ORGDET_GAP_TEST_SLUG,
        ORGDET_TEST_SLUG,
    )

    orgdet_gap_spec = importlib.util.spec_from_file_location(
        "collect_learned_unet_orgdet_gap_test_v1",
        ROOT / "scripts" / "collect_learned_unet_orgdet_gap_test_v1.py",
    )
    assert orgdet_gap_spec and orgdet_gap_spec.loader
    orgdet_gap_mod = importlib.util.module_from_spec(orgdet_gap_spec)
    orgdet_gap_spec.loader.exec_module(orgdet_gap_mod)
    assert orgdet_gap_mod.ORGDET_GAP_TEST_SLUG == ORGDET_GAP_TEST_SLUG
    assert ORGDET_GAP_TEST_CONFIG_HASH == "1ce4837ae16bee52"
    assert ORGDET_GAP_TEST_SLUG in fetch_mod.ALLOWED_SLUGS
    assert ORGDET_GAP_TEST_SLUG != ORGDET_TEST_SLUG
    assert ORGDET_GAP_TEST_SLUG != GAP_TEST_SLUG
