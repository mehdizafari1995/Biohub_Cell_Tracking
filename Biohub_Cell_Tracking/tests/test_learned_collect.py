from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

from biohub_tracking.learned_collect import (
    TRAIN_SLUG,
    assert_helper_dest,
    collect_decision,
    extract_learned_infer_output,
    extract_learned_train_output,
    refuse_submission_outputs,
    safe_kernel_status,
    summarize_collected_train,
)
from biohub_tracking.learned_infer import INFER_KERNEL_SLUG, TRAIN_CONFIG_HASH
from biohub_tracking.learned_split import HoldoutLeakError, KERNEL_SLUG

ROOT = Path(__file__).resolve().parents[1]


def _zip(path: Path, members: dict[str, bytes | str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in members.items():
            data = payload.encode() if isinstance(payload, str) else payload
            zf.writestr(name, data)
    return path


def test_collect_decision_and_safe_status():
    running = safe_kernel_status({"status": "running", "hasFailureMessage": False}, TRAIN_SLUG)
    assert running["is_running"] is True
    assert running["is_complete"] is False
    assert running["is_competition_submission"] is False
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


def test_assert_helper_dest_stays_under_artifacts_kaggle(tmp_path: Path):
    with pytest.raises(ValueError, match="artifacts/kaggle"):
        assert_helper_dest(tmp_path / "not_ignored")


def test_collect_script_locks_train_slug_and_allows_status_only():
    spec = importlib.util.spec_from_file_location(
        "collect_learned_unet_v1",
        ROOT / "scripts" / "collect_learned_unet_v1.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.TRAIN_SLUG == KERNEL_SLUG
    assert INFER_KERNEL_SLUG != mod.TRAIN_SLUG
    fetch = importlib.util.spec_from_file_location(
        "fetch_kernel_output",
        ROOT / "scripts" / "fetch_kernel_output.py",
    )
    assert fetch and fetch.loader
    fetch_mod = importlib.util.module_from_spec(fetch)
    fetch.loader.exec_module(fetch_mod)
    assert KERNEL_SLUG in fetch_mod.ALLOWED_SLUGS
    assert INFER_KERNEL_SLUG in fetch_mod.ALLOWED_SLUGS
