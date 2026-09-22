from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from biohub_tracking.learned_collect import (  # noqa: E402
    GAP_TEST_SLUG,
    HOLDOUT_SLUG,
    INFER_SLUG,
    PRUNE_TEST_SLUG,
    TEST_SLUG,
    TRAIN_SLUG,
)
from biohub_tracking.learned_gap_test import (  # noqa: E402
    ASSOCIATION_V2_PUBLIC_SCORE,
    ASSOCIATION_V2_SUBMISSION_REF,
    ESTIMATE_HIDDEN_VOLUMES,
    GAP_TEST_KERNEL_SLUG,
)
from biohub_tracking.learned_orgdet import LOCKED_ORGDET_HASH  # noqa: E402
from biohub_tracking.learned_orgdet_holdout import (  # noqa: E402
    LOCKED_ORGDET_HOLDOUT_HASH,
    ORGDET_HOLDOUT_KERNEL_SLUG,
)
from biohub_tracking.learned_orgdet_test import (  # noqa: E402
    LOCKED_ORGDET_TEST_HASH,
    ORGDET_TEST_KERNEL_SLUG,
    classify_orgdet_test_submit_poll,
    hidden_projection_from_rehearsal,
    orgdet_test_config_hash as locked_hash,
    orgdet_test_config_payload as locked_payload,
    public_lb_not_a_tune_target,
    refuse_overwrite_locked_slugs,
    refuse_public_orgdet_test_weights,
    require_orgdet_holdout_keep,
    runtime_estimate,
)
from biohub_tracking.learned_orgop import (  # noqa: E402
    BANNED_PUBLIC_WEIGHT_MARKERS,
    LOCKED_ORGOP_HASH as REJECTED_ORGOP_HASH,
)
from biohub_tracking.learned_prune_test import PRUNE_TEST_KERNEL_SLUG  # noqa: E402
from biohub_tracking.learned_split import HoldoutLeakError  # noqa: E402
from biohub_tracking.learned_test import TEST_KERNEL_SLUG  # noqa: E402
from kaggle_learned_unet_orgdet_test_v1 import (  # noqa: E402
    CONFIG_LABEL,
    GAP_TEST_KERNEL_SLUG as NB_GAP_TEST_SLUG,
    KERNEL_SLUG,
    LOCKED_ORGDET_TEST_HASH as NB_HASH,
    PRUNE_TEST_KERNEL_SLUG as NB_PRUNE_TEST_SLUG,
    TEST_KERNEL_SLUG as NB_TEST_SLUG,
    HoldoutLeakError as NbHoldoutLeakError,
    infer_config_hash as nb_hash,
    infer_config_payload as nb_payload,
    main as run_orgdet_test_kernel,
    refuse_public_weight_sources as nb_refuse_public,
    runtime_estimate as nb_estimate,
)

_MAKE = importlib.util.spec_from_file_location("make_kaggle_kernel", ROOT / "scripts" / "make_kaggle_kernel.py")
assert _MAKE and _MAKE.loader
_make_mod = importlib.util.module_from_spec(_MAKE)
_MAKE.loader.exec_module(_make_mod)


def test_orgdet_test_payload_matches_notebook_yaml_and_locked_hash():
    payload = locked_payload()
    assert payload == nb_payload()
    assert locked_hash(payload) == nb_hash(payload)
    assert locked_hash(payload) == LOCKED_ORGDET_TEST_HASH
    assert locked_hash(payload) == NB_HASH
    assert locked_hash(payload) == "c45c859610d29e90"
    assert payload["orgdet_config_hash"] == LOCKED_ORGDET_HASH == "80b20589198148b7"
    assert payload["orgdet_holdout_config_hash"] == LOCKED_ORGDET_HOLDOUT_HASH == "912f0c19b572a054"
    assert payload["rejected_orgop_hash"] == REJECTED_ORGOP_HASH == "250c5bfacd4a8198"
    assert payload["prune_test_parent_hash"] == "074a71dec68304e8"
    assert payload["writes_submission_csv"] is True
    assert payload["discovers_test_zarrs"] is True
    assert payload["hardcoded_test_stems"] == []
    assert payload["opens_train_volumes"] is False
    assert payload["max_gap_frames"] == 0
    assert payload["drop_degree0"] is False
    assert payload["interpolate_gaps"] is False
    assert payload["linker"] == "greedy_gate_12um"
    assert payload["det_threshold"] == 0.99
    assert payload["pool_kernel_um"] == 3.0
    assert payload["kernel_slug"] == ORGDET_TEST_KERNEL_SLUG
    assert payload["prune_test_kernel_slug"] == PRUNE_TEST_KERNEL_SLUG
    assert payload["orgdet_holdout_kernel_slug"] == ORGDET_HOLDOUT_KERNEL_SLUG
    assert payload["kernel_slug"] != payload["prune_test_kernel_slug"]
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_orgdet_test_v1.yaml").read_text())
    assert yaml_cfg["label"] == CONFIG_LABEL
    assert yaml_cfg["kernel_slug"] == KERNEL_SLUG == ORGDET_TEST_KERNEL_SLUG
    assert yaml_cfg["prune_test_kernel_slug"] == PRUNE_TEST_SLUG
    assert yaml_cfg["orgdet_holdout_kernel_slug"] == ORGDET_HOLDOUT_KERNEL_SLUG
    assert yaml_cfg["det_threshold"] == 0.99
    assert yaml_cfg["pool_kernel_um"] == 3.0
    assert yaml_cfg["writes_submission_csv"] is True
    assert yaml_cfg["drop_degree0"] is False


def test_runtime_estimate_hidden_test_under_12h():
    hidden = runtime_estimate(ESTIMATE_HIDDEN_VOLUMES)
    public = runtime_estimate(4)
    assert hidden == nb_estimate(ESTIMATE_HIDDEN_VOLUMES)
    assert hidden["under_limit"] is True
    assert hidden["estimate_hours"] < 12.0
    assert 4.5 <= hidden["estimate_hours"] <= 5.5
    assert public["under_limit"] is True
    assert public["estimate_sec"] < 8 * 60
    projected = hidden_projection_from_rehearsal(347.6)
    assert projected["hidden_test"]["under_limit"] is True
    assert projected["fail_closed_approaches_12h"] is False


def test_kernel_discovers_test_and_packages_new_gpu_slug(tmp_path: Path, monkeypatch):
    source = (ROOT / "notebooks" / "kaggle_learned_unet_orgdet_test_v1.py").read_text()
    assert "apply_locked_gap_overlay" not in source
    assert "drop_degree0_arrays" not in source
    assert "submission.csv" in source
    assert "run_manifest.json" in source
    assert "list_test_zarrs" in source
    assert "INFER_STEMS" not in source
    assert "6bba_05db0fb1" not in source
    assert "44b6_0113de3b" not in source
    assert "0.99" in source
    assert "pool_kernel_um" in source
    assert KERNEL_SLUG != TEST_KERNEL_SLUG
    assert KERNEL_SLUG != GAP_TEST_KERNEL_SLUG
    assert KERNEL_SLUG != PRUNE_TEST_KERNEL_SLUG
    assert KERNEL_SLUG != ORGDET_HOLDOUT_KERNEL_SLUG
    assert NB_TEST_SLUG == TEST_KERNEL_SLUG
    assert NB_GAP_TEST_SLUG == GAP_TEST_KERNEL_SLUG
    assert NB_PRUNE_TEST_SLUG == PRUNE_TEST_KERNEL_SLUG
    spec = _make_mod.KERNELS["learned_unet_orgdet_test_v1"]
    assert spec["kernel_id"] == ORGDET_TEST_KERNEL_SLUG
    assert spec["kernel_id"] != PRUNE_TEST_SLUG
    assert spec["kernel_id"] != TEST_SLUG
    assert spec["kernel_id"] != GAP_TEST_SLUG
    assert spec.get("enable_gpu") is True
    assert TRAIN_SLUG in spec.get("kernel_sources", [])
    meta = _make_mod.metadata_payload(spec)
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is False
    assert meta["is_private"] is True
    assert meta["id"] == ORGDET_TEST_KERNEL_SLUG
    nb = _make_mod.build_notebook(source, spec["markdown"], enable_gpu=True)
    assert nb["metadata"]["kaggle"]["accelerator"] == "gpu"
    assert nb["metadata"]["kaggle"]["isInternetEnabled"] is False

    data = tmp_path / "data"
    (data / "test" / "pub_aaa.zarr").mkdir(parents=True)
    (data / "test" / "pub_bbb.zarr").mkdir()
    monkeypatch.setenv("BIOHUB_DATA_ROOT", str(data))
    monkeypatch.setenv("BIOHUB_LEARNED_DRY_RUN", "1")
    monkeypatch.chdir(tmp_path)
    result = run_orgdet_test_kernel()
    assert result["status"] == "dry_run"
    assert result["config_hash"] == LOCKED_ORGDET_TEST_HASH
    assert result["is_competition_submission"] is False
    assert result["datasets"] == ["pub_aaa", "pub_bbb"]
    assert result["estimate"]["under_limit"] is True
    assert (tmp_path / "run_manifest.json").exists()
    assert not (tmp_path / "submission.csv").exists()


def test_blocked_no_weights_records_search_and_writes_no_submission(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    (data / "test" / "pub_aaa.zarr").mkdir(parents=True)
    monkeypatch.setenv("BIOHUB_DATA_ROOT", str(data))
    monkeypatch.delenv("BIOHUB_LEARNED_DRY_RUN", raising=False)
    monkeypatch.delenv("BIOHUB_LEARNED_WEIGHTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    result = run_orgdet_test_kernel()
    assert result["status"] == "blocked_no_weights"
    assert result["config_hash"] == LOCKED_ORGDET_TEST_HASH
    assert "weight_search" in result
    assert result["is_competition_submission"] is False
    assert not (tmp_path / "submission.csv").exists()


def test_push_keeps_locked_slugs_forbidden_and_allows_new_orgdet_test_slug():
    try:
        refuse_overwrite_locked_slugs(TEST_SLUG)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        refuse_overwrite_locked_slugs(GAP_TEST_SLUG)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        refuse_overwrite_locked_slugs(PRUNE_TEST_SLUG)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        refuse_overwrite_locked_slugs(ORGDET_HOLDOUT_KERNEL_SLUG)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    refuse_overwrite_locked_slugs(ORGDET_TEST_KERNEL_SLUG)
    spec = importlib.util.spec_from_file_location(
        "push_kaggle_kernel",
        ROOT / "scripts" / "push_kaggle_kernel.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert TRAIN_SLUG in mod.FORBIDDEN_SLUGS
    assert INFER_SLUG in mod.FORBIDDEN_SLUGS
    assert HOLDOUT_SLUG in mod.FORBIDDEN_SLUGS
    assert TEST_SLUG in mod.FORBIDDEN_SLUGS
    assert GAP_TEST_SLUG in mod.FORBIDDEN_SLUGS
    assert PRUNE_TEST_SLUG in mod.FORBIDDEN_SLUGS
    assert ORGDET_HOLDOUT_KERNEL_SLUG in mod.FORBIDDEN_SLUGS
    assert ORGDET_TEST_KERNEL_SLUG in mod.FORBIDDEN_SLUGS
    fetch = importlib.util.spec_from_file_location(
        "fetch_kernel_output",
        ROOT / "scripts" / "fetch_kernel_output.py",
    )
    assert fetch and fetch.loader
    fetch_mod = importlib.util.module_from_spec(fetch)
    fetch.loader.exec_module(fetch_mod)
    assert ORGDET_TEST_KERNEL_SLUG in fetch_mod.ALLOWED_SLUGS
    assert PRUNE_TEST_SLUG in fetch_mod.ALLOWED_SLUGS
    assert GAP_TEST_SLUG in fetch_mod.ALLOWED_SLUGS
    assert TEST_SLUG in fetch_mod.ALLOWED_SLUGS
    assert ORGDET_HOLDOUT_KERNEL_SLUG in fetch_mod.ALLOWED_SLUGS


def test_refuses_public_weights_and_public_lb_tune_target(tmp_path: Path):
    try:
        refuse_public_orgdet_test_weights(["pilkwang/biohub-deepcenter-unet3d-center-prior"])
        raise AssertionError("expected HoldoutLeakError")
    except HoldoutLeakError:
        pass
    try:
        nb_refuse_public(["giorgosi/biohub-divnet"])
        raise AssertionError("expected HoldoutLeakError")
    except (HoldoutLeakError, NbHoldoutLeakError):
        pass
    refuse_public_orgdet_test_weights(["mahdimzafari/biohub-learned-unet-v1"])
    try:
        public_lb_not_a_tune_target("0.553")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    public_lb_not_a_tune_target(0.705226)
    assert "pilkwang/biohub-deepcenter" in BANNED_PUBLIC_WEIGHT_MARKERS
    keep = tmp_path / "keep.json"
    keep.write_text(
        '{"decision":"keep_candidate_learned_unet_orgdet_holdout_v1","config_hash":"912f0c19b572a054"}'
    )
    assert require_orgdet_holdout_keep(keep)["decision"].startswith("keep_")
    reject = tmp_path / "reject.json"
    reject.write_text(
        '{"decision":"reject_learned_unet_orgdet_holdout_v1","config_hash":"912f0c19b572a054"}'
    )
    try:
        require_orgdet_holdout_keep(reject)
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert "reject" in str(exc)
    assert ASSOCIATION_V2_PUBLIC_SCORE == "0.553"


def test_classify_orgdet_test_submit_poll_requires_ui_when_only_association_exists():
    association = {
        "ref": ASSOCIATION_V2_SUBMISSION_REF,
        "status": "complete",
        "publicScore": ASSOCIATION_V2_PUBLIC_SCORE,
        "private_score_returned": False,
        "date": "2026-09-20T15:12:52.307Z",
    }
    poll = classify_orgdet_test_submit_poll(
        kernel_complete=True,
        kernel_failed=False,
        output_names=["run_manifest.json", "submission.csv"],
        submissions=[association],
    )
    assert poll["decision"] == "human_action_required_ui_submit"
    assert poll["orgdet_test_submission_present"] is False
    assert poll["association_v2_submission_present"] is True
    assert poll["is_competition_submission"] is False
    assert poll["is_leaderboard_score"] is False
    assert "Kaggle UI" in poll["human_action"]
    assert "CLI" in poll["human_action"]
    assert "orgdet-test-v1" in poll["human_action"]
    assert "prune-test" in poll["human_action"]
    assert classify_orgdet_test_submit_poll(
        kernel_complete=True,
        kernel_failed=False,
        output_names=["run_manifest.json"],
        submissions=[association],
    )["decision"] == "kernel_missing_csv"
    extra = classify_orgdet_test_submit_poll(
        kernel_complete=True,
        kernel_failed=False,
        output_names=["submission.csv"],
        submissions=[association, {"ref": 999, "status": "pending", "publicScore": None}],
    )
    assert extra["decision"] == "new_submission_present"
    assert extra["orgdet_test_submission_present"] is False
    assert extra["orgdet_test_identity_proven"] is False
    assert extra["new_submission_identity"] == ["identity_unproven"]
    proven = classify_orgdet_test_submit_poll(
        kernel_complete=True,
        kernel_failed=False,
        output_names=["submission.csv"],
        submissions=[
            association,
            {
                "ref": 56466744,
                "status": "pending",
                "publicScore": "",
                "date": "2026-09-22T14:13:11.68Z",
                "kernelSlug": "mahdimzafari/biohub-learned-unet-orgdet-test-v1",
                "scriptVersionId": 351557863,
            },
        ],
    )
    assert proven["orgdet_test_submission_present"] is True
    assert proven["orgdet_test_identity_proven"] is True
    assert proven["new_submission_identity"] == ["orgdet_test"]
    assert proven["new_submission_script_version_ids"] == [351557863]
    prune = classify_orgdet_test_submit_poll(
        kernel_complete=True,
        kernel_failed=False,
        output_names=["submission.csv"],
        submissions=[
            association,
            {
                "ref": 1,
                "kernelSlug": "mahdimzafari/biohub-learned-unet-prune-test-v1",
            },
        ],
    )
    assert prune["decision"] == "new_submission_present"
    assert prune["orgdet_test_identity_proven"] is False
    assert prune["new_submission_identity"] == ["prune_test"]
    gap = classify_orgdet_test_submit_poll(
        kernel_complete=True,
        kernel_failed=False,
        output_names=["submission.csv"],
        submissions=[
            association,
            {
                "ref": 2,
                "kernelSlug": "mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1",
            },
        ],
    )
    assert gap["orgdet_test_identity_proven"] is False
    assert gap["new_submission_identity"] == ["orgdet_gap_test"]
    poll_src = (ROOT / "scripts" / "poll_orgdet_test_submit.py").read_text()
    assert "list_competition_submissions" in poll_src
    assert "competitions/submit" not in poll_src
    assert "kernels/push" not in poll_src
    live = json.loads((ROOT / "reports" / "learned_unet_orgdet_test_v1_submit_poll.json").read_text())
    assert live["orgdet_test_config_hash"] == LOCKED_ORGDET_TEST_HASH
    assert live["is_leaderboard_score"] is False
    assert live["is_competition_submission"] is False
    assert live["decision"] in {
        "human_action_required_ui_submit",
        "new_submission_present",
        "wait_kernel",
        "kernel_missing_csv",
        "record_kernel_error",
    }
    if live["decision"] == "new_submission_present":
        assert "orgdet_test_identity_proven" in live
        assert live["new_submission_identity"]
        if live["orgdet_test_identity_proven"]:
            assert live["orgdet_test_submission_present"] is True
            assert "orgdet_test" in live["new_submission_identity"]
            assert live["new_submission_kernel_slugs"] == [
                "mahdimzafari/biohub-learned-unet-orgdet-test-v1"
            ]


def test_keep_orgdet_test_complete_report_matches_holdout_overlap():
    report = json.loads((ROOT / "reports" / "learned_unet_orgdet_test_v1_complete.json").read_text())
    assert report["status"] == "inferred"
    assert report["config_hash"] == LOCKED_ORGDET_TEST_HASH
    assert report["audit_ok"] is True
    assert report["independent_audit_ok"] is True
    assert report["is_leaderboard_score"] is False
    assert report["n_nodes"] == 188509
    assert report["n_edges"] == 172148
    assert report["holdout_overlap_6bba_05db0fb1"]["matches_orgdet_holdout_v1"] is True
    assert report["holdout_overlap_6bba_05db0fb1"]["n_nodes"] == 84328
    assert report["hidden_projection"]["under_limit"] is True
    assert report["hidden_projection"]["estimate_hours"] < 12.0
