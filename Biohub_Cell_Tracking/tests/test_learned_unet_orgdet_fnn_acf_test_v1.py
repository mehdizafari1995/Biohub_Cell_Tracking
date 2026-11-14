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
    ORGDET_ELF_TEST_SLUG,
    ORGDET_FNN_ACF_TEST_CONFIG_HASH,
    ORGDET_FNN_ACF_TEST_SLUG,
    ORGDET_FNN_FAST_TEST_SLUG,
    ORGDET_FNN_TEST_SLUG,
)
from biohub_tracking.learned_orgdet_fnn_acf import LOCKED_ORGDET_FNN_ACF_HASH  # noqa: E402
from biohub_tracking.learned_orgdet_fnn_acf_test import (  # noqa: E402
    ASSOCIATION_V2_PUBLIC_SCORE,
    ASSOCIATION_V2_SUBMISSION_REF,
    LOCKED_ORGDET_FNN_ACF_TEST_HASH,
    ORGDET_FNN_ACF_TEST_KERNEL_SLUG,
    ORGDET_FNN_ACF_TEST_PRIVATE_SCORE,
    ORGDET_FNN_ACF_TEST_PUBLIC_SCORE,
    ORGDET_FNN_ACF_TEST_SCRIPT_VERSION_ID,
    ORGDET_FNN_ACF_TEST_SUBMISSION_REF,
    ORGDET_GAP_TEST_PUBLIC_SCORE,
    ORGDET_GAP_TEST_SUBMISSION_REF,
    ORGDET_TEST_PUBLIC_SCORE,
    ORGDET_TEST_SUBMISSION_REF,
    classify_orgdet_fnn_acf_test_submit_poll,
    hidden_projection_from_rehearsal,
    orgdet_fnn_acf_test_config_hash as locked_hash,
    orgdet_fnn_acf_test_config_payload as locked_payload,
    refuse_overwrite_locked_slugs,
    runtime_estimate,
)
from biohub_tracking.learned_orgdet_test import classify_submission_notebook  # noqa: E402
from biohub_tracking.learned_orgdet_fnn_fast_test import (  # noqa: E402
    LOCKED_ORGDET_FNN_FAST_TEST_HASH,
)
from kaggle_learned_unet_orgdet_fnn_acf_test_v1 import (  # noqa: E402
    KERNEL_SLUG,
    infer_config_hash as nb_hash,
    infer_config_payload as nb_payload,
    runtime_estimate as nb_estimate,
)

_MAKE = importlib.util.spec_from_file_location("make_kaggle_kernel", ROOT / "scripts" / "make_kaggle_kernel.py")
assert _MAKE and _MAKE.loader
_make_mod = importlib.util.module_from_spec(_MAKE)
_MAKE.loader.exec_module(_make_mod)

_PUSH = importlib.util.spec_from_file_location("push_kaggle_kernel", ROOT / "scripts" / "push_kaggle_kernel.py")
assert _PUSH and _PUSH.loader
_push_mod = importlib.util.module_from_spec(_PUSH)
_PUSH.loader.exec_module(_push_mod)

_FETCH = importlib.util.spec_from_file_location("fetch_kernel_output", ROOT / "scripts" / "fetch_kernel_output.py")
assert _FETCH and _FETCH.loader
_fetch_mod = importlib.util.module_from_spec(_FETCH)
_FETCH.loader.exec_module(_fetch_mod)


def test_payload_matches_notebook_yaml_and_locked_hash():
    payload = locked_payload()
    assert payload == nb_payload()
    assert locked_hash(payload) == nb_hash(payload)
    assert locked_hash(payload) == LOCKED_ORGDET_FNN_ACF_TEST_HASH
    assert locked_hash(payload) == ORGDET_FNN_ACF_TEST_CONFIG_HASH
    assert locked_hash(payload) == "2d13d0e3c90df093"
    assert locked_hash(payload) != LOCKED_ORGDET_FNN_FAST_TEST_HASH
    assert payload["orgdet_fnn_acf_config_hash"] == LOCKED_ORGDET_FNN_ACF_HASH == "a66075085ac64400"
    assert payload["orgdet_fnn_fast_test_parent_hash"] == LOCKED_ORGDET_FNN_FAST_TEST_HASH == "9b9763e26bc0f037"
    assert payload["linker"] == "orgdet_fnn_then_drop_inconsistent_dt1_appearance"
    assert payload["max_rel_appearance"] == 0.38
    assert payload["appearance_feature"] == "patch_3x3x3_mean"
    assert payload["drop_inconsistent_dt1"] is True
    assert payload["writes_submission_csv"] is True
    assert payload["discovers_test_zarrs"] is True
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_orgdet_fnn_acf_test_v1.yaml").read_text())
    assert yaml_cfg["kernel_slug"] == KERNEL_SLUG == ORGDET_FNN_ACF_TEST_SLUG == ORGDET_FNN_ACF_TEST_KERNEL_SLUG
    assert yaml_cfg["orgdet_fnn_acf_config_hash"] == payload["orgdet_fnn_acf_config_hash"]
    assert yaml_cfg["orgdet_fnn_fast_test_parent_hash"] == payload["orgdet_fnn_fast_test_parent_hash"]
    assert yaml_cfg["linker"] == payload["linker"]
    assert yaml_cfg["max_rel_appearance"] == 0.38
    assert yaml_cfg["drop_inconsistent_dt1"] is True


def test_refuses_overwrite_of_locked_slugs():
    refuse_overwrite_locked_slugs(ORGDET_FNN_ACF_TEST_KERNEL_SLUG)
    for slug, match in (
        (ORGDET_FNN_FAST_TEST_SLUG, "orgdet_fnn_fast_test"),
        (ORGDET_FNN_TEST_SLUG, "orgdet_fnn_test"),
        (ORGDET_ELF_TEST_SLUG, "orgdet_elf_test"),
    ):
        try:
            refuse_overwrite_locked_slugs(slug)
        except ValueError as exc:
            assert match in str(exc)
        else:
            raise AssertionError(f"expected refuse for {slug}")


def test_new_slug_is_locked_after_complete_and_fetchable():
    assert ORGDET_FNN_ACF_TEST_SLUG in _push_mod.FORBIDDEN_SLUGS
    assert ORGDET_FNN_FAST_TEST_SLUG in _push_mod.FORBIDDEN_SLUGS
    assert ORGDET_FNN_TEST_SLUG in _push_mod.FORBIDDEN_SLUGS
    assert ORGDET_ELF_TEST_SLUG in _push_mod.FORBIDDEN_SLUGS
    assert ORGDET_FNN_ACF_TEST_SLUG in _fetch_mod.ALLOWED_SLUGS
    assert "learned_unet_orgdet_fnn_acf_test_v1" in _make_mod.KERNELS
    assert _make_mod.KERNELS["learned_unet_orgdet_fnn_acf_test_v1"]["kernel_id"] == ORGDET_FNN_ACF_TEST_SLUG
    assert _make_mod.KERNELS["learned_unet_orgdet_fnn_acf_test_v1"]["enable_gpu"] is True


def test_keep_orgdet_fnn_acf_test_complete_report_matches_holdout_overlap():
    report = json.loads((ROOT / "reports" / "learned_unet_orgdet_fnn_acf_test_v1_complete.json").read_text())
    assert report["status"] == "inferred"
    assert report["config_hash"] == LOCKED_ORGDET_FNN_ACF_TEST_HASH
    assert report["audit_ok"] is True
    assert report["independent_audit_ok"] is True
    assert report["is_leaderboard_score"] is False
    assert report["ready_for_manual_submission_review"] is True
    assert report["n_nodes"] == 196290
    assert report["n_edges"] == 178473
    assert report["n_dropped_inconsistent_dt1_edges"] == 1652
    assert report["n_dropped_non_forward_nn_edges"] == 7710
    assert report["n_dropped_long_dt1_edges"] == 7441
    assert report["n_gap_links"] == 9499
    assert report["holdout_overlap_6bba_05db0fb1"]["matches_orgdet_fnn_acf_holdout_v1"] is True
    assert report["holdout_overlap_6bba_05db0fb1"]["n_nodes"] == 87109
    assert report["holdout_overlap_6bba_05db0fb1"]["n_edges"] == 80506
    assert report["compare_vs_orgdet_fnn_fast_test"]["ok"] is True
    assert report["hidden_projection"]["under_limit"] is True
    assert report["hidden_projection"]["fail_closed_approaches_12h"] is False


def test_classify_orgdet_fnn_acf_test_submit_poll_records_returned_scores():
    association = {
        "ref": ASSOCIATION_V2_SUBMISSION_REF,
        "status": "complete",
        "publicScore": ASSOCIATION_V2_PUBLIC_SCORE,
        "kernelSlug": "mahdimzafari/biohub-association-v2",
    }
    orgdet_test = {
        "ref": ORGDET_TEST_SUBMISSION_REF,
        "status": "complete",
        "publicScore": ORGDET_TEST_PUBLIC_SCORE,
        "kernelSlug": "mahdimzafari/biohub-learned-unet-orgdet-test-v1",
    }
    gap_test = {
        "ref": ORGDET_GAP_TEST_SUBMISSION_REF,
        "status": "complete",
        "publicScore": ORGDET_GAP_TEST_PUBLIC_SCORE,
        "kernelSlug": "mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1",
    }
    ui_needed = classify_orgdet_fnn_acf_test_submit_poll(
        kernel_complete=True,
        kernel_failed=False,
        output_names=["run_manifest.json", "submission.csv"],
        submissions=[association, orgdet_test, gap_test],
    )
    assert ui_needed["decision"] == "human_action_required_ui_submit"
    assert ui_needed["orgdet_fnn_acf_test_submission_present"] is False
    assert ui_needed["is_leaderboard_score"] is False
    acf = {
        "ref": ORGDET_FNN_ACF_TEST_SUBMISSION_REF,
        "status": "complete",
        "publicScore": ORGDET_FNN_ACF_TEST_PUBLIC_SCORE,
        "privateScore": ORGDET_FNN_ACF_TEST_PRIVATE_SCORE,
        "date": "2026-09-28T15:14:43.263Z",
        "kernelSlug": ORGDET_FNN_ACF_TEST_KERNEL_SLUG,
        "scriptVersionId": ORGDET_FNN_ACF_TEST_SCRIPT_VERSION_ID,
    }
    assert classify_submission_notebook(acf["kernelSlug"], acf["ref"]) == "orgdet_fnn_acf_test"
    scored = classify_orgdet_fnn_acf_test_submit_poll(
        kernel_complete=True,
        kernel_failed=False,
        output_names=["run_manifest.json", "submission.csv"],
        submissions=[association, orgdet_test, gap_test, acf],
    )
    assert scored["decision"] == "public_score_returned"
    assert scored["orgdet_fnn_acf_test_identity_proven"] is True
    assert scored["new_submission_public_score"] == ORGDET_FNN_ACF_TEST_PUBLIC_SCORE
    assert scored["new_submission_private_score"] == ORGDET_FNN_ACF_TEST_PRIVATE_SCORE
    assert scored["is_leaderboard_score"] is False
    assert "not a tune target" in scored["human_action"]
    poll_src = (ROOT / "scripts" / "poll_orgdet_fnn_acf_test_submit.py").read_text()
    assert "list_competition_submissions" in poll_src
    assert "competitions/submit" not in poll_src
    assert "kernels/push" not in poll_src
    assert "download_kernel_output" not in poll_src
    live = json.loads(
        (ROOT / "reports" / "learned_unet_orgdet_fnn_acf_test_v1_submit_poll.json").read_text()
    )
    assert live["orgdet_fnn_acf_test_config_hash"] == LOCKED_ORGDET_FNN_ACF_TEST_HASH
    assert live["is_leaderboard_score"] is False
    assert live["is_competition_submission"] is False
    assert live["decision"] == "public_score_returned"
    assert live["new_submission_public_score"] == ORGDET_FNN_ACF_TEST_PUBLIC_SCORE
    assert live["new_submission_private_score"] == ORGDET_FNN_ACF_TEST_PRIVATE_SCORE
    assert live["new_submission_refs"] == [ORGDET_FNN_ACF_TEST_SUBMISSION_REF]
    assert live["orgdet_fnn_acf_test_identity_proven"] is True
    blob = json.dumps(live)
    assert "token=" not in blob
    assert "https://" not in blob


def test_runtime_estimate_under_limit():
    est = runtime_estimate(199)
    assert est["under_limit"] is True
    assert est["estimate_hours"] < 10.8
    assert est["is_leaderboard_score"] is False
    assert nb_estimate(4)["under_limit"] is True
    hidden = hidden_projection_from_rehearsal(324.0)
    assert hidden["fail_closed_approaches_12h"] is False
    assert hidden["is_leaderboard_score"] is False
