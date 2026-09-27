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
    LOCKED_ORGDET_FNN_ACF_TEST_HASH,
    ORGDET_FNN_ACF_TEST_KERNEL_SLUG,
    hidden_projection_from_rehearsal,
    orgdet_fnn_acf_test_config_hash as locked_hash,
    orgdet_fnn_acf_test_config_payload as locked_payload,
    refuse_overwrite_locked_slugs,
    runtime_estimate,
)
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


def test_runtime_estimate_under_limit():
    est = runtime_estimate(199)
    assert est["under_limit"] is True
    assert est["estimate_hours"] < 10.8
    assert est["is_leaderboard_score"] is False
    assert nb_estimate(4)["under_limit"] is True
    hidden = hidden_projection_from_rehearsal(324.0)
    assert hidden["fail_closed_approaches_12h"] is False
    assert hidden["is_leaderboard_score"] is False
