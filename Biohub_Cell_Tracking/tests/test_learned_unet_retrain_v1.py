from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH  # noqa: E402
from biohub_tracking.learned_retrain import (  # noqa: E402
    ADDITIONAL_EPOCHS,
    INFER_LOCKED_TUNE,
    LAUNCH_AUTHORIZED,
    LOCKED_RETRAIN_HASH,
    PRUNE_LOCKED_HOLDOUT,
    RETRAIN_KERNEL_SLUG,
    TRAIN_KERNEL_SLUG,
    allowed_resume_weight_names,
    assert_kernel_not_launched,
    assert_launch_authorized,
    assert_tuning_only,
    public_lb_not_a_tune_target,
    refuse_holdout_design,
    refuse_public_weight_sources,
    retrain_config_hash,
    retrain_config_payload,
    runtime_estimate,
    tuning_decision,
)
from biohub_tracking.learned_split import HoldoutLeakError, INTERNAL_VAL_STEMS  # noqa: E402

_MAKE = importlib.util.spec_from_file_location(
    "make_kaggle_kernel", ROOT / "scripts" / "make_kaggle_kernel.py"
)
assert _MAKE and _MAKE.loader
_make_mod = importlib.util.module_from_spec(_MAKE)
_MAKE.loader.exec_module(_make_mod)

_PUSH = importlib.util.spec_from_file_location(
    "push_kaggle_kernel", ROOT / "scripts" / "push_kaggle_kernel.py"
)
assert _PUSH and _PUSH.loader
_push_mod = importlib.util.module_from_spec(_PUSH)
_PUSH.loader.exec_module(_push_mod)


def _yaml() -> dict:
    return yaml.safe_load((ROOT / "configs" / "learned_unet_retrain_v1.yaml").read_text())


def test_retrain_hash_is_immutable_and_matches_yaml():
    payload = retrain_config_payload()
    cfg = _yaml()
    assert retrain_config_hash() == "92a419926fa7e40b"
    assert LOCKED_RETRAIN_HASH == "92a419926fa7e40b"
    assert payload["parent_train_hash"] == TRAIN_CONFIG_HASH == "4c38035d6657f2ad"
    assert payload["infer_config_hash"] == LOCKED_INFER_HASH == "d7578b5f64ddac08"
    for key, value in payload.items():
        assert cfg[key] == value, key


def test_retrain_is_44b6_continuation_not_from_scratch():
    payload = retrain_config_payload()
    assert payload["additional_epochs"] == ADDITIONAL_EPOCHS == 3
    assert payload["parent_epochs"] == 3
    assert payload["total_epochs_equivalent"] == 6
    assert payload["from_scratch"] is False
    assert payload["resume_from_weights"] == "learned_unet_v1_last.pt"
    assert payload["tuning_specimen"] == "44b6"
    assert payload["internal_val"] == list(INTERNAL_VAL_STEMS)
    assert payload["writes_submission_csv"] is False
    assert payload["infer_after_train_uses_locked_infer_op"] is True
    assert payload["refuses_public_199_video_weights"] is True
    assert payload["kernel_slug"] == RETRAIN_KERNEL_SLUG
    assert payload["parent_kernel_slug"] == TRAIN_KERNEL_SLUG
    assert payload["kernel_slug"] != TRAIN_KERNEL_SLUG


def test_retrain_refuses_holdout_while_designing():
    with pytest.raises(HoldoutLeakError):
        refuse_holdout_design(["6bba_05db0fb1"])
    with pytest.raises(HoldoutLeakError):
        refuse_holdout_design(["44b6_aaa", "6bba_c328f2fd"])
    refuse_holdout_design(["44b6_aaa", "44b6_12dfb391", "44b6_267148e4"])
    with pytest.raises(SystemExit, match="holdout"):
        assert_tuning_only("holdout")
    assert_tuning_only("tuning")


def test_retrain_refuses_public_and_organizer_split_weights():
    with pytest.raises(HoldoutLeakError):
        refuse_public_weight_sources(
            ["pilkwang/biohub-tracking-support-pack-50ep-v1"]
        )
    with pytest.raises(HoldoutLeakError):
        refuse_public_weight_sources(
            ["/kaggle/input/datasets/thibautgoldsborough/cellmot-baseline-artifacts"]
        )
    with pytest.raises(HoldoutLeakError):
        refuse_public_weight_sources(["giorgosi/biohub-divnet-v2"])
    refuse_public_weight_sources(
        [
            "mahdimzafari/biohub-learned-unet-v1/learned_unet_v1_last.pt",
            TRAIN_KERNEL_SLUG,
        ]
    )
    allowed_resume_weight_names(["learned_unet_v1_best.pt", "learned_unet_v1_last.pt"])
    with pytest.raises(HoldoutLeakError):
        allowed_resume_weight_names(["checkpoint_last.pth"])


def test_retrain_is_specified_not_launched_and_does_not_overwrite():
    assert LAUNCH_AUTHORIZED is False
    with pytest.raises(SystemExit, match="not authorized"):
        assert_kernel_not_launched()
    with pytest.raises(SystemExit, match="not authorized"):
        assert_launch_authorized()
    assert "learned_unet_retrain_v1" not in _make_mod.KERNELS
    assert RETRAIN_KERNEL_SLUG not in _push_mod.FORBIDDEN_SLUGS
    locked = {
        "mahdimzafari/biohub-learned-unet-v1",
        "mahdimzafari/biohub-learned-unet-infer-v1",
        "mahdimzafari/biohub-learned-unet-holdout-v1",
        "mahdimzafari/biohub-learned-unet-test-v1",
        "mahdimzafari/biohub-learned-unet-gap-test-v1",
        "mahdimzafari/biohub-learned-unet-prune-test-v1",
        "mahdimzafari/biohub-learned-unet-orgop-v1",
    }
    assert locked <= _push_mod.FORBIDDEN_SLUGS
    assert RETRAIN_KERNEL_SLUG == "mahdimzafari/biohub-learned-unet-retrain-v1"


def test_retrain_does_not_use_public_lb_and_estimates_under_12h():
    with pytest.raises(ValueError, match="0.553"):
        public_lb_not_a_tune_target(0.553)
    with pytest.raises(ValueError, match="0.553"):
        public_lb_not_a_tune_target("0.553")
    public_lb_not_a_tune_target(INFER_LOCKED_TUNE)
    public_lb_not_a_tune_target(PRUNE_LOCKED_HOLDOUT)
    estimate = runtime_estimate()
    assert estimate["launched"] is False
    assert estimate["overwrites_parent_slug"] is False
    assert estimate["under_12h"] is True
    assert estimate["hidden_test_not_projected"] is True
    assert 3.0 <= estimate["expected_hours_low"] < estimate["expected_hours_high"] <= 4.0
    assert estimate["max_hours"] == 11.0
    assert 0.655 < INFER_LOCKED_TUNE < 0.66
    assert 0.701 < PRUNE_LOCKED_HOLDOUT < 0.702
    assert tuning_decision(0.6557507105877639).startswith("reject")
    assert tuning_decision(0.6557507105877640).startswith("keep")
