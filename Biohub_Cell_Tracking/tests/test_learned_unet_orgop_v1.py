from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

from biohub_tracking.learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH  # noqa: E402
from biohub_tracking.learned_orgop import (  # noqa: E402
    BANNED_PUBLIC_WEIGHT_MARKERS,
    INFER_LOCKED_TUNE,
    INFER_STEMS,
    LAUNCH_AUTHORIZED,
    LOCKED_ORGOP_HASH,
    ORGANIZER_DET_THRESHOLD,
    ORGANIZER_ILP_APPEARANCE_WEIGHT,
    ORGANIZER_ILP_DISAPPEARANCE_WEIGHT,
    ORGANIZER_ILP_DIVISION_WEIGHT,
    ORGANIZER_ILP_EDGE_WEIGHT,
    ORGANIZER_POOL_KERNEL_UM,
    ORGANIZER_USE_ILP,
    ORGOP_KERNEL_SLUG,
    PRUNE_LOCKED_HOLDOUT,
    TRAIN_KERNEL_SLUG,
    allowed_weight_names,
    assert_kernel_not_launched,
    assert_tuning_only,
    orgop_config_hash,
    orgop_config_payload,
    public_lb_not_a_tune_target,
    refuse_holdout_design,
    refuse_public_weight_sources,
    runtime_estimate,
)
from biohub_tracking.learned_split import HoldoutLeakError  # noqa: E402

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
    return yaml.safe_load((ROOT / "configs" / "learned_unet_orgop_v1.yaml").read_text())


def test_orgop_hash_is_immutable_and_matches_yaml():
    payload = orgop_config_payload()
    cfg = _yaml()
    assert orgop_config_hash() == "250c5bfacd4a8198"
    assert LOCKED_ORGOP_HASH == "250c5bfacd4a8198"
    assert payload["train_config_hash"] == TRAIN_CONFIG_HASH == "4c38035d6657f2ad"
    assert payload["infer_config_hash"] == LOCKED_INFER_HASH == "d7578b5f64ddac08"
    for key, value in payload.items():
        assert cfg[key] == value, key


def test_orgop_matches_organizer_ilp_and_threshold_defaults():
    payload = orgop_config_payload()
    assert payload["det_threshold"] == ORGANIZER_DET_THRESHOLD == 0.99
    assert payload["pool_kernel_um"] == ORGANIZER_POOL_KERNEL_UM == 3.0
    assert payload["use_ilp"] is ORGANIZER_USE_ILP is True
    assert payload["ilp_edge_weight"] == ORGANIZER_ILP_EDGE_WEIGHT == -1.0
    assert payload["ilp_appearance_weight"] == ORGANIZER_ILP_APPEARANCE_WEIGHT == 0.1
    assert payload["ilp_disappearance_weight"] == ORGANIZER_ILP_DISAPPEARANCE_WEIGHT == 0.1
    assert payload["ilp_division_weight"] == ORGANIZER_ILP_DIVISION_WEIGHT == 1.0
    assert payload["linker"] == "tracksdata_ilp"
    assert payload["weights_source"] == "locked_learned_unet_v1_44b6_only"
    assert payload["refuses_public_199_video_weights"] is True
    assert payload["writes_submission_csv"] is False
    assert payload["infer_stems"] == list(INFER_STEMS)


def test_orgop_refuses_holdout_while_designing():
    with pytest.raises(HoldoutLeakError):
        refuse_holdout_design(["6bba_05db0fb1"])
    with pytest.raises(HoldoutLeakError):
        refuse_holdout_design(["44b6_12dfb391", "6bba_c328f2fd"])
    refuse_holdout_design(list(INFER_STEMS))
    with pytest.raises(SystemExit, match="holdout"):
        assert_tuning_only("holdout")
    assert_tuning_only("tuning")


def test_orgop_refuses_public_and_organizer_split_weights():
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
    with pytest.raises(HoldoutLeakError):
        refuse_public_weight_sources(["weights/unet_transformer/split_0/edge_predictor_best.pth"])
    refuse_public_weight_sources(
        [
            "mahdimzafari/biohub-learned-unet-v1/learned_unet_v1_best.pt",
            TRAIN_KERNEL_SLUG,
        ]
    )
    allowed_weight_names(["learned_unet_v1_best.pt", "learned_unet_v1_last.pt"])
    with pytest.raises(HoldoutLeakError):
        allowed_weight_names(["checkpoint_last.pth"])
    assert "pilkwang/biohub-deepcenter" in BANNED_PUBLIC_WEIGHT_MARKERS


def test_orgop_kernel_is_specified_not_launched():
    assert LAUNCH_AUTHORIZED is False
    with pytest.raises(SystemExit, match="not authorized"):
        assert_kernel_not_launched()
    assert "learned_unet_orgop_v1" not in _make_mod.KERNELS
    assert ORGOP_KERNEL_SLUG not in _push_mod.FORBIDDEN_SLUGS
    locked = {
        "mahdimzafari/biohub-learned-unet-v1",
        "mahdimzafari/biohub-learned-unet-infer-v1",
        "mahdimzafari/biohub-learned-unet-holdout-v1",
        "mahdimzafari/biohub-learned-unet-test-v1",
        "mahdimzafari/biohub-learned-unet-gap-test-v1",
        "mahdimzafari/biohub-learned-unet-prune-test-v1",
    }
    assert locked <= _push_mod.FORBIDDEN_SLUGS
    assert ORGOP_KERNEL_SLUG == "mahdimzafari/biohub-learned-unet-orgop-v1"


def test_orgop_does_not_use_public_lb_or_claim_094():
    with pytest.raises(ValueError, match="0.553"):
        public_lb_not_a_tune_target(0.553)
    with pytest.raises(ValueError, match="0.553"):
        public_lb_not_a_tune_target("0.553")
    public_lb_not_a_tune_target(INFER_LOCKED_TUNE)
    public_lb_not_a_tune_target(PRUNE_LOCKED_HOLDOUT)
    estimate = runtime_estimate()
    assert estimate["launched"] is False
    assert estimate["volumes"] == 2
    assert estimate["hidden_test_not_projected"] is True
    assert 0.655 < INFER_LOCKED_TUNE < 0.66
    assert 0.701 < PRUNE_LOCKED_HOLDOUT < 0.702
