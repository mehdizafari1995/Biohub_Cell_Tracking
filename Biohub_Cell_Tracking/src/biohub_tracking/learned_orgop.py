"""Proposed organizer operating point on locked 44b6-only weights.

``learned_unet_orgop_v1`` is a specification, not a launched kernel.
It reproduces the organizer inference defaults
(``det_threshold=0.99`` + tracksdata ILP) on
``learned_unet_v1`` weights only.

Do not attach public 199-video / DeepCenter / secondary-seed weights.
Do not open ``6bba`` until a documented 44b6 keep.
Do not write ``submission.csv``.
Do not launch a Kaggle helper until an authorized cycle.
Not a competition submission.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .learned_infer import (  # noqa: F401
    INFER_KERNEL_SLUG,
    INFER_STEMS,
    LOCKED_INFER_HASH,
    TRAIN_CONFIG_HASH,
    TRAIN_KERNEL_SLUG,
    FROZEN_HOLDOUT_STEMS,
    HOLDOUT_SPECIMENS,
    SCALE_ZYX,
    TUNING_SPECIMEN,
    WEIGHTS_NAME,
    HoldoutLeakError,
    assert_no_submission_csv,
    assert_tuning_only,
    refuse_non_infer_stems,
)
from .learned_split import FALLBACK_WEIGHTS_NAME, specimen_of

ORGOP_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgop-v1"
INFER_LOCKED_TUNE = 0.6557507105877639
PRUNE_LOCKED_HOLDOUT = 0.7013778258203534
LAUNCH_AUTHORIZED = False

# Organizer ``predict_unet_transformer.py`` CLI / notebook defaults.
ORGANIZER_DET_THRESHOLD = 0.99
ORGANIZER_POOL_KERNEL_UM = 3.0
ORGANIZER_EDGE_ACTIVATION = "softmax"
ORGANIZER_EDGE_THRESHOLD = 0.5
ORGANIZER_USE_ILP = True
ORGANIZER_ILP_EDGE_WEIGHT = -1.0
ORGANIZER_ILP_APPEARANCE_WEIGHT = 0.1
ORGANIZER_ILP_DISAPPEARANCE_WEIGHT = 0.1
ORGANIZER_ILP_DIVISION_WEIGHT = 1.0

BANNED_PUBLIC_WEIGHT_MARKERS = (
    "pilkwang/biohub-tracking-support-pack",
    "pilkwang/biohub-temporal-unet3d",
    "pilkwang/biohub-deepcenter",
    "reyhanksatria/biohub-tracking-support-pack",
    "reyhanksatria/biohub-deepcenter",
    "reyhanksatria/biohub-temporalunet",
    "thibautgoldsborough/cellmot-baseline-artifacts",
    "giorgosi/biohub-divnet",
    "biohub-tracking-support-pack-50ep",
    "biohub-temporal-unet3d-seed314159",
    "biohub-deepcenter-unet3d-center-prior",
    "edge_predictor_best.pth",
    "checkpoint_last.pth",
)


def orgop_config_payload() -> dict[str, Any]:
    """First organizer-default operating point. Do not retune from holdout."""
    return {
        "label": "learned_unet_orgop_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "infer_stems": list(INFER_STEMS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "weights_source": "locked_learned_unet_v1_44b6_only",
        "weights_name": WEIGHTS_NAME,
        "fallback_weights_name": FALLBACK_WEIGHTS_NAME,
        "refuses_public_199_video_weights": True,
        "refuses_deepcenter": True,
        "refuses_secondary_seed_blend": True,
        "uses_organizer_predict_path": True,
        "det_threshold": ORGANIZER_DET_THRESHOLD,
        "det_tta": True,
        "pool_kernel_um": ORGANIZER_POOL_KERNEL_UM,
        "edge_activation": ORGANIZER_EDGE_ACTIVATION,
        "edge_threshold": ORGANIZER_EDGE_THRESHOLD,
        "linker": "tracksdata_ilp",
        "use_ilp": ORGANIZER_USE_ILP,
        "ilp_edge_weight": ORGANIZER_ILP_EDGE_WEIGHT,
        "ilp_appearance_weight": ORGANIZER_ILP_APPEARANCE_WEIGHT,
        "ilp_disappearance_weight": ORGANIZER_ILP_DISAPPEARANCE_WEIGHT,
        "ilp_division_weight": ORGANIZER_ILP_DIVISION_WEIGHT,
        "max_gap_frames": 0,
        "interpolate_gaps": False,
        "downsample": [1, 4, 4],
        "normalize": "window_minmax",
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": False,
        "kernel_slug": ORGOP_KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
    }


def orgop_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or orgop_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGOP_HASH = orgop_config_hash()
assert LOCKED_ORGOP_HASH == "250c5bfacd4a8198"


def assert_kernel_not_launched() -> None:
    if not LAUNCH_AUTHORIZED:
        raise SystemExit(
            "learned_unet_orgop_v1 kernel launch is not authorized; "
            "post the 0.94 audit and wait for a reviewed cycle"
        )


def refuse_public_weight_sources(sources: Iterable[str]) -> None:
    """Reject Pilkwang / organizer / DivNet public checkpoints for this slug."""
    cleaned = [str(s).strip().lower() for s in sources if str(s).strip()]
    hits = []
    for src in cleaned:
        for marker in BANNED_PUBLIC_WEIGHT_MARKERS:
            if marker.lower() in src:
                hits.append(src)
                break
    if hits:
        raise HoldoutLeakError(
            "learned_unet_orgop_v1 refuses public 199-video / DeepCenter / "
            "organizer-split / DivNet weights; got " + ", ".join(hits)
        )


def allowed_weight_names(names: Iterable[str]) -> None:
    allowed = {WEIGHTS_NAME, FALLBACK_WEIGHTS_NAME}
    extra = [Path(n).name for n in names if Path(n).name not in allowed]
    if extra:
        raise HoldoutLeakError(
            "learned_unet_orgop_v1 only loads locked 44b6 weights; got "
            + ", ".join(extra)
        )


def refuse_holdout_design(stems: Iterable[str]) -> None:
    refuse_non_infer_stems(stems)
    leaked = [s for s in stems if specimen_of(s) in HOLDOUT_SPECIMENS]
    if leaked:
        raise HoldoutLeakError(
            "learned_unet_orgop_v1 must not inspect 6bba while designing; got "
            + ", ".join(leaked)
        )


def public_lb_not_a_tune_target(score: float | str) -> None:
    text = str(score).strip()
    if text in {"0.553", "0.5530"} or abs(float(score) - 0.553) < 1e-9:
        raise ValueError(
            "public leaderboard 0.553 is association_v2 only and is not a "
            "learned_unet_orgop_v1 tune target"
        )


def runtime_estimate() -> dict[str, Any]:
    """Two frozen-tuning volumes on GPU T4. Not a hidden-test projection."""
    return {
        "accelerator": "gpu_t4_x2",
        "volumes": 2,
        "stems": list(INFER_STEMS),
        "expected_minutes_low": 5,
        "expected_minutes_high": 15,
        "ilp_runtime_multiplier_vs_greedy": 3.8,
        "hidden_test_not_projected": True,
        "launched": False,
    }
