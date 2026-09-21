"""Specify ``learned_unet_retrain_v1``: +3-epoch 44b6-only continuation.

Hypothesis (after ``reject_learned_unet_orgop_v1``): additional epochs on the
locked specimen-separated weights improve detection enough to beat greedy
infer 0.655751. Public 199-video / DeepCenter checkpoints stay refused.

Do **not** launch until a reviewed cycle. Do **not** overwrite the locked
train helper. Do **not** open ``6bba``. Do **not** write ``submission.csv``.
Not a competition submission.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .learned_infer import (  # noqa: F401
    FROZEN_HOLDOUT_STEMS,
    HOLDOUT_SPECIMENS,
    INFER_KERNEL_SLUG,
    LOCKED_INFER_HASH,
    TRAIN_CONFIG_HASH,
    TRAIN_KERNEL_SLUG,
    TUNING_SPECIMEN,
    HoldoutLeakError,
    assert_no_submission_csv,
    assert_tuning_only,
)
from .learned_orgop import BANNED_PUBLIC_WEIGHT_MARKERS, INFER_LOCKED_TUNE, PRUNE_LOCKED_HOLDOUT
from .learned_split import (
    FALLBACK_WEIGHTS_NAME,
    INTERNAL_VAL_STEMS,
    WEIGHTS_NAME,
    refuse_holdout_stems,
)

RETRAIN_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-retrain-v1"
LAUNCH_AUTHORIZED = False
PARENT_EPOCHS = 3
ADDITIONAL_EPOCHS = 3
TOTAL_EPOCHS_EQUIVALENT = 6
RESUME_WEIGHTS_NAME = FALLBACK_WEIGHTS_NAME
PARENT_TRAIN_RUNTIME_SEC = 11732.0
PARENT_PEAK_MB = 1793.0


def retrain_config_payload() -> dict[str, Any]:
    """First 44b6-only continuation. Do not retune from holdout."""
    return {
        "label": "learned_unet_retrain_v1",
        "parent_label": "learned_unet_v1",
        "parent_train_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "internal_val": list(INTERNAL_VAL_STEMS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "resume_from_weights": RESUME_WEIGHTS_NAME,
        "parent_epochs": PARENT_EPOCHS,
        "additional_epochs": ADDITIONAL_EPOCHS,
        "total_epochs_equivalent": TOTAL_EPOCHS_EQUIVALENT,
        "from_scratch": False,
        "batch_size": 1,
        "grad_accum": 8,
        "lr": 0.0001,
        "downsample": [1, 4, 4],
        "unet_layers": [32, 64, 128],
        "unet_out_channels": 32,
        "window_size": 2,
        "pool_kernel_um": 5.0,
        "det_loss_weight": 1.0,
        "det_neg_weight": 0.01,
        "skip_fullres_temporal": True,
        "seed": 0,
        "max_hours": 11.0,
        "writes_submission_csv": False,
        "edge_supervision": "gt_node_index",
        "refuses_public_199_video_weights": True,
        "refuses_deepcenter": True,
        "infer_after_train_uses_locked_infer_op": True,
        "kernel_slug": RETRAIN_KERNEL_SLUG,
        "parent_kernel_slug": TRAIN_KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
    }


def retrain_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or retrain_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_RETRAIN_HASH = retrain_config_hash()
assert LOCKED_RETRAIN_HASH == "92a419926fa7e40b"


def assert_kernel_not_launched() -> None:
    if not LAUNCH_AUTHORIZED:
        raise SystemExit(
            "learned_unet_retrain_v1 kernel launch is not authorized; "
            "wait for a reviewed cycle. Do not overwrite "
            f"{TRAIN_KERNEL_SLUG}"
        )


def assert_launch_authorized() -> None:
    if not LAUNCH_AUTHORIZED:
        raise SystemExit("learned_unet_retrain_v1 launch is not authorized")


def refuse_holdout_design(stems: Iterable[str]) -> None:
    refuse_holdout_stems(stems)


def refuse_public_weight_sources(sources: Iterable[str]) -> None:
    cleaned = [str(s).strip().lower() for s in sources if str(s).strip()]
    hits = []
    for src in cleaned:
        for marker in BANNED_PUBLIC_WEIGHT_MARKERS:
            if marker.lower() in src:
                hits.append(src)
                break
    if hits:
        raise HoldoutLeakError(
            "learned_unet_retrain_v1 refuses public 199-video / DeepCenter / "
            "organizer-split / DivNet weights; got " + ", ".join(hits)
        )


def allowed_resume_weight_names(names: Iterable[str]) -> None:
    allowed = {WEIGHTS_NAME, FALLBACK_WEIGHTS_NAME}
    extra = [Path(n).name for n in names if Path(n).name not in allowed]
    if extra:
        raise HoldoutLeakError(
            "learned_unet_retrain_v1 only resumes locked 44b6 weights; got "
            + ", ".join(extra)
        )


def public_lb_not_a_tune_target(score: float | str) -> None:
    text = str(score).strip()
    if text in {"0.553", "0.5530"} or abs(float(score) - 0.553) < 1e-9:
        raise ValueError(
            "public leaderboard 0.553 is association_v2 only and is not a "
            "learned_unet_retrain_v1 tune target"
        )


def tuning_decision(combined: float) -> str:
    if float(combined) > INFER_LOCKED_TUNE:
        return "keep_candidate_learned_unet_retrain_v1"
    return "reject_learned_unet_retrain_v1"


def runtime_estimate() -> dict[str, Any]:
    """Continuation of locked 3-epoch helper. Not a hidden-test projection."""
    per_epoch = PARENT_TRAIN_RUNTIME_SEC / float(PARENT_EPOCHS)
    expected_sec = per_epoch * float(ADDITIONAL_EPOCHS)
    return {
        "accelerator": "gpu_t4_x2",
        "parent_runtime_sec": PARENT_TRAIN_RUNTIME_SEC,
        "parent_peak_mb": PARENT_PEAK_MB,
        "additional_epochs": ADDITIONAL_EPOCHS,
        "expected_seconds_low": int(expected_sec * 0.95),
        "expected_seconds_high": int(expected_sec * 1.15),
        "expected_hours_low": round(expected_sec * 0.95 / 3600.0, 2),
        "expected_hours_high": round(expected_sec * 1.15 / 3600.0, 2),
        "max_hours": 11.0,
        "under_12h": True,
        "hidden_test_not_projected": True,
        "launched": False,
        "overwrites_parent_slug": False,
    }
