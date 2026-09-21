"""Specimen-separated edge-train canary after rejected orgop v1.

Proves the locked architecture can learn peaked softmax edges. Does not
open ``6bba``. Does not write ``submission.csv``. Does not push a kernel.
Does not attach 199-video weights. Not a competition submission.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from .learned_checkpoint_audit import dense_softmax_rarely_exceeds_half
from .learned_split import (
    FROZEN_HOLDOUT_STEMS,
    HOLDOUT_SPECIMENS,
    TRAIN_CONFIG_HASH,
    TUNING_SPECIMEN,
    HoldoutLeakError,
    specimen_of,
)

CANARY_LABEL = "learned_unet_edge_train_canary_v1"
PARENT_ORGOP_HASH = "250c5bfacd4a8198"
LOCKED_TRAIN_SEC = 11731.948331356049
LOCKED_TRAIN_EPOCHS = 3
KERNEL_PUSH = False


def canary_config_payload() -> dict[str, Any]:
    """Immutable CPU canary. Do not retune from holdout or public 0.553."""
    return {
        "label": CANARY_LABEL,
        "train_label": "learned_unet_v1",
        "parent_orgop_hash": PARENT_ORGOP_HASH,
        "train_config_hash": TRAIN_CONFIG_HASH,
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "refuses_public_199_video_weights": True,
        "writes_submission_csv": False,
        "kernel_push": False,
        "n_src": 12,
        "n_tgt": 12,
        "n_steps": 60,
        "lr": 0.001,
        "seed": 0,
        "feat_dim": 64,
        "freeze_unet": True,
        "edge_activation": "softmax",
        "edge_threshold": 0.5,
        "edge_supervision": "gt_node_index",
        "min_prob_max": 0.5,
        "min_n_gt_half": 1,
        "opens_6bba": False,
        "uses_199_video_weights": False,
    }


def canary_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or canary_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_CANARY_HASH = canary_config_hash()
assert LOCKED_CANARY_HASH == "1a3d524e4d922701"


def refuse_holdout_design(stems: Iterable[str]) -> None:
    leaked = [s for s in stems if specimen_of(str(s)) in HOLDOUT_SPECIMENS]
    if leaked:
        raise HoldoutLeakError(
            "learned_unet_edge_train_canary_v1 must not inspect 6bba; got "
            + ", ".join(leaked)
        )


def public_lb_not_a_tune_target(score: float | str) -> None:
    text = str(score).strip()
    if text in {"0.553", "0.5530"} or abs(float(score) - 0.553) < 1e-9:
        raise ValueError(
            "public leaderboard 0.553 is association_v2 only and is not a "
            "learned_unet_edge_train_canary_v1 tune target"
        )


def runtime_projection() -> dict[str, Any]:
    """Scale the locked 3-epoch 44b6 train helper. Not a hidden-test score."""
    sec_per_epoch = LOCKED_TRAIN_SEC / float(LOCKED_TRAIN_EPOCHS)
    rows = []
    for epochs in (3, 10, 20, 50):
        hours = sec_per_epoch * epochs / 3600.0
        rows.append(
            {
                "epochs": epochs,
                "videos": 69,
                "hours": round(hours, 2),
                "under_12h": hours < 12.0,
            }
        )
    return {
        "sec_per_epoch": sec_per_epoch,
        "source": "mahdimzafari/biohub-learned-unet-v1 v3",
        "rows": rows,
        "recommended_next_budget_epochs": 10,
        "organizer_50_epoch_exceeds_12h": True,
        "kernel_push": KERNEL_PUSH,
        "opened_holdout": False,
    }


def dense_threshold_blocker() -> dict[str, Any]:
    return {
        n_src: dense_softmax_rarely_exceeds_half(n_src, 80, peak=3.0)
        for n_src in (8, 64, 400, 800)
    }
