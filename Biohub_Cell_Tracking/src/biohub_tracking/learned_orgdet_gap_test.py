"""Test-time helpers for learned_unet_orgdet_gap_test_v1.

Locked orgdet detections plus locked association_v2 gap-close on
discovered ``test/*.zarr``. Do not overwrite orgdet-test. Do not
retune. Do not score frozen ``6bba`` again.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .learned_gap import LOCKED_GAP_HASH
from .learned_gap_test import (
    ASSOCIATION_V2_PUBLIC_SCORE,
    ASSOCIATION_V2_SUBMISSION_REF,
    GAP_TEST_KERNEL_SLUG,
    LOCKED_GAP_TEST_HASH,
    LOCKED_TEST_HASH,
)
from .learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from .learned_orgdet import LOCKED_ORGDET_HASH, ORGDET_KERNEL_SLUG
from .learned_orgdet_gap import LOCKED_ORGDET_GAP_HASH, ORGDET_LOCKED_HOLDOUT
from .learned_orgdet_holdout import LOCKED_ORGDET_HOLDOUT_HASH, ORGDET_HOLDOUT_KERNEL_SLUG
from .learned_orgdet_test import LOCKED_ORGDET_TEST_HASH, ORGDET_TEST_KERNEL_SLUG
from .learned_orgop import LOCKED_ORGOP_HASH as REJECTED_ORGOP_HASH
from .learned_orgop import ORGOP_KERNEL_SLUG
from .learned_prune_test import LOCKED_PRUNE_TEST_HASH, PRUNE_TEST_KERNEL_SLUG
from .learned_split import KERNEL_SLUG as TRAIN_KERNEL_SLUG
from .learned_test import INFER_KERNEL_SLUG, TEST_KERNEL_SLUG, WEIGHTS_NAME

ORGDET_GAP_TEST_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1"
ORGDET_GAP_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-gap-v1"


def orgdet_gap_test_config_payload() -> dict[str, Any]:
    return {
        "label": "learned_unet_orgdet_gap_test_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "orgdet_label": "learned_unet_orgdet_v1",
        "orgdet_holdout_label": "learned_unet_orgdet_holdout_v1",
        "orgdet_gap_label": "learned_unet_orgdet_gap_v1",
        "orgdet_test_label": "learned_unet_orgdet_test_v1",
        "orgop_label": "learned_unet_orgop_v1",
        "test_label": "learned_unet_test_v1",
        "gap_test_label": "learned_unet_gap_test_v1",
        "prune_test_label": "learned_unet_prune_test_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "orgdet_config_hash": LOCKED_ORGDET_HASH,
        "orgdet_holdout_config_hash": LOCKED_ORGDET_HOLDOUT_HASH,
        "orgdet_gap_config_hash": LOCKED_ORGDET_GAP_HASH,
        "orgdet_test_parent_hash": LOCKED_ORGDET_TEST_HASH,
        "rejected_orgop_hash": REJECTED_ORGOP_HASH,
        "test_parent_hash": LOCKED_TEST_HASH,
        "gap_test_parent_hash": LOCKED_GAP_TEST_HASH,
        "prune_test_parent_hash": LOCKED_PRUNE_TEST_HASH,
        "discovers_test_zarrs": True,
        "hardcoded_test_stems": [],
        "opens_train_volumes": False,
        "det_threshold": 0.99,
        "det_tta": True,
        "pool_kernel_um": 3.0,
        "linker": "orgdet_then_association_v2_gap_overlay",
        "motion_model": "last_position",
        "gate_um": 12.0,
        "max_link_um": 12.0,
        "max_gap_frames": 2,
        "gap_scale": "sqrt_dt",
        "assignment": "greedy",
        "interpolate_gaps": True,
        "drop_degree0": False,
        "max_parents_per_node": 1,
        "max_children_per_node": 1,
        "intensity_weight": 0.0,
        "downsample": [1, 4, 4],
        "normalize": "window_minmax",
        "scale_zyx": [1.625, 0.40625, 0.40625],
        "writes_submission_csv": True,
        "writes_run_manifest": True,
        "one_shot_holdout": False,
        "kernel_slug": ORGDET_GAP_TEST_KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
        "orgdet_kernel_slug": ORGDET_KERNEL_SLUG,
        "orgdet_holdout_kernel_slug": ORGDET_HOLDOUT_KERNEL_SLUG,
        "orgdet_gap_kernel_slug": ORGDET_GAP_KERNEL_SLUG,
        "orgdet_test_kernel_slug": ORGDET_TEST_KERNEL_SLUG,
        "orgop_kernel_slug": ORGOP_KERNEL_SLUG,
        "test_kernel_slug": TEST_KERNEL_SLUG,
        "gap_test_kernel_slug": GAP_TEST_KERNEL_SLUG,
        "prune_test_kernel_slug": PRUNE_TEST_KERNEL_SLUG,
        "weights_name": WEIGHTS_NAME,
    }


def orgdet_gap_test_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(
        payload or orgdet_gap_test_config_payload(), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGDET_GAP_TEST_HASH = orgdet_gap_test_config_hash()
ORGDET_GAP_TEST_LAST_RUN_UTC = "2026-09-22T09:05:18Z"
assert LOCKED_ORGDET_GAP_TEST_HASH == "1ce4837ae16bee52"
assert LOCKED_ORGDET_GAP_HASH == "ee31f36151325f9a"
assert LOCKED_ORGDET_TEST_HASH == "c45c859610d29e90"
assert LOCKED_GAP_HASH == "f4d888d190c45a4a"
assert ORGDET_LOCKED_HOLDOUT == 0.7052260455024029
assert ASSOCIATION_V2_SUBMISSION_REF == 56399003
assert ASSOCIATION_V2_PUBLIC_SCORE == "0.553"


def classify_orgdet_gap_test_submit_poll(
    *,
    kernel_complete: bool,
    kernel_failed: bool,
    output_names: Sequence[str],
    submissions: Sequence[Mapping[str, Any]],
    association_ref: int = ASSOCIATION_V2_SUBMISSION_REF,
) -> dict[str, Any]:
    """Classify a read-only orgdet-gap-test submit poll. Never submits.

    The preferred worker is ``mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1``.
    Association_v2 submission ``56399003`` (public API string ``0.553``) is
    not that kernel. Do not treat that public score as this slug's result.
    """
    names = [str(name) for name in output_names]
    rows = [dict(row) for row in submissions]
    refs = [row.get("ref") for row in rows]
    extra = [row for row in rows if str(row.get("ref")) != str(association_ref)]
    association = [row for row in rows if str(row.get("ref")) == str(association_ref)]
    if kernel_failed:
        decision = "record_kernel_error"
    elif not kernel_complete:
        decision = "wait_kernel"
    elif "submission.csv" not in names:
        decision = "kernel_missing_csv"
    elif extra:
        decision = "new_submission_present"
    else:
        decision = "human_action_required_ui_submit"
    return {
        "slug": ORGDET_GAP_TEST_KERNEL_SLUG,
        "decision": decision,
        "kernel_complete": bool(kernel_complete),
        "kernel_failed": bool(kernel_failed),
        "output_names": names,
        "n_submissions": len(rows),
        "submission_refs": refs,
        "association_v2_submission_present": bool(association),
        "orgdet_gap_test_submission_present": bool(extra),
        "is_competition_submission": False,
        "is_leaderboard_score": False,
        "human_action": (
            "Mahdi must Submit to Competition from the Kaggle UI on "
            "mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1 version 1 "
            "before the 2026-09-22 23:59 UTC entry deadline. "
            "Do not submit from the CLI. Do not submit orgdet-test, "
            "prune-test, or infer-family gap-test."
            if decision == "human_action_required_ui_submit"
            else None
        ),
    }
