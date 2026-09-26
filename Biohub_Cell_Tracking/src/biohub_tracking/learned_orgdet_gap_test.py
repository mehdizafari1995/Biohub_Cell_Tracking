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
from .learned_orgdet_test import (
    LOCKED_ORGDET_TEST_HASH,
    ORGDET_TEST_KERNEL_SLUG,
    ORGDET_TEST_PUBLIC_SCORE,
    ORGDET_TEST_SUBMISSION_REF,
    classify_submission_notebook,
)
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
ORGDET_GAP_TEST_SUBMISSION_REF = 56583783
ORGDET_GAP_TEST_PUBLIC_SCORE = "0.602"
assert LOCKED_ORGDET_GAP_TEST_HASH == "1ce4837ae16bee52"
assert LOCKED_ORGDET_GAP_HASH == "ee31f36151325f9a"
assert LOCKED_ORGDET_TEST_HASH == "c45c859610d29e90"
assert LOCKED_GAP_HASH == "f4d888d190c45a4a"
assert ORGDET_LOCKED_HOLDOUT == 0.7052260455024029
assert ASSOCIATION_V2_SUBMISSION_REF == 56399003
assert ASSOCIATION_V2_PUBLIC_SCORE == "0.553"
assert ORGDET_TEST_SUBMISSION_REF == 56466744
assert ORGDET_TEST_PUBLIC_SCORE == "0.612"
assert ORGDET_GAP_TEST_SUBMISSION_REF == 56583783
assert ORGDET_GAP_TEST_PUBLIC_SCORE == "0.602"


def classify_orgdet_gap_test_submit_poll(
    *,
    kernel_complete: bool,
    kernel_failed: bool,
    output_names: Sequence[str],
    submissions: Sequence[Mapping[str, Any]],
    association_ref: int = ASSOCIATION_V2_SUBMISSION_REF,
    orgdet_test_ref: int = ORGDET_TEST_SUBMISSION_REF,
    orgdet_gap_test_ref: int = ORGDET_GAP_TEST_SUBMISSION_REF,
) -> dict[str, Any]:
    """Classify a read-only orgdet-gap-test submit poll. Never submits.

    Proven notebook identity decides whether a competition row is this
    kernel. Association_v2 ``56399003`` (public ``0.553``) and orgdet-test
    ``56466744`` (public ``0.612``) are not this slug. Do not treat those
    scores as an orgdet-gap-test result. Do not retune from ``0.602``.
    """
    names = [str(name) for name in output_names]
    rows = [dict(row) for row in submissions]
    refs = [row.get("ref") for row in rows]
    association = [row for row in rows if str(row.get("ref")) == str(association_ref)]
    orgdet_test = [row for row in rows if str(row.get("ref")) == str(orgdet_test_ref)]
    labels = [classify_submission_notebook(row.get("kernelSlug"), row.get("ref")) for row in rows]
    gap_rows = [
        row
        for row, label in zip(rows, labels)
        if label == "orgdet_gap_test" or str(row.get("ref")) == str(orgdet_gap_test_ref)
    ]
    extra = [
        row
        for row, label in zip(rows, labels)
        if str(row.get("ref")) not in {str(association_ref), str(orgdet_test_ref)}
        and label != "orgdet_gap_test"
        and str(row.get("ref")) != str(orgdet_gap_test_ref)
    ]
    gap_newest = gap_rows[0] if gap_rows else None
    gap_score = str((gap_newest or {}).get("publicScore") or "").strip()
    gap_status = str((gap_newest or {}).get("status") or "").strip().lower()
    if kernel_failed:
        decision = "record_kernel_error"
    elif not kernel_complete:
        decision = "wait_kernel"
    elif "submission.csv" not in names:
        decision = "kernel_missing_csv"
    elif gap_rows and gap_status == "complete" and gap_score:
        decision = "public_score_returned"
    elif extra or (gap_rows and not (gap_status == "complete" and gap_score)):
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
        "orgdet_test_submission_present": bool(orgdet_test),
        "orgdet_gap_test_submission_present": bool(gap_rows),
        "orgdet_gap_test_identity_proven": bool(gap_rows),
        "new_submission_public_score": gap_score or None,
        "new_submission_status": (gap_newest or {}).get("status"),
        "new_submission_refs": [row.get("ref") for row in gap_rows],
        "is_competition_submission": False,
        "is_leaderboard_score": False,
        "human_action": (
            "Do not resubmit. Recorded publicScore 0.602 is orgdet-gap-test "
            "56583783 only and is not a tune target. Do not invent extra "
            "digits. Do not poll this ref as pending."
            if decision == "public_score_returned"
            else (
                "Mahdi must Submit to Competition from the Kaggle UI on "
                "mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1 version 1 "
                "only if a later message authorizes that extra click. "
                "Do not submit from the CLI. Do not submit orgdet-test, "
                "prune-test, infer-family gap-test, or orgdet-gap-prune-test "
                "unless separately authorized."
                if decision == "human_action_required_ui_submit"
                else (
                    "Do not resubmit. Poll until Kaggle returns a terminal "
                    "status and publicScore. Do not use any returned public "
                    "score for tuning."
                    if decision == "new_submission_present"
                    else None
                )
            )
        ),
    }
