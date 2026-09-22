"""Test-time inference helpers for learned_unet_orgdet_test_v1.

Applies the locked ``learned_unet_orgdet_holdout_v1`` operating point
(organizer detections ``det_threshold=0.99``, ``pool_kernel_um=3.0``,
flip TTA, greedy 12 µm last-position, no gap, no ILP) to every
discovered ``test/*.zarr``. Never opens ``train/`` volumes. Writes
``submission.csv`` + ``run_manifest.json``. Do not overwrite locked
prune-test, gap-test, no-gap test, or orgdet-holdout slugs. Do not
retune. Do not score frozen ``6bba`` again. Not a competition
submission until a human submits from the Kaggle UI.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .learned_gap_test import (
    ASSOCIATION_V2_PUBLIC_SCORE,
    ASSOCIATION_V2_SUBMISSION_REF,
    GAP_TEST_KERNEL_SLUG,
    LOCKED_GAP_TEST_HASH,
    LOCKED_TEST_HASH,
)
from .learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from .learned_orgdet import LOCKED_ORGDET_HASH, ORGDET_KERNEL_SLUG
from .learned_orgdet_holdout import (
    LOCKED_ORGDET_HOLDOUT_HASH,
    ORGDET_HOLDOUT_KERNEL_SLUG,
)
from .learned_orgop import (
    BANNED_PUBLIC_WEIGHT_MARKERS,
    INFER_KERNEL_SLUG,
    LOCKED_ORGOP_HASH as REJECTED_ORGOP_HASH,
    ORGANIZER_DET_THRESHOLD,
    ORGANIZER_POOL_KERNEL_UM,
    ORGOP_KERNEL_SLUG,
    refuse_public_weight_sources,
)
from .learned_prune_test import LOCKED_PRUNE_TEST_HASH, PRUNE_TEST_KERNEL_SLUG
from .learned_split import KERNEL_SLUG as TRAIN_KERNEL_SLUG
from .learned_test import (
    ESTIMATE_HIDDEN_VOLUMES,
    HOLDOUT_KERNEL_SLUG,
    KAGGLE_GPU_LIMIT_HOURS,
    SCALE_ZYX,
    TEST_KERNEL_SLUG,
    WEIGHTS_NAME,
    list_test_zarrs,
    refuse_hardcoded_public_stems,
    refuse_train_volume,
)

ORGDET_TEST_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-test-v1"
ASSOCIATION_V2_KERNEL_SLUG = "mahdimzafari/biohub-association-v2"
ORGDET_GAP_TEST_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1"
ORGDET_TEST_PUBLIC_SCORE = "0.612"
ORGDET_TEST_SUBMISSION_REF = 56466744


def classify_submission_notebook(slug: str | None, ref: Any = None) -> str:
    """Label a competition row from a proven notebook slug. Never guesses."""
    text = str(slug or "").strip()
    if text == ORGDET_TEST_KERNEL_SLUG:
        return "orgdet_test"
    if text == PRUNE_TEST_KERNEL_SLUG:
        return "prune_test"
    if text == GAP_TEST_KERNEL_SLUG:
        return "gap_test"
    if text == ORGDET_GAP_TEST_KERNEL_SLUG:
        return "orgdet_gap_test"
    if text == ASSOCIATION_V2_KERNEL_SLUG or str(ref) == str(ASSOCIATION_V2_SUBMISSION_REF):
        return "association_v2"
    if text:
        return "other_notebook"
    return "identity_unproven"
# Saved-kernel poll time for orgdet-test v1 COMPLETE (public-test worker, not a scoring rerun).
ORGDET_TEST_LAST_RUN_UTC = "2026-09-21T11:07:02Z"
ORGDET_HOLDOUT_MEASURED_SEC = 173.79521799087524
ORGDET_HOLDOUT_MEASURED_VOLUMES = 2
ORGDET_HOLDOUT_MEASURED_PEAK_MEMORY_MB = 1185.26171875
ORGDET_SEC_PER_VOLUME = ORGDET_HOLDOUT_MEASURED_SEC / ORGDET_HOLDOUT_MEASURED_VOLUMES
MAX_LINK_UM = 12.0


def orgdet_test_config_payload() -> dict[str, Any]:
    """Locked orgdet holdout operating point on discovered test volumes."""
    return {
        "label": "learned_unet_orgdet_test_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "orgdet_label": "learned_unet_orgdet_v1",
        "orgdet_holdout_label": "learned_unet_orgdet_holdout_v1",
        "orgop_label": "learned_unet_orgop_v1",
        "test_label": "learned_unet_test_v1",
        "gap_test_label": "learned_unet_gap_test_v1",
        "prune_test_label": "learned_unet_prune_test_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "orgdet_config_hash": LOCKED_ORGDET_HASH,
        "orgdet_holdout_config_hash": LOCKED_ORGDET_HOLDOUT_HASH,
        "rejected_orgop_hash": REJECTED_ORGOP_HASH,
        "test_parent_hash": LOCKED_TEST_HASH,
        "gap_test_parent_hash": LOCKED_GAP_TEST_HASH,
        "prune_test_parent_hash": LOCKED_PRUNE_TEST_HASH,
        "discovers_test_zarrs": True,
        "hardcoded_test_stems": [],
        "opens_train_volumes": False,
        "det_threshold": ORGANIZER_DET_THRESHOLD,
        "det_tta": True,
        "pool_kernel_um": ORGANIZER_POOL_KERNEL_UM,
        "linker": "greedy_gate_12um",
        "motion_model": "last_position",
        "gate_um": MAX_LINK_UM,
        "max_link_um": MAX_LINK_UM,
        "max_gap_frames": 0,
        "assignment": "greedy",
        "interpolate_gaps": False,
        "drop_degree0": False,
        "max_parents_per_node": 1,
        "max_children_per_node": 1,
        "intensity_weight": 0.0,
        "downsample": [1, 4, 4],
        "normalize": "window_minmax",
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": True,
        "writes_run_manifest": True,
        "one_shot_holdout": False,
        "kernel_slug": ORGDET_TEST_KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
        "orgdet_kernel_slug": ORGDET_KERNEL_SLUG,
        "orgdet_holdout_kernel_slug": ORGDET_HOLDOUT_KERNEL_SLUG,
        "orgop_kernel_slug": ORGOP_KERNEL_SLUG,
        "test_kernel_slug": TEST_KERNEL_SLUG,
        "gap_test_kernel_slug": GAP_TEST_KERNEL_SLUG,
        "prune_test_kernel_slug": PRUNE_TEST_KERNEL_SLUG,
        "weights_name": WEIGHTS_NAME,
    }


def orgdet_test_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or orgdet_test_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGDET_TEST_HASH = orgdet_test_config_hash(orgdet_test_config_payload())
assert LOCKED_ORGDET_TEST_HASH == "c45c859610d29e90"
assert LOCKED_ORGDET_HOLDOUT_HASH == "912f0c19b572a054"
assert LOCKED_ORGDET_HASH == "80b20589198148b7"
assert REJECTED_ORGOP_HASH == "250c5bfacd4a8198"
assert LOCKED_PRUNE_TEST_HASH == "074a71dec68304e8"


def runtime_estimate(
    n_volumes: int,
    sec_per_volume: float = ORGDET_SEC_PER_VOLUME,
    limit_hours: float = KAGGLE_GPU_LIMIT_HOURS,
    peak_memory_mb: float = ORGDET_HOLDOUT_MEASURED_PEAK_MEMORY_MB,
) -> dict[str, Any]:
    """Estimate GPU time from measured orgdet-holdout 173.8 s / 2 volumes."""
    n = int(n_volumes)
    sec = float(n) * float(sec_per_volume)
    hours = sec / 3600.0
    return {
        "n_volumes": n,
        "sec_per_volume": float(sec_per_volume),
        "estimate_sec": sec,
        "estimate_hours": hours,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_observed": float(peak_memory_mb),
        "source": (
            "learned_unet_orgdet_holdout_v1 v1 measured 173.8 s / 2 volumes "
            "(86.9 s/volume) at locked 0.99 / pool 3.0 / TTA + greedy 12 µm"
        ),
    }


def hidden_projection_from_rehearsal(
    public_runtime_sec: float,
    n_public: int = 4,
    n_hidden: int = ESTIMATE_HIDDEN_VOLUMES,
    peak_memory_mb: float = ORGDET_HOLDOUT_MEASURED_PEAK_MEMORY_MB,
    limit_hours: float = KAGGLE_GPU_LIMIT_HOURS,
) -> dict[str, Any]:
    """Project hidden-test wall time from a completed public rehearsal."""
    n_pub = int(n_public)
    sec_per = float(public_runtime_sec) / max(n_pub, 1)
    hidden = runtime_estimate(int(n_hidden), sec_per_volume=sec_per, peak_memory_mb=peak_memory_mb)
    fail_closed = (not hidden["under_limit"]) or hidden["estimate_hours"] >= 0.9 * float(limit_hours)
    return {
        "public_runtime_sec": float(public_runtime_sec),
        "n_public": n_pub,
        "sec_per_volume": sec_per,
        "hidden_test": hidden,
        "fail_closed_approaches_12h": fail_closed,
        "is_leaderboard_score": False,
    }


def classify_orgdet_test_submit_poll(
    *,
    kernel_complete: bool,
    kernel_failed: bool,
    output_names: Sequence[str],
    submissions: Sequence[Mapping[str, Any]],
    association_ref: int = ASSOCIATION_V2_SUBMISSION_REF,
) -> dict[str, Any]:
    """Classify a read-only orgdet-test submit poll. Never submits.

    The preferred worker is ``mahdimzafari/biohub-learned-unet-orgdet-test-v1``.
    Association_v2 submission ``56399003`` (public API string ``0.553``) is
    not that kernel: it was created before the orgdet-test lastRunTime.
    """
    names = [str(name) for name in output_names]
    rows = [dict(row) for row in submissions]
    refs = [row.get("ref") for row in rows]
    extra = [row for row in rows if str(row.get("ref")) != str(association_ref)]
    association = [row for row in rows if str(row.get("ref")) == str(association_ref)]
    extra_labels = [
        classify_submission_notebook(row.get("kernelSlug"), row.get("ref")) for row in extra
    ]
    orgdet_rows = [row for row, label in zip(extra, extra_labels) if label == "orgdet_test"]
    identity_proven = bool(orgdet_rows)
    newest = extra[0] if extra else None
    newest_score = str((newest or {}).get("publicScore") or "").strip()
    newest_status = str((newest or {}).get("status") or "").strip().lower()
    if kernel_failed:
        decision = "record_kernel_error"
    elif not kernel_complete:
        decision = "wait_kernel"
    elif "submission.csv" not in names:
        decision = "kernel_missing_csv"
    elif extra and newest_status == "complete" and newest_score:
        decision = "public_score_returned"
    elif extra:
        decision = "new_submission_present"
    else:
        decision = "human_action_required_ui_submit"
    return {
        "slug": ORGDET_TEST_KERNEL_SLUG,
        "decision": decision,
        "kernel_complete": bool(kernel_complete),
        "kernel_failed": bool(kernel_failed),
        "output_names": names,
        "n_submissions": len(rows),
        "submission_refs": refs,
        "association_v2_submission_present": bool(association),
        "orgdet_test_submission_present": identity_proven,
        "orgdet_test_identity_proven": identity_proven,
        "new_submission_identity": extra_labels,
        "new_submission_kernel_slugs": [row.get("kernelSlug") for row in extra],
        "new_submission_script_version_ids": [row.get("scriptVersionId") for row in extra],
        "new_submission_refs": [row.get("ref") for row in extra],
        "new_submission_status": (newest or {}).get("status"),
        "new_submission_date": (newest or {}).get("date"),
        "new_submission_public_score": newest_score or None,
        "is_competition_submission": False,
        "is_leaderboard_score": False,
        "human_action": (
            "Mahdi must Submit to Competition from the Kaggle UI on "
            "mahdimzafari/biohub-learned-unet-orgdet-test-v1 version 1 "
            "before the 2026-09-22 23:59 UTC entry deadline. "
            "Do not submit from the CLI. Do not submit prune-test or gap-test now."
            if decision == "human_action_required_ui_submit"
            else (
                "Do not resubmit. Recorded publicScore is not a tune target. "
                "Do not invent extra digits. Do not poll this ref as pending."
                if decision == "public_score_returned"
                else (
                    "Do not resubmit. Poll until Kaggle returns a terminal status "
                    "and publicScore. Do not use any returned public score for tuning."
                    if decision == "new_submission_present"
                    else None
                )
            )
        ),
    }


def refuse_overwrite_locked_slugs(slug: str) -> None:
    locked = {
        TEST_KERNEL_SLUG: "learned_unet_test_v1",
        GAP_TEST_KERNEL_SLUG: "learned_unet_gap_test_v1",
        PRUNE_TEST_KERNEL_SLUG: "learned_unet_prune_test_v1",
        ORGDET_HOLDOUT_KERNEL_SLUG: "learned_unet_orgdet_holdout_v1",
    }
    if slug in locked:
        raise ValueError(f"refusing to overwrite locked {locked[slug]} slug")


def require_orgdet_holdout_keep(path: str | Path) -> dict[str, Any]:
    dest = Path(path)
    if not dest.exists():
        raise SystemExit("refusing orgdet test-time; orgdet holdout keep report missing.")
    payload = json.loads(dest.read_text())
    decision = str(payload.get("decision") or "")
    if decision != "keep_candidate_learned_unet_orgdet_holdout_v1":
        raise SystemExit(f"refusing orgdet test-time; holdout decision is {decision}")
    if payload.get("config_hash") != LOCKED_ORGDET_HOLDOUT_HASH:
        raise SystemExit("refusing orgdet test-time; orgdet holdout hash drifted")
    return payload


def refuse_public_orgdet_test_weights(sources: Sequence[str]) -> None:
    refuse_public_weight_sources(sources)


def public_lb_not_a_tune_target(score: float | str) -> None:
    text = str(score).strip()
    forbidden = {
        "0.553": "public leaderboard 0.553 is association_v2 only and is not a "
        "learned_unet_orgdet_test_v1 tune target",
        "0.5530": "public leaderboard 0.553 is association_v2 only and is not a "
        "learned_unet_orgdet_test_v1 tune target",
        "0.612": "public leaderboard 0.612 is orgdet-test 56466744 only and is "
        "not a learned_unet_orgdet_test_v1 tune target",
        "0.6120": "public leaderboard 0.612 is orgdet-test 56466744 only and is "
        "not a learned_unet_orgdet_test_v1 tune target",
    }
    if text in forbidden:
        raise ValueError(forbidden[text])
    value = float(score)
    if abs(value - 0.553) < 1e-9:
        raise ValueError(forbidden["0.553"])
    if abs(value - 0.612) < 1e-9:
        raise ValueError(forbidden["0.612"])


def write_estimate_json(
    path: str | Path,
    n_public: int = 4,
    n_hidden: int = ESTIMATE_HIDDEN_VOLUMES,
) -> dict[str, Any]:
    dest = Path(path)
    payload = {
        "label": "learned_unet_orgdet_test_v1",
        "config_hash": LOCKED_ORGDET_TEST_HASH,
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "orgdet_config_hash": LOCKED_ORGDET_HASH,
        "orgdet_holdout_config_hash": LOCKED_ORGDET_HOLDOUT_HASH,
        "rejected_orgop_hash": REJECTED_ORGOP_HASH,
        "test_parent_hash": LOCKED_TEST_HASH,
        "gap_test_parent_hash": LOCKED_GAP_TEST_HASH,
        "prune_test_parent_hash": LOCKED_PRUNE_TEST_HASH,
        "sec_per_volume": ORGDET_SEC_PER_VOLUME,
        "source": runtime_estimate(1)["source"],
        "peak_memory_mb_observed": ORGDET_HOLDOUT_MEASURED_PEAK_MEMORY_MB,
        "public_test": {k: v for k, v in runtime_estimate(n_public).items() if k != "source"},
        "hidden_test_train_sized": {
            k: v for k, v in runtime_estimate(n_hidden).items() if k != "source"
        },
        "kaggle_gpu_limit_hours": KAGGLE_GPU_LIMIT_HOURS,
        "is_leaderboard_score": False,
        "note": (
            "Estimate only. Not a Kaggle run and not a leaderboard score. "
            "Do not overwrite mahdimzafari/biohub-learned-unet-prune-test-v1, "
            "mahdimzafari/biohub-learned-unet-gap-test-v1, "
            "mahdimzafari/biohub-learned-unet-test-v1, or "
            "mahdimzafari/biohub-learned-unet-orgdet-holdout-v1."
        ),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


__all__ = [
    "ASSOCIATION_V2_KERNEL_SLUG",
    "ASSOCIATION_V2_PUBLIC_SCORE",
    "ASSOCIATION_V2_SUBMISSION_REF",
    "ORGDET_TEST_PUBLIC_SCORE",
    "ORGDET_TEST_SUBMISSION_REF",
    "BANNED_PUBLIC_WEIGHT_MARKERS",
    "LOCKED_ORGDET_TEST_HASH",
    "ORGDET_TEST_KERNEL_SLUG",
    "ORGDET_TEST_LAST_RUN_UTC",
    "classify_orgdet_test_submit_poll",
    "classify_submission_notebook",
    "hidden_projection_from_rehearsal",
    "list_test_zarrs",
    "orgdet_test_config_hash",
    "orgdet_test_config_payload",
    "public_lb_not_a_tune_target",
    "refuse_hardcoded_public_stems",
    "refuse_overwrite_locked_slugs",
    "refuse_public_orgdet_test_weights",
    "refuse_train_volume",
    "require_orgdet_holdout_keep",
    "runtime_estimate",
    "write_estimate_json",
]
