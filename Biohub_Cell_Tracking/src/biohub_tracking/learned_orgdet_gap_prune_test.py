"""Test-time helpers for learned_unet_orgdet_gap_prune_test_v1.

Locked orgdet detections, locked association_v2 gap-close, then drop
leftover degree-0 nodes on discovered ``test/*.zarr``. Do not overwrite
orgdet-gap-test, orgdet-test, prune-test, or gap-test. Do not retune.
Do not score frozen ``6bba`` again.
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
from .learned_orgdet_gap_prune import (
    LEARNED_ORGDET_GAP_PRUNE_LOCKED_HOLDOUT,
    LEARNED_ORGDET_GAP_PRUNE_LOCKED_TUNE,
    LOCKED_ORGDET_GAP_PRUNE_HASH,
)
from .learned_orgdet_gap_test import (
    LOCKED_ORGDET_GAP_TEST_HASH,
    ORGDET_GAP_TEST_KERNEL_SLUG,
    ORGDET_GAP_TEST_PUBLIC_SCORE,
    ORGDET_GAP_TEST_SUBMISSION_REF,
)
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
from .learned_test import (
    ESTIMATE_HIDDEN_VOLUMES,
    INFER_KERNEL_SLUG,
    KAGGLE_GPU_LIMIT_HOURS,
    TEST_KERNEL_SLUG,
    WEIGHTS_NAME,
    list_test_zarrs,
    refuse_hardcoded_public_stems,
    refuse_train_volume,
)

ORGDET_GAP_PRUNE_TEST_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-gap-prune-test-v1"
ORGDET_GAP_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-gap-v1"
ORGDET_GAP_TEST_MEASURED_PUBLIC_SEC = 312.69036865234375
ORGDET_GAP_TEST_MEASURED_PUBLIC_VOLUMES = 4
ORGDET_GAP_TEST_MEASURED_PEAK_MEMORY_MB = 1176.0
PRUNE_OVERHEAD_SEC_PER_VOLUME = 0.5
ORGDET_GAP_TEST_SEC_PER_VOLUME = (
    ORGDET_GAP_TEST_MEASURED_PUBLIC_SEC / ORGDET_GAP_TEST_MEASURED_PUBLIC_VOLUMES
)
PRUNE_TEST_SEC_PER_VOLUME = ORGDET_GAP_TEST_SEC_PER_VOLUME + PRUNE_OVERHEAD_SEC_PER_VOLUME


def orgdet_gap_prune_test_config_payload() -> dict[str, Any]:
    return {
        "label": "learned_unet_orgdet_gap_prune_test_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "orgdet_label": "learned_unet_orgdet_v1",
        "orgdet_holdout_label": "learned_unet_orgdet_holdout_v1",
        "orgdet_gap_label": "learned_unet_orgdet_gap_v1",
        "orgdet_gap_prune_label": "learned_unet_orgdet_gap_prune_v1",
        "orgdet_gap_test_label": "learned_unet_orgdet_gap_test_v1",
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
        "orgdet_gap_prune_config_hash": LOCKED_ORGDET_GAP_PRUNE_HASH,
        "orgdet_gap_test_parent_hash": LOCKED_ORGDET_GAP_TEST_HASH,
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
        "linker": "orgdet_gap_then_drop_degree0",
        "motion_model": "last_position",
        "gate_um": 12.0,
        "max_link_um": 12.0,
        "max_gap_frames": 2,
        "gap_scale": "sqrt_dt",
        "assignment": "greedy",
        "interpolate_gaps": True,
        "drop_degree0": True,
        "max_parents_per_node": 1,
        "max_children_per_node": 1,
        "intensity_weight": 0.0,
        "downsample": [1, 4, 4],
        "normalize": "window_minmax",
        "scale_zyx": [1.625, 0.40625, 0.40625],
        "writes_submission_csv": True,
        "writes_run_manifest": True,
        "one_shot_holdout": False,
        "kernel_slug": ORGDET_GAP_PRUNE_TEST_KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
        "orgdet_kernel_slug": ORGDET_KERNEL_SLUG,
        "orgdet_holdout_kernel_slug": ORGDET_HOLDOUT_KERNEL_SLUG,
        "orgdet_gap_kernel_slug": ORGDET_GAP_KERNEL_SLUG,
        "orgdet_gap_test_kernel_slug": ORGDET_GAP_TEST_KERNEL_SLUG,
        "orgdet_test_kernel_slug": ORGDET_TEST_KERNEL_SLUG,
        "orgop_kernel_slug": ORGOP_KERNEL_SLUG,
        "test_kernel_slug": TEST_KERNEL_SLUG,
        "gap_test_kernel_slug": GAP_TEST_KERNEL_SLUG,
        "prune_test_kernel_slug": PRUNE_TEST_KERNEL_SLUG,
        "weights_name": WEIGHTS_NAME,
    }


def orgdet_gap_prune_test_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(
        payload or orgdet_gap_prune_test_config_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGDET_GAP_PRUNE_TEST_HASH = orgdet_gap_prune_test_config_hash()
assert LOCKED_ORGDET_GAP_PRUNE_TEST_HASH == "52a30a5843e0a880"
assert LOCKED_ORGDET_GAP_PRUNE_HASH == "862e3e7f7ac0330e"
assert LOCKED_ORGDET_GAP_TEST_HASH == "1ce4837ae16bee52"
assert LOCKED_ORGDET_GAP_HASH == "ee31f36151325f9a"
assert LOCKED_ORGDET_TEST_HASH == "c45c859610d29e90"
assert LOCKED_GAP_HASH == "f4d888d190c45a4a"
assert LEARNED_ORGDET_GAP_PRUNE_LOCKED_HOLDOUT == 0.7230176371507055
assert LEARNED_ORGDET_GAP_PRUNE_LOCKED_TUNE == 0.6934371804770478
assert ORGDET_LOCKED_HOLDOUT == 0.7052260455024029
assert ASSOCIATION_V2_SUBMISSION_REF == 56399003
assert ASSOCIATION_V2_PUBLIC_SCORE == "0.553"
assert ORGDET_TEST_SUBMISSION_REF == 56466744
assert ORGDET_TEST_PUBLIC_SCORE == "0.612"
assert ORGDET_GAP_TEST_SUBMISSION_REF == 56583783
assert ORGDET_GAP_TEST_PUBLIC_SCORE == "0.602"


def runtime_estimate(
    n_volumes: int,
    sec_per_volume: float = PRUNE_TEST_SEC_PER_VOLUME,
    limit_hours: float = KAGGLE_GPU_LIMIT_HOURS,
    peak_memory_mb: float = ORGDET_GAP_TEST_MEASURED_PEAK_MEMORY_MB,
) -> dict[str, Any]:
    """Estimate GPU time from measured orgdet-gap-test plus isolate-drop overhead."""
    n = int(n_volumes)
    sec = float(n) * float(sec_per_volume)
    return {
        "n_volumes": n,
        "sec_per_volume": float(sec_per_volume),
        "orgdet_gap_test_sec_per_volume": ORGDET_GAP_TEST_SEC_PER_VOLUME,
        "prune_overhead_sec_per_volume": PRUNE_OVERHEAD_SEC_PER_VOLUME,
        "estimate_sec": sec,
        "estimate_hours": sec / 3600.0,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_observed": float(peak_memory_mb),
        "source": (
            "learned_unet_orgdet_gap_test_v1 measured 312.69 s / 4 volumes "
            "(78.17 s/volume) + 0.5 s isolate drop"
        ),
    }


def hidden_projection_from_rehearsal(
    public_runtime_sec: float,
    n_public: int = ORGDET_GAP_TEST_MEASURED_PUBLIC_VOLUMES,
    n_hidden: int = ESTIMATE_HIDDEN_VOLUMES,
    peak_memory_mb: float = ORGDET_GAP_TEST_MEASURED_PEAK_MEMORY_MB,
    limit_hours: float = KAGGLE_GPU_LIMIT_HOURS,
) -> dict[str, Any]:
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


def classify_orgdet_gap_prune_test_submit_poll(
    *,
    kernel_complete: bool,
    kernel_failed: bool,
    output_names: Sequence[str],
    submissions: Sequence[Mapping[str, Any]],
    association_ref: int = ASSOCIATION_V2_SUBMISSION_REF,
    orgdet_test_ref: int = ORGDET_TEST_SUBMISSION_REF,
    orgdet_gap_test_ref: int = ORGDET_GAP_TEST_SUBMISSION_REF,
) -> dict[str, Any]:
    """Classify a read-only orgdet-gap-prune-test submit poll. Never submits.

    Known competition rows (association_v2 ``56399003``, orgdet-test
    ``56466744``, and orgdet-gap-test ``56583783``) are not this kernel.
    Do not treat their public scores as this slug's result.
    """
    names = [str(name) for name in output_names]
    rows = [dict(row) for row in submissions]
    refs = [row.get("ref") for row in rows]
    known = {str(association_ref), str(orgdet_test_ref), str(orgdet_gap_test_ref)}
    labels = [classify_submission_notebook(row.get("kernelSlug"), row.get("ref")) for row in rows]
    prune_rows = [row for row, label in zip(rows, labels) if label == "orgdet_gap_prune_test"]
    extra = [
        row
        for row, label in zip(rows, labels)
        if str(row.get("ref")) not in known and label != "orgdet_gap_prune_test"
    ]
    association = [row for row in rows if str(row.get("ref")) == str(association_ref)]
    orgdet_test = [row for row in rows if str(row.get("ref")) == str(orgdet_test_ref)]
    orgdet_gap_test = [row for row in rows if str(row.get("ref")) == str(orgdet_gap_test_ref)]
    if kernel_failed:
        decision = "record_kernel_error"
    elif not kernel_complete:
        decision = "wait_kernel"
    elif "submission.csv" not in names:
        decision = "kernel_missing_csv"
    elif prune_rows or extra:
        decision = "new_submission_present"
    else:
        decision = "human_action_required_ui_submit"
    return {
        "slug": ORGDET_GAP_PRUNE_TEST_KERNEL_SLUG,
        "decision": decision,
        "kernel_complete": bool(kernel_complete),
        "kernel_failed": bool(kernel_failed),
        "output_names": names,
        "n_submissions": len(rows),
        "submission_refs": refs,
        "association_v2_submission_present": bool(association),
        "orgdet_test_submission_present": bool(orgdet_test),
        "orgdet_gap_test_submission_present": bool(orgdet_gap_test),
        "orgdet_gap_prune_test_submission_present": bool(prune_rows),
        "is_competition_submission": False,
        "is_leaderboard_score": False,
        "human_action": (
            "Mahdi must Submit to Competition from the Kaggle UI on "
            "mahdimzafari/biohub-learned-unet-orgdet-gap-prune-test-v1 "
            "version 1 only after READY_FOR_MANUAL_SUBMISSION_REVIEW and a "
            "later authorizing message. Do not submit from the CLI. Do not "
            "submit orgdet-gap-test, orgdet-test, prune-test, or infer-family "
            "gap-test unless separately authorized. Public 0.602 is "
            "orgdet-gap-test 56583783 only and is not this slug."
            if decision == "human_action_required_ui_submit"
            else None
        ),
    }


def refuse_overwrite_locked_slugs(slug: str) -> None:
    if slug == TEST_KERNEL_SLUG:
        raise ValueError("refusing to overwrite locked learned_unet_test_v1 slug")
    if slug == GAP_TEST_KERNEL_SLUG:
        raise ValueError("refusing to overwrite locked learned_unet_gap_test_v1 slug")
    if slug == PRUNE_TEST_KERNEL_SLUG:
        raise ValueError("refusing to overwrite locked learned_unet_prune_test_v1 slug")
    if slug == ORGDET_TEST_KERNEL_SLUG:
        raise ValueError("refusing to overwrite locked learned_unet_orgdet_test_v1 slug")
    if slug == ORGDET_GAP_TEST_KERNEL_SLUG:
        raise ValueError("refusing to overwrite locked learned_unet_orgdet_gap_test_v1 slug")
    if slug == ORGDET_GAP_PRUNE_TEST_KERNEL_SLUG:
        raise ValueError("refusing to overwrite locked learned_unet_orgdet_gap_prune_test_v1 slug")


def compare_prune_vs_gap_counts(
    prune: Mapping[str, Any],
    gap: Mapping[str, Any],
) -> dict[str, Any]:
    """Prune may only drop degree-0 nodes. Edges and gap counts must match."""
    errors: list[str] = []
    prune_edges = int(prune.get("n_edges") or 0)
    gap_edges = int(gap.get("n_edges") or 0)
    if prune_edges != gap_edges:
        errors.append(f"edges changed: prune {prune_edges} vs gap {gap_edges}")
    for key in ("n_gap_links", "n_interpolated", "n_predicted_divisions"):
        if int(prune.get(key) or 0) != int(gap.get(key) or 0):
            errors.append(f"{key} changed: prune {prune.get(key)} vs gap {gap.get(key)}")
    prune_nodes = int(prune.get("n_nodes") or 0)
    gap_nodes = int(gap.get("n_nodes") or 0)
    dropped = int(prune.get("n_dropped_isolates") or (gap_nodes - prune_nodes))
    if prune_nodes > gap_nodes:
        errors.append(f"prune added nodes: {prune_nodes} > {gap_nodes}")
    if dropped < 0:
        errors.append(f"negative isolate drop {dropped}")
    if prune_nodes != gap_nodes - dropped and "n_dropped_isolates" in prune:
        errors.append(
            f"node accounting: prune {prune_nodes} != gap {gap_nodes} - dropped {dropped}"
        )
    return {
        "ok": not errors,
        "errors": errors,
        "n_nodes_gap": gap_nodes,
        "n_nodes_prune": prune_nodes,
        "n_edges_gap": gap_edges,
        "n_edges_prune": prune_edges,
        "n_dropped_isolates": dropped,
    }


__all__ = [
    "ASSOCIATION_V2_PUBLIC_SCORE",
    "ASSOCIATION_V2_SUBMISSION_REF",
    "LOCKED_ORGDET_GAP_PRUNE_TEST_HASH",
    "ORGDET_GAP_PRUNE_TEST_KERNEL_SLUG",
    "ORGDET_GAP_TEST_KERNEL_SLUG",
    "ORGDET_GAP_TEST_PUBLIC_SCORE",
    "ORGDET_GAP_TEST_SUBMISSION_REF",
    "classify_orgdet_gap_prune_test_submit_poll",
    "compare_prune_vs_gap_counts",
    "hidden_projection_from_rehearsal",
    "list_test_zarrs",
    "orgdet_gap_prune_test_config_hash",
    "orgdet_gap_prune_test_config_payload",
    "refuse_hardcoded_public_stems",
    "refuse_overwrite_locked_slugs",
    "refuse_train_volume",
    "runtime_estimate",
]
