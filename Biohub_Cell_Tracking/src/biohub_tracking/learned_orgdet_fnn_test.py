"""Test-time helpers for learned_unet_orgdet_fnn_test_v1.

Locked orgdet+gap+prune+ELF, then keep only forward nearest-neighbor
dt=1 edges and newly created degree-0 leftovers, on discovered
``test/*.zarr``. Do not overwrite orgdet-elf-test or earlier locked
slugs. Do not retune ``nn_tie_eps``. Do not score frozen ``6bba``
again.
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
from .learned_orgdet_elf import LOCKED_ORGDET_ELF_HASH, MAX_DT1_EDGE_UM
from .learned_orgdet_elf_test import (
    LOCKED_ORGDET_ELF_TEST_HASH,
    ORGDET_ELF_TEST_KERNEL_SLUG,
)
from .learned_orgdet_fnn import (
    LEARNED_ORGDET_FNN_LOCKED_HOLDOUT,
    LEARNED_ORGDET_FNN_LOCKED_TUNE,
    LOCKED_ORGDET_FNN_HASH,
    NN_TIE_EPS,
)
from .learned_orgdet_gap import LOCKED_ORGDET_GAP_HASH, ORGDET_LOCKED_HOLDOUT
from .learned_orgdet_gap_prune import (
    LEARNED_ORGDET_GAP_PRUNE_LOCKED_HOLDOUT,
    LEARNED_ORGDET_GAP_PRUNE_LOCKED_TUNE,
    LOCKED_ORGDET_GAP_PRUNE_HASH,
)
from .learned_orgdet_gap_prune_test import (
    LOCKED_ORGDET_GAP_PRUNE_TEST_HASH,
    ORGDET_GAP_PRUNE_TEST_KERNEL_SLUG,
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

ORGDET_FNN_TEST_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-fnn-test-v1"
ORGDET_GAP_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-gap-v1"
ORGDET_ELF_TEST_MEASURED_PUBLIC_SEC = 352.8744547367096
ORGDET_ELF_TEST_MEASURED_PUBLIC_VOLUMES = 4
ORGDET_ELF_TEST_MEASURED_PEAK_MEMORY_MB = 1161.0859375
FNN_OVERHEAD_SEC_PER_VOLUME = 1.0
ORGDET_ELF_TEST_SEC_PER_VOLUME = (
    ORGDET_ELF_TEST_MEASURED_PUBLIC_SEC / ORGDET_ELF_TEST_MEASURED_PUBLIC_VOLUMES
)
FNN_TEST_SEC_PER_VOLUME = ORGDET_ELF_TEST_SEC_PER_VOLUME + FNN_OVERHEAD_SEC_PER_VOLUME


def orgdet_fnn_test_config_payload() -> dict[str, Any]:
    return {
        "label": "learned_unet_orgdet_fnn_test_v1",
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
        "orgdet_elf_label": "learned_unet_orgdet_elf_v1",
        "orgdet_fnn_label": "learned_unet_orgdet_fnn_v1",
        "orgdet_gap_prune_test_label": "learned_unet_orgdet_gap_prune_test_v1",
        "orgdet_elf_test_label": "learned_unet_orgdet_elf_test_v1",
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
        "orgdet_elf_config_hash": LOCKED_ORGDET_ELF_HASH,
        "orgdet_fnn_config_hash": LOCKED_ORGDET_FNN_HASH,
        "orgdet_gap_prune_test_parent_hash": LOCKED_ORGDET_GAP_PRUNE_TEST_HASH,
        "orgdet_elf_test_parent_hash": LOCKED_ORGDET_ELF_TEST_HASH,
        "discovers_test_zarrs": True,
        "hardcoded_test_stems": [],
        "opens_train_volumes": False,
        "det_threshold": 0.99,
        "det_tta": True,
        "pool_kernel_um": 3.0,
        "linker": "orgdet_elf_then_keep_forward_nn_dt1",
        "motion_model": "last_position",
        "gate_um": 12.0,
        "max_link_um": 12.0,
        "max_gap_frames": 2,
        "gap_scale": "sqrt_dt",
        "assignment": "greedy",
        "interpolate_gaps": True,
        "drop_degree0": True,
        "max_dt1_edge_um": MAX_DT1_EDGE_UM,
        "drop_long_dt1_edges": True,
        "keep_forward_nn_only": True,
        "drop_degree0_after": True,
        "nn_tie_eps": NN_TIE_EPS,
        "max_parents_per_node": 1,
        "max_children_per_node": 1,
        "intensity_weight": 0.0,
        "downsample": [1, 4, 4],
        "normalize": "window_minmax",
        "scale_zyx": [1.625, 0.40625, 0.40625],
        "writes_submission_csv": True,
        "writes_run_manifest": True,
        "one_shot_holdout": False,
        "kernel_slug": ORGDET_FNN_TEST_KERNEL_SLUG,
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
        "orgdet_gap_prune_test_kernel_slug": ORGDET_GAP_PRUNE_TEST_KERNEL_SLUG,
        "orgdet_elf_test_kernel_slug": ORGDET_ELF_TEST_KERNEL_SLUG,
        "weights_name": WEIGHTS_NAME,
    }


def orgdet_fnn_test_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(
        payload or orgdet_fnn_test_config_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGDET_FNN_TEST_HASH = orgdet_fnn_test_config_hash()
assert LOCKED_ORGDET_FNN_TEST_HASH == "1af763e6ffc23188"
assert LOCKED_ORGDET_FNN_HASH == "95e9300eb84ba71c"
assert LOCKED_ORGDET_ELF_TEST_HASH == "ad375d39e13098d1"
assert LOCKED_ORGDET_ELF_HASH == "292c4f1f968f948d"
assert LOCKED_ORGDET_GAP_PRUNE_TEST_HASH == "52a30a5843e0a880"
assert LOCKED_ORGDET_GAP_PRUNE_HASH == "862e3e7f7ac0330e"
assert LOCKED_ORGDET_GAP_TEST_HASH == "1ce4837ae16bee52"
assert LOCKED_ORGDET_GAP_HASH == "ee31f36151325f9a"
assert LOCKED_ORGDET_TEST_HASH == "c45c859610d29e90"
assert LOCKED_GAP_HASH == "f4d888d190c45a4a"
assert LEARNED_ORGDET_FNN_LOCKED_HOLDOUT == 0.7520914330648746
assert LEARNED_ORGDET_FNN_LOCKED_TUNE == 0.7246526336456757
assert LEARNED_ORGDET_GAP_PRUNE_LOCKED_HOLDOUT == 0.7230176371507055
assert LEARNED_ORGDET_GAP_PRUNE_LOCKED_TUNE == 0.6934371804770478
assert ORGDET_LOCKED_HOLDOUT == 0.7052260455024029
assert ASSOCIATION_V2_SUBMISSION_REF == 56399003
assert ASSOCIATION_V2_PUBLIC_SCORE == "0.553"
assert ORGDET_TEST_SUBMISSION_REF == 56466744
assert ORGDET_TEST_PUBLIC_SCORE == "0.612"
assert ORGDET_GAP_TEST_SUBMISSION_REF == 56583783
assert ORGDET_GAP_TEST_PUBLIC_SCORE == "0.602"
assert float(NN_TIE_EPS) == 1e-9


def runtime_estimate(
    n_volumes: int,
    sec_per_volume: float = FNN_TEST_SEC_PER_VOLUME,
    limit_hours: float = KAGGLE_GPU_LIMIT_HOURS,
    peak_memory_mb: float = ORGDET_ELF_TEST_MEASURED_PEAK_MEMORY_MB,
) -> dict[str, Any]:
    n = int(n_volumes)
    sec = float(n) * float(sec_per_volume)
    return {
        "n_volumes": n,
        "sec_per_volume": float(sec_per_volume),
        "orgdet_elf_test_sec_per_volume": ORGDET_ELF_TEST_SEC_PER_VOLUME,
        "fnn_overhead_sec_per_volume": FNN_OVERHEAD_SEC_PER_VOLUME,
        "estimate_sec": sec,
        "estimate_hours": sec / 3600.0,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_observed": float(peak_memory_mb),
        "source": (
            "learned_unet_orgdet_elf_test_v1 measured 352.87 s / 4 volumes "
            "(88.22 s/volume) + 1.0 s forward-NN keep"
        ),
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
    }


def hidden_projection_from_rehearsal(
    public_runtime_sec: float,
    n_public: int = ORGDET_ELF_TEST_MEASURED_PUBLIC_VOLUMES,
    n_hidden: int = ESTIMATE_HIDDEN_VOLUMES,
    peak_memory_mb: float = ORGDET_ELF_TEST_MEASURED_PEAK_MEMORY_MB,
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
    if slug == ORGDET_ELF_TEST_KERNEL_SLUG:
        raise ValueError("refusing to overwrite locked learned_unet_orgdet_elf_test_v1 slug")
    if slug == ORGDET_FNN_TEST_KERNEL_SLUG:
        return


def classify_orgdet_fnn_test_submit_poll(
    *,
    kernel_complete: bool,
    kernel_failed: bool,
    output_names: Sequence[str],
    submissions: Sequence[Mapping[str, Any]],
    association_ref: int = ASSOCIATION_V2_SUBMISSION_REF,
    orgdet_test_ref: int = ORGDET_TEST_SUBMISSION_REF,
    orgdet_gap_test_ref: int = ORGDET_GAP_TEST_SUBMISSION_REF,
) -> dict[str, Any]:
    """Classify a read-only orgdet-fnn-test submit poll. Never submits."""
    names = [str(name) for name in output_names]
    rows = [dict(row) for row in submissions]
    refs = [row.get("ref") for row in rows]
    known = {str(association_ref), str(orgdet_test_ref), str(orgdet_gap_test_ref)}
    labels = [classify_submission_notebook(row.get("kernelSlug"), row.get("ref")) for row in rows]
    fnn_rows = [
        row
        for row, label in zip(rows, labels)
        if str(row.get("kernelSlug") or "") == ORGDET_FNN_TEST_KERNEL_SLUG
    ]
    extra = [
        row
        for row, label in zip(rows, labels)
        if str(row.get("ref")) not in known
        and str(row.get("kernelSlug") or "") != ORGDET_FNN_TEST_KERNEL_SLUG
        and label != "orgdet_fnn_acf_test"
    ]
    if kernel_failed:
        decision = "record_kernel_error"
    elif not kernel_complete:
        decision = "wait_kernel"
    elif "submission.csv" not in names:
        decision = "kernel_missing_csv"
    elif fnn_rows or extra:
        decision = "new_submission_present"
    else:
        decision = "human_action_required_ui_submit"
    return {
        "slug": ORGDET_FNN_TEST_KERNEL_SLUG,
        "decision": decision,
        "kernel_complete": bool(kernel_complete),
        "kernel_failed": bool(kernel_failed),
        "output_names": names,
        "n_submissions": len(rows),
        "submission_refs": refs,
        "is_competition_submission": False,
        "is_leaderboard_score": False,
        "human_action": (
            "Mahdi must Submit to Competition from the Kaggle UI on "
            "mahdimzafari/biohub-learned-unet-orgdet-fnn-test-v1 version 1 "
            "only after READY_FOR_MANUAL_SUBMISSION_REVIEW and a later "
            "authorizing message. Do not submit from the CLI."
            if decision == "human_action_required_ui_submit"
            else None
        ),
    }
