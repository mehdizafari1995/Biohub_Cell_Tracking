"""One-shot frozen-holdout GPU helper for locked learned_unet_orgdet_v1.

Applies the locked 44b6 orgdet operating point (organizer detections
``det_threshold=0.99``, ``pool_kernel_um=3.0``, flip TTA, greedy 12 µm
last-position linker) to frozen ``6bba`` only. Never opens ``44b6``
volumes. Does not retune hash ``80b20589198148b7``. Does not overwrite
rejected orgop or locked infer/holdout/test/prune slugs. Does not write
``submission.csv``. Not a competition submission. Score holdout once
against champion ``learned_unet_prune_v1`` 0.701378.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .learned_holdout import (  # noqa: F401
    HOLDOUT_STEMS,
    TUNING_INFER_STEMS,
    assert_holdout_only,
    graph_from_payload,
    graph_payload,
    load_graph_json,
    refuse_existing_holdout_score,
    refuse_non_holdout_stems,
    refuse_tuning_stems,
    write_graph_json,
)
from .learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH, TRAIN_KERNEL_SLUG
from .learned_orgdet import LOCKED_ORGDET_HASH, ORGDET_KERNEL_SLUG
from .learned_orgop import (
    BANNED_PUBLIC_WEIGHT_MARKERS,
    FALLBACK_WEIGHTS_NAME,
    INFER_KERNEL_SLUG,
    LOCKED_ORGOP_HASH as REJECTED_ORGOP_HASH,
    ORGANIZER_DET_THRESHOLD,
    ORGANIZER_POOL_KERNEL_UM,
    ORGOP_KERNEL_SLUG,
    PRUNE_LOCKED_HOLDOUT,
    WEIGHTS_NAME,
    refuse_public_weight_sources,
)
from .learned_split import (
    FROZEN_HOLDOUT_STEMS,
    HOLDOUT_SPECIMENS,
    TUNING_SPECIMEN,
)

ORGDET_HOLDOUT_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-holdout-v1"
SCALE_ZYX = (1.625, 0.40625, 0.40625)
MAX_LINK_UM = 12.0
ORGDET_LOCKED_TUNE = 0.6795779484295005
INFER_LOCKED_HOLDOUT = 0.6936047711123506


def orgdet_holdout_config_payload() -> dict[str, Any]:
    """Locked orgdet operating point on frozen 6bba. Do not retune."""
    return {
        "label": "learned_unet_orgdet_holdout_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "orgdet_label": "learned_unet_orgdet_v1",
        "orgop_label": "learned_unet_orgop_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "orgdet_config_hash": LOCKED_ORGDET_HASH,
        "rejected_orgop_hash": REJECTED_ORGOP_HASH,
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "infer_stems": list(HOLDOUT_STEMS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "refused_tuning_stems": list(TUNING_INFER_STEMS),
        "weights_source": "locked_learned_unet_v1_44b6_only",
        "weights_name": WEIGHTS_NAME,
        "fallback_weights_name": FALLBACK_WEIGHTS_NAME,
        "refuses_public_199_video_weights": True,
        "refuses_deepcenter": True,
        "refuses_secondary_seed_blend": True,
        "source_graphs": "locked_orgdet_operating_point",
        "retunes_detection": False,
        "uses_intensity": False,
        "uses_transformer": False,
        "uses_velocity": False,
        "uses_ilp": False,
        "linker": "greedy_gate_12um",
        "motion_model": "last_position",
        "gate_um": MAX_LINK_UM,
        "max_link_um": MAX_LINK_UM,
        "max_gap_frames": 0,
        "assignment": "greedy",
        "interpolate_gaps": False,
        "max_parents_per_node": 1,
        "max_children_per_node": 1,
        "intensity_weight": 0.0,
        "det_threshold": ORGANIZER_DET_THRESHOLD,
        "det_tta": True,
        "pool_kernel_um": ORGANIZER_POOL_KERNEL_UM,
        "downsample": [1, 4, 4],
        "normalize": "window_minmax",
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": False,
        "one_shot_holdout": True,
        "kernel_slug": ORGDET_HOLDOUT_KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
        "orgdet_kernel_slug": ORGDET_KERNEL_SLUG,
        "orgop_kernel_slug": ORGOP_KERNEL_SLUG,
    }


def orgdet_holdout_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(
        payload or orgdet_holdout_config_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGDET_HOLDOUT_HASH = orgdet_holdout_config_hash()
assert LOCKED_ORGDET_HOLDOUT_HASH == "912f0c19b572a054"
assert LOCKED_ORGDET_HASH == "80b20589198148b7"
assert REJECTED_ORGOP_HASH == "250c5bfacd4a8198"


def assert_no_submission_csv(working: str | Path) -> None:
    banned = Path(working) / "submission.csv"
    if banned.exists():
        banned.unlink()
        raise RuntimeError("learned_unet_orgdet_holdout_v1 must not write submission.csv")


def require_orgdet_tuning_keep(path: str | Path) -> dict[str, Any]:
    dest = Path(path)
    if not dest.exists():
        raise SystemExit("refusing holdout; orgdet 44b6 keep report missing.")
    payload = json.loads(dest.read_text())
    decision = str(payload.get("decision") or "")
    if decision != "keep_candidate_learned_unet_orgdet_v1":
        raise SystemExit(f"refusing holdout; orgdet tuning decision is {decision}")
    if payload.get("config_hash") != LOCKED_ORGDET_HASH:
        raise SystemExit("refusing holdout; orgdet 44b6 hash drifted")
    return payload


def holdout_decision(learned_combined: float) -> str:
    if float(learned_combined) > PRUNE_LOCKED_HOLDOUT:
        return "keep_candidate_learned_unet_orgdet_holdout_v1"
    return "reject_learned_unet_orgdet_holdout_v1"


def runtime_estimate(
    n_volumes: int = 2,
    infer_sec_per_volume: float = 92.2,
    limit_hours: float = 12.0,
    peak_memory_mb: float = 1300.0,
) -> dict[str, Any]:
    """GPU UNet at locked 0.99 / pool 3.0 plus greedy 12 µm. No ILP."""
    n = int(n_volumes)
    sec = float(n) * float(infer_sec_per_volume)
    return {
        "n_volumes": n,
        "infer_sec_per_volume": float(infer_sec_per_volume),
        "estimate_sec": sec,
        "estimate_hours": sec / 3600.0,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_budget": float(peak_memory_mb),
        "source": (
            "GPU locked orgdet operating point (0.99 / pool 3.0 / TTA + greedy "
            "12 µm) on frozen 6bba; no transformer; no ILP; no gap overlay. "
            "Per-volume time from learned_unet_holdout_v1 184.4 s / 2 volumes."
        ),
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
    }


def refuse_public_orgdet_holdout_weights(sources: Iterable[str]) -> None:
    refuse_public_weight_sources(sources)


def public_lb_not_a_tune_target(score: float | str) -> None:
    text = str(score).strip()
    if text in {"0.553", "0.5530"} or abs(float(score) - 0.553) < 1e-9:
        raise ValueError(
            "public leaderboard 0.553 is association_v2 only and is not a "
            "learned_unet_orgdet_holdout_v1 tune target"
        )
