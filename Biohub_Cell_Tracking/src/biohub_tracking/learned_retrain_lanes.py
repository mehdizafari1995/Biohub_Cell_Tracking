"""Preregistered 44b6-only post-retrain evaluation lanes.

Immutable hashes are locked before any 44b6 score is seen. Do not sweep.
Do not open ``6bba`` unless a lane strictly beats its comparator.
Not a competition submission.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from .learned_orgdet import LOCKED_ORGDET_HASH
from .learned_orgdet_holdout import ORGDET_LOCKED_TUNE
from .learned_orgop import INFER_LOCKED_TUNE, LOCKED_ORGOP_HASH
from .learned_retrain import LOCKED_RETRAIN_HASH, RETRAIN_KERNEL_SLUG
from .learned_split import (
    FROZEN_HOLDOUT_STEMS,
    HOLDOUT_SPECIMENS,
    INTERNAL_VAL_STEMS,
    TUNING_SPECIMEN,
    HoldoutLeakError,
    refuse_holdout_stems,
)

RETRAIN_WEIGHTS_BEST = "learned_unet_retrain_v1_best.pt"
RETRAIN_WEIGHTS_LAST = "learned_unet_retrain_v1_last.pt"
LANE_A_COMPARATOR = INFER_LOCKED_TUNE
LANE_B_COMPARATOR = ORGDET_LOCKED_TUNE
ORGDET_HOLDOUT_CHAMPION = 0.7052260455024029
LANE_A_SLUG = "mahdimzafari/biohub-learned-unet-retrain-infer-v1"
LANE_B_SLUG = "mahdimzafari/biohub-learned-unet-retrain-orgdet-v1"
LANE_C_SLUG = "mahdimzafari/biohub-learned-unet-retrain-orgop-v1"
LANE_LABELS = ("A", "B", "C")


def _hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def lane_a_payload() -> dict[str, Any]:
    """Locked infer operating point on retrain weights vs 0.655751."""
    return {
        "label": "learned_unet_retrain_infer_v1",
        "lane": "A",
        "parent_retrain_hash": LOCKED_RETRAIN_HASH,
        "parent_train_hash": TRAIN_CONFIG_HASH,
        "parent_infer_hash": LOCKED_INFER_HASH,
        "comparator": LANE_A_COMPARATOR,
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "infer_stems": list(INTERNAL_VAL_STEMS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "weights_name": RETRAIN_WEIGHTS_BEST,
        "fallback_weights_name": RETRAIN_WEIGHTS_LAST,
        "det_threshold": 0.5,
        "det_tta": True,
        "pool_kernel_um": 5.0,
        "linker": "greedy_gate_12um",
        "gate_um": 12.0,
        "uses_transformer": False,
        "uses_ilp": False,
        "writes_submission_csv": False,
        "opens_6bba": False,
        "kernel_slug": "mahdimzafari/biohub-learned-unet-retrain-infer-v1",
        "train_kernel_slug": RETRAIN_KERNEL_SLUG,
    }


def lane_b_payload() -> dict[str, Any]:
    """Locked orgdet geometry on retrain weights vs 0.679578."""
    return {
        "label": "learned_unet_retrain_orgdet_v1",
        "lane": "B",
        "parent_retrain_hash": LOCKED_RETRAIN_HASH,
        "parent_train_hash": TRAIN_CONFIG_HASH,
        "parent_orgdet_hash": LOCKED_ORGDET_HASH,
        "comparator": LANE_B_COMPARATOR,
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "infer_stems": list(INTERNAL_VAL_STEMS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "weights_name": RETRAIN_WEIGHTS_BEST,
        "fallback_weights_name": RETRAIN_WEIGHTS_LAST,
        "det_threshold": 0.99,
        "det_tta": True,
        "pool_kernel_um": 3.0,
        "linker": "greedy_gate_12um",
        "gate_um": 12.0,
        "uses_transformer": False,
        "uses_ilp": False,
        "writes_submission_csv": False,
        "opens_6bba": False,
        "kernel_slug": "mahdimzafari/biohub-learned-unet-retrain-orgdet-v1",
        "train_kernel_slug": RETRAIN_KERNEL_SLUG,
    }


def lane_c_payload() -> dict[str, Any]:
    """Organizer 0.99/3µm + softmax 0.5 + fixed ILP vs lane B."""
    return {
        "label": "learned_unet_retrain_orgop_v1",
        "lane": "C",
        "parent_retrain_hash": LOCKED_RETRAIN_HASH,
        "parent_train_hash": TRAIN_CONFIG_HASH,
        "parent_orgop_hash": LOCKED_ORGOP_HASH,
        "comparator": "lane_B_official_combined",
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "infer_stems": list(INTERNAL_VAL_STEMS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "weights_name": RETRAIN_WEIGHTS_BEST,
        "fallback_weights_name": RETRAIN_WEIGHTS_LAST,
        "det_threshold": 0.99,
        "det_tta": True,
        "pool_kernel_um": 3.0,
        "edge_activation": "softmax",
        "edge_threshold": 0.5,
        "linker": "tracksdata_ilp",
        "use_ilp": True,
        "ilp_edge_weight": -1.0,
        "ilp_appearance_weight": 0.1,
        "ilp_disappearance_weight": 0.1,
        "ilp_division_weight": 1.0,
        "uses_transformer": True,
        "writes_submission_csv": False,
        "opens_6bba": False,
        "kernel_slug": "mahdimzafari/biohub-learned-unet-retrain-orgop-v1",
        "train_kernel_slug": RETRAIN_KERNEL_SLUG,
    }


LANE_A_HASH = _hash(lane_a_payload())
LANE_B_HASH = _hash(lane_b_payload())
LANE_C_HASH = _hash(lane_c_payload())
assert LANE_A_HASH == "1db21f87d00ce083"
assert LANE_B_HASH == "594f6fbc03c974cf"
assert LANE_C_HASH == "f924802e5f716a61"


def lane_decision(combined: float, comparator: float) -> str:
    if float(combined) > float(comparator):
        return "keep_lane"
    return "reject_lane"


def select_winner(scores: Mapping[str, float], comparators: Mapping[str, float]) -> str | None:
    kept = {
        name: float(score)
        for name, score in scores.items()
        if float(score) > float(comparators[name])
    }
    if not kept:
        return None
    return max(kept, key=kept.get)


def assert_tuning_only(role: str) -> None:
    if role != "tuning":
        raise SystemExit(
            "refusing to score holdout; learned_unet_retrain lanes are 44b6-only "
            "until a documented locked winner"
        )


def refuse_incomplete_retrain(status: str | None, output_names: Iterable[str] | None = None) -> None:
    names = [str(name) for name in (output_names or [])]
    if "submission.csv" in names:
        raise SystemExit("retrain helper must not write submission.csv")
    state = str(status or "").strip().lower()
    if state not in {"complete", "completed"}:
        raise SystemExit(
            f"retrain helper is {state or 'unknown'}; do not invent a 44b6 score"
        )


def require_lane_graph_paths(
    graph_dir: str | Path,
    stems: Sequence[str] | None = None,
) -> list[Path]:
    dest = Path(graph_dir)
    wanted = tuple(stems or INTERNAL_VAL_STEMS)
    refuse_holdout_stems(wanted)
    if not dest.exists():
        raise FileNotFoundError(f"missing lane graph dir {dest}; do not invent a score")
    found = [path.stem for path in dest.iterdir() if path.suffix == ".json"]
    refuse_holdout_stems(found)
    extra = [stem for stem in found if stem not in wanted and not stem.endswith("manifest")]
    if extra:
        raise HoldoutLeakError(
            "learned_unet_retrain lanes only open frozen-tuning stems; got " + ", ".join(extra)
        )
    missing = [stem for stem in wanted if not (dest / f"{stem}.json").exists()]
    if missing:
        raise FileNotFoundError(
            "missing 44b6 lane graphs " + ", ".join(missing) + "; do not invent a score"
        )
    return [dest / f"{stem}.json" for stem in wanted]


def lane_comparators(scores: Mapping[str, float]) -> dict[str, float]:
    missing = [name for name in LANE_LABELS if name not in scores]
    if missing:
        raise ValueError(f"missing lane scores: {', '.join(missing)}; do not invent a score")
    return {
        "A": float(LANE_A_COMPARATOR),
        "B": float(LANE_B_COMPARATOR),
        "C": float(scores["B"]),
    }


def decide_all_lanes(scores: Mapping[str, float]) -> dict[str, Any]:
    comparators = lane_comparators(scores)
    decisions = {name: lane_decision(scores[name], comparators[name]) for name in LANE_LABELS}
    winner = select_winner(scores, comparators)
    return {
        "decisions": decisions,
        "comparators": comparators,
        "winner": winner,
        "may_open_6bba": winner is not None,
        "opens_6bba": False,
        "holdout_champion": ORGDET_HOLDOUT_CHAMPION,
    }
