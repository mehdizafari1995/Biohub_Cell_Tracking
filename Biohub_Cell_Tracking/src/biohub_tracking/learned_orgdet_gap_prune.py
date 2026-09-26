"""Drop leftover degree-0 detections after locked orgdet+gap.

Applies locked ``learned_unet_orgdet_gap_v1`` (orgdet relink when the
source graph is empty-edge orgop, then association_v2 gap-close), then
removes nodes that still have in-degree 0 and out-degree 0. Detection,
consecutive dt=1 links, and gap gates are not retuned. This is not
another 12 µm assignment: leftover isolates inflate ``T_pred`` without
contributing edges. Same isolate-drop rule as ``learned_unet_prune_v1``,
now on the holdout champion graphs.

Never opens ``6bba`` during tuning. Does not write ``submission.csv``.
Not a competition submission. A local official score is not a
leaderboard number. Do not overwrite locked slugs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .graphs import TrackGraph
from .learned_gap import LOCKED_GAP_HASH, LOCKED_GATES as GAP_LOCKED_GATES
from .learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from .learned_orgdet import LOCKED_ORGDET_HASH
from .learned_orgdet_gap import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LOCKED_ORGDET_GAP_HASH,
    LOCKED_ORGDET_HOLDOUT_NODE_COUNTS,
    LOCKED_ORGDET_NODE_COUNTS,
    apply_orgdet_gap_overlay,
    ensure_orgdet_tuning_graph,
    load_role_graphs,
    refuse_non_holdout_stems as refuse_orgdet_gap_holdout,
    refuse_non_orgdet_node_count,
    refuse_non_tuning_stems as refuse_orgdet_gap_tuning,
)
from .learned_orgop import (
    LOCKED_ORGOP_HASH as REJECTED_ORGOP_HASH,
    ORGANIZER_DET_THRESHOLD,
    ORGANIZER_POOL_KERNEL_UM,
)
from .learned_prune import drop_degree0
from .learned_split import HOLDOUT_SPECIMENS, TUNING_SPECIMEN, specimen_of

# Official locked orgdet+gap combined scores. Do not retune those overlays.
ORGDET_GAP_LOCKED_TUNE = 0.6872212709981039
ORGDET_GAP_LOCKED_HOLDOUT = 0.7228052416579369
# Official one-shot scores of this overlay. Do not retune. Do not rescore.
LEARNED_ORGDET_GAP_PRUNE_LOCKED_TUNE = 0.6934371804770478
LEARNED_ORGDET_GAP_PRUNE_LOCKED_HOLDOUT = 0.7230176371507055
SCALE_ZYX = (1.625, 0.40625, 0.40625)

LOCKED_GATES = {
    "gate_um": GAP_LOCKED_GATES["gate_um"],
    "max_gap_frames": GAP_LOCKED_GATES["max_gap_frames"],
    "gap_scale": GAP_LOCKED_GATES["gap_scale"],
    "assignment": GAP_LOCKED_GATES["assignment"],
    "interpolate_gaps": GAP_LOCKED_GATES["interpolate_gaps"],
    "drop_degree0": True,
    "max_parents_per_node": GAP_LOCKED_GATES["max_parents_per_node"],
    "max_children_per_node": GAP_LOCKED_GATES["max_children_per_node"],
    "intensity_weight": GAP_LOCKED_GATES["intensity_weight"],
}


def orgdet_gap_prune_config_payload() -> dict[str, Any]:
    """First orgdet+gap isolate-prune operating point. Do not retune from holdout."""
    return {
        "label": "learned_unet_orgdet_gap_prune_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "orgdet_label": "learned_unet_orgdet_v1",
        "orgdet_gap_label": "learned_unet_orgdet_gap_v1",
        "gap_label": "learned_unet_gap_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "orgdet_config_hash": LOCKED_ORGDET_HASH,
        "orgdet_gap_config_hash": LOCKED_ORGDET_GAP_HASH,
        "gap_config_hash": LOCKED_GAP_HASH,
        "source_graphs": "locked_learned_unet_orgdet_graphs",
        "retunes_detection": False,
        "retunes_consecutive_links": False,
        "retunes_gap_gates": False,
        "uses_intensity": False,
        "uses_transformer": False,
        "uses_velocity": False,
        "uses_ilp": False,
        "linker": "orgdet_gap_then_drop_degree0",
        "gate_um": LOCKED_GATES["gate_um"],
        "max_gap_frames": LOCKED_GATES["max_gap_frames"],
        "gap_scale": LOCKED_GATES["gap_scale"],
        "assignment": LOCKED_GATES["assignment"],
        "interpolate_gaps": LOCKED_GATES["interpolate_gaps"],
        "drop_degree0": LOCKED_GATES["drop_degree0"],
        "max_parents_per_node": LOCKED_GATES["max_parents_per_node"],
        "max_children_per_node": LOCKED_GATES["max_children_per_node"],
        "intensity_weight": LOCKED_GATES["intensity_weight"],
        "det_threshold": ORGANIZER_DET_THRESHOLD,
        "det_tta": True,
        "pool_kernel_um": ORGANIZER_POOL_KERNEL_UM,
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": False,
    }


def orgdet_gap_prune_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(
        payload or orgdet_gap_prune_config_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGDET_GAP_PRUNE_HASH = orgdet_gap_prune_config_hash(orgdet_gap_prune_config_payload())
assert LOCKED_ORGDET_GAP_PRUNE_HASH == "862e3e7f7ac0330e"
assert LOCKED_ORGDET_GAP_HASH == "ee31f36151325f9a"
assert LOCKED_ORGDET_HASH == "80b20589198148b7"
assert LOCKED_GAP_HASH == "f4d888d190c45a4a"
assert REJECTED_ORGOP_HASH == "250c5bfacd4a8198"


def _clean_stem(raw: str) -> str:
    stem = Path(raw).name
    for suffix in (".zarr", ".geff", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def refuse_non_tuning_stems(stems: Iterable[str]) -> None:
    for raw in stems:
        stem = _clean_stem(str(raw))
        if specimen_of(stem) != TUNING_SPECIMEN or stem not in INFER_STEMS:
            raise ValueError(f"learned_unet_orgdet_gap_prune_v1 tuning refuses stem {stem}")


def refuse_non_holdout_stems(stems: Iterable[str]) -> None:
    for raw in stems:
        stem = _clean_stem(str(raw))
        if specimen_of(stem) in HOLDOUT_SPECIMENS or stem in HOLDOUT_STEMS:
            if stem not in HOLDOUT_STEMS:
                raise ValueError(f"learned_unet_orgdet_gap_prune_v1 holdout refuses extra stem {stem}")
            continue
        raise ValueError(f"learned_unet_orgdet_gap_prune_v1 holdout refuses non-holdout stem {stem}")


def apply_orgdet_gap_prune_overlay(graph: TrackGraph) -> TrackGraph:
    """Locked orgdet+gap overlay, then drop leftover isolates. Do not retune."""
    if orgdet_gap_prune_config_hash() != LOCKED_ORGDET_GAP_PRUNE_HASH:
        raise RuntimeError("learned_unet_orgdet_gap_prune_v1 hash drifted; refusing to retune")
    if LOCKED_GATES["gap_scale"] != "sqrt_dt" or float(LOCKED_GATES["gate_um"]) != 12.0:
        raise RuntimeError("learned_unet_orgdet_gap_prune_v1 would retune gap gates; refusing")
    if not LOCKED_GATES["drop_degree0"]:
        raise RuntimeError("learned_unet_orgdet_gap_prune_v1 drop_degree0 disabled; refusing")
    gapped = apply_orgdet_gap_overlay(graph)
    out = drop_degree0(gapped)
    out.metadata["overlay"] = "learned_unet_orgdet_gap_prune_v1"
    out.metadata["config_hash"] = LOCKED_ORGDET_GAP_PRUNE_HASH
    out.metadata["parent_orgdet_hash"] = LOCKED_ORGDET_HASH
    out.metadata["parent_orgdet_gap_hash"] = LOCKED_ORGDET_GAP_HASH
    out.metadata["parent_gap_hash"] = LOCKED_GAP_HASH
    out.metadata["rejected_orgop_hash"] = REJECTED_ORGOP_HASH
    out.metadata["uses_intensity"] = False
    out.metadata["uses_transformer"] = False
    out.metadata["uses_ilp"] = False
    out.metadata["retunes_detection"] = False
    out.metadata["retunes_consecutive_links"] = False
    out.metadata["retunes_gap_gates"] = False
    out.metadata["drop_degree0"] = True
    return out


def overlay_summary(before: TrackGraph, after: TrackGraph) -> dict[str, Any]:
    return {
        "dataset": after.dataset,
        "n_nodes_before": before.n_nodes,
        "n_nodes_after": after.n_nodes,
        "n_edges_before": before.n_edges,
        "n_edges_after": after.n_edges,
        "n_extra_nodes": int(after.n_nodes - before.n_nodes),
        "n_extra_edges": int(after.n_edges - before.n_edges),
        "n_dropped_isolates": int(after.metadata.get("n_dropped_isolates") or 0),
        "n_gap_links": int(after.metadata.get("n_gap_links") or 0),
        "n_interpolated": int(after.metadata.get("n_interpolated") or 0),
        "n_nodes_before_prune": int(after.metadata.get("n_nodes_before_prune") or after.n_nodes),
        "overlay": "learned_unet_orgdet_gap_prune_v1",
        "config_hash": LOCKED_ORGDET_GAP_PRUNE_HASH,
    }


def refuse_existing_holdout_score(path: str | Path) -> None:
    dest = Path(path)
    if not dest.exists():
        return
    payload = json.loads(dest.read_text())
    if payload.get("status") == "holdout_scored" and payload.get("learned_orgdet_gap_prune_combined") is not None:
        raise SystemExit(
            f"refusing to score frozen holdout twice; existing decision={payload.get('decision')}"
        )


def require_tuning_keep(path: str | Path) -> dict[str, Any]:
    dest = Path(path)
    if not dest.exists():
        raise SystemExit("refusing holdout; tuning report missing. Score 44b6 first.")
    payload = json.loads(dest.read_text())
    decision = str(payload.get("decision") or "")
    if not decision.startswith("keep_"):
        raise SystemExit(f"refusing holdout; tuning decision is {decision}")
    return payload


def tuning_decision(learned_combined: float, n_dropped: int) -> str:
    if float(learned_combined) <= ORGDET_GAP_LOCKED_TUNE:
        return "reject_learned_unet_orgdet_gap_prune_v1"
    if int(n_dropped) == 0:
        return "keep_as_noop_learned_unet_orgdet_gap_prune_v1"
    return "keep_candidate_learned_unet_orgdet_gap_prune_v1"


def holdout_decision(learned_combined: float) -> str:
    if float(learned_combined) <= ORGDET_GAP_LOCKED_HOLDOUT:
        return "reject_learned_unet_orgdet_gap_prune_v1"
    return "keep_candidate_learned_unet_orgdet_gap_prune_v1"


def runtime_estimate(
    n_volumes: int = 2,
    overlay_sec_per_volume: float = 5.5,
    prune_overhead: float = 0.5,
    limit_hours: float = 12.0,
    peak_memory_mb: float = 900.0,
) -> dict[str, Any]:
    """CPU orgdet+gap overlay plus isolate drop. No new UNet pass."""
    n = int(n_volumes)
    sec = float(n) * (float(overlay_sec_per_volume) + float(prune_overhead))
    return {
        "n_volumes": n,
        "overlay_sec_per_volume": float(overlay_sec_per_volume),
        "prune_overhead": float(prune_overhead),
        "estimate_sec": sec,
        "estimate_hours": sec / 3600.0,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_budget": float(peak_memory_mb),
        "source": (
            "CPU locked orgdet+gap overlay then drop degree-0 leftovers on "
            "locked organizer detections; no transformer; no ILP; no new UNet"
        ),
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
    }


# Re-export helpers used by the scoring script.
__all__ = [
    "HOLDOUT_STEMS",
    "INFER_STEMS",
    "LOCKED_ORGDET_GAP_HASH",
    "LOCKED_ORGDET_GAP_PRUNE_HASH",
    "LOCKED_ORGDET_HOLDOUT_NODE_COUNTS",
    "LOCKED_ORGDET_NODE_COUNTS",
    "LEARNED_ORGDET_GAP_PRUNE_LOCKED_HOLDOUT",
    "LEARNED_ORGDET_GAP_PRUNE_LOCKED_TUNE",
    "ORGDET_GAP_LOCKED_HOLDOUT",
    "ORGDET_GAP_LOCKED_TUNE",
    "apply_orgdet_gap_prune_overlay",
    "ensure_orgdet_tuning_graph",
    "holdout_decision",
    "load_role_graphs",
    "orgdet_gap_prune_config_hash",
    "orgdet_gap_prune_config_payload",
    "overlay_summary",
    "refuse_existing_holdout_score",
    "refuse_non_holdout_stems",
    "refuse_non_orgdet_node_count",
    "refuse_non_tuning_stems",
    "require_tuning_keep",
    "runtime_estimate",
    "tuning_decision",
    "refuse_orgdet_gap_holdout",
    "refuse_orgdet_gap_tuning",
]
