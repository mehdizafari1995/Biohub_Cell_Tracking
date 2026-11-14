"""Conservative second-child overlay on locked FNN graphs.

Applies locked ``division_v2`` gates to locked
``learned_unet_orgdet_fnn_v1`` graphs. Does not retune detection, the
12 µm linker, gap, ELF, FNN, or division thresholds. Learned graphs
have no intensity, so the intensity-sum gate is unused. Not a
competition submission. A local official score is not a leaderboard
number. Do not overwrite locked slugs. Do not use public 0.602 / 0.612
/ 0.553.

This is not ``learned_unet_division_v1`` (that overlay sat on infer
graphs) and not a rank_v1 retune.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import LinkingConfig
from .division import propose_divisions
from .graphs import TrackGraph
from .learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from .learned_orgdet import LOCKED_ORGDET_HASH
from .learned_orgdet_elf import LOCKED_ORGDET_ELF_HASH
from .learned_orgdet_fnn import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LEARNED_ORGDET_FNN_LOCKED_HOLDOUT,
    LEARNED_ORGDET_FNN_LOCKED_TUNE,
    LOCKED_ORGDET_FNN_HASH,
    apply_orgdet_fnn_overlay,
    ensure_orgdet_tuning_graph,
    load_role_graphs,
    refuse_non_orgdet_node_count,
)
from .learned_orgdet_gap import LOCKED_ORGDET_GAP_HASH
from .learned_orgdet_gap_prune import LOCKED_ORGDET_GAP_PRUNE_HASH
from .learned_orgop import (
    LOCKED_ORGOP_HASH as REJECTED_ORGOP_HASH,
    ORGANIZER_DET_THRESHOLD,
    ORGANIZER_POOL_KERNEL_UM,
)
from .learned_split import HOLDOUT_SPECIMENS, TUNING_SPECIMEN, specimen_of

DIVISION_CONFIG_HASH = "0fa756963e5d42d1"
SCALE_ZYX = (1.625, 0.40625, 0.40625)
ORGDET_FNN_LOCKED_TUNE = LEARNED_ORGDET_FNN_LOCKED_TUNE
ORGDET_FNN_LOCKED_HOLDOUT = LEARNED_ORGDET_FNN_LOCKED_HOLDOUT
TUNE_MARGIN = 0.002000
TUNE_KEEP_THRESHOLD = ORGDET_FNN_LOCKED_TUNE + TUNE_MARGIN

# Locked division_v2 gates from 44b6 GEFF geometry. Do not retune.
LOCKED_GATES = {
    "max_parent_child_um": 10.0,
    "min_parent_child_um": 0.0,
    "min_child_separation_um": 8.3,
    "max_sibling_alignment": 0.05,
    "require_parent_predecessor": True,
    "require_no_alternative_parent": True,
    "alternative_parent_gate_um": 12.0,
    "min_displacement_um": 1.0,
    "max_children": 2,
    "intensity_sum_tolerance": 0.45,
    "gate_um": 12.0,
}


def orgdet_fnn_div_config_payload() -> dict[str, Any]:
    """First FNN + locked division_v2 operating point. Lock before the 44b6 score."""
    return {
        "label": "learned_unet_orgdet_fnn_div_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "orgdet_label": "learned_unet_orgdet_v1",
        "orgdet_gap_label": "learned_unet_orgdet_gap_v1",
        "orgdet_gap_prune_label": "learned_unet_orgdet_gap_prune_v1",
        "orgdet_elf_label": "learned_unet_orgdet_elf_v1",
        "orgdet_fnn_label": "learned_unet_orgdet_fnn_v1",
        "division_label": "division_v2",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "orgdet_config_hash": LOCKED_ORGDET_HASH,
        "orgdet_gap_config_hash": LOCKED_ORGDET_GAP_HASH,
        "orgdet_gap_prune_config_hash": LOCKED_ORGDET_GAP_PRUNE_HASH,
        "orgdet_elf_config_hash": LOCKED_ORGDET_ELF_HASH,
        "orgdet_fnn_config_hash": LOCKED_ORGDET_FNN_HASH,
        "division_config_hash": DIVISION_CONFIG_HASH,
        "source_graphs": "locked_learned_unet_orgdet_fnn_graphs",
        "retunes_detection": False,
        "retunes_consecutive_links": False,
        "retunes_gap_gates": False,
        "retunes_max_dt1_edge_um": False,
        "retunes_nn_tie_eps": False,
        "retunes_division_gates": False,
        "uses_intensity": False,
        "uses_transformer": False,
        "uses_velocity": False,
        "uses_ilp": False,
        "linker": "orgdet_fnn_then_locked_division_v2_second_child",
        "allow_divisions": True,
        "max_parent_child_um": LOCKED_GATES["max_parent_child_um"],
        "min_parent_child_um": LOCKED_GATES["min_parent_child_um"],
        "min_child_separation_um": LOCKED_GATES["min_child_separation_um"],
        "max_sibling_alignment": LOCKED_GATES["max_sibling_alignment"],
        "require_parent_predecessor": LOCKED_GATES["require_parent_predecessor"],
        "require_no_alternative_parent": LOCKED_GATES["require_no_alternative_parent"],
        "alternative_parent_gate_um": LOCKED_GATES["alternative_parent_gate_um"],
        "min_displacement_um": LOCKED_GATES["min_displacement_um"],
        "max_children": LOCKED_GATES["max_children"],
        "intensity_sum_tolerance": LOCKED_GATES["intensity_sum_tolerance"],
        "gate_um": LOCKED_GATES["gate_um"],
        "assignment": "unchanged_locked_orgdet_fnn",
        "tune_keep_threshold": TUNE_KEEP_THRESHOLD,
        "tune_margin": TUNE_MARGIN,
        "det_threshold": ORGANIZER_DET_THRESHOLD,
        "det_tta": True,
        "pool_kernel_um": ORGANIZER_POOL_KERNEL_UM,
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": False,
    }


def orgdet_fnn_div_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(
        payload or orgdet_fnn_div_config_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGDET_FNN_DIV_HASH = orgdet_fnn_div_config_hash(orgdet_fnn_div_config_payload())
assert LOCKED_ORGDET_FNN_DIV_HASH == "e184e6ce5bd90e46"
assert LOCKED_ORGDET_FNN_HASH == "95e9300eb84ba71c"
assert DIVISION_CONFIG_HASH == "0fa756963e5d42d1"
assert TUNE_KEEP_THRESHOLD == 0.7266526336456757
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
            raise ValueError(f"learned_unet_orgdet_fnn_div_v1 tuning refuses stem {stem}")


def refuse_non_holdout_stems(stems: Iterable[str]) -> None:
    for raw in stems:
        stem = _clean_stem(str(raw))
        if specimen_of(stem) in HOLDOUT_SPECIMENS or stem in HOLDOUT_STEMS:
            if stem not in HOLDOUT_STEMS:
                raise ValueError(f"learned_unet_orgdet_fnn_div_v1 holdout refuses extra stem {stem}")
            continue
        raise ValueError(f"learned_unet_orgdet_fnn_div_v1 holdout refuses non-holdout stem {stem}")


def locked_linking_config() -> LinkingConfig:
    return LinkingConfig(
        method="learned_unet_orgdet_fnn_div_v1",
        max_distance_um=12.0,
        allow_divisions=True,
        division_gap_um=8.0,
        motion_model="none",
        gate_um=float(LOCKED_GATES["gate_um"]),
        max_gap_frames=0,
        assignment="greedy",
        interpolate_gaps=False,
        min_child_separation_um=float(LOCKED_GATES["min_child_separation_um"]),
        max_parent_child_um=float(LOCKED_GATES["max_parent_child_um"]),
        min_parent_child_um=float(LOCKED_GATES["min_parent_child_um"]),
        intensity_sum_tolerance=float(LOCKED_GATES["intensity_sum_tolerance"]),
        max_children=int(LOCKED_GATES["max_children"]),
        require_parent_predecessor=bool(LOCKED_GATES["require_parent_predecessor"]),
        require_no_alternative_parent=bool(LOCKED_GATES["require_no_alternative_parent"]),
        alternative_parent_gate_um=float(LOCKED_GATES["alternative_parent_gate_um"]),
        max_sibling_alignment=float(LOCKED_GATES["max_sibling_alignment"]),
        min_displacement_um=float(LOCKED_GATES["min_displacement_um"]),
    )


def n_predicted_forks(graph: TrackGraph) -> int:
    return int(sum(1 for children in graph.out_edges().values() if len(children) >= 2))


def apply_locked_fnn_div(graph: TrackGraph) -> TrackGraph:
    """Add at most one extra child per FNN parent when locked gates pass."""
    if orgdet_fnn_div_config_hash() != LOCKED_ORGDET_FNN_DIV_HASH:
        raise RuntimeError("learned_unet_orgdet_fnn_div_v1 hash drifted; refusing to retune")
    if LOCKED_GATES["max_parent_child_um"] != 10.0 or LOCKED_GATES["min_child_separation_um"] != 8.3:
        raise RuntimeError("learned_unet_orgdet_fnn_div_v1 would retune division gates; refusing")
    out = propose_divisions(graph, None, locked_linking_config(), SCALE_ZYX)
    out.metadata = dict(out.metadata)
    out.metadata["overlay"] = "learned_unet_orgdet_fnn_div_v1"
    out.metadata["config_hash"] = LOCKED_ORGDET_FNN_DIV_HASH
    out.metadata["parent_orgdet_fnn_hash"] = LOCKED_ORGDET_FNN_HASH
    out.metadata["division_config_hash"] = DIVISION_CONFIG_HASH
    out.metadata["uses_intensity"] = False
    out.metadata["uses_transformer"] = False
    out.metadata["uses_ilp"] = False
    out.metadata["retunes_detection"] = False
    out.metadata["retunes_consecutive_links"] = False
    out.metadata["retunes_gap_gates"] = False
    out.metadata["retunes_division_gates"] = False
    out.metadata["is_12um_assignment"] = False
    return out


def apply_orgdet_fnn_div_overlay(graph: TrackGraph) -> TrackGraph:
    """Locked FNN, then locked division_v2 second-child proposals."""
    fnn = apply_orgdet_fnn_overlay(graph)
    return apply_locked_fnn_div(fnn)


def overlay_summary(before: TrackGraph, after: TrackGraph) -> dict[str, Any]:
    return {
        "dataset": after.dataset,
        "n_nodes": after.n_nodes,
        "n_edges_before": before.n_edges,
        "n_edges_after": after.n_edges,
        "n_extra_edges": int(after.n_edges - before.n_edges),
        "n_forks_before": n_predicted_forks(before),
        "n_forks_after": n_predicted_forks(after),
        "n_division_proposals": int(after.metadata.get("n_division_proposals") or 0),
        "n_division_rejected": dict(after.metadata.get("n_division_rejected") or {}),
        "overlay": "learned_unet_orgdet_fnn_div_v1",
        "config_hash": LOCKED_ORGDET_FNN_DIV_HASH,
    }


def refuse_existing_holdout_score(path: str | Path) -> None:
    dest = Path(path)
    if not dest.exists():
        return
    payload = json.loads(dest.read_text())
    if payload.get("status") == "holdout_scored" and payload.get("learned_orgdet_fnn_div_combined") is not None:
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


def tuning_decision(learned_combined: float, n_extra_edges: int) -> str:
    if float(learned_combined) < float(TUNE_KEEP_THRESHOLD):
        return "reject_learned_unet_orgdet_fnn_div_v1"
    if int(n_extra_edges) == 0:
        return "keep_as_noop_learned_unet_orgdet_fnn_div_v1"
    return "keep_candidate_learned_unet_orgdet_fnn_div_v1"


def holdout_decision(learned_combined: float) -> str:
    if float(learned_combined) <= float(ORGDET_FNN_LOCKED_HOLDOUT):
        return "reject_learned_unet_orgdet_fnn_div_v1"
    return "keep_candidate_learned_unet_orgdet_fnn_div_v1"


def runtime_estimate(
    n_volumes: int = 2,
    overlay_sec_per_volume: float = 36.0,
    division_overhead: float = 2.0,
    limit_hours: float = 12.0,
    peak_memory_mb: float = 900.0,
) -> dict[str, Any]:
    n = int(n_volumes)
    sec = float(n) * (float(overlay_sec_per_volume) + float(division_overhead))
    return {
        "n_volumes": n,
        "overlay_sec_per_volume": float(overlay_sec_per_volume),
        "division_overhead": float(division_overhead),
        "estimate_sec": sec,
        "estimate_hours": sec / 3600.0,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_budget": float(peak_memory_mb),
        "source": (
            "CPU locked orgdet+gap+prune+ELF+FNN then locked division_v2 "
            "second-child proposals; no transformer; no ILP; no new UNet"
        ),
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
    }


__all__ = [
    "DIVISION_CONFIG_HASH",
    "HOLDOUT_STEMS",
    "INFER_STEMS",
    "LOCKED_GATES",
    "LOCKED_ORGDET_FNN_DIV_HASH",
    "ORGDET_FNN_LOCKED_HOLDOUT",
    "ORGDET_FNN_LOCKED_TUNE",
    "TUNE_KEEP_THRESHOLD",
    "TUNE_MARGIN",
    "apply_locked_fnn_div",
    "apply_orgdet_fnn_div_overlay",
    "ensure_orgdet_tuning_graph",
    "holdout_decision",
    "load_role_graphs",
    "n_predicted_forks",
    "orgdet_fnn_div_config_hash",
    "orgdet_fnn_div_config_payload",
    "overlay_summary",
    "refuse_existing_holdout_score",
    "refuse_non_holdout_stems",
    "refuse_non_orgdet_node_count",
    "refuse_non_tuning_stems",
    "require_tuning_keep",
    "runtime_estimate",
    "tuning_decision",
]
