"""Locked association_v2 gap-close overlay on locked orgdet graphs.

``learned_unet_orgdet_v1`` is greedy 12 µm last-position with
``max_gap_frames=0``. The same locked gap-close that improved infer
(0.655751 → 0.658452 tune; holdout 0.693605 → 0.701026) is applied
only to leftover unmatched orgdet nodes. Detection, consecutive dt=1
links, and gap gates are not retuned.

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
from .learned_gap import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LOCKED_GAP_HASH,
    LOCKED_GATES as GAP_LOCKED_GATES,
    apply_locked_overlay as apply_gap_overlay,
    load_graph_json,
    overlay_summary as gap_overlay_summary,
)
from .learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from .learned_orgdet import LOCKED_ORGDET_HASH, apply_orgdet_relink
from .learned_orgop import (
    LOCKED_ORGOP_HASH as REJECTED_ORGOP_HASH,
    ORGANIZER_DET_THRESHOLD,
    ORGANIZER_POOL_KERNEL_UM,
)
from .learned_split import HOLDOUT_SPECIMENS, TUNING_SPECIMEN, specimen_of

# Official locked orgdet combined scores. Do not retune orgdet or gap.
ORGDET_LOCKED_TUNE = 0.6795779484295005
ORGDET_LOCKED_HOLDOUT = 0.7052260455024029
SCALE_ZYX = (1.625, 0.40625, 0.40625)
LOCKED_ORGDET_NODE_COUNTS = {
    "44b6_12dfb391": 62480,
    "44b6_267148e4": 28672,
}
LOCKED_ORGDET_HOLDOUT_NODE_COUNTS = {
    "6bba_05db0fb1": 84328,
    "6bba_c328f2fd": 37482,
}

LOCKED_GATES = {
    "gate_um": GAP_LOCKED_GATES["gate_um"],
    "max_gap_frames": GAP_LOCKED_GATES["max_gap_frames"],
    "gap_scale": GAP_LOCKED_GATES["gap_scale"],
    "assignment": GAP_LOCKED_GATES["assignment"],
    "interpolate_gaps": GAP_LOCKED_GATES["interpolate_gaps"],
    "max_parents_per_node": GAP_LOCKED_GATES["max_parents_per_node"],
    "max_children_per_node": GAP_LOCKED_GATES["max_children_per_node"],
    "intensity_weight": GAP_LOCKED_GATES["intensity_weight"],
}


def orgdet_gap_config_payload() -> dict[str, Any]:
    """First orgdet+gap operating point. Do not retune from holdout."""
    return {
        "label": "learned_unet_orgdet_gap_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "orgdet_label": "learned_unet_orgdet_v1",
        "gap_label": "learned_unet_gap_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "orgdet_config_hash": LOCKED_ORGDET_HASH,
        "gap_config_hash": LOCKED_GAP_HASH,
        "source_graphs": "locked_learned_unet_orgdet_graphs",
        "retunes_detection": False,
        "retunes_consecutive_links": False,
        "retunes_gap_gates": False,
        "uses_intensity": False,
        "uses_transformer": False,
        "uses_velocity": False,
        "uses_ilp": False,
        "linker": "orgdet_then_association_v2_gap_overlay",
        "gate_um": LOCKED_GATES["gate_um"],
        "max_gap_frames": LOCKED_GATES["max_gap_frames"],
        "gap_scale": LOCKED_GATES["gap_scale"],
        "assignment": LOCKED_GATES["assignment"],
        "interpolate_gaps": LOCKED_GATES["interpolate_gaps"],
        "max_parents_per_node": LOCKED_GATES["max_parents_per_node"],
        "max_children_per_node": LOCKED_GATES["max_children_per_node"],
        "intensity_weight": LOCKED_GATES["intensity_weight"],
        "det_threshold": ORGANIZER_DET_THRESHOLD,
        "det_tta": True,
        "pool_kernel_um": ORGANIZER_POOL_KERNEL_UM,
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": False,
    }


def orgdet_gap_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or orgdet_gap_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGDET_GAP_HASH = orgdet_gap_config_hash(orgdet_gap_config_payload())
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
            raise ValueError(f"learned_unet_orgdet_gap_v1 tuning refuses stem {stem}")


def refuse_non_holdout_stems(stems: Iterable[str]) -> None:
    for raw in stems:
        stem = _clean_stem(str(raw))
        if specimen_of(stem) in HOLDOUT_SPECIMENS or stem in HOLDOUT_STEMS:
            if stem not in HOLDOUT_STEMS:
                raise ValueError(f"learned_unet_orgdet_gap_v1 holdout refuses extra stem {stem}")
            continue
        raise ValueError(f"learned_unet_orgdet_gap_v1 holdout refuses non-holdout stem {stem}")


def refuse_non_orgdet_node_count(payload: Mapping[str, Any], role: str) -> None:
    stem = _clean_stem(str(payload.get("dataset") or ""))
    n_nodes = len(payload.get("node_id") or [])
    expected = (
        LOCKED_ORGDET_HOLDOUT_NODE_COUNTS if role == "holdout" else LOCKED_ORGDET_NODE_COUNTS
    )
    if stem not in expected:
        raise ValueError(f"learned_unet_orgdet_gap_v1 {role} refuses unknown stem {stem}")
    if int(n_nodes) != int(expected[stem]):
        raise ValueError(
            f"learned_unet_orgdet_gap_v1 refuses {stem}: "
            f"{n_nodes} nodes != locked orgdet {expected[stem]}"
        )


def ensure_orgdet_tuning_graph(graph: TrackGraph) -> TrackGraph:
    """Rebuild locked orgdet edges from orgop detections when the graph is empty."""
    if specimen_of(graph.dataset) != TUNING_SPECIMEN:
        raise ValueError(f"learned_unet_orgdet_gap_v1 tuning refuses stem {graph.dataset}")
    if graph.n_edges == 0:
        return apply_orgdet_relink(graph)
    return graph


def apply_orgdet_gap_overlay(graph: TrackGraph) -> TrackGraph:
    """Apply locked gap-close to leftover unmatched orgdet nodes only."""
    if orgdet_gap_config_hash() != LOCKED_ORGDET_GAP_HASH:
        raise RuntimeError("learned_unet_orgdet_gap_v1 hash drifted; refusing to retune")
    after = apply_gap_overlay(graph)
    after.metadata["overlay"] = "learned_unet_orgdet_gap_v1"
    after.metadata["config_hash"] = LOCKED_ORGDET_GAP_HASH
    after.metadata["parent_orgdet_hash"] = LOCKED_ORGDET_HASH
    after.metadata["parent_gap_hash"] = LOCKED_GAP_HASH
    after.metadata["rejected_orgop_hash"] = REJECTED_ORGOP_HASH
    after.metadata["retunes_detection"] = False
    after.metadata["retunes_consecutive_links"] = False
    after.metadata["retunes_gap_gates"] = False
    after.metadata["uses_intensity"] = False
    after.metadata["uses_transformer"] = False
    after.metadata["uses_ilp"] = False
    return after


def overlay_summary(before: TrackGraph, after: TrackGraph) -> dict[str, Any]:
    summary = gap_overlay_summary(before, after)
    summary["overlay"] = "learned_unet_orgdet_gap_v1"
    summary["config_hash"] = LOCKED_ORGDET_GAP_HASH
    return summary


def load_role_graphs(graph_dir: str | Path, stems: Sequence[str]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for stem in stems:
        path = Path(graph_dir) / f"{stem}.json"
        if not path.exists():
            nested = Path(graph_dir) / "graphs" / f"{stem}.json"
            path = nested if nested.exists() else path
        if not path.exists():
            raise FileNotFoundError(f"missing orgdet/orgop graph {path}; do not invent a score")
        payload = load_graph_json(path)
        if str(payload.get("dataset")) != stem:
            raise ValueError(f"{path} dataset {payload.get('dataset')!r} != {stem}")
        payloads.append(payload)
    return payloads


def refuse_existing_holdout_score(path: str | Path) -> None:
    dest = Path(path)
    if not dest.exists():
        return
    payload = json.loads(dest.read_text())
    if payload.get("status") == "holdout_scored" and payload.get("learned_orgdet_gap_combined") is not None:
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
    if float(learned_combined) <= ORGDET_LOCKED_TUNE:
        return "reject_learned_unet_orgdet_gap_v1"
    if int(n_extra_edges) == 0:
        return "keep_as_noop_learned_unet_orgdet_gap_v1"
    return "keep_candidate_learned_unet_orgdet_gap_v1"


def holdout_decision(learned_combined: float) -> str:
    if float(learned_combined) <= ORGDET_LOCKED_HOLDOUT:
        return "reject_learned_unet_orgdet_gap_v1"
    return "keep_candidate_learned_unet_orgdet_gap_v1"


def runtime_estimate(
    n_volumes: int = 2,
    limit_hours: float = 12.0,
    peak_memory_mb: float = 1300.0,
) -> dict[str, Any]:
    """CPU gap overlay on locked orgdet graphs. No new UNet pass."""
    n = int(n_volumes)
    sec = 15.0 * n
    return {
        "n_volumes": n,
        "estimate_sec": sec,
        "estimate_hours": sec / 3600.0,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_budget": float(peak_memory_mb),
        "source": (
            "CPU association_v2 gap-close on locked orgdet detections; "
            "no transformer; no ILP; no new UNet pass"
        ),
    }
