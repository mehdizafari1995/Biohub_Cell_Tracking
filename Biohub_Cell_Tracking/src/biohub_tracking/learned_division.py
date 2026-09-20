"""Conservative division overlay on locked learned_unet graphs.

Applies locked ``division_v2`` gates to locked ``learned_unet_infer_v1`` /
``learned_unet_holdout_v1`` graphs. Does not retune detection, linking, or
division thresholds. Learned graphs have no intensity, so the intensity-sum
gate is unused. Not a competition submission. A local official score is not
a leaderboard number. Do not overwrite locked train/infer/holdout/test slugs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import LinkingConfig
from .division import propose_divisions
from .graphs import TrackGraph
from .learned_holdout import LOCKED_HOLDOUT_HASH
from .learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from .learned_split import (
    FROZEN_HOLDOUT_STEMS,
    HOLDOUT_SPECIMENS,
    INTERNAL_VAL_STEMS,
    TUNING_SPECIMEN,
    specimen_of,
)

DIVISION_CONFIG_HASH = "0fa756963e5d42d1"
LEARNED_LOCKED_TUNE = 0.6557507105877639
LEARNED_LOCKED_HOLDOUT = 0.6936047711123506
SCALE_ZYX = (1.625, 0.40625, 0.40625)
INFER_STEMS = INTERNAL_VAL_STEMS
HOLDOUT_STEMS = FROZEN_HOLDOUT_STEMS

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


def division_config_payload() -> dict[str, Any]:
    """Locked overlay operating point. Do not retune from holdout."""
    return {
        "label": "learned_unet_division_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "holdout_label": "learned_unet_holdout_v1",
        "division_label": "division_v2",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "holdout_config_hash": LOCKED_HOLDOUT_HASH,
        "division_config_hash": DIVISION_CONFIG_HASH,
        "source_graphs": "locked_learned_unet_graphs",
        "retunes_detection": False,
        "retunes_division_gates": False,
        "uses_intensity": False,
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
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": False,
    }


def division_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or division_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_DIVISION_HASH = division_config_hash(division_config_payload())
assert LOCKED_DIVISION_HASH == "8761fb66cc2254f3"


def locked_linking_config() -> LinkingConfig:
    return LinkingConfig(
        method="learned_unet_division_v1",
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
            raise ValueError(f"learned_unet_division_v1 tuning refuses stem {stem}")


def refuse_non_holdout_stems(stems: Iterable[str]) -> None:
    for raw in stems:
        stem = _clean_stem(str(raw))
        if specimen_of(stem) in HOLDOUT_SPECIMENS or stem in HOLDOUT_STEMS:
            if stem not in HOLDOUT_STEMS:
                raise ValueError(f"learned_unet_division_v1 holdout refuses extra stem {stem}")
            continue
        raise ValueError(f"learned_unet_division_v1 holdout refuses non-holdout stem {stem}")


def graph_from_payload(payload: Mapping[str, Any]) -> TrackGraph:
    edges = payload.get("edges") or []
    return TrackGraph(
        dataset=str(payload["dataset"]),
        node_id=np.asarray(payload.get("node_id") or [], dtype=np.int64),
        t=np.asarray(payload.get("t") or [], dtype=np.int64),
        z=np.asarray(payload.get("z") or [], dtype=np.int64),
        y=np.asarray(payload.get("y") or [], dtype=np.int64),
        x=np.asarray(payload.get("x") or [], dtype=np.int64),
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2) if edges else np.zeros((0, 2), dtype=np.int64),
    )


def load_graph_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def apply_locked_overlay(graph: TrackGraph) -> TrackGraph:
    """Add at most one extra child per parent when locked gates pass.

    Intensity is omitted: locked learned graphs do not store per-node intensity.
    """
    if division_config_hash() != LOCKED_DIVISION_HASH:
        raise RuntimeError("learned_unet_division_v1 hash drifted; refusing to retune")
    out = propose_divisions(graph, None, locked_linking_config(), SCALE_ZYX)
    out.metadata = dict(out.metadata)
    out.metadata["overlay"] = "learned_unet_division_v1"
    out.metadata["config_hash"] = LOCKED_DIVISION_HASH
    out.metadata["parent_infer_hash"] = LOCKED_INFER_HASH
    out.metadata["uses_intensity"] = False
    return out


def n_predicted_forks(graph: TrackGraph) -> int:
    return int(sum(1 for children in graph.out_edges().values() if len(children) >= 2))


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
    }


def load_role_graphs(graph_dir: str | Path, stems: Sequence[str]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for stem in stems:
        path = Path(graph_dir) / f"{stem}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing learned graph {path}; do not invent a score")
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
    if payload.get("status") == "holdout_scored" and payload.get("learned_division_combined") is not None:
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
    if float(learned_combined) < LEARNED_LOCKED_TUNE:
        return "reject_learned_unet_division_v1"
    if int(n_extra_edges) == 0:
        return "keep_as_noop_learned_unet_division_v1"
    return "keep_candidate_learned_unet_division_v1"


def holdout_decision(learned_combined: float) -> str:
    if float(learned_combined) < LEARNED_LOCKED_HOLDOUT:
        return "reject_learned_unet_division_v1"
    return "keep_candidate_learned_unet_division_v1"
