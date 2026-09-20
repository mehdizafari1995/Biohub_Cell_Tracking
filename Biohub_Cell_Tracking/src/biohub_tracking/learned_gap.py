"""Conservative gap-close overlay on locked learned_unet graphs.

Applies locked ``association_v2`` gap-close gates (12 µm, max_gap_frames=2,
interpolate) only to leftover unmatched nodes. Existing dt=1 1-1 edges from
locked ``learned_unet_infer_v1`` / ``learned_unet_holdout_v1`` graphs are
kept. Detection is not retuned. Learned graphs have no intensity, so the
association intensity term is unused. Not a competition submission. A local
official score is not a leaderboard number. Do not overwrite locked
train/infer/holdout/test slugs.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

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

LEARNED_LOCKED_TUNE = 0.6557507105877639
LEARNED_LOCKED_HOLDOUT = 0.6936047711123506
SCALE_ZYX = (1.625, 0.40625, 0.40625)
INFER_STEMS = INTERNAL_VAL_STEMS
HOLDOUT_STEMS = FROZEN_HOLDOUT_STEMS

# Locked association_v2 gap-close gates. Do not retune.
LOCKED_GATES = {
    "gate_um": 12.0,
    "max_gap_frames": 2,
    "gap_scale": "sqrt_dt",
    "assignment": "greedy",
    "interpolate_gaps": True,
    "max_parents_per_node": 1,
    "max_children_per_node": 1,
    "intensity_weight": 0.0,
}


def gap_config_payload() -> dict[str, Any]:
    """Locked overlay operating point. Do not retune from holdout."""
    return {
        "label": "learned_unet_gap_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "holdout_label": "learned_unet_holdout_v1",
        "association_label": "association_v2",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "holdout_config_hash": LOCKED_HOLDOUT_HASH,
        "source_graphs": "locked_learned_unet_graphs",
        "retunes_detection": False,
        "retunes_consecutive_links": False,
        "uses_intensity": False,
        "linker": "association_v2_gap_overlay",
        "gate_um": LOCKED_GATES["gate_um"],
        "max_gap_frames": LOCKED_GATES["max_gap_frames"],
        "gap_scale": LOCKED_GATES["gap_scale"],
        "assignment": LOCKED_GATES["assignment"],
        "interpolate_gaps": LOCKED_GATES["interpolate_gaps"],
        "max_parents_per_node": LOCKED_GATES["max_parents_per_node"],
        "max_children_per_node": LOCKED_GATES["max_children_per_node"],
        "intensity_weight": LOCKED_GATES["intensity_weight"],
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": False,
    }


def gap_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or gap_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_GAP_HASH = gap_config_hash(gap_config_payload())
assert LOCKED_GAP_HASH == "f4d888d190c45a4a"


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
            raise ValueError(f"learned_unet_gap_v1 tuning refuses stem {stem}")


def refuse_non_holdout_stems(stems: Iterable[str]) -> None:
    for raw in stems:
        stem = _clean_stem(str(raw))
        if specimen_of(stem) in HOLDOUT_SPECIMENS or stem in HOLDOUT_STEMS:
            if stem not in HOLDOUT_STEMS:
                raise ValueError(f"learned_unet_gap_v1 holdout refuses extra stem {stem}")
            continue
        raise ValueError(f"learned_unet_gap_v1 holdout refuses non-holdout stem {stem}")


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


def _physical(z: float, y: float, x: float) -> np.ndarray:
    return np.asarray([z, y, x], dtype=np.float64) * np.asarray(SCALE_ZYX, dtype=np.float64)


def apply_locked_overlay(graph: TrackGraph) -> TrackGraph:
    """Add interpolated gap-close links for unmatched nodes only.

    Existing outgoing/incoming edges stay. Intensity is omitted: locked
    learned graphs do not store per-node intensity.
    """
    if gap_config_hash() != LOCKED_GAP_HASH:
        raise RuntimeError("learned_unet_gap_v1 hash drifted; refusing to retune")

    nids = [int(x) for x in graph.node_id]
    ts = [int(x) for x in graph.t]
    zs = [int(x) for x in graph.z]
    ys = [int(x) for x in graph.y]
    xs = [int(x) for x in graph.x]
    edges = [(int(s), int(d)) for s, d in np.asarray(graph.edges).reshape(-1, 2)] if graph.n_edges else []

    idx = {nid: i for i, nid in enumerate(nids)}
    has_out = {int(src) for src, _dst in edges}
    has_in = {int(dst) for _src, dst in edges}
    by_t: dict[int, list[int]] = defaultdict(list)
    for nid in nids:
        by_t[ts[idx[nid]]].append(nid)

    gate = float(LOCKED_GATES["gate_um"])
    max_dt = int(LOCKED_GATES["max_gap_frames"]) + 1
    candidates: list[tuple[float, int, int, int]] = []
    for parent in nids:
        if parent in has_out:
            continue
        t0 = ts[idx[parent]]
        p0 = _physical(zs[idx[parent]], ys[idx[parent]], xs[idx[parent]])
        for dt in range(2, max_dt + 1):
            for child in by_t.get(t0 + dt, []):
                if child in has_in:
                    continue
                p1 = _physical(zs[idx[child]], ys[idx[child]], xs[idx[child]])
                dist = float(np.linalg.norm(p0 - p1))
                if dist <= gate * float(dt) ** 0.5:
                    candidates.append((dist, parent, child, dt))

    candidates.sort()
    used_parent: set[int] = set()
    used_child: set[int] = set()
    accepted: list[tuple[float, int, int, int]] = []
    for dist, parent, child, dt in candidates:
        if parent in used_parent or parent in has_out:
            continue
        if child in used_child or child in has_in:
            continue
        used_parent.add(parent)
        used_child.add(child)
        accepted.append((dist, parent, child, dt))

    next_id = (max(nids) + 1) if nids else 1
    n_interp = 0
    for _dist, parent, child, dt in accepted:
        i0 = idx[parent]
        i1 = idx[child]
        p0 = np.asarray([zs[i0], ys[i0], xs[i0]], dtype=np.float64)
        p1 = np.asarray([zs[i1], ys[i1], xs[i1]], dtype=np.float64)
        prev = parent
        t0 = ts[i0]
        if LOCKED_GATES["interpolate_gaps"]:
            for k in range(1, dt):
                alpha = k / dt
                pos = (1.0 - alpha) * p0 + alpha * p1
                nid = next_id
                next_id += 1
                nids.append(nid)
                ts.append(t0 + k)
                zs.append(int(round(float(pos[0]))))
                ys.append(int(round(float(pos[1]))))
                xs.append(int(round(float(pos[2]))))
                edges.append((prev, nid))
                prev = nid
                n_interp += 1
        edges.append((prev, child))

    out = TrackGraph(
        dataset=graph.dataset,
        node_id=np.asarray(nids, dtype=np.int64),
        t=np.asarray(ts, dtype=np.int64),
        z=np.asarray(zs, dtype=np.int64),
        y=np.asarray(ys, dtype=np.int64),
        x=np.asarray(xs, dtype=np.int64),
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2) if edges else np.zeros((0, 2), dtype=np.int64),
        estimated_n_nodes=graph.estimated_n_nodes,
    )
    out.metadata = dict(graph.metadata)
    out.metadata["overlay"] = "learned_unet_gap_v1"
    out.metadata["config_hash"] = LOCKED_GAP_HASH
    out.metadata["parent_infer_hash"] = LOCKED_INFER_HASH
    out.metadata["uses_intensity"] = False
    out.metadata["n_gap_links"] = len(accepted)
    out.metadata["n_interpolated"] = n_interp
    out.metadata["retunes_consecutive_links"] = False
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
        "n_gap_links": int(after.metadata.get("n_gap_links") or 0),
        "n_interpolated": int(after.metadata.get("n_interpolated") or 0),
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
    if payload.get("status") == "holdout_scored" and payload.get("learned_gap_combined") is not None:
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
        return "reject_learned_unet_gap_v1"
    if int(n_extra_edges) == 0:
        return "keep_as_noop_learned_unet_gap_v1"
    return "keep_candidate_learned_unet_gap_v1"


def holdout_decision(learned_combined: float) -> str:
    if float(learned_combined) < LEARNED_LOCKED_HOLDOUT:
        return "reject_learned_unet_gap_v1"
    return "keep_candidate_learned_unet_gap_v1"
