"""Greedy 12 µm relink of locked organizer detections.

``learned_unet_orgop_v1`` failed because transformer softmax>0.5 wrote
zero candidate edges (official 44b6 0.0 vs infer 0.655751). This overlay
keeps those locked 0.99 / pool-3.0 / TTA detections and applies the
locked infer greedy 12 µm last-position linker. It is not another
assignment on infer detections and not a retune of orgop ILP.

Never opens ``6bba`` during tuning. Does not write ``submission.csv``.
Not a competition submission. Do not overwrite locked slugs.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .graphs import TrackGraph
from .learned_infer import (  # noqa: F401
    ASSOC_LOCKED_TUNE,
    INFER_KERNEL_SLUG,
    INFER_STEMS,
    LOCKED_INFER_HASH,
    TRAIN_CONFIG_HASH,
    TRAIN_KERNEL_SLUG,
    FROZEN_HOLDOUT_STEMS,
    HOLDOUT_SPECIMENS,
    SCALE_ZYX,
    TUNING_SPECIMEN,
    HoldoutLeakError,
    assert_no_submission_csv,
    assert_tuning_only,
    graph_from_payload,
    graph_payload,
    greedy_gate_links,
    load_graph_json,
    refuse_non_infer_stems,
    write_graph_json,
)
from .learned_orgop import (  # noqa: F401
    LOCKED_ORGOP_HASH as REJECTED_ORGOP_HASH,
    ORGANIZER_DET_THRESHOLD,
    ORGANIZER_POOL_KERNEL_UM,
    ORGOP_KERNEL_SLUG,
)
from .learned_split import specimen_of

ORGDET_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-v1"
INFER_LOCKED_TUNE = 0.6557507105877639
MAX_LINK_UM = 12.0

LOCKED_GATES = {
    "motion_model": "last_position",
    "gate_um": MAX_LINK_UM,
    "max_link_um": MAX_LINK_UM,
    "max_gap_frames": 0,
    "assignment": "greedy",
    "interpolate_gaps": False,
    "max_parents_per_node": 1,
    "max_children_per_node": 1,
    "intensity_weight": 0.0,
}


def orgdet_config_payload() -> dict[str, Any]:
    """First greedy-on-orgop-detections operating point. Do not retune from holdout."""
    return {
        "label": "learned_unet_orgdet_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "orgop_label": "learned_unet_orgop_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "rejected_orgop_hash": REJECTED_ORGOP_HASH,
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "infer_stems": list(INFER_STEMS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "source_graphs": "locked_learned_unet_orgop_graphs",
        "retunes_detection": False,
        "uses_intensity": False,
        "uses_transformer": False,
        "uses_velocity": False,
        "uses_ilp": False,
        "linker": "greedy_gate_12um",
        "motion_model": LOCKED_GATES["motion_model"],
        "gate_um": LOCKED_GATES["gate_um"],
        "max_link_um": LOCKED_GATES["max_link_um"],
        "max_gap_frames": LOCKED_GATES["max_gap_frames"],
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
        "kernel_slug": ORGDET_KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
        "orgop_kernel_slug": ORGOP_KERNEL_SLUG,
    }


def orgdet_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or orgdet_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGDET_HASH = orgdet_config_hash()
assert LOCKED_ORGDET_HASH == "80b20589198148b7"


def apply_orgdet_relink_arrays(
    node_id: Sequence[int],
    t: Sequence[int],
    z: Sequence[int],
    y: Sequence[int],
    x: Sequence[int],
) -> list[tuple[int, int]]:
    """Locked infer-style greedy last-position 1-1 on consecutive frames."""
    nids = [int(v) for v in node_id]
    ts = [int(v) for v in t]
    zs = [int(v) for v in z]
    ys = [int(v) for v in y]
    xs = [int(v) for v in x]
    by_t: dict[int, list[int]] = defaultdict(list)
    for i, tk in enumerate(ts):
        by_t[tk].append(i)
    times = sorted(by_t)
    edges: list[tuple[int, int]] = []
    for prev, cur in zip(times, times[1:]):
        if cur - prev != 1:
            continue
        idx_a = by_t[prev]
        idx_b = by_t[cur]
        coords_a = np.stack(
            [
                np.asarray([zs[i] for i in idx_a], dtype=np.float64),
                np.asarray([ys[i] for i in idx_a], dtype=np.float64),
                np.asarray([xs[i] for i in idx_a], dtype=np.float64),
            ],
            axis=1,
        )
        coords_b = np.stack(
            [
                np.asarray([zs[i] for i in idx_b], dtype=np.float64),
                np.asarray([ys[i] for i in idx_b], dtype=np.float64),
                np.asarray([xs[i] for i in idx_b], dtype=np.float64),
            ],
            axis=1,
        )
        for i, j, _dist in greedy_gate_links(coords_a, coords_b, SCALE_ZYX, MAX_LINK_UM):
            edges.append((int(nids[idx_a[i]]), int(nids[idx_b[j]])))
    return edges


def apply_orgdet_relink(graph: TrackGraph) -> TrackGraph:
    """Replace edges with greedy 12 µm last-position 1-1. Keep orgop detections."""
    if orgdet_config_hash() != LOCKED_ORGDET_HASH:
        raise RuntimeError("learned_unet_orgdet_v1 hash drifted; refusing to retune")
    if specimen_of(graph.dataset) != TUNING_SPECIMEN:
        raise HoldoutLeakError(
            f"learned_unet_orgdet_v1 tuning refuses stem {graph.dataset}"
        )
    refuse_non_infer_stems([graph.dataset])
    edges = apply_orgdet_relink_arrays(graph.node_id, graph.t, graph.z, graph.y, graph.x)
    out = TrackGraph(
        dataset=graph.dataset,
        node_id=np.asarray(graph.node_id, dtype=np.int64),
        t=np.asarray(graph.t, dtype=np.int64),
        z=np.asarray(graph.z, dtype=np.int64),
        y=np.asarray(graph.y, dtype=np.int64),
        x=np.asarray(graph.x, dtype=np.int64),
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2) if edges else np.zeros((0, 2), dtype=np.int64),
        estimated_n_nodes=graph.estimated_n_nodes,
    )
    if out.n_nodes != graph.n_nodes:
        raise RuntimeError("orgdet relink changed detection count; refusing")
    before_ids = set(int(x) for x in graph.node_id)
    after_ids = set(int(x) for x in out.node_id)
    if before_ids != after_ids:
        raise RuntimeError("orgdet relink changed node identities; refusing")
    children = out.out_edges()
    parents = out.in_edges()
    if any(len(v) > 1 for v in children.values()) or any(len(v) > 1 for v in parents.values()):
        raise RuntimeError("orgdet relink is not one-to-one; refusing")
    out.metadata = dict(graph.metadata)
    out.metadata["overlay"] = "learned_unet_orgdet_v1"
    out.metadata["config_hash"] = LOCKED_ORGDET_HASH
    out.metadata["parent_infer_hash"] = LOCKED_INFER_HASH
    out.metadata["rejected_orgop_hash"] = REJECTED_ORGOP_HASH
    out.metadata["uses_intensity"] = False
    out.metadata["uses_transformer"] = False
    out.metadata["uses_velocity"] = False
    out.metadata["uses_ilp"] = False
    out.metadata["retunes_detection"] = False
    out.metadata["n_greedy_links"] = len(edges)
    out.metadata["n_orgop_edges"] = graph.n_edges
    return out


def apply_orgdet_relink_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    graph = graph_from_payload(payload)
    out = apply_orgdet_relink(graph)
    return {
        "dataset": out.dataset,
        "node_id": [int(x) for x in out.node_id],
        "t": [int(x) for x in out.t],
        "z": [int(x) for x in out.z],
        "y": [int(x) for x in out.y],
        "x": [int(x) for x in out.x],
        "edges": [[int(s), int(d)] for s, d in np.asarray(out.edges).reshape(-1, 2)]
        if out.n_edges
        else [],
        "metadata": dict(out.metadata),
    }


def relink_summary(before: TrackGraph, after: TrackGraph) -> dict[str, Any]:
    before_edges = {tuple(map(int, row)) for row in np.asarray(before.edges).reshape(-1, 2)} if before.n_edges else set()
    after_edges = {tuple(map(int, row)) for row in np.asarray(after.edges).reshape(-1, 2)} if after.n_edges else set()
    return {
        "dataset": after.dataset,
        "n_nodes_before": before.n_nodes,
        "n_nodes_after": after.n_nodes,
        "n_edges_before": before.n_edges,
        "n_edges_after": after.n_edges,
        "n_shared_edges": len(before_edges & after_edges),
        "n_replaced_edges": len(before_edges - after_edges),
        "n_new_edges": len(after_edges - before_edges),
        "n_greedy_links": int(after.metadata.get("n_greedy_links") or 0),
        "detection_counts_match": before.n_nodes == after.n_nodes,
    }


def edge_geometry(graph: TrackGraph, scale_zyx: Sequence[float] = SCALE_ZYX) -> dict[str, Any]:
    if graph.n_edges == 0:
        return {
            "dataset": graph.dataset,
            "dt_unique": [],
            "max_out_degree": 0,
            "max_in_degree": 0,
            "n_div_pred": 0,
            "edge_um_mean": 0.0,
            "edge_um_median": 0.0,
            "edge_um_p95": 0.0,
            "edge_um_max": 0.0,
            "n_over_12um": 0,
        }
    idx = graph.id_to_index()
    dts: list[int] = []
    dists: list[float] = []
    scale = np.asarray(scale_zyx, dtype=np.float64)
    for src, dst in np.asarray(graph.edges).reshape(-1, 2):
        i = idx[int(src)]
        j = idx[int(dst)]
        dts.append(int(graph.t[j] - graph.t[i]))
        p0 = np.asarray([graph.z[i], graph.y[i], graph.x[i]], dtype=np.float64) * scale
        p1 = np.asarray([graph.z[j], graph.y[j], graph.x[j]], dtype=np.float64) * scale
        dists.append(float(np.linalg.norm(p1 - p0)))
    out_deg = [len(v) for v in graph.out_edges().values()]
    in_deg = [len(v) for v in graph.in_edges().values()]
    arr = np.asarray(dists, dtype=np.float64)
    return {
        "dataset": graph.dataset,
        "dt_unique": sorted(set(dts)),
        "max_out_degree": max(out_deg) if out_deg else 0,
        "max_in_degree": max(in_deg) if in_deg else 0,
        "n_div_pred": sum(1 for d in out_deg if d >= 2),
        "edge_um_mean": float(arr.mean()),
        "edge_um_median": float(np.median(arr)),
        "edge_um_p95": float(np.quantile(arr, 0.95)),
        "edge_um_max": float(arr.max()),
        "n_over_12um": int(np.sum(arr > MAX_LINK_UM + 1e-9)),
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
    if payload.get("status") == "holdout_scored" and payload.get("learned_combined") is not None:
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


def tuning_decision(learned_combined: float) -> str:
    if float(learned_combined) > INFER_LOCKED_TUNE:
        return "keep_candidate_learned_unet_orgdet_v1"
    return "reject_learned_unet_orgdet_v1"


def runtime_estimate(
    n_volumes: int = 2,
    infer_sec_per_volume: float = 76.9,
    linker_overhead: float = 0.05,
    limit_hours: float = 12.0,
    peak_memory_mb: float = 1300.0,
) -> dict[str, Any]:
    """CPU greedy relink of locked orgop detections. No new UNet pass required."""
    n = int(n_volumes)
    relink_sec = 10.0 * n
    infer_sec = float(n) * float(infer_sec_per_volume) * float(linker_overhead)
    sec = relink_sec + infer_sec
    return {
        "n_volumes": n,
        "infer_sec_per_volume": float(infer_sec_per_volume),
        "linker_overhead": float(linker_overhead),
        "relink_sec": relink_sec,
        "estimate_sec": sec,
        "estimate_hours": sec / 3600.0,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_budget": float(peak_memory_mb),
        "source": (
            "CPU greedy 12 µm last-position relink of locked orgop detections; "
            "no transformer; no ILP; no new UNet pass required"
        ),
    }


def find_orgop_graphs(roots: Iterable[str | Path] | None = None) -> dict[str, Path]:
    """Locate locked 44b6 orgop JSON graphs. Skip Zarr/GEFF trees."""
    if roots is None:
        roots = [
            Path("/kaggle/input/notebooks"),
            Path("/kaggle/input"),
            Path("artifacts/kaggle/learned_unet_orgop_v1"),
        ]
    found: dict[str, Path] = {}
    for raw in roots:
        root = Path(raw)
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if any(part.endswith((".zarr", ".geff")) for part in path.parts):
                continue
            stem = path.name[: -len(".json")] if path.name.endswith(".json") else path.name
            if stem in INFER_STEMS and stem not in found:
                found[stem] = path
            if len(found) == len(INFER_STEMS):
                return found
    return found
