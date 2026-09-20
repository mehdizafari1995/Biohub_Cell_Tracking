"""Constant-velocity relink of locked learned_unet detections.

Keeps locked ``learned_unet_infer_v1`` nodes and replaces consecutive
edges with greedy 1-1 assignment to a constant-velocity prediction,
plus the locked association_v2 12 µm spatial gate. Does not use the
rejected transformer linker. Never opens ``6bba`` during tuning.
Does not write ``submission.csv``. Not a competition submission.
Do not overwrite locked train/infer/holdout/test/gap-test/link slugs.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .graphs import TrackGraph
from .learned_infer import (  # noqa: F401 — re-export tuning guards / graph I/O
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
    load_graph_json,
    refuse_non_infer_stems,
    write_graph_json,
)
from .learned_link import LOCKED_LINK_HASH as REJECTED_LINK_V1_HASH
from .learned_link_v2 import LOCKED_LINK_V2_HASH as REJECTED_LINK_V2_HASH
from .learned_link_v2 import REJECTED_LINK_V1_SLUG
from .learned_split import specimen_of

MOTION_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-motion-v1"
REJECTED_LINK_V2_SLUG = "mahdimzafari/biohub-learned-unet-link-v2"
INFER_LOCKED_TUNE = 0.6557507105877639
MAX_LINK_UM = 12.0
VOLUME_SHAPE = (64, 256, 256)

LOCKED_GATES = {
    "motion_model": "constant_velocity",
    "gate_um": MAX_LINK_UM,
    "max_link_um": MAX_LINK_UM,
    "max_gap_frames": 0,
    "assignment": "greedy",
    "interpolate_gaps": False,
    "max_parents_per_node": 1,
    "max_children_per_node": 1,
    "intensity_weight": 0.0,
}


def motion_config_payload() -> dict[str, Any]:
    """First motion-aware operating point. Do not retune from holdout."""
    return {
        "label": "learned_unet_motion_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "rejected_link_v1_hash": REJECTED_LINK_V1_HASH,
        "rejected_link_v2_hash": REJECTED_LINK_V2_HASH,
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "infer_stems": list(INFER_STEMS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "source_graphs": "locked_learned_unet_infer_graphs",
        "retunes_detection": False,
        "uses_intensity": False,
        "uses_transformer": False,
        "linker": "constant_velocity_greedy_1to1",
        "motion_model": LOCKED_GATES["motion_model"],
        "gate_um": LOCKED_GATES["gate_um"],
        "max_link_um": LOCKED_GATES["max_link_um"],
        "max_gap_frames": LOCKED_GATES["max_gap_frames"],
        "assignment": LOCKED_GATES["assignment"],
        "interpolate_gaps": LOCKED_GATES["interpolate_gaps"],
        "max_parents_per_node": LOCKED_GATES["max_parents_per_node"],
        "max_children_per_node": LOCKED_GATES["max_children_per_node"],
        "intensity_weight": LOCKED_GATES["intensity_weight"],
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": False,
        "kernel_slug": MOTION_KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
        "rejected_link_v1_slug": REJECTED_LINK_V1_SLUG,
        "rejected_link_v2_slug": REJECTED_LINK_V2_SLUG,
    }


def motion_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or motion_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_MOTION_HASH = motion_config_hash()
assert LOCKED_MOTION_HASH == "66853b6f1445c18b"


def predicted_voxel(positions: Sequence[np.ndarray], times: Sequence[int], t: int) -> np.ndarray:
    """Constant-velocity forecast. Falls back to last position when history is short."""
    pos = np.asarray(positions[-1], dtype=np.float64)
    if len(positions) < 2:
        return pos
    dt_hist = float(times[-1] - times[-2])
    if dt_hist <= 0:
        return pos
    vel = (np.asarray(positions[-1], dtype=np.float64) - np.asarray(positions[-2], dtype=np.float64)) / dt_hist
    return pos + vel * float(t - times[-1])


def physical(coords: np.ndarray, scale_zyx: Sequence[float] = SCALE_ZYX) -> np.ndarray:
    return np.asarray(coords, dtype=np.float64).reshape(-1, 3) * np.asarray(scale_zyx, dtype=np.float64)


def greedy_assign(cost: np.ndarray, allowed: np.ndarray) -> list[tuple[int, int]]:
    """Greedy 1-1 on finite allowed costs. Lower cost wins."""
    if cost.size == 0:
        return []
    mask = np.asarray(allowed, dtype=bool)
    finite = np.argwhere(mask)
    if finite.size == 0:
        return []
    order = np.argsort(cost[finite[:, 0], finite[:, 1]], kind="stable")
    used_i: set[int] = set()
    used_j: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for k in order:
        i, j = int(finite[k, 0]), int(finite[k, 1])
        if i in used_i or j in used_j:
            continue
        pairs.append((i, j))
        used_i.add(i)
        used_j.add(j)
    return pairs


def assign_motion_links(
    last_zyx: np.ndarray,
    pred_zyx: np.ndarray,
    det_zyx: np.ndarray,
    max_link_um: float = MAX_LINK_UM,
    scale_zyx: Sequence[float] = SCALE_ZYX,
) -> list[tuple[int, int]]:
    """Motion ranking inside the locked last-position 12 µm gate.

    Candidates farther than ``max_link_um`` from the last observed centroid
    are refused. Remaining pairs are assigned 1-1 by distance to the
    constant-velocity prediction.
    """
    last = np.asarray(last_zyx, dtype=np.float64).reshape(-1, 3)
    pred = np.asarray(pred_zyx, dtype=np.float64).reshape(-1, 3)
    det = np.asarray(det_zyx, dtype=np.float64).reshape(-1, 3)
    if last.size == 0 or det.size == 0:
        return []
    last_dist = np.linalg.norm(physical(last, scale_zyx)[:, None, :] - physical(det, scale_zyx)[None, :, :], axis=2)
    pred_dist = np.linalg.norm(physical(pred, scale_zyx)[:, None, :] - physical(det, scale_zyx)[None, :, :], axis=2)
    allowed = last_dist <= float(max_link_um)
    return greedy_assign(pred_dist, allowed)


def apply_motion_relink_arrays(
    node_id: Sequence[int],
    t: Sequence[int],
    z: Sequence[int],
    y: Sequence[int],
    x: Sequence[int],
) -> list[tuple[int, int]]:
    """Standalone constant-velocity 1-1 relink. Returns (src, dst) node ids."""
    nids = [int(v) for v in node_id]
    ts = [int(v) for v in t]
    zs = [int(v) for v in z]
    ys = [int(v) for v in y]
    xs = [int(v) for v in x]
    by_t: dict[int, list[int]] = defaultdict(list)
    for i, tk in enumerate(ts):
        by_t[tk].append(i)
    tracks: list[dict[str, Any]] = []
    edges: list[tuple[int, int]] = []
    for tk in sorted(by_t):
        idxs = by_t[tk]
        coords = np.stack(
            [
                np.asarray([zs[i] for i in idxs], dtype=np.float64),
                np.asarray([ys[i] for i in idxs], dtype=np.float64),
                np.asarray([xs[i] for i in idxs], dtype=np.float64),
            ],
            axis=1,
        )
        live = [tr for tr in tracks if tk - tr["times"][-1] == 1]
        used_dets: set[int] = set()
        if live and len(coords):
            last = np.stack([tr["positions"][-1] for tr in live])
            pred = np.stack([predicted_voxel(tr["positions"], tr["times"], tk) for tr in live])
            for li, j in assign_motion_links(last, pred, coords):
                tr = live[li]
                edges.append((int(tr["node_ids"][-1]), int(nids[idxs[j]])))
                tr["node_ids"].append(int(nids[idxs[j]]))
                tr["times"].append(tk)
                tr["positions"].append(coords[j])
                used_dets.add(j)
        for j, i in enumerate(idxs):
            if j in used_dets:
                continue
            tracks.append(
                {
                    "node_ids": [int(nids[i])],
                    "times": [tk],
                    "positions": [coords[j]],
                }
            )
    return edges


def apply_motion_relink(graph: TrackGraph) -> TrackGraph:
    """Replace edges with constant-velocity 1-1 links. Keep detections."""
    if motion_config_hash() != LOCKED_MOTION_HASH:
        raise RuntimeError("learned_unet_motion_v1 hash drifted; refusing to retune")
    if specimen_of(graph.dataset) != TUNING_SPECIMEN:
        raise HoldoutLeakError(
            f"learned_unet_motion_v1 tuning refuses stem {graph.dataset}"
        )
    refuse_non_infer_stems([graph.dataset])
    edges = apply_motion_relink_arrays(graph.node_id, graph.t, graph.z, graph.y, graph.x)
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
        raise RuntimeError("motion relink changed detection count; refusing")
    before_ids = set(int(x) for x in graph.node_id)
    after_ids = set(int(x) for x in out.node_id)
    if before_ids != after_ids:
        raise RuntimeError("motion relink changed node identities; refusing")
    n_velocity, n_identity = _link_kinds(out)
    out.metadata = dict(graph.metadata)
    out.metadata["overlay"] = "learned_unet_motion_v1"
    out.metadata["config_hash"] = LOCKED_MOTION_HASH
    out.metadata["parent_infer_hash"] = LOCKED_INFER_HASH
    out.metadata["uses_intensity"] = False
    out.metadata["uses_transformer"] = False
    out.metadata["retunes_detection"] = False
    out.metadata["n_velocity_links"] = n_velocity
    out.metadata["n_identity_pred_links"] = n_identity
    return out


def apply_motion_relink_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    graph = graph_from_payload(payload)
    out = apply_motion_relink(graph)
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


def _link_kinds(graph: TrackGraph) -> tuple[int, int]:
    incoming = graph.in_edges()
    n_velocity = 0
    n_identity = 0
    for src, _dst in np.asarray(graph.edges).reshape(-1, 2) if graph.n_edges else []:
        if incoming.get(int(src)):
            n_velocity += 1
        else:
            n_identity += 1
    return n_velocity, n_identity


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
        "n_velocity_links": int(after.metadata.get("n_velocity_links") or 0),
        "n_identity_pred_links": int(after.metadata.get("n_identity_pred_links") or 0),
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
    if float(learned_combined) < INFER_LOCKED_TUNE:
        return "reject_learned_unet_motion_v1"
    return "keep_candidate_learned_unet_motion_v1"


def runtime_estimate(
    n_volumes: int = 2,
    infer_sec_per_volume: float = 93.5,
    linker_overhead: float = 0.08,
    limit_hours: float = 12.0,
    peak_memory_mb: float = 1500.0,
) -> dict[str, Any]:
    """CPU relink of locked detections. Optional later GPU infer+relink budget."""
    n = int(n_volumes)
    relink_sec = 8.0 * n
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
            "CPU constant-velocity relink of locked infer detections; "
            "no transformer; no new UNet pass required"
        ),
    }


def find_infer_graphs(roots: Iterable[str | Path] | None = None) -> dict[str, Path]:
    """Locate locked 44b6 infer JSON graphs. Skip Zarr/GEFF trees."""
    if roots is None:
        roots = [
            Path("/kaggle/input/notebooks"),
            Path("/kaggle/input"),
            Path("artifacts/kaggle/learned_unet_infer_v1"),
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
