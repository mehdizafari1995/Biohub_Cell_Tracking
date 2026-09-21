"""Hungarian last-position relink of locked learned_unet detections.

Keeps locked ``learned_unet_infer_v1`` nodes and replaces consecutive
edges with Hungarian 1-1 assignment on last-position distance inside
the locked association_v2 12 µm gate. Does not use rejected transformer
or constant-velocity rankers. Never opens ``6bba`` during tuning.
Does not write ``submission.csv``. Not a competition submission.
Do not overwrite locked train/infer/holdout/test/gap-test/link slugs.
Do not retune rejected ``learned_unet_motion_v1``.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

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
from .learned_motion import LOCKED_MOTION_HASH as REJECTED_MOTION_HASH
from .learned_motion import MOTION_KERNEL_SLUG as REJECTED_MOTION_SLUG
from .learned_split import specimen_of

HUNG_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-hung-v1"
REJECTED_LINK_V2_SLUG = "mahdimzafari/biohub-learned-unet-link-v2"
INFER_LOCKED_TUNE = 0.6557507105877639
MAX_LINK_UM = 12.0
INF = 1.0e6

LOCKED_GATES = {
    "motion_model": "last_position",
    "gate_um": MAX_LINK_UM,
    "max_link_um": MAX_LINK_UM,
    "max_gap_frames": 0,
    "assignment": "hungarian",
    "interpolate_gaps": False,
    "max_parents_per_node": 1,
    "max_children_per_node": 1,
    "intensity_weight": 0.0,
}


def hung_config_payload() -> dict[str, Any]:
    """First Hungarian last-position operating point. Do not retune from holdout."""
    return {
        "label": "learned_unet_hung_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "rejected_link_v1_hash": REJECTED_LINK_V1_HASH,
        "rejected_link_v2_hash": REJECTED_LINK_V2_HASH,
        "rejected_motion_hash": REJECTED_MOTION_HASH,
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "infer_stems": list(INFER_STEMS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "source_graphs": "locked_learned_unet_infer_graphs",
        "retunes_detection": False,
        "uses_intensity": False,
        "uses_transformer": False,
        "uses_velocity": False,
        "linker": "last_position_hungarian_1to1",
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
        "kernel_slug": HUNG_KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
        "rejected_link_v1_slug": REJECTED_LINK_V1_SLUG,
        "rejected_link_v2_slug": REJECTED_LINK_V2_SLUG,
        "rejected_motion_slug": REJECTED_MOTION_SLUG,
    }


def hung_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or hung_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_HUNG_HASH = hung_config_hash()
assert LOCKED_HUNG_HASH == "f67bc97d0521dc78"


def physical(coords: np.ndarray, scale_zyx: Sequence[float] = SCALE_ZYX) -> np.ndarray:
    return np.asarray(coords, dtype=np.float64).reshape(-1, 3) * np.asarray(scale_zyx, dtype=np.float64)


def last_position_distances(
    last_zyx: np.ndarray,
    det_zyx: np.ndarray,
    scale_zyx: Sequence[float] = SCALE_ZYX,
) -> np.ndarray:
    last = np.asarray(last_zyx, dtype=np.float64).reshape(-1, 3)
    det = np.asarray(det_zyx, dtype=np.float64).reshape(-1, 3)
    if last.size == 0 or det.size == 0:
        return np.zeros((last.shape[0], det.shape[0]), dtype=np.float64)
    return np.linalg.norm(physical(last, scale_zyx)[:, None, :] - physical(det, scale_zyx)[None, :, :], axis=2)


def greedy_assign(cost: np.ndarray, allowed: np.ndarray) -> list[tuple[int, int]]:
    """Greedy 1-1 on finite allowed costs. Lower cost wins. Diagnostic only."""
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


def hungarian_assign(cost: np.ndarray, allowed: np.ndarray) -> list[tuple[int, int]]:
    """Hungarian 1-1 on allowed costs. Disallowed pairs are never returned."""
    if cost.size == 0:
        return []
    mask = np.asarray(allowed, dtype=bool)
    if not mask.any():
        return []
    work = np.full(np.asarray(cost).shape, INF, dtype=np.float64)
    work[mask] = np.asarray(cost, dtype=np.float64)[mask]
    rows, cols = linear_sum_assignment(work)
    pairs: list[tuple[int, int]] = []
    for i, j in zip(rows, cols):
        ii, jj = int(i), int(j)
        if mask[ii, jj] and work[ii, jj] < INF / 2:
            pairs.append((ii, jj))
    return pairs


def assign_hung_links(
    last_zyx: np.ndarray,
    det_zyx: np.ndarray,
    max_link_um: float = MAX_LINK_UM,
    scale_zyx: Sequence[float] = SCALE_ZYX,
) -> list[tuple[int, int]]:
    """Hungarian 1-1 inside the locked last-position 12 µm gate."""
    last = np.asarray(last_zyx, dtype=np.float64).reshape(-1, 3)
    det = np.asarray(det_zyx, dtype=np.float64).reshape(-1, 3)
    if last.size == 0 or det.size == 0:
        return []
    dist = last_position_distances(last, det, scale_zyx)
    return hungarian_assign(dist, dist <= float(max_link_um))


def assign_greedy_last_position_links(
    last_zyx: np.ndarray,
    det_zyx: np.ndarray,
    max_link_um: float = MAX_LINK_UM,
    scale_zyx: Sequence[float] = SCALE_ZYX,
) -> list[tuple[int, int]]:
    """Locked infer-style greedy last-position 1-1. Used only for disagreement counts."""
    last = np.asarray(last_zyx, dtype=np.float64).reshape(-1, 3)
    det = np.asarray(det_zyx, dtype=np.float64).reshape(-1, 3)
    if last.size == 0 or det.size == 0:
        return []
    dist = last_position_distances(last, det, scale_zyx)
    return greedy_assign(dist, dist <= float(max_link_um))


def apply_hung_relink_arrays(
    node_id: Sequence[int],
    t: Sequence[int],
    z: Sequence[int],
    y: Sequence[int],
    x: Sequence[int],
) -> list[tuple[int, int]]:
    """Standalone Hungarian last-position 1-1 relink. Returns (src, dst) node ids."""
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
            for li, j in assign_hung_links(last, coords):
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


def apply_greedy_last_position_arrays(
    node_id: Sequence[int],
    t: Sequence[int],
    z: Sequence[int],
    y: Sequence[int],
    x: Sequence[int],
) -> list[tuple[int, int]]:
    """Diagnostic greedy last-position relink on the same detections."""
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
            for li, j in assign_greedy_last_position_links(last, coords):
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


def apply_hung_relink(graph: TrackGraph) -> TrackGraph:
    """Replace edges with Hungarian last-position 1-1 links. Keep detections."""
    if hung_config_hash() != LOCKED_HUNG_HASH:
        raise RuntimeError("learned_unet_hung_v1 hash drifted; refusing to retune")
    if specimen_of(graph.dataset) != TUNING_SPECIMEN:
        raise HoldoutLeakError(
            f"learned_unet_hung_v1 tuning refuses stem {graph.dataset}"
        )
    refuse_non_infer_stems([graph.dataset])
    edges = apply_hung_relink_arrays(graph.node_id, graph.t, graph.z, graph.y, graph.x)
    greedy_edges = apply_greedy_last_position_arrays(
        graph.node_id, graph.t, graph.z, graph.y, graph.x
    )
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
        raise RuntimeError("hungarian relink changed detection count; refusing")
    before_ids = set(int(x) for x in graph.node_id)
    after_ids = set(int(x) for x in out.node_id)
    if before_ids != after_ids:
        raise RuntimeError("hungarian relink changed node identities; refusing")
    hung_set = set(edges)
    greedy_set = set(greedy_edges)
    out.metadata = dict(graph.metadata)
    out.metadata["overlay"] = "learned_unet_hung_v1"
    out.metadata["config_hash"] = LOCKED_HUNG_HASH
    out.metadata["parent_infer_hash"] = LOCKED_INFER_HASH
    out.metadata["uses_intensity"] = False
    out.metadata["uses_transformer"] = False
    out.metadata["uses_velocity"] = False
    out.metadata["retunes_detection"] = False
    out.metadata["n_hungarian_links"] = len(edges)
    out.metadata["n_greedy_last_position_links"] = len(greedy_edges)
    out.metadata["n_assignment_disagreements"] = len(hung_set.symmetric_difference(greedy_set))
    return out


def apply_hung_relink_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    graph = graph_from_payload(payload)
    out = apply_hung_relink(graph)
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
        "n_hungarian_links": int(after.metadata.get("n_hungarian_links") or 0),
        "n_greedy_last_position_links": int(after.metadata.get("n_greedy_last_position_links") or 0),
        "n_assignment_disagreements": int(after.metadata.get("n_assignment_disagreements") or 0),
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
        return "reject_learned_unet_hung_v1"
    return "keep_candidate_learned_unet_hung_v1"


def runtime_estimate(
    n_volumes: int = 2,
    infer_sec_per_volume: float = 93.5,
    linker_overhead: float = 0.08,
    limit_hours: float = 12.0,
    peak_memory_mb: float = 1500.0,
) -> dict[str, Any]:
    """CPU Hungarian relink of locked detections. Optional later GPU infer+relink budget."""
    n = int(n_volumes)
    relink_sec = 15.0 * n
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
            "CPU Hungarian last-position relink of locked infer detections; "
            "no transformer; no velocity; no new UNet pass required"
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
