"""Conflict-only Hungarian relink of locked learned_unet detections.

Keeps unique last-position nearest-neighbor matches and applies
Hungarian 1-1 only when two or more tracks share the same nearest
neighbor inside the locked association_v2 12 µm gate. Does not reopen
rejected full-frame ``learned_unet_hung_v1``. Never opens ``6bba``
during tuning. Does not write ``submission.csv``. Not a competition
submission. Do not overwrite locked train/infer/holdout/test/gap-test/
link slugs. Do not retune rejected motion/hung hashes.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .graphs import TrackGraph
from .learned_hung import (  # noqa: F401 — reuse locked geometry / greedy helpers
    INFER_LOCKED_TUNE,
    LOCKED_HUNG_HASH as REJECTED_HUNG_HASH,
    MAX_LINK_UM,
    apply_greedy_last_position_arrays,
    assign_greedy_last_position_links,
    assign_hung_links,
    edge_geometry,
    find_infer_graphs,
    hungarian_assign,
    last_position_distances,
    load_role_graphs,
    refuse_existing_holdout_score,
    require_tuning_keep,
)
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

CONFLICT_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-conflict-v1"
REJECTED_LINK_V2_SLUG = "mahdimzafari/biohub-learned-unet-link-v2"
REJECTED_HUNG_SLUG = "mahdimzafari/biohub-learned-unet-hung-v1"
HUNG_KERNEL_SLUG = REJECTED_HUNG_SLUG

LOCKED_GATES = {
    "motion_model": "last_position",
    "gate_um": MAX_LINK_UM,
    "max_link_um": MAX_LINK_UM,
    "max_gap_frames": 0,
    "assignment": "hungarian_on_shared_nearest",
    "interpolate_gaps": False,
    "max_parents_per_node": 1,
    "max_children_per_node": 1,
    "intensity_weight": 0.0,
}


def conflict_config_payload() -> dict[str, Any]:
    """First conflict-Hungarian operating point. Do not retune from holdout."""
    return {
        "label": "learned_unet_conflict_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "rejected_link_v1_hash": REJECTED_LINK_V1_HASH,
        "rejected_link_v2_hash": REJECTED_LINK_V2_HASH,
        "rejected_motion_hash": REJECTED_MOTION_HASH,
        "rejected_hung_hash": REJECTED_HUNG_HASH,
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "infer_stems": list(INFER_STEMS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "source_graphs": "locked_learned_unet_infer_graphs",
        "retunes_detection": False,
        "uses_intensity": False,
        "uses_transformer": False,
        "uses_velocity": False,
        "linker": "last_position_conflict_hungarian_1to1",
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
        "kernel_slug": CONFLICT_KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
        "rejected_link_v1_slug": REJECTED_LINK_V1_SLUG,
        "rejected_link_v2_slug": REJECTED_LINK_V2_SLUG,
        "rejected_motion_slug": REJECTED_MOTION_SLUG,
        "rejected_hung_slug": REJECTED_HUNG_SLUG,
    }


def conflict_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or conflict_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_CONFLICT_HASH = conflict_config_hash()
assert LOCKED_CONFLICT_HASH == "8503d804fb9e9f1d"


def nearest_neighbors(dist: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Index of each source's nearest allowed dest, or -1 when none exist."""
    n_src = int(np.asarray(dist).shape[0])
    nn = np.full(n_src, -1, dtype=np.int64)
    mask = np.asarray(allowed, dtype=bool)
    if dist.size == 0 or not mask.any():
        return nn
    work = np.where(mask, np.asarray(dist, dtype=np.float64), np.inf)
    best = np.argmin(work, axis=1)
    finite = np.isfinite(work[np.arange(n_src), best])
    nn[finite] = best[finite].astype(np.int64)
    return nn


def assign_conflict_links(
    last_zyx: np.ndarray,
    det_zyx: np.ndarray,
    max_link_um: float = MAX_LINK_UM,
    scale_zyx: Sequence[float] = SCALE_ZYX,
) -> list[tuple[int, int]]:
    """Keep unique last-position NNs; Hungarian only on shared-nearest collisions."""
    last = np.asarray(last_zyx, dtype=np.float64).reshape(-1, 3)
    det = np.asarray(det_zyx, dtype=np.float64).reshape(-1, 3)
    if last.size == 0 or det.size == 0:
        return []
    dist = last_position_distances(last, det, scale_zyx)
    allowed = dist <= float(max_link_um)
    nn = nearest_neighbors(dist, allowed)
    n_dst = det.shape[0]
    claim_count = np.zeros(n_dst, dtype=np.int64)
    for j in nn:
        if int(j) >= 0:
            claim_count[int(j)] += 1
    pairs: list[tuple[int, int]] = []
    used_dst: set[int] = set()
    collision_src: list[int] = []
    for i, j in enumerate(nn.tolist()):
        if j < 0:
            continue
        if int(claim_count[j]) == 1:
            pairs.append((i, int(j)))
            used_dst.add(int(j))
        elif int(claim_count[j]) >= 2:
            collision_src.append(i)
    if not collision_src:
        return pairs
    collision_dst: list[int] = []
    seen_dst: set[int] = set()
    for i in collision_src:
        for j in np.flatnonzero(allowed[i]):
            jj = int(j)
            if jj in used_dst or jj in seen_dst:
                continue
            seen_dst.add(jj)
            collision_dst.append(jj)
    if not collision_dst:
        return pairs
    sub_cost = dist[np.ix_(collision_src, collision_dst)]
    sub_allowed = allowed[np.ix_(collision_src, collision_dst)]
    for li, lj in hungarian_assign(sub_cost, sub_allowed):
        pairs.append((collision_src[li], collision_dst[lj]))
    return pairs


def conflict_frame_stats(
    last_zyx: np.ndarray,
    det_zyx: np.ndarray,
    max_link_um: float = MAX_LINK_UM,
    scale_zyx: Sequence[float] = SCALE_ZYX,
) -> dict[str, int]:
    """Diagnostic counts for one frame pair. Not used for scoring."""
    last = np.asarray(last_zyx, dtype=np.float64).reshape(-1, 3)
    det = np.asarray(det_zyx, dtype=np.float64).reshape(-1, 3)
    if last.size == 0 or det.size == 0:
        return {"n_unique_greedy_links": 0, "n_hungarian_collision_links": 0, "n_collision_sources": 0}
    dist = last_position_distances(last, det, scale_zyx)
    allowed = dist <= float(max_link_um)
    nn = nearest_neighbors(dist, allowed)
    n_dst = det.shape[0]
    claim_count = np.zeros(n_dst, dtype=np.int64)
    for j in nn:
        if int(j) >= 0:
            claim_count[int(j)] += 1
    n_unique = int(sum(1 for j in nn if int(j) >= 0 and int(claim_count[int(j)]) == 1))
    n_collision = int(sum(1 for j in nn if int(j) >= 0 and int(claim_count[int(j)]) >= 2))
    pairs = assign_conflict_links(last, det, max_link_um, scale_zyx)
    return {
        "n_unique_greedy_links": n_unique,
        "n_hungarian_collision_links": max(0, len(pairs) - n_unique),
        "n_collision_sources": n_collision,
    }


def apply_conflict_relink_arrays(
    node_id: Sequence[int],
    t: Sequence[int],
    z: Sequence[int],
    y: Sequence[int],
    x: Sequence[int],
) -> list[tuple[int, int]]:
    """Standalone conflict-Hungarian last-position 1-1 relink. Returns (src, dst) ids."""
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
            for li, j in assign_conflict_links(last, coords):
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


def apply_conflict_relink(graph: TrackGraph) -> TrackGraph:
    """Replace edges with conflict-Hungarian last-position 1-1 links. Keep detections."""
    if conflict_config_hash() != LOCKED_CONFLICT_HASH:
        raise RuntimeError("learned_unet_conflict_v1 hash drifted; refusing to retune")
    if specimen_of(graph.dataset) != TUNING_SPECIMEN:
        raise HoldoutLeakError(
            f"learned_unet_conflict_v1 tuning refuses stem {graph.dataset}"
        )
    refuse_non_infer_stems([graph.dataset])
    edges = apply_conflict_relink_arrays(graph.node_id, graph.t, graph.z, graph.y, graph.x)
    greedy_edges = apply_greedy_last_position_arrays(
        graph.node_id, graph.t, graph.z, graph.y, graph.x
    )
    hung_edges = []
    nids = [int(v) for v in graph.node_id]
    ts = [int(v) for v in graph.t]
    zs = [int(v) for v in graph.z]
    ys = [int(v) for v in graph.y]
    xs = [int(v) for v in graph.x]
    by_t: dict[int, list[int]] = defaultdict(list)
    for i, tk in enumerate(ts):
        by_t[tk].append(i)
    n_unique = 0
    n_collision_src = 0
    n_hung_collision = 0
    tracks: list[dict[str, Any]] = []
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
            stats = conflict_frame_stats(last, coords)
            n_unique += stats["n_unique_greedy_links"]
            n_collision_src += stats["n_collision_sources"]
            n_hung_collision += stats["n_hungarian_collision_links"]
            for li, j in assign_hung_links(last, coords):
                hung_edges.append((int(live[li]["node_ids"][-1]), int(nids[idxs[j]])))
            for li, j in assign_conflict_links(last, coords):
                tr = live[li]
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
        raise RuntimeError("conflict relink changed detection count; refusing")
    before_ids = set(int(x) for x in graph.node_id)
    after_ids = set(int(x) for x in out.node_id)
    if before_ids != after_ids:
        raise RuntimeError("conflict relink changed node identities; refusing")
    conflict_set = set(edges)
    greedy_set = set(greedy_edges)
    hung_set = set(hung_edges)
    out.metadata = dict(graph.metadata)
    out.metadata["overlay"] = "learned_unet_conflict_v1"
    out.metadata["config_hash"] = LOCKED_CONFLICT_HASH
    out.metadata["parent_infer_hash"] = LOCKED_INFER_HASH
    out.metadata["rejected_hung_hash"] = REJECTED_HUNG_HASH
    out.metadata["uses_intensity"] = False
    out.metadata["uses_transformer"] = False
    out.metadata["uses_velocity"] = False
    out.metadata["retunes_detection"] = False
    out.metadata["n_conflict_links"] = len(edges)
    out.metadata["n_unique_greedy_links"] = n_unique
    out.metadata["n_hungarian_collision_links"] = n_hung_collision
    out.metadata["n_collision_sources"] = n_collision_src
    out.metadata["n_greedy_last_position_links"] = len(greedy_edges)
    out.metadata["n_full_frame_hungarian_links"] = len(hung_edges)
    out.metadata["n_assignment_disagreements"] = len(conflict_set.symmetric_difference(greedy_set))
    out.metadata["n_disagreements_vs_full_frame_hungarian"] = len(conflict_set.symmetric_difference(hung_set))
    return out


def apply_conflict_relink_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    graph = graph_from_payload(payload)
    out = apply_conflict_relink(graph)
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
        "n_conflict_links": int(after.metadata.get("n_conflict_links") or 0),
        "n_unique_greedy_links": int(after.metadata.get("n_unique_greedy_links") or 0),
        "n_hungarian_collision_links": int(after.metadata.get("n_hungarian_collision_links") or 0),
        "n_collision_sources": int(after.metadata.get("n_collision_sources") or 0),
        "n_greedy_last_position_links": int(after.metadata.get("n_greedy_last_position_links") or 0),
        "n_full_frame_hungarian_links": int(after.metadata.get("n_full_frame_hungarian_links") or 0),
        "n_assignment_disagreements": int(after.metadata.get("n_assignment_disagreements") or 0),
        "n_disagreements_vs_full_frame_hungarian": int(
            after.metadata.get("n_disagreements_vs_full_frame_hungarian") or 0
        ),
        "detection_counts_match": before.n_nodes == after.n_nodes,
    }


def tuning_decision(learned_combined: float) -> str:
    if float(learned_combined) < INFER_LOCKED_TUNE:
        return "reject_learned_unet_conflict_v1"
    return "keep_candidate_learned_unet_conflict_v1"


def runtime_estimate(
    n_volumes: int = 2,
    infer_sec_per_volume: float = 93.5,
    linker_overhead: float = 0.08,
    limit_hours: float = 12.0,
    peak_memory_mb: float = 1500.0,
) -> dict[str, Any]:
    """CPU conflict-Hungarian relink of locked detections. Optional later GPU infer+relink budget."""
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
            "CPU conflict-Hungarian last-position relink of locked infer detections; "
            "unique NNs kept; Hungarian only on shared-nearest collisions; "
            "no transformer; no velocity; no new UNet pass required"
        ),
    }
