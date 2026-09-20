from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from .config import LinkingConfig
from .graphs import TrackGraph
from .link import physical_coords


def sibling_alignment_cosine(
    parent_xyz: np.ndarray,
    child_a_xyz: np.ndarray,
    child_b_xyz: np.ndarray,
    min_displacement_um: float = 1.0,
) -> float | None:
    """Cosine of the two parent→child displacement vectors.

    Returns None when either displacement is shorter than ``min_displacement_um``
    so a near-coincident daughter does not dominate the angle check.
    """
    d1 = np.asarray(child_a_xyz, dtype=np.float64).reshape(3) - np.asarray(parent_xyz, dtype=np.float64).reshape(3)
    d2 = np.asarray(child_b_xyz, dtype=np.float64).reshape(3) - np.asarray(parent_xyz, dtype=np.float64).reshape(3)
    n1 = float(np.linalg.norm(d1))
    n2 = float(np.linalg.norm(d2))
    if n1 < min_displacement_um or n2 < min_displacement_um:
        return None
    return float(np.dot(d1, d2) / (n1 * n2))


def alternative_parent_exists(
    child_xyz: np.ndarray,
    parent_id: int,
    parent_time_ids: list[int],
    index: dict[int, int],
    coords: np.ndarray,
    scale_zyx: tuple[float, float, float],
    gate_um: float,
) -> bool:
    """True when another node at the parent time sits within ``gate_um`` of the child."""
    others = [nid for nid in parent_time_ids if nid != parent_id and nid in index]
    if not others:
        return False
    other_idx = np.array([index[n] for n in others], dtype=np.int64)
    other_xyz = physical_coords(coords[other_idx], scale_zyx)
    child = physical_coords(np.asarray(child_xyz, dtype=np.float64).reshape(1, 3), scale_zyx)
    dists = cdist(child, other_xyz)[0]
    return bool(np.any(dists <= gate_um))


def propose_divisions(
    graph: TrackGraph,
    intensity_by_id: dict[int, float] | None,
    config: LinkingConfig,
    scale_zyx: tuple[float, float, float],
) -> TrackGraph:
    """Add at most one extra child when a second daughter is geometrically plausible.

    Ordinary one-child linking stays the default. A parent may receive a second
    child only if every configured check passes. A child is never given to two parents.

    Conservative flags (``require_parent_predecessor``, ``require_no_alternative_parent``,
    ``max_sibling_alignment``) default off so ``division_v1`` stays unchanged.
    """
    if not config.allow_divisions or graph.n_nodes == 0:
        return graph
    max_parent = float(config.max_parent_child_um if config.max_parent_child_um is not None else config.division_gap_um)
    min_parent = float(config.min_parent_child_um)
    min_sep = float(config.min_child_separation_um)
    alt_gate = config.alternative_parent_gate_um
    if alt_gate is None:
        alt_gate = float(config.gate_um if config.gate_um is not None else config.max_distance_um)
    index = graph.id_to_index()
    coords = graph.coords_zyx()
    out = graph.out_edges()
    incoming = graph.in_edges()
    by_t: dict[int, list[int]] = {}
    for nid, t in zip(graph.node_id, graph.t):
        by_t.setdefault(int(t), []).append(int(nid))

    parent_candidates = [int(nid) for nid, children in out.items() if len(children) == 1]
    used_children: set[int] = set()
    for children in out.values():
        used_children.update(int(c) for c in children)

    rejected = {
        "predecessor": 0,
        "dt": 0,
        "parent_child": 0,
        "sibling_sep": 0,
        "alignment": 0,
        "alternative_parent": 0,
        "intensity": 0,
    }
    proposals: list[tuple[float, int, int]] = []
    for parent in parent_candidates:
        if config.require_parent_predecessor and len(incoming.get(parent, [])) == 0:
            rejected["predecessor"] += 1
            continue
        pidx = index[parent]
        t_parent = int(graph.t[pidx])
        child1 = int(out[parent][0])
        if child1 not in index:
            continue
        t_child = int(graph.t[index[child1]])
        if t_child != t_parent + 1:
            rejected["dt"] += 1
            continue
        free = [
            nid
            for nid in by_t.get(t_child, [])
            if nid != child1 and nid not in used_children and len(incoming.get(nid, [])) == 0
        ]
        if not free:
            continue
        parent_xyz = physical_coords(coords[pidx : pidx + 1], scale_zyx)
        child1_xyz = physical_coords(coords[index[child1] : index[child1] + 1], scale_zyx)
        if config.require_parent_predecessor or min_parent > 0:
            d_parent_child1 = float(np.linalg.norm(child1_xyz[0] - parent_xyz[0]))
            if d_parent_child1 > max_parent or d_parent_child1 < min_parent:
                rejected["parent_child"] += 1
                continue
        free_idx = np.array([index[n] for n in free], dtype=np.int64)
        free_xyz = physical_coords(coords[free_idx], scale_zyx)
        d_parent = cdist(parent_xyz, free_xyz)[0]
        d_sib = cdist(child1_xyz, free_xyz)[0]
        parent_time_ids = by_t.get(t_parent, [])
        for j, nid in enumerate(free):
            if d_parent[j] > max_parent or d_parent[j] < min_parent:
                rejected["parent_child"] += 1
                continue
            if d_sib[j] < min_sep:
                rejected["sibling_sep"] += 1
                continue
            if config.max_sibling_alignment is not None:
                cosine = sibling_alignment_cosine(
                    parent_xyz[0],
                    child1_xyz[0],
                    free_xyz[j],
                    min_displacement_um=float(config.min_displacement_um),
                )
                if cosine is not None and cosine > float(config.max_sibling_alignment):
                    rejected["alignment"] += 1
                    continue
            if config.require_no_alternative_parent and alternative_parent_exists(
                coords[index[int(nid)]],
                parent,
                parent_time_ids,
                index,
                coords,
                scale_zyx,
                float(alt_gate),
            ):
                rejected["alternative_parent"] += 1
                continue
            if intensity_by_id is not None:
                ip = intensity_by_id.get(parent)
                i1 = intensity_by_id.get(child1)
                i2 = intensity_by_id.get(int(nid))
                if ip and i1 and i2 and ip > 0:
                    rel = abs((i1 + i2) - ip) / (ip + 1e-6)
                    if rel > config.intensity_sum_tolerance:
                        rejected["intensity"] += 1
                        continue
            proposals.append((float(d_parent[j]), parent, int(nid)))

    proposals.sort()
    claimed_parent: set[int] = set()
    claimed_child: set[int] = set()
    extra: list[tuple[int, int]] = []
    for _dist, parent, child in proposals:
        if parent in claimed_parent or child in claimed_child:
            continue
        if len(out.get(parent, [])) >= int(config.max_children):
            continue
        extra.append((parent, child))
        claimed_parent.add(parent)
        claimed_child.add(child)
        used_children.add(child)

    out_graph = graph.copy()
    out_graph.metadata = dict(graph.metadata)
    out_graph.metadata["n_division_proposals"] = len(extra)
    out_graph.metadata["n_division_rejected"] = rejected
    if extra:
        out_graph.edges = np.concatenate([graph.edges, np.asarray(extra, dtype=np.int64).reshape(-1, 2)])
    return out_graph
