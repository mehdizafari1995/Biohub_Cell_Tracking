from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from .config import LinkingConfig
from .graphs import TrackGraph
from .link import physical_coords


def propose_divisions(
    graph: TrackGraph,
    intensity_by_id: dict[int, float] | None,
    config: LinkingConfig,
    scale_zyx: tuple[float, float, float],
) -> TrackGraph:
    """Add at most one extra child when a second daughter is geometrically plausible.

    Ordinary one-child linking stays the default. A parent may receive a second
    child only if every configured check passes. A child is never given to two parents.
    """
    if not config.allow_divisions or graph.n_nodes == 0:
        return graph
    max_parent = float(config.max_parent_child_um if config.max_parent_child_um is not None else config.division_gap_um)
    min_sep = float(config.min_child_separation_um)
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

    proposals: list[tuple[float, int, int]] = []
    for parent in parent_candidates:
        pidx = index[parent]
        t_parent = int(graph.t[pidx])
        child1 = int(out[parent][0])
        if child1 not in index:
            continue
        t_child = int(graph.t[index[child1]])
        if t_child != t_parent + 1:
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
        free_idx = np.array([index[n] for n in free], dtype=np.int64)
        free_xyz = physical_coords(coords[free_idx], scale_zyx)
        d_parent = cdist(parent_xyz, free_xyz)[0]
        d_sib = cdist(child1_xyz, free_xyz)[0]
        for j, nid in enumerate(free):
            if d_parent[j] > max_parent:
                continue
            if d_sib[j] < min_sep:
                continue
            if intensity_by_id is not None:
                ip = intensity_by_id.get(parent)
                i1 = intensity_by_id.get(child1)
                i2 = intensity_by_id.get(int(nid))
                if ip and i1 and i2 and ip > 0:
                    rel = abs((i1 + i2) - ip) / (ip + 1e-6)
                    if rel > config.intensity_sum_tolerance:
                        continue
            proposals.append((float(d_parent[j]), parent, int(nid)))

    proposals.sort()
    claimed_parent: set[int] = set()
    claimed_child: set[int] = set()
    extra: list[tuple[int, int]] = []
    for _dist, parent, child in proposals:
        if parent in claimed_parent or child in claimed_child:
            continue
        if len(out.get(parent, [])) >= 2:
            continue
        extra.append((parent, child))
        claimed_parent.add(parent)
        claimed_child.add(child)
        used_children.add(child)

    if not extra:
        return graph
    new_edges = np.concatenate([graph.edges, np.asarray(extra, dtype=np.int64).reshape(-1, 2)])
    out_graph = graph.copy()
    out_graph.edges = new_edges
    out_graph.metadata = dict(graph.metadata)
    out_graph.metadata["n_division_proposals"] = len(extra)
    return out_graph
