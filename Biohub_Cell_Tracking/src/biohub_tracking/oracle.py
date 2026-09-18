from __future__ import annotations

from .config import LinkingConfig
from .graphs import TrackGraph
from .link import link_frames


def link_gt_centroids(gt: TrackGraph, linking: LinkingConfig, scale_zyx: tuple[float, float, float]) -> TrackGraph:
    """Oracle diagnostic: treat GT centroids as detections and run the linker."""
    times = sorted({int(t) for t in gt.t})
    coords = {t: gt.coords_zyx()[gt.t == t] for t in times}
    next_id = 1
    ids: dict[int, list[int]] = {}
    for t in times:
        n = len(coords[t])
        ids[t] = list(range(next_id, next_id + n))
        next_id += n
    edges: list[tuple[int, int]] = []
    for t_prev, t_curr in zip(times, times[1:]):
        if t_curr != t_prev + 1:
            continue
        for i, j in link_frames(coords[t_prev], coords[t_curr], linking, scale_zyx):
            edges.append((ids[t_prev][i], ids[t_curr][j]))
    return TrackGraph(
        dataset=gt.dataset + "_gt_detections",
        node_id=list(range(1, next_id)),
        t=gt.t,
        z=gt.z,
        y=gt.y,
        x=gt.x,
        edges=edges,
        estimated_n_nodes=gt.estimated_n_nodes,
        metadata={"oracle": True, "note": "GT centroids linked; not a detection result"},
    )
