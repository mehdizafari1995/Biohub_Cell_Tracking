from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .config import PipelineConfig
from .detect import detect_cells
from .graphs import TrackGraph, graph_from_detections
from .io_zarr import VolumeMeta, iter_timepoints
from .link import link_frames


def track_volume(
    array: Any,
    meta: VolumeMeta,
    config: PipelineConfig,
    timepoints: Iterable[int] | None = None,
) -> TrackGraph:
    detections: list[tuple[int, np.ndarray]] = []
    per_t_coords: dict[int, np.ndarray] = {}
    if timepoints is None:
        frames = iter_timepoints(
            array, meta, start=config.timepoint_start, max_timepoints=config.max_timepoints
        )
    else:
        allowed = {int(t) for t in timepoints}
        frames = ((t, np.asarray(array[t])) for t in meta.available_timepoints if t in allowed)

    for t, image in frames:
        coords = detect_cells(image, config.detection, config.scale_zyx)
        detections.append((int(t), coords))
        per_t_coords[int(t)] = coords

    times = [t for t, _ in detections]
    node_ids_by_t: dict[int, np.ndarray] = {}
    next_id = 1
    for t, coords in detections:
        n = len(coords)
        node_ids_by_t[t] = np.arange(next_id, next_id + n, dtype=np.int64)
        next_id += n

    edges: list[tuple[int, int]] = []
    for t_prev, t_curr in zip(times, times[1:]):
        if t_curr != t_prev + 1:
            continue
        pairs = link_frames(per_t_coords[t_prev], per_t_coords[t_curr], config.linking, config.scale_zyx)
        ids_prev = node_ids_by_t[t_prev]
        ids_curr = node_ids_by_t[t_curr]
        for i, j in pairs:
            edges.append((int(ids_prev[i]), int(ids_curr[j])))

    return graph_from_detections(meta.name, detections, edges)
