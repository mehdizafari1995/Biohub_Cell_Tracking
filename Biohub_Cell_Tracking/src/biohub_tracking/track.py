from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .association import FrameDetections, track_from_detections
from .config import LinkingConfig, PipelineConfig
from .detect import detect_cells, detect_cells_with_appearance
from .division import propose_divisions
from .graphs import TrackGraph, graph_from_detections
from .io_zarr import VolumeMeta, iter_timepoints
from .link import link_frames


def _use_association(config: LinkingConfig) -> bool:
    return (
        config.method in {"association", "association_v2"}
        or config.motion_model not in {"none", "", "identity"}
        or int(config.max_gap_frames) > 0
        or config.assignment in {"gated_hungarian"}
        or float(config.intensity_weight) > 0
        or float(config.track_start_penalty_um) > 0
        or float(config.track_end_penalty_um) > 0
        or float(config.boundary_margin_um) > 0
    )


def track_volume(
    array: Any,
    meta: VolumeMeta,
    config: PipelineConfig,
    timepoints: Iterable[int] | None = None,
) -> TrackGraph:
    detections: list[tuple[int, np.ndarray]] = []
    per_t_coords: dict[int, np.ndarray] = {}
    frames: list[FrameDetections] = []
    intensity_by_id: dict[int, float] = {}
    if timepoints is None:
        iter_frames = iter_timepoints(
            array, meta, start=config.timepoint_start, max_timepoints=config.max_timepoints
        )
    else:
        allowed = {int(t) for t in timepoints}
        iter_frames = ((t, np.asarray(array[t])) for t in meta.available_timepoints if t in allowed)

    next_id = 1
    for t, image in iter_frames:
        if _use_association(config.linking) or config.linking.allow_divisions:
            coords, intensity, appearance = detect_cells_with_appearance(image, config.detection, config.scale_zyx)
        else:
            coords = detect_cells(image, config.detection, config.scale_zyx)
            intensity = np.zeros(len(coords), dtype=np.float64)
            appearance = intensity
        detections.append((int(t), coords))
        per_t_coords[int(t)] = coords
        n = len(coords)
        node_ids = np.arange(next_id, next_id + n, dtype=np.int64)
        for nid, inten in zip(node_ids, intensity):
            intensity_by_id[int(nid)] = float(inten)
        frames.append(
            FrameDetections(
                t=int(t),
                coords=np.asarray(coords, dtype=np.float64),
                intensity=np.asarray(intensity, dtype=np.float64),
                appearance=np.asarray(appearance, dtype=np.float64),
                node_ids=node_ids,
            )
        )
        next_id += n

    if _use_association(config.linking):
        graph = track_from_detections(
            meta.name,
            frames,
            config.linking,
            config.scale_zyx,
            volume_shape=(int(meta.shape[1]), int(meta.shape[2]), int(meta.shape[3])),
        )
        if config.linking.allow_divisions:
            graph = propose_divisions(graph, intensity_by_id, config.linking, config.scale_zyx)
        return graph

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

    graph = graph_from_detections(meta.name, detections, edges)
    if config.linking.allow_divisions and config.linking.method in {"association", "association_v2"}:
        graph = propose_divisions(graph, intensity_by_id, config.linking, config.scale_zyx)
    return graph


def graph_from_cached_frames(
    dataset: str,
    frames: list[FrameDetections],
    config: PipelineConfig,
    volume_shape: tuple[int, int, int],
    estimated_n_nodes: float | None = None,
) -> TrackGraph:
    if _use_association(config.linking):
        graph = track_from_detections(
            dataset, frames, config.linking, config.scale_zyx, volume_shape, estimated_n_nodes
        )
    else:
        detections = [(f.t, f.coords) for f in frames]
        times = [f.t for f in frames]
        coords = {f.t: f.coords for f in frames}
        ids = {f.t: f.node_ids for f in frames}
        edges: list[tuple[int, int]] = []
        for t_prev, t_curr in zip(times, times[1:]):
            if t_curr != t_prev + 1:
                continue
            for i, j in link_frames(coords[t_prev], coords[t_curr], config.linking, config.scale_zyx):
                edges.append((int(ids[t_prev][i]), int(ids[t_curr][j])))
        graph = graph_from_detections(dataset, detections, edges, estimated_n_nodes=estimated_n_nodes)
    if config.linking.allow_divisions:
        intensity_by_id = {}
        for frame in frames:
            for nid, inten in zip(frame.node_ids, frame.intensity):
                intensity_by_id[int(nid)] = float(inten)
        graph = propose_divisions(graph, intensity_by_id, config.linking, config.scale_zyx)
    return graph
