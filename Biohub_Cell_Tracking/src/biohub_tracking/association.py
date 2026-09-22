from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .config import LinkingConfig
from .graphs import TrackGraph
from .link import physical_coords

INF = 1.0e6


@dataclass
class FrameDetections:
    t: int
    coords: np.ndarray
    intensity: np.ndarray
    appearance: np.ndarray
    node_ids: np.ndarray


@dataclass
class TrackState:
    track_id: int
    node_ids: list[int] = field(default_factory=list)
    times: list[int] = field(default_factory=list)
    positions: list[np.ndarray] = field(default_factory=list)
    intensity: list[float] = field(default_factory=list)
    appearance: list[float] = field(default_factory=list)

    def predicted_voxel(self, t: int, motion_model: str) -> np.ndarray:
        pos = np.asarray(self.positions[-1], dtype=np.float64)
        if motion_model != "constant_velocity" or len(self.positions) < 2:
            return pos
        dt_hist = float(self.times[-1] - self.times[-2])
        if dt_hist <= 0:
            return pos
        vel = (np.asarray(self.positions[-1], dtype=np.float64) - np.asarray(self.positions[-2], dtype=np.float64)) / dt_hist
        return pos + vel * float(t - self.times[-1])


def gate_um_of(config: LinkingConfig) -> float:
    return float(config.gate_um if config.gate_um is not None else config.max_distance_um)


def near_boundary(
    coord_zyx: np.ndarray,
    volume_shape: tuple[int, int, int],
    scale_zyx: tuple[float, float, float],
    margin_um: float,
) -> bool:
    if margin_um <= 0:
        return False
    z, y, x = [float(v) for v in coord_zyx]
    nz, ny, nx = volume_shape
    sz, sy, sx = scale_zyx
    dist = min(
        z * sz,
        (nz - 1 - z) * sz,
        y * sy,
        (ny - 1 - y) * sy,
        x * sx,
        (nx - 1 - x) * sx,
    )
    return dist < margin_um


def appearance_cost(a: float, b: float, weight: float, scale_um: float) -> float:
    if weight <= 0:
        return 0.0
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        return 0.0
    ratio = min(a, b) / (max(a, b) + 1e-6)
    if ratio < 0.25:
        return 0.0
    rel = abs(a - b) / (a + b + 1e-6)
    return float(weight) * rel * float(scale_um)


def associate_frames(
    tracks: list[TrackState],
    frame: FrameDetections,
    config: LinkingConfig,
    scale_zyx: tuple[float, float, float],
    volume_shape: tuple[int, int, int],
) -> list[tuple[int, int, float]]:
    """Return (track_index, detection_index, cost) for one-to-one gated assignment."""
    gate = gate_um_of(config)
    if not tracks or len(frame.coords) == 0:
        return []
    predicted = np.stack([tr.predicted_voxel(frame.t, config.motion_model) for tr in tracks])
    pred_phys = physical_coords(predicted, scale_zyx)
    det_phys = physical_coords(frame.coords, scale_zyx)
    dist = cdist(pred_phys, det_phys)
    cost = np.full(dist.shape, INF, dtype=np.float64)
    dt = np.array([frame.t - tr.times[-1] for tr in tracks], dtype=np.int64)
    max_dt = max(int(config.max_gap_frames) + 1, 1)
    for i, tr in enumerate(tracks):
        if dt[i] < 1 or dt[i] > max_dt:
            continue
        scaled_gate = gate * np.sqrt(float(dt[i]))
        for j in range(len(frame.coords)):
            if dist[i, j] > scaled_gate:
                continue
            app = appearance_cost(
                tr.appearance[-1] if tr.appearance else tr.intensity[-1],
                float(frame.appearance[j] if len(frame.appearance) else frame.intensity[j]),
                config.intensity_weight,
                gate,
            )
            cost[i, j] = float(dist[i, j]) + app
    if config.assignment == "greedy":
        return _greedy_gated(cost, config)
    if config.assignment in {"gated_hungarian", "hungarian"}:
        return _gated_hungarian(cost, tracks, frame, config, scale_zyx, volume_shape)
    raise ValueError(f"Unknown assignment: {config.assignment}")


def _pairs_from_cost(cost: np.ndarray) -> list[tuple[int, int, float]]:
    pairs = []
    finite = np.argwhere(cost < INF / 2)
    order = np.argsort(cost[finite[:, 0], finite[:, 1]])
    used_i, used_j = set(), set()
    for k in order:
        i, j = int(finite[k, 0]), int(finite[k, 1])
        if i in used_i or j in used_j:
            continue
        pairs.append((i, j, float(cost[i, j])))
        used_i.add(i)
        used_j.add(j)
    return pairs


def _greedy_gated(cost: np.ndarray, config: LinkingConfig) -> list[tuple[int, int, float]]:
    limit = None
    if config.track_start_penalty_um > 0 or config.track_end_penalty_um > 0:
        limit = config.track_start_penalty_um + config.track_end_penalty_um
    pairs = []
    for i, j, c in _pairs_from_cost(cost):
        if limit is not None and c > limit:
            continue
        pairs.append((i, j, c))
    return pairs


def _gated_hungarian(
    cost: np.ndarray,
    tracks: list[TrackState],
    frame: FrameDetections,
    config: LinkingConfig,
    scale_zyx: tuple[float, float, float],
    volume_shape: tuple[int, int, int],
) -> list[tuple[int, int, float]]:
    n_r, n_c = cost.shape
    use_penalties = config.track_start_penalty_um > 0 or config.track_end_penalty_um > 0
    if not use_penalties:
        work = cost.copy()
        rows, cols = linear_sum_assignment(work)
        pairs = []
        for i, j in zip(rows, cols):
            if work[i, j] < INF / 2:
                pairs.append((int(i), int(j), float(work[i, j])))
        return pairs
    n = n_r + n_c
    big = np.full((n, n), INF, dtype=np.float64)
    big[:n_r, :n_c] = cost
    for i, tr in enumerate(tracks):
        end_pen = config.track_end_penalty_um
        if near_boundary(tr.positions[-1], volume_shape, scale_zyx, config.boundary_margin_um):
            end_pen *= 0.15
        big[i, n_c + i] = end_pen
    for j in range(n_c):
        start_pen = config.track_start_penalty_um
        if near_boundary(frame.coords[j], volume_shape, scale_zyx, config.boundary_margin_um):
            start_pen *= 0.15
        big[n_r + j, j] = start_pen
    big[n_r:, n_c:] = 0.0
    rows, cols = linear_sum_assignment(big)
    pairs = []
    for i, j in zip(rows, cols):
        if i < n_r and j < n_c and big[i, j] < INF / 2:
            pairs.append((int(i), int(j), float(big[i, j])))
    return pairs


def _interpolate(
    t0: int,
    p0: np.ndarray,
    t1: int,
    p1: np.ndarray,
    intensity0: float,
    intensity1: float,
) -> list[tuple[int, np.ndarray, float]]:
    dt = int(t1 - t0)
    if dt <= 1:
        return []
    out = []
    for k in range(1, dt):
        alpha = k / dt
        pos = (1.0 - alpha) * np.asarray(p0, dtype=np.float64) + alpha * np.asarray(p1, dtype=np.float64)
        inten = (1.0 - alpha) * intensity0 + alpha * intensity1
        out.append((t0 + k, pos, inten))
    return out


def track_from_detections(
    dataset: str,
    frames: list[FrameDetections],
    config: LinkingConfig,
    scale_zyx: tuple[float, float, float],
    volume_shape: tuple[int, int, int] | None = None,
    estimated_n_nodes: float | None = None,
) -> TrackGraph:
    """Link detections with gated association. Interpolates skip-gaps into consecutive edges."""
    if not frames:
        return graph_from_detections(dataset, [], [])
    shape = volume_shape or (2**16, 2**16, 2**16)
    frames = sorted(frames, key=lambda f: f.t)
    next_id = 1
    for frame in frames:
        if frame.node_ids is not None and len(frame.node_ids) == len(frame.coords) and len(frame.coords) > 0:
            next_id = max(next_id, int(np.max(frame.node_ids)) + 1)
            continue
        n = len(frame.coords)
        frame.node_ids = np.arange(next_id, next_id + n, dtype=np.int64)
        next_id += n

    tracks: list[TrackState] = []
    interp_nodes: list[tuple[int, int, float, float, float]] = []
    edges: list[tuple[int, int]] = []
    next_track = 1

    def start_track(frame: FrameDetections, j: int) -> TrackState:
        nonlocal next_track
        tr = TrackState(track_id=next_track)
        next_track += 1
        tr.node_ids.append(int(frame.node_ids[j]))
        tr.times.append(int(frame.t))
        tr.positions.append(np.asarray(frame.coords[j], dtype=np.float64))
        tr.intensity.append(float(frame.intensity[j]) if len(frame.intensity) else 0.0)
        tr.appearance.append(float(frame.appearance[j]) if len(frame.appearance) else tr.intensity[-1])
        return tr

    for frame in frames:
        live = [tr for tr in tracks if frame.t - tr.times[-1] <= max(int(config.max_gap_frames) + 1, 1)]
        pairs = associate_frames(live, frame, config, scale_zyx, shape)
        used_dets: set[int] = set()
        used_tracks: set[int] = set()
        for li, j, _cost in pairs:
            if config.max_children <= 1 and li in used_tracks:
                continue
            if j in used_dets:
                continue
            tr = live[li]
            parent_id = tr.node_ids[-1]
            child_id = int(frame.node_ids[j])
            t0, t1 = tr.times[-1], int(frame.t)
            p0, p1 = tr.positions[-1], np.asarray(frame.coords[j], dtype=np.float64)
            i0 = tr.intensity[-1]
            i1 = float(frame.intensity[j]) if len(frame.intensity) else 0.0
            if t1 == t0 + 1:
                edges.append((parent_id, child_id))
            elif config.interpolate_gaps and t1 > t0 + 1:
                prev_id = parent_id
                for tk, pos, inten in _interpolate(t0, p0, t1, p1, i0, i1):
                    nid = next_id
                    next_id += 1
                    interp_nodes.append((nid, tk, float(pos[0]), float(pos[1]), float(pos[2])))
                    edges.append((prev_id, nid))
                    prev_id = nid
                    tr.node_ids.append(nid)
                    tr.times.append(tk)
                    tr.positions.append(pos)
                    tr.intensity.append(inten)
                    tr.appearance.append(inten)
                edges.append((prev_id, child_id))
            else:
                continue
            tr.node_ids.append(child_id)
            tr.times.append(t1)
            tr.positions.append(p1)
            tr.intensity.append(i1)
            tr.appearance.append(float(frame.appearance[j]) if len(frame.appearance) else i1)
            used_dets.add(j)
            used_tracks.add(li)
        for j in range(len(frame.coords)):
            if j in used_dets:
                continue
            tracks.append(start_track(frame, j))

    node_id: list[int] = []
    t_list: list[int] = []
    z_list: list[int] = []
    y_list: list[int] = []
    x_list: list[int] = []
    for frame in frames:
        for nid, (z, y, x) in zip(frame.node_ids, np.asarray(frame.coords, dtype=np.float64).reshape(-1, 3)):
            node_id.append(int(nid))
            t_list.append(int(frame.t))
            z_list.append(int(round(float(z))))
            y_list.append(int(round(float(y))))
            x_list.append(int(round(float(x))))
    for nid, tk, z, y, x in interp_nodes:
        node_id.append(int(nid))
        t_list.append(int(tk))
        z_list.append(int(round(float(z))))
        y_list.append(int(round(float(y))))
        x_list.append(int(round(float(x))))
    graph = TrackGraph(
        dataset=dataset,
        node_id=np.asarray(node_id, dtype=np.int64),
        t=np.asarray(t_list, dtype=np.int64),
        z=np.asarray(z_list, dtype=np.int64),
        y=np.asarray(y_list, dtype=np.int64),
        x=np.asarray(x_list, dtype=np.int64),
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2) if edges else np.zeros((0, 2), dtype=np.int64),
        estimated_n_nodes=estimated_n_nodes,
        metadata={"association": {"n_interpolated": len(interp_nodes), "n_tracks": len(tracks)}},
    )
    return graph
