from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .config import LinkingConfig


def physical_coords(coords_zyx: np.ndarray, scale_zyx: tuple[float, float, float]) -> np.ndarray:
    coords = np.asarray(coords_zyx, dtype=np.float64).reshape(-1, 3)
    return coords * np.asarray(scale_zyx, dtype=np.float64)


def link_frames(
    coords_prev: np.ndarray,
    coords_curr: np.ndarray,
    config: LinkingConfig,
    scale_zyx: tuple[float, float, float],
) -> list[tuple[int, int]]:
    prev = physical_coords(coords_prev, scale_zyx)
    curr = physical_coords(coords_curr, scale_zyx)
    if len(prev) == 0 or len(curr) == 0:
        return []
    method = config.method.lower()
    if method == "greedy":
        return _greedy(prev, curr, config)
    if method == "hungarian":
        return _hungarian(prev, curr, config)
    raise ValueError(f"Unknown linking method: {config.method}")


def _greedy(prev: np.ndarray, curr: np.ndarray, config: LinkingConfig) -> list[tuple[int, int]]:
    dists = cdist(prev, curr)
    links: list[tuple[int, int]] = []
    used_curr: set[int] = set()
    used_prev: set[int] = set()
    order = np.argsort(dists.min(axis=1))
    for i in order:
        js = np.argsort(dists[i])
        for j in js:
            if j in used_curr:
                continue
            if dists[i, j] <= config.max_distance_um:
                links.append((int(i), int(j)))
                used_curr.add(int(j))
                used_prev.add(int(i))
            break
    if config.allow_divisions:
        links.extend(_extra_division_links(dists, used_prev, used_curr, config))
    return links


def _hungarian(prev: np.ndarray, curr: np.ndarray, config: LinkingConfig) -> list[tuple[int, int]]:
    dists = cdist(prev, curr)
    cost = dists.copy()
    inf = config.max_distance_um * 1000.0 + 1.0
    cost[cost > config.max_distance_um] = inf
    rows, cols = linear_sum_assignment(cost)
    links = []
    used_curr: set[int] = set()
    used_prev: set[int] = set()
    for i, j in zip(rows, cols):
        if dists[i, j] <= config.max_distance_um:
            links.append((int(i), int(j)))
            used_curr.add(int(j))
            used_prev.add(int(i))
    if config.allow_divisions:
        links.extend(_extra_division_links(dists, used_prev, used_curr, config))
    return links


def _extra_division_links(
    dists: np.ndarray,
    used_prev: set[int],
    used_curr: set[int],
    config: LinkingConfig,
) -> list[tuple[int, int]]:
    """Attach a second nearby child to a parent that already has one link."""
    extra: list[tuple[int, int]] = []
    for i in list(used_prev):
        js = np.argsort(dists[i])
        attached = 1
        for j in js:
            if attached >= 2:
                break
            if int(j) in used_curr:
                continue
            if dists[i, j] <= min(config.max_distance_um, config.division_gap_um):
                extra.append((int(i), int(j)))
                used_curr.add(int(j))
                attached += 1
    return extra
