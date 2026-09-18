from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


@dataclass
class TrackGraph:
    dataset: str
    node_id: np.ndarray
    t: np.ndarray
    z: np.ndarray
    y: np.ndarray
    x: np.ndarray
    edges: np.ndarray
    estimated_n_nodes: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.node_id = np.asarray(self.node_id, dtype=np.int64)
        self.t = np.asarray(self.t, dtype=np.int64)
        self.z = np.asarray(self.z, dtype=np.int64)
        self.y = np.asarray(self.y, dtype=np.int64)
        self.x = np.asarray(self.x, dtype=np.int64)
        self.edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2) if len(self.edges) else np.zeros((0, 2), dtype=np.int64)

    @property
    def n_nodes(self) -> int:
        return int(self.node_id.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    def coords_zyx(self) -> np.ndarray:
        return np.stack([self.z, self.y, self.x], axis=1).astype(np.float64)

    def id_to_index(self) -> dict[int, int]:
        return {int(nid): i for i, nid in enumerate(self.node_id)}

    def filter_timepoints(self, timepoints: Iterable[int]) -> "TrackGraph":
        allowed = {int(t) for t in timepoints}
        keep = np.array([int(t) in allowed for t in self.t], dtype=bool)
        kept_ids = set(int(x) for x in self.node_id[keep])
        edge_keep = np.array(
            [int(s) in kept_ids and int(t) in kept_ids for s, t in self.edges],
            dtype=bool,
        ) if self.n_edges else np.zeros((0,), dtype=bool)
        estimated = self.estimated_n_nodes
        if estimated is not None:
            orig_t = len({int(t) for t in self.t})
            new_t = len({int(t) for t in self.t[keep]}) if keep.any() else 0
            if orig_t:
                estimated = estimated * new_t / orig_t
        return TrackGraph(
            dataset=self.dataset,
            node_id=self.node_id[keep],
            t=self.t[keep],
            z=self.z[keep],
            y=self.y[keep],
            x=self.x[keep],
            edges=self.edges[edge_keep] if self.n_edges else self.edges,
            estimated_n_nodes=estimated,
            metadata=dict(self.metadata),
        )

    def out_edges(self) -> dict[int, list[int]]:
        mapping: dict[int, list[int]] = defaultdict(list)
        for src, dst in self.edges:
            mapping[int(src)].append(int(dst))
        return mapping

    def in_edges(self) -> dict[int, list[int]]:
        mapping: dict[int, list[int]] = defaultdict(list)
        for src, dst in self.edges:
            mapping[int(dst)].append(int(src))
        return mapping

    def consecutive_edges(self) -> np.ndarray:
        index = self.id_to_index()
        keep = []
        for src, dst in self.edges:
            if int(src) not in index or int(dst) not in index:
                continue
            if int(self.t[index[int(dst)]]) == int(self.t[index[int(src)]]) + 1:
                keep.append((int(src), int(dst)))
        return np.asarray(keep, dtype=np.int64).reshape(-1, 2)

    def track_ids(self) -> np.ndarray:
        """Assign linear track IDs; a division starts new daughter tracks."""
        index = self.id_to_index()
        parent = {int(nid): int(nid) for nid in self.node_id}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        out = self.out_edges()
        for src, dst in self.consecutive_edges():
            if len(out.get(int(src), [])) <= 1:
                union(int(src), int(dst))
        roots = {}
        next_id = 1
        labels = np.zeros(self.n_nodes, dtype=np.int64)
        order = np.argsort(self.t, kind="stable")
        for i in order:
            nid = int(self.node_id[i])
            root = find(nid)
            if root not in roots:
                roots[root] = next_id
                next_id += 1
            labels[i] = roots[root]
        return labels

    def copy(self) -> "TrackGraph":
        return TrackGraph(
            dataset=self.dataset,
            node_id=self.node_id.copy(),
            t=self.t.copy(),
            z=self.z.copy(),
            y=self.y.copy(),
            x=self.x.copy(),
            edges=self.edges.copy(),
            estimated_n_nodes=self.estimated_n_nodes,
            metadata=dict(self.metadata),
        )


def graph_from_detections(
    dataset: str,
    detections: list[tuple[int, np.ndarray]],
    edges: list[tuple[int, int]] | np.ndarray,
    estimated_n_nodes: float | None = None,
    start_node_id: int = 1,
) -> TrackGraph:
    node_id = []
    t_list, z_list, y_list, x_list = [], [], [], []
    next_id = start_node_id
    for t, coords in detections:
        coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        for z, y, x in coords:
            node_id.append(next_id)
            t_list.append(int(t))
            z_list.append(int(round(float(z))))
            y_list.append(int(round(float(y))))
            x_list.append(int(round(float(x))))
            next_id += 1
    edge_arr = np.asarray(edges, dtype=np.int64).reshape(-1, 2) if len(edges) else np.zeros((0, 2), dtype=np.int64)
    return TrackGraph(
        dataset=dataset,
        node_id=np.asarray(node_id, dtype=np.int64),
        t=np.asarray(t_list, dtype=np.int64),
        z=np.asarray(z_list, dtype=np.int64),
        y=np.asarray(y_list, dtype=np.int64),
        x=np.asarray(x_list, dtype=np.int64),
        edges=edge_arr,
        estimated_n_nodes=estimated_n_nodes,
    )
