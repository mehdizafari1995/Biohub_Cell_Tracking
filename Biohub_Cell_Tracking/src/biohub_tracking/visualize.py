from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .graphs import TrackGraph


def max_intensity_xy(volume_zyx: np.ndarray) -> np.ndarray:
    return np.max(np.asarray(volume_zyx), axis=0)


def save_detection_overlay(
    image_zyx: np.ndarray,
    path: str | Path,
    pred: TrackGraph | None = None,
    gt: TrackGraph | None = None,
    timepoint: int = 0,
    title: str = "",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mip = max_intensity_xy(image_zyx)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mip, cmap="gray", origin="upper")
    if gt is not None:
        mask = gt.t == timepoint
        ax.scatter(gt.x[mask], gt.y[mask], s=40, facecolors="none", edgecolors="#4FC3F7", linewidths=1.4, label="GT")
    if pred is not None:
        mask = pred.t == timepoint
        ax.scatter(pred.x[mask], pred.y[mask], s=12, c="#FF8A65", alpha=0.8, label="pred")
    ax.set_title(title or f"t={timepoint} max-intensity XY")
    ax.set_xlabel("x (voxels)")
    ax.set_ylabel("y (voxels)")
    if gt is not None or pred is not None:
        ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_xy_slice(
    image_zyx: np.ndarray,
    path: str | Path,
    z_index: int,
    gt: TrackGraph | None = None,
    timepoint: int = 0,
    title: str = "",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sl = np.asarray(image_zyx[z_index])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(sl, cmap="gray", origin="upper")
    if gt is not None:
        mask = (gt.t == timepoint) & (np.abs(gt.z - z_index) <= 1)
        ax.scatter(gt.x[mask], gt.y[mask], s=50, facecolors="none", edgecolors="#80CBC4", linewidths=1.5, label="GT")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title or f"t={timepoint} z={z_index} XY slice")
    ax.set_xlabel("x (voxels)")
    ax.set_ylabel("y (voxels)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_track_plot(graph: TrackGraph, path: str | Path, title: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = graph.track_ids()
    fig, ax = plt.subplots(figsize=(6, 6))
    index = graph.id_to_index()
    for src, dst in graph.consecutive_edges():
        i, j = index[int(src)], index[int(dst)]
        ax.plot([graph.x[i], graph.x[j]], [graph.y[i], graph.y[j]], color="#90CAF9", lw=0.8, alpha=0.8)
    scatter = ax.scatter(graph.x, graph.y, c=graph.t, s=12, cmap="viridis")
    fig.colorbar(scatter, ax=ax, label="t")
    ax.set_title(title or f"{graph.dataset} tracks (XY)")
    ax.set_xlabel("x (voxels)")
    ax.set_ylabel("y (voxels)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_track_time(graph: TrackGraph, path: str | Path, title: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = graph.track_ids()
    fig, ax = plt.subplots(figsize=(7, 4))
    for track in np.unique(labels):
        mask = labels == track
        order = np.argsort(graph.t[mask])
        ax.plot(graph.t[mask][order], graph.x[mask][order], lw=1.0, alpha=0.85)
    ax.set_xlabel("t (frames)")
    ax.set_ylabel("x (voxels)")
    ax.set_title(title or f"{graph.dataset} track continuity")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
