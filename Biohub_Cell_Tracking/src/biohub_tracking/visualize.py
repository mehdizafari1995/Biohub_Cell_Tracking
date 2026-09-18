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


def save_error_overlay(
    image_zyx: np.ndarray,
    path: str | Path,
    pred: TrackGraph,
    gt: TrackGraph,
    missed_gt_ids: list[int],
    fp_edges: list[tuple[int, int]],
    fn_edges: list[tuple[int, int]],
    recovered_divisions: list[int],
    missed_divisions: list[int],
    timepoint: int,
    title: str = "",
) -> Path:
    """XY MIP overlay: missed GT, incorrect/fragmented links, division markers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mip = max_intensity_xy(image_zyx)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(mip, cmap="gray", origin="upper")
    gt_index = gt.id_to_index()
    pred_index = pred.id_to_index()
    gt_t = gt.t == timepoint
    pred_t = pred.t == timepoint
    ax.scatter(pred.x[pred_t], pred.y[pred_t], s=10, c="#FFCC80", alpha=0.55, label="pred")
    ax.scatter(gt.x[gt_t], gt.y[gt_t], s=36, facecolors="none", edgecolors="#4FC3F7", linewidths=1.2, label="GT")
    missed_mask = np.array([int(nid) in set(missed_gt_ids) for nid in gt.node_id], dtype=bool) & gt_t
    if missed_mask.any():
        ax.scatter(gt.x[missed_mask], gt.y[missed_mask], s=70, marker="x", c="#EF5350", linewidths=1.6, label="missed GT")
    for src, dst in fn_edges:
        if src in gt_index and dst in gt_index:
            i, j = gt_index[src], gt_index[dst]
            if int(gt.t[i]) == timepoint or int(gt.t[j]) == timepoint:
                ax.plot([gt.x[i], gt.x[j]], [gt.y[i], gt.y[j]], color="#EF5350", lw=1.4, alpha=0.9)
    for src, dst in fp_edges:
        if src in pred_index and dst in pred_index:
            i, j = pred_index[src], pred_index[dst]
            if int(pred.t[i]) == timepoint or int(pred.t[j]) == timepoint:
                ax.plot([pred.x[i], pred.x[j]], [pred.y[i], pred.y[j]], color="#FFD54F", lw=1.2, ls="--", alpha=0.9)
    rec = np.array([int(nid) in set(recovered_divisions) for nid in gt.node_id], dtype=bool) & gt_t
    miss_div = np.array([int(nid) in set(missed_divisions) for nid in gt.node_id], dtype=bool) & gt_t
    if rec.any():
        ax.scatter(gt.x[rec], gt.y[rec], s=90, marker="*", c="#66BB6A", label="recovered division")
    if miss_div.any():
        ax.scatter(gt.x[miss_div], gt.y[miss_div], s=90, marker="*", c="#AB47BC", label="missed division")
    ax.set_title(title or f"{pred.dataset} t={timepoint} error overlay")
    ax.set_xlabel("x (voxels)")
    ax.set_ylabel("y (voxels)")
    ax.legend(loc="upper right", fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_error_panel(analysis_summary: dict, path: str | Path, title: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        "missed GT",
        "unmatched pred*",
        "edge TP",
        "edge FP",
        "edge FN / fragments",
        "ID switches",
        "GT divisions",
        "recovered div",
        "missed div",
    ]
    keys = [
        "missed_annotated",
        "unmatched_predictions",
        "tp_edges",
        "fp_edges",
        "fragmented_tracks",
        "identity_switches",
        "gt_divisions",
        "recovered_divisions",
        "missed_divisions",
    ]
    values = [int(analysis_summary.get(k, 0) or 0) for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(range(len(labels)), values, color="#80CBC4")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("count")
    ax.set_title(title or f"{analysis_summary.get('dataset', '')} error counts")
    fig.text(
        0.01,
        0.01,
        "* unmatched predictions are not scored as node false positives (sparse GT).",
        fontsize=7,
    )
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
