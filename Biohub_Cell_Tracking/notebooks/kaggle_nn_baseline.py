"""
Kaggle notebook: Biohub cell tracking nearest-neighbor baseline.

Copy this file into a Kaggle notebook attached to
`biohub-cell-tracking-during-development`. Internet must be OFF for submit.

This is a CPU baseline: per-frame blob_dog detection + nearest-neighbor linking.
It follows the documented getting-started recipe, not the organizer UNet.

Public getting-started kernel:
https://www.kaggle.com/code/inversion/cell-tracking-getting-started-w-nearest-neighbor
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.spatial.distance import cdist
from skimage.feature import blob_dog

CANDIDATE_ROOTS = [
    Path("/kaggle/input/biohub-cell-tracking-during-development"),
    Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
    Path("data"),
]
SCALE_ZYX = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
MAX_LINK_UM = 15.0
BLOB_MIN_SIGMA = 2.0
BLOB_MAX_SIGMA = 5.0
BLOB_THRESHOLD = 0.05
COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]


def find_root() -> Path:
    for path in CANDIDATE_ROOTS:
        if (path / "test").is_dir():
            return path
    raise FileNotFoundError("Could not find the competition test directory")


def detect(frame: np.ndarray) -> np.ndarray:
    lo = float(frame.min())
    hi = float(frame.max())
    if hi <= lo:
        return np.zeros((0, 3), dtype=np.float64)
    norm = (frame.astype(np.float32) - lo) / (hi - lo)
    blobs = blob_dog(norm, min_sigma=BLOB_MIN_SIGMA, max_sigma=BLOB_MAX_SIGMA, threshold=BLOB_THRESHOLD)
    return blobs[:, :3] if len(blobs) else np.zeros((0, 3), dtype=np.float64)


def link(prev: np.ndarray, curr: np.ndarray) -> list[tuple[int, int]]:
    if len(prev) == 0 or len(curr) == 0:
        return []
    dists = cdist(prev * SCALE_ZYX, curr * SCALE_ZYX)
    used = set()
    pairs = []
    for i in np.argsort(dists.min(axis=1)):
        for j in np.argsort(dists[i]):
            if int(j) in used:
                continue
            if dists[i, j] <= MAX_LINK_UM:
                pairs.append((int(i), int(j)))
                used.add(int(j))
            break
    return pairs


def track_dataset(zarr_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = zarr.open_group(zarr_path, mode="r")
    volume = group["0"]
    name = zarr_path.name.removesuffix(".zarr")
    n_time = int(volume.shape[0])
    node_rows = []
    edge_rows = []
    prev_coords = None
    prev_ids = None
    next_id = 1
    for t in range(n_time):
        coords = detect(np.asarray(volume[t]))
        ids = list(range(next_id, next_id + len(coords)))
        next_id += len(coords)
        for nid, (z, y, x) in zip(ids, coords):
            node_rows.append(
                {
                    "dataset": name,
                    "row_type": "node",
                    "node_id": int(nid),
                    "t": int(t),
                    "z": int(round(float(z))),
                    "y": int(round(float(y))),
                    "x": int(round(float(x))),
                    "source_id": -1,
                    "target_id": -1,
                }
            )
        if prev_coords is not None:
            for i, j in link(prev_coords, coords):
                edge_rows.append(
                    {
                        "dataset": name,
                        "row_type": "edge",
                        "node_id": -1,
                        "t": -1,
                        "z": -1,
                        "y": -1,
                        "x": -1,
                        "source_id": int(prev_ids[i]),
                        "target_id": int(ids[j]),
                    }
                )
        prev_coords, prev_ids = coords, ids
        print(f"{name} t={t}/{n_time - 1} detections={len(coords)}", flush=True)
    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


def main() -> None:
    root = find_root()
    test_dir = root / "test"
    frames = []
    for zarr_path in sorted(p for p in test_dir.iterdir() if p.is_dir() and p.name.endswith(".zarr")):
        nodes, edges = track_dataset(zarr_path)
        frames.extend([nodes, edges])
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    df.insert(0, "id", range(len(df)))
    df = df[COLUMNS].astype(
        {
            "id": "int64",
            "node_id": "int64",
            "t": "int64",
            "z": "int64",
            "y": "int64",
            "x": "int64",
            "source_id": "int64",
            "target_id": "int64",
        }
    )
    df.to_csv("submission.csv", index=False)
    print(f"wrote submission.csv rows={len(df)} datasets={df['dataset'].nunique()}")


if __name__ == "__main__":
    main()
