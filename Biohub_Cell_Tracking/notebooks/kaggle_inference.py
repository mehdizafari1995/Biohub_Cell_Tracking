"""
Kaggle inference notebook: Biohub cell tracking.

Paste into a notebook attached to `biohub-cell-tracking-during-development`.
Internet must be OFF for submit. Output file must be `submission.csv`.

This discovers every `test/*.zarr` directory at runtime. It does not hard-code
the public four-sample list. Hidden test is swapped in on scoring rerun.

This file is a local-or-Kaggle inference path. A run against `data/test` on
this machine is a LOCAL SIMULATION, not a Kaggle leaderboard submission.
"""

from __future__ import annotations

import time
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
SUPPRESS_DUPLICATES = True
MIN_DISTANCE_UM = 4.0
ANISOTROPIC = False
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


def environment_label(root: Path) -> str:
    text = str(root)
    if text.startswith("/kaggle/input"):
        return "kaggle_input"
    return "local_simulation"


def list_test_zarrs(test_dir: Path) -> list[Path]:
    return sorted(p for p in test_dir.iterdir() if p.is_dir() and p.name.endswith(".zarr"))


def anisotropic_sigmas(min_sigma: float, max_sigma: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    xy = float(SCALE_ZYX[2])
    mins = tuple(max(float(min_sigma) * xy / float(s), 0.25) for s in SCALE_ZYX)
    maxs = tuple(max(float(max_sigma) * xy / float(s), mins[i]) for i, s in enumerate(SCALE_ZYX))
    return mins, maxs  # type: ignore[return-value]


def suppress_close(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    if len(coords) == 0 or MIN_DISTANCE_UM <= 0:
        return coords
    phys = coords * SCALE_ZYX
    keep: list[int] = []
    for i, point in enumerate(phys):
        if keep and np.any(np.linalg.norm(phys[keep] - point, axis=1) < MIN_DISTANCE_UM):
            continue
        keep.append(i)
    return coords[np.asarray(keep, dtype=np.int64)]


def detect(frame: np.ndarray) -> np.ndarray:
    lo = float(frame.min())
    hi = float(frame.max())
    if hi <= lo:
        return np.zeros((0, 3), dtype=np.float64)
    norm = (frame.astype(np.float32) - lo) / (hi - lo)
    min_sigma: float | tuple[float, float, float] = BLOB_MIN_SIGMA
    max_sigma: float | tuple[float, float, float] = BLOB_MAX_SIGMA
    if ANISOTROPIC:
        min_sigma, max_sigma = anisotropic_sigmas(BLOB_MIN_SIGMA, BLOB_MAX_SIGMA)
    blobs = blob_dog(norm, min_sigma=min_sigma, max_sigma=max_sigma, threshold=BLOB_THRESHOLD)
    coords = blobs[:, :3] if len(blobs) else np.zeros((0, 3), dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    if SUPPRESS_DUPLICATES:
        coords = suppress_close(coords)
    return coords


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


def track_dataset(zarr_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    group = zarr.open_group(zarr_path, mode="r")
    volume = group["0"]
    name = zarr_path.name.removesuffix(".zarr")
    shape = tuple(int(x) for x in volume.shape)
    if len(shape) != 4:
        raise ValueError(f"{name}: expected TZYX volume, got shape {shape}")
    n_time, n_z, n_y, n_x = shape
    node_rows = []
    edge_rows = []
    prev_coords = None
    prev_ids = None
    next_id = 1
    t0 = time.perf_counter()
    last_t = -1
    for t in range(n_time):
        if t <= last_t:
            raise ValueError(f"{name}: time indices must increase (got {t} after {last_t})")
        last_t = t
        coords = detect(np.asarray(volume[t]))
        ids = list(range(next_id, next_id + len(coords)))
        next_id += len(coords)
        for nid, (z, y, x) in zip(ids, coords):
            zi, yi, xi = int(round(float(z))), int(round(float(y))), int(round(float(x)))
            node_rows.append(
                {
                    "dataset": name,
                    "row_type": "node",
                    "node_id": int(nid),
                    "t": int(t),
                    "z": zi,
                    "y": yi,
                    "x": xi,
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
    stats = {
        "dataset": name,
        "shape": shape,
        "n_nodes": len(node_rows),
        "n_edges": len(edge_rows),
        "runtime_sec": time.perf_counter() - t0,
        "bounds_z": (0, n_z - 1),
        "bounds_y": (0, n_y - 1),
        "bounds_x": (0, n_x - 1),
    }
    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows), stats


def validate_graph_frames(nodes: pd.DataFrame, edges: pd.DataFrame, shape: tuple[int, int, int, int], name: str) -> list[str]:
    errors: list[str] = []
    n_time, n_z, n_y, n_x = shape
    if nodes.empty:
        errors.append(f"{name}: no predicted nodes")
        return errors
    if nodes["node_id"].duplicated().any():
        errors.append(f"{name}: duplicate node_id")
    ids = set(nodes["node_id"].tolist())
    if (nodes["t"] < 0).any() or (nodes["t"] >= n_time).any():
        errors.append(f"{name}: node t outside [0, {n_time - 1}]")
    if (nodes["z"] < 0).any() or (nodes["z"] >= n_z).any():
        errors.append(f"{name}: node z outside [0, {n_z - 1}]")
    if (nodes["y"] < 0).any() or (nodes["y"] >= n_y).any():
        errors.append(f"{name}: node y outside [0, {n_y - 1}]")
    if (nodes["x"] < 0).any() or (nodes["x"] >= n_x).any():
        errors.append(f"{name}: node x outside [0, {n_x - 1}]")
    if not nodes["t"].is_monotonic_increasing:
        # nodes are written in time order per dataset; warn if not
        if nodes["t"].min() > nodes["t"].max():
            errors.append(f"{name}: time direction reversed")
    for _, row in edges.iterrows():
        if int(row["source_id"]) not in ids or int(row["target_id"]) not in ids:
            errors.append(f"{name}: edge references missing node_id")
            break
    if not edges.empty:
        src = edges["source_id"].map(dict(zip(nodes["node_id"], nodes["t"])))
        dst = edges["target_id"].map(dict(zip(nodes["node_id"], nodes["t"])))
        if ((dst - src) < 0).any():
            errors.append(f"{name}: backward-in-time edges present")
    return errors


def validate_submission(df: pd.DataFrame, required_datasets: list[str]) -> list[str]:
    errors: list[str] = []
    if list(df.columns) != COLUMNS:
        errors.append(f"columns must be {COLUMNS}")
    if df["id"].tolist() != list(range(len(df))):
        errors.append("id must be consecutive integers starting at 0")
    if set(df["row_type"].unique()) - {"node", "edge"}:
        errors.append("row_type must be node or edge")
    nodes = df[df["row_type"] == "node"]
    edges = df[df["row_type"] == "edge"]
    if (nodes[["source_id", "target_id"]] != -1).any().any():
        errors.append("node rows must use source_id=target_id=-1")
    if (edges[["node_id", "t", "z", "y", "x"]] != -1).any().any():
        errors.append("edge rows must use node_id,t,z,y,x=-1")
    present = set(df["dataset"].astype(str))
    missing = [d for d in required_datasets if d not in present]
    extra = sorted(present - set(required_datasets))
    if missing:
        errors.append(f"missing datasets: {missing}")
    if extra:
        errors.append(f"unexpected datasets: {extra}")
    return errors


def main() -> None:
    root = find_root()
    label = environment_label(root)
    test_dir = root / "test"
    zarrs = list_test_zarrs(test_dir)
    if not zarrs:
        raise SystemExit(f"No test/*.zarr under {test_dir}")
    print(f"environment={label} root={root} n_test={len(zarrs)}", flush=True)
    print("datasets=" + ",".join(p.name.removesuffix('.zarr') for p in zarrs), flush=True)
    frames = []
    required = []
    runtimes = []
    errors: list[str] = []
    t0 = time.perf_counter()
    for zarr_path in zarrs:
        nodes, edges, stats = track_dataset(zarr_path)
        required.append(stats["dataset"])
        runtimes.append(stats)
        errors.extend(validate_graph_frames(nodes, edges, tuple(stats["shape"]), stats["dataset"]))
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
    errors.extend(validate_submission(df, required))
    if errors:
        raise SystemExit("submission invalid:\n" + "\n".join(errors))
    df.to_csv("submission.csv", index=False)
    elapsed = time.perf_counter() - t0
    n_complete_proxy = max((r["shape"][0] for r in runtimes), default=0)
    mean_per_frame = elapsed / max(sum(r["shape"][0] for r in runtimes), 1)
    estimated_hidden = mean_per_frame * 100 * 200
    print(
        {
            "environment": label,
            "rows": len(df),
            "datasets": df["dataset"].nunique(),
            "runtime_sec": elapsed,
            "sec_per_frame": mean_per_frame,
            "longest_T": n_complete_proxy,
            "est_hidden_200x100_hours": estimated_hidden / 3600,
            "est_with_2x_margin_hours": 2 * estimated_hidden / 3600,
            "kaggle_limit_hours": 12,
            "note": "LOCAL SIMULATION" if label == "local_simulation" else "running under /kaggle/input",
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
