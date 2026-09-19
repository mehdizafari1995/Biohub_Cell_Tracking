"""
Kaggle kernel: Biohub cell tracking baseline_v1

anisotropic DoG + 4 µm NMS + greedy linking.

Attach competition `biohub-cell-tracking-during-development`. Internet OFF.
Discovers every test/*.zarr at runtime. Writes /kaggle/working/submission.csv.
This file is the kernel source. A local run is not a Kaggle execution.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from skimage.feature import blob_dog

CONFIG_LABEL = "baseline_v1"
SCALE_ZYX = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
MAX_LINK_UM = 15.0
BLOB_MIN_SIGMA = 2.0
BLOB_MAX_SIGMA = 5.0
BLOB_THRESHOLD = 0.05
SUPPRESS_DUPLICATES = True
MIN_DISTANCE_UM = 4.0
ANISOTROPIC = True
ALLOW_DIVISIONS = False
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
CANDIDATE_ROOTS = [
    Path("/kaggle/input/biohub-cell-tracking-during-development"),
    Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
    Path("data"),
]
if os.environ.get("BIOHUB_DATA_ROOT"):
    CANDIDATE_ROOTS.insert(0, Path(os.environ["BIOHUB_DATA_ROOT"]))


def ensure_blosc_decoder() -> None:
    """Install a local numcodecs wheel if Kaggle has neither zarr nor a blosc decoder."""
    import importlib.util
    import subprocess

    if any(importlib.util.find_spec(name) for name in ("zarr", "imagecodecs", "numcodecs", "blosc2")):
        return
    roots = [Path.cwd(), Path("/kaggle/working"), Path("/kaggle/usr/lib"), Path("/tmp")]
    wheels: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        wheels.extend(sorted(root.glob("wheels/numcodecs*.whl")))
        wheels.extend(sorted(root.glob("numcodecs*.whl")))
    if not wheels:
        return
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(wheels[0])],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def find_root() -> Path:
    for path in CANDIDATE_ROOTS:
        if (path / "test").is_dir():
            return path
    raise FileNotFoundError("Could not find the competition test directory")


def environment_label(root: Path) -> str:
    return "kaggle_input" if str(root).startswith("/kaggle/input") else "local_simulation"


def output_dir() -> Path:
    kaggle = Path("/kaggle/working")
    return kaggle if kaggle.is_dir() else Path(".")


def list_test_zarrs(test_dir: Path) -> list[Path]:
    return sorted(p for p in test_dir.iterdir() if p.is_dir() and p.name.endswith(".zarr"))


def decode_blosc(payload: bytes) -> bytes:
    errors: list[str] = []
    try:
        import imagecodecs

        return bytes(imagecodecs.blosc_decode(payload))
    except Exception as exc:
        errors.append(f"imagecodecs:{type(exc).__name__}")
    try:
        from numcodecs import Blosc

        return bytes(Blosc().decode(payload))
    except Exception as exc:
        errors.append(f"numcodecs:{type(exc).__name__}")
    try:
        import blosc2

        return bytes(blosc2.decompress(payload))
    except Exception as exc:
        errors.append(f"blosc2:{type(exc).__name__}")
    raise ModuleNotFoundError(
        "Cannot decode blosc zarr chunks (zarr/imagecodecs/numcodecs/blosc2 missing). " + ",".join(errors)
    )


class ZarrV3FrameArray:
    """Read OME-Zarr v3 TZYX volumes one timepoint at a time without the zarr package."""

    def __init__(self, array_dir: Path):
        meta = json.loads((array_dir / "zarr.json").read_text())
        self.array_dir = array_dir
        self.shape = tuple(int(x) for x in meta["shape"])
        self.dtype = np.dtype(meta["data_type"])
        self.chunk_shape = tuple(int(x) for x in meta["chunk_grid"]["configuration"]["chunk_shape"])
        self.codecs = list(meta.get("codecs") or [])
        encoding = meta.get("chunk_key_encoding") or {}
        self.separator = str((encoding.get("configuration") or {}).get("separator") or "/")

    def _chunk_path(self, t: int) -> Path:
        key = self.separator.join(["c", str(int(t)), "0", "0", "0"])
        return self.array_dir / key

    def __getitem__(self, t: int) -> np.ndarray:
        path = self._chunk_path(int(t))
        if not path.exists():
            raise FileNotFoundError(path)
        raw = path.read_bytes()
        for codec in reversed(self.codecs):
            name = str(codec.get("name") or "")
            if name == "blosc":
                raw = decode_blosc(raw)
            elif name == "bytes":
                continue
            else:
                raise ValueError(f"unsupported zarr codec {name!r}")
        arr = np.frombuffer(raw, dtype=self.dtype)
        expected = int(np.prod(self.chunk_shape))
        if arr.size != expected:
            raise ValueError(f"{path}: decoded {arr.size} values, expected {expected}")
        frame = arr.reshape(self.chunk_shape)
        if frame.shape[0] == 1:
            frame = frame[0]
        return np.asarray(frame)


def open_volume_array(zarr_path: Path):
    try:
        import zarr

        return zarr.open_group(zarr_path, mode="r")["0"]
    except Exception:
        return ZarrV3FrameArray(Path(zarr_path) / "0")


def sample_submission_datasets(root: Path) -> list[str]:
    path = root / "sample_submission.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "dataset" not in df.columns:
        return []
    return sorted({str(x) for x in df["dataset"].tolist()})


def anisotropic_sigmas(min_sigma: float, max_sigma: float):
    xy = float(SCALE_ZYX[2])
    mins = tuple(max(float(min_sigma) * xy / float(s), 0.25) for s in SCALE_ZYX)
    maxs = tuple(max(float(max_sigma) * xy / float(s), mins[i]) for i, s in enumerate(SCALE_ZYX))
    return mins, maxs


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
    min_sigma: float | tuple = BLOB_MIN_SIGMA
    max_sigma: float | tuple = BLOB_MAX_SIGMA
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


def peak_memory_mb() -> float:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if usage > 10_000_000:
            return usage / (1024 * 1024)
        return usage / 1024
    except Exception:
        return float("nan")


def package_versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "scikit-image", "pandas", "zarr", "imagecodecs", "numcodecs", "blosc2"):
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = "missing"
    return out


def config_payload() -> dict:
    return {
        "label": CONFIG_LABEL,
        "scale_zyx": SCALE_ZYX.tolist(),
        "max_link_um": MAX_LINK_UM,
        "blob_min_sigma": BLOB_MIN_SIGMA,
        "blob_max_sigma": BLOB_MAX_SIGMA,
        "blob_threshold": BLOB_THRESHOLD,
        "suppress_duplicates": SUPPRESS_DUPLICATES,
        "min_distance_um": MIN_DISTANCE_UM,
        "anisotropic": ANISOTROPIC,
        "allow_divisions": ALLOW_DIVISIONS,
        "linking": "greedy",
    }


def config_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def track_dataset(zarr_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    volume = open_volume_array(zarr_path)
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
    n_div = 0
    if edge_rows:
        counts: dict[int, int] = {}
        for row in edge_rows:
            counts[int(row["source_id"])] = counts.get(int(row["source_id"]), 0) + 1
        n_div = sum(1 for v in counts.values() if v >= 2)
    stats = {
        "dataset": name,
        "shape": list(shape),
        "n_nodes": len(node_rows),
        "n_edges": len(edge_rows),
        "n_predicted_divisions": n_div,
        "runtime_sec": time.perf_counter() - t0,
    }
    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows), stats


def audit_submission(
    df: pd.DataFrame,
    discovered: list[str],
    shapes: dict[str, tuple[int, int, int, int]],
    sample_names: list[str],
) -> list[str]:
    errors: list[str] = []
    if df is None or len(df) == 0:
        return ["submission.csv is empty"]
    if list(df.columns) != COLUMNS:
        errors.append(f"columns must be {COLUMNS}, got {list(df.columns)}")
        return errors
    if df["id"].tolist() != list(range(len(df))):
        errors.append("id must be consecutive integers starting at 0")
    if set(df["row_type"].astype(str).unique()) - {"node", "edge"}:
        errors.append("row_type must be node or edge")
    present = set(df["dataset"].astype(str))
    missing_disc = [d for d in discovered if d not in present]
    extra = sorted(present - set(discovered))
    if missing_disc:
        errors.append(f"missing discovered datasets: {missing_disc}")
    if extra:
        errors.append(f"unexpected datasets: {extra}")
    sample_in_test = [d for d in sample_names if d in discovered]
    sample_missing = [d for d in sample_in_test if d not in present]
    if sample_missing:
        errors.append(f"sample_submission datasets present in test/ but missing from CSV: {sample_missing}")
    nodes = df[df["row_type"] == "node"]
    edges = df[df["row_type"] == "edge"]
    if (nodes[["source_id", "target_id"]] != -1).any().any():
        errors.append("node rows must use source_id=target_id=-1")
    if len(edges) and (edges[["node_id", "t", "z", "y", "x"]] != -1).any().any():
        errors.append("edge rows must use node_id,t,z,y,x=-1")
    for dataset, group in df.groupby("dataset"):
        name = str(dataset)
        gnodes = group[group["row_type"] == "node"]
        gedges = group[group["row_type"] == "edge"]
        if gnodes["node_id"].duplicated().any():
            errors.append(f"{name}: duplicate node_id")
        if len(gedges):
            dup_e = gedges.duplicated(subset=["source_id", "target_id"]).any()
            if dup_e:
                errors.append(f"{name}: duplicate edges")
        ids = set(gnodes["node_id"].tolist())
        for col in ("t", "z", "y", "x"):
            vals = pd.to_numeric(gnodes[col], errors="coerce")
            if vals.isna().any() or np.isinf(vals.to_numpy(dtype=float)).any():
                errors.append(f"{name}: non-finite {col}")
            if not np.allclose(vals.to_numpy(dtype=float), np.round(vals.to_numpy(dtype=float))):
                errors.append(f"{name}: {col} is not integer-valued")
        shape = shapes.get(name)
        if shape is not None:
            n_time, n_z, n_y, n_x = shape
            if (gnodes["t"] < 0).any() or (gnodes["t"] >= n_time).any():
                errors.append(f"{name}: t outside [0, {n_time - 1}]")
            if (gnodes["z"] < 0).any() or (gnodes["z"] >= n_z).any():
                errors.append(f"{name}: z outside bounds")
            if (gnodes["y"] < 0).any() or (gnodes["y"] >= n_y).any():
                errors.append(f"{name}: y outside bounds")
            if (gnodes["x"] < 0).any() or (gnodes["x"] >= n_x).any():
                errors.append(f"{name}: x outside bounds")
        t_by_id = dict(zip(gnodes["node_id"].astype("int64"), gnodes["t"].astype("int64")))
        for _, row in gedges.iterrows():
            s, tgt = int(row["source_id"]), int(row["target_id"])
            if s not in ids or tgt not in ids:
                errors.append(f"{name}: edge {s}->{tgt} references missing node_id")
                break
            if t_by_id[tgt] <= t_by_id[s]:
                errors.append(f"{name}: source time does not precede target time ({s}->{tgt})")
                break
    return errors


def main() -> None:
    ensure_blosc_decoder()
    root = find_root()
    label = environment_label(root)
    out = output_dir()
    test_dir = root / "test"
    zarrs = list_test_zarrs(test_dir)
    if not zarrs:
        raise SystemExit(f"No test/*.zarr under {test_dir}")
    sample_names = sample_submission_datasets(root)
    discovered = [p.name.removesuffix(".zarr") for p in zarrs]
    cfg = config_payload()
    print(f"environment={label} root={root} config={CONFIG_LABEL} n_test={len(zarrs)}", flush=True)
    print("datasets=" + ",".join(discovered), flush=True)
    frames = []
    runtimes = []
    shapes: dict[str, tuple[int, int, int, int]] = {}
    t0 = time.perf_counter()
    for zarr_path in zarrs:
        nodes, edges, stats = track_dataset(zarr_path)
        runtimes.append(stats)
        shapes[stats["dataset"]] = tuple(stats["shape"])  # type: ignore[arg-type]
        frames.extend([nodes, edges])
        print(
            f"DONE {stats['dataset']} nodes={stats['n_nodes']} edges={stats['n_edges']} "
            f"divisions={stats['n_predicted_divisions']} sec={stats['runtime_sec']:.1f}",
            flush=True,
        )
    nonempty = [f for f in frames if len(f)]
    if not nonempty:
        raise SystemExit("no predicted nodes or edges")
    df = pd.concat(nonempty, ignore_index=True)
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
    errors = audit_submission(df, discovered, shapes, sample_names)
    dest = out / "submission.csv"
    df.to_csv(dest, index=False)
    size = dest.stat().st_size
    elapsed = time.perf_counter() - t0
    memory = peak_memory_mb()
    n_nodes = int((df["row_type"] == "node").sum())
    n_edges = int((df["row_type"] == "edge").sum())
    n_div = 0
    if n_edges:
        src_counts = df.loc[df["row_type"] == "edge"].groupby(["dataset", "source_id"]).size()
        n_div = int((src_counts >= 2).sum())
    n_frames = sum(int(r["shape"][0]) for r in runtimes)
    summary = {
        "environment": label,
        "config_label": CONFIG_LABEL,
        "config_hash": config_hash(cfg),
        "config": cfg,
        "package_versions": package_versions(),
        "datasets": discovered,
        "sample_submission_datasets": sample_names,
        "per_dataset": runtimes,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_predicted_divisions": n_div,
        "rows": int(len(df)),
        "submission_path": str(dest),
        "submission_bytes": int(size),
        "submission_empty": size == 0,
        "runtime_sec": elapsed,
        "sec_per_frame": elapsed / max(n_frames, 1),
        "peak_memory_mb": memory,
        "audit_errors": errors,
        "audit_ok": not errors,
        "note": "LOCAL SIMULATION" if label == "local_simulation" else "Kaggle kernel execution (public test unless this is a scoring rerun)",
    }
    (out / "run_manifest.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in (
        "environment", "config_label", "config_hash", "n_nodes", "n_edges",
        "n_predicted_divisions", "rows", "submission_bytes", "runtime_sec",
        "peak_memory_mb", "audit_ok", "audit_errors", "note",
    )}, indent=2), flush=True)
    if errors:
        raise SystemExit("submission invalid:\n" + "\n".join(errors))
    if size == 0:
        raise SystemExit("submission.csv is empty")


if __name__ == "__main__":
    main()
