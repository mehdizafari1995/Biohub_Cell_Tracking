from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr

from .graphs import TrackGraph


def _open_array(path: Path) -> Any:
    return zarr.open_array(path, mode="r")


def load_geff(geff_path: str | Path) -> TrackGraph:
    geff_path = Path(geff_path)
    group = zarr.open_group(geff_path, mode="r")
    attrs = dict(group.attrs)
    geff_meta = attrs.get("geff") or {}
    extra = geff_meta.get("extra") or {}
    estimated = extra.get("estimated_number_of_nodes")
    node_ids = np.asarray(_open_array(geff_path / "nodes" / "ids")[:])
    props = {}
    for key in ("t", "z", "y", "x"):
        props[key] = np.asarray(_open_array(geff_path / "nodes" / "props" / key / "values")[:])
    edges = np.asarray(_open_array(geff_path / "edges" / "ids")[:], dtype=np.int64)
    if edges.ndim == 1:
        edges = edges.reshape(-1, 2)
    name = geff_path.name.removesuffix(".geff")
    return TrackGraph(
        dataset=name,
        node_id=np.asarray(node_ids, dtype=np.int64),
        t=np.asarray(props["t"], dtype=np.int64),
        z=np.asarray(props["z"], dtype=np.int64),
        y=np.asarray(props["y"], dtype=np.int64),
        x=np.asarray(props["x"], dtype=np.int64),
        edges=edges,
        estimated_n_nodes=float(estimated) if estimated is not None else None,
        metadata={"geff": geff_meta},
    )


def find_geff(zarr_or_name: str | Path, search_dirs: list[Path] | None = None) -> Path | None:
    path = Path(zarr_or_name)
    name = path.name.removesuffix(".zarr").removesuffix(".geff")
    candidates = []
    if path.suffix == ".zarr":
        candidates.append(path.with_suffix(".geff"))
        candidates.append(path.parent / f"{name}.geff")
    else:
        candidates.append(path if path.suffix == ".geff" else Path(f"{path}.geff"))
    for folder in search_dirs or []:
        candidates.append(folder / f"{name}.geff")
    for cand in candidates:
        if cand.exists():
            return cand
    return None
