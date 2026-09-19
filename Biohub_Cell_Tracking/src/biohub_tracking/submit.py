from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .graphs import TrackGraph

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


def graph_to_rows(graph: TrackGraph) -> pd.DataFrame:
    nodes = pd.DataFrame(
        {
            "dataset": graph.dataset,
            "row_type": "node",
            "node_id": graph.node_id.astype("int64"),
            "t": graph.t.astype("int64"),
            "z": graph.z.astype("int64"),
            "y": graph.y.astype("int64"),
            "x": graph.x.astype("int64"),
            "source_id": np_neg(len(graph.node_id)),
            "target_id": np_neg(len(graph.node_id)),
        }
    )
    n_edges = graph.n_edges
    edges = pd.DataFrame(
        {
            "dataset": graph.dataset,
            "row_type": "edge",
            "node_id": np_neg(n_edges),
            "t": np_neg(n_edges),
            "z": np_neg(n_edges),
            "y": np_neg(n_edges),
            "x": np_neg(n_edges),
            "source_id": graph.edges[:, 0].astype("int64") if n_edges else pd.Series(dtype="int64"),
            "target_id": graph.edges[:, 1].astype("int64") if n_edges else pd.Series(dtype="int64"),
        }
    )
    return pd.concat([nodes, edges], ignore_index=True)


def np_neg(n: int) -> pd.Series:
    return pd.Series([-1] * int(n), dtype="int64")


def graphs_to_submission(graphs: list[TrackGraph]) -> pd.DataFrame:
    frames = [graph_to_rows(g) for g in graphs if g.n_nodes or g.n_edges]
    if not frames:
        df = pd.DataFrame(columns=COLUMNS[1:])
    else:
        df = pd.concat(frames, ignore_index=True)
    df.insert(0, "id", range(len(df)))
    df = df[COLUMNS]
    df = df.astype(
        {
            "id": "int64",
            "dataset": "string",
            "row_type": "string",
            "node_id": "int64",
            "t": "int64",
            "z": "int64",
            "y": "int64",
            "x": "int64",
            "source_id": "int64",
            "target_id": "int64",
        }
    )
    return df


def validate_submission(
    df: pd.DataFrame,
    required_datasets: list[str] | None = None,
    volume_shapes: dict[str, tuple[int, int, int, int]] | None = None,
) -> list[str]:
    errors: list[str] = []
    if df is None or len(df) == 0:
        return ["submission.csv is empty"]
    missing_cols = [c for c in COLUMNS if c not in df.columns]
    if missing_cols:
        return [f"missing columns: {missing_cols}"]
    if list(df.columns[: len(COLUMNS)]) != COLUMNS:
        errors.append(f"column order should be {COLUMNS}")
    if df["id"].tolist() != list(range(len(df))):
        errors.append("id must be consecutive integers starting at 0")
    if set(df["row_type"].astype(str).unique()) - {"node", "edge"}:
        errors.append(f"invalid row_type values: {sorted(df['row_type'].unique())}")
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
            errors.append(f"{name}: duplicate node_id values")
        if len(gedges) and gedges.duplicated(subset=["source_id", "target_id"]).any():
            errors.append(f"{name}: duplicate edges")
        node_ids = set(gnodes["node_id"].tolist())
        for col in ("t", "z", "y", "x"):
            vals = pd.to_numeric(gnodes[col], errors="coerce")
            arr = vals.to_numpy(dtype=float)
            if vals.isna().any() or np.isinf(arr).any():
                errors.append(f"{name}: non-finite {col}")
            elif not np.allclose(arr, np.round(arr)):
                errors.append(f"{name}: {col} is not integer-valued")
        shape = (volume_shapes or {}).get(name)
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
            source_id, target_id = int(row["source_id"]), int(row["target_id"])
            if source_id not in node_ids or target_id not in node_ids:
                errors.append(f"{name}: edge {source_id}->{target_id} references missing node_id")
                break
            if t_by_id[target_id] <= t_by_id[source_id]:
                errors.append(f"{name}: source time does not precede target time ({source_id}->{target_id})")
                break
    if required_datasets is not None:
        present = set(df["dataset"].astype(str))
        missing = [d for d in required_datasets if d not in present]
        extra = sorted(present - set(required_datasets))
        if missing:
            errors.append(f"missing datasets: {missing}")
        if extra:
            errors.append(f"unexpected datasets: {extra}")
    return errors


def write_submission(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def read_submission(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def submission_to_graphs(df: pd.DataFrame) -> list[TrackGraph]:
    graphs = []
    for dataset, group in df.groupby("dataset", sort=True):
        nodes = group[group["row_type"] == "node"]
        edges = group[group["row_type"] == "edge"]
        graphs.append(
            TrackGraph(
                dataset=str(dataset),
                node_id=nodes["node_id"].to_numpy(dtype="int64"),
                t=nodes["t"].to_numpy(dtype="int64"),
                z=nodes["z"].to_numpy(dtype="int64"),
                y=nodes["y"].to_numpy(dtype="int64"),
                x=nodes["x"].to_numpy(dtype="int64"),
                edges=edges[["source_id", "target_id"]].to_numpy(dtype="int64")
                if len(edges)
                else [],
            )
        )
    return graphs
