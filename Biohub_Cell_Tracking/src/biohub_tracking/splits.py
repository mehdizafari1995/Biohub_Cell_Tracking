from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .graphs import TrackGraph
from .io_geff import load_geff
from .io_zarr import embryo_id, list_zarr_datasets


def embryo_holdout_splits(train_dir: str | Path) -> dict[str, dict[str, list[str]]]:
    """Hold out entire embryos. Never split neighboring frames."""
    datasets = [p.name.removesuffix(".zarr") for p in list_zarr_datasets(train_dir)]
    by_embryo: dict[str, list[str]] = defaultdict(list)
    for name in datasets:
        by_embryo[embryo_id(name)].append(name)
    embryos = sorted(by_embryo)
    splits: dict[str, dict[str, list[str]]] = {}
    if not embryos:
        return splits
    if len(embryos) == 1:
        names = by_embryo[embryos[0]]
        splits["single_embryo_warning"] = {
            "train": names,
            "val": [],
            "note": "Only one embryo is present locally; do not randomly split frames.",
        }
        return splits
    for held in embryos:
        train = [n for e, names in by_embryo.items() if e != held for n in names]
        splits[f"holdout_{held}"] = {"train": sorted(train), "val": sorted(by_embryo[held])}
    return splits


def summarise_geff(graph: TrackGraph) -> dict[str, Any]:
    out = graph.out_edges()
    n_div = sum(1 for children in out.values() if len(children) >= 2)
    times = [int(t) for t in graph.t] if graph.n_nodes else []
    return {
        "dataset": graph.dataset,
        "embryo": embryo_id(graph.dataset),
        "n_nodes": graph.n_nodes,
        "n_edges": graph.n_edges,
        "n_divisions": n_div,
        "t_min": min(times) if times else None,
        "t_max": max(times) if times else None,
        "n_timepoints_annotated": len(set(times)),
        "estimated_n_nodes": graph.estimated_n_nodes,
        "coords_integer": bool(
            graph.n_nodes
            and np_all_int(graph.z)
            and np_all_int(graph.y)
            and np_all_int(graph.x)
            and np_all_int(graph.t)
        ),
    }


def np_all_int(arr) -> bool:
    import numpy as np

    values = np.asarray(arr)
    return bool(np.allclose(values, np.round(values)))


def summarise_geff_dir(geff_dir: str | Path) -> list[dict[str, Any]]:
    folder = Path(geff_dir)
    rows = []
    if not folder.exists():
        return rows
    for path in sorted(folder.glob("*.geff")):
        if not path.is_dir() and not path.exists():
            continue
        try:
            rows.append(summarise_geff(load_geff(path)))
        except Exception as exc:
            rows.append({"dataset": path.name.removesuffix(".geff"), "error": str(exc)})
    return rows


def build_fixed_split(
    summaries: list[dict[str, Any]],
    selected: list[str] | None = None,
    tuning_embryos: list[str] | None = None,
    max_per_embryo: int = 2,
) -> dict[str, Any]:
    """Create a frozen embryo-level split. Crops from one specimen stay together.

    Prefer sequences with annotated divisions. `selected` overrides automatic
    choice once GEFF summaries exist.
    """
    usable = [r for r in summaries if "error" not in r]
    by_embryo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        by_embryo[str(row["embryo"])].append(row)
    embryos = sorted(by_embryo)
    if selected is None:
        selected_rows: list[dict[str, Any]] = []
        for embryo in embryos:
            ranked = sorted(
                by_embryo[embryo],
                key=lambda r: (-int(r.get("n_divisions") or 0), -int(r.get("n_nodes") or 0), r["dataset"]),
            )
            selected_rows.extend(ranked[:max_per_embryo])
        selected_names = [r["dataset"] for r in selected_rows]
    else:
        selected_names = list(selected)
        selected_rows = [r for r in usable if r["dataset"] in selected_names]

    selected_embryos = sorted({embryo_id(n) for n in selected_names})
    if tuning_embryos is None:
        if len(selected_embryos) >= 2:
            tuning_embryos = [selected_embryos[0]]
        else:
            tuning_embryos = selected_embryos
    tuning_set = set(tuning_embryos)
    tuning = sorted(n for n in selected_names if embryo_id(n) in tuning_set)
    validation = sorted(n for n in selected_names if embryo_id(n) not in tuning_set)
    note = []
    if len(selected_embryos) < 2:
        note.append(
            "Only one specimen identifier is present among selected sequences; "
            "tuning and validation cannot be specimen-disjoint."
        )
    if not any(int(r.get("n_divisions") or 0) > 0 for r in selected_rows):
        note.append("No annotated divisions in the selected GEFFs.")
    lookup = {r["dataset"]: r for r in selected_rows}
    return {
        "version": "v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rule": "Hold out entire embryos. Crops from the same specimen stay together. Never split neighboring frames.",
        "specimen_key": "folder prefix before first underscore (e.g. 44b6, 6bba)",
        "tuning_embryos": list(tuning_embryos),
        "validation_embryos": sorted({embryo_id(n) for n in validation}),
        "tuning": tuning,
        "validation": validation,
        "selected": selected_names,
        "geff_summaries": lookup,
        "notes": note,
    }


def save_split_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return path


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
