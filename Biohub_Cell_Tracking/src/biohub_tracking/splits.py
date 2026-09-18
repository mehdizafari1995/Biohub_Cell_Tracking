from __future__ import annotations

from collections import defaultdict
from pathlib import Path

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
