from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import zarr

DEFAULT_SCALE_TZYX = (1.0, 1.625, 0.40625, 0.40625)


@dataclass
class VolumeMeta:
    path: Path
    name: str
    shape: tuple[int, int, int, int]
    dtype: str
    scale_tzyx: tuple[float, float, float, float]
    time_unit: str | None
    space_unit: str | None
    quantiles: dict[str, float]
    available_timepoints: list[int]
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def n_timepoints(self) -> int:
        return int(self.shape[0])

    @property
    def scale_zyx(self) -> tuple[float, float, float]:
        return self.scale_tzyx[1], self.scale_tzyx[2], self.scale_tzyx[3]


def discover_competition_root(explicit: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            Path("/kaggle/input/biohub-cell-tracking-during-development"),
            Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
            Path("data"),
        ]
    )
    for path in candidates:
        if (path / "train").is_dir() or (path / "test").is_dir() or (path / "sample_submission.csv").exists():
            return path.resolve()
        if path.is_dir() and any(path.glob("*.zarr")):
            return path.resolve()
    raise FileNotFoundError(
        "Could not find the competition data root. Pass --data-dir, or place files under data/."
    )


def list_zarr_datasets(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_dir() and p.name.endswith(".zarr"))


def embryo_id(dataset_name: str) -> str:
    return dataset_name.split("_", 1)[0]


def _parse_scale(attrs: dict[str, Any]) -> tuple[tuple[float, float, float, float], str | None, str | None]:
    multi = attrs.get("multiscales") or []
    if not multi:
        return DEFAULT_SCALE_TZYX, "second", "micrometer"
    axes = multi[0].get("axes") or []
    transforms = multi[0].get("datasets", [{}])[0].get("coordinateTransformations") or []
    scale = None
    for transform in transforms:
        if transform.get("type") == "scale":
            scale = tuple(float(x) for x in transform["scale"])
            break
    if scale is None:
        scale = DEFAULT_SCALE_TZYX
    if len(scale) != 4:
        raise ValueError(f"Expected 4-element TZYX scale, got {scale}")
    time_unit = next((a.get("unit") for a in axes if a.get("name") in {"T", "t"}), None)
    space_unit = next((a.get("unit") for a in axes if a.get("name") in {"Z", "z"}), None)
    return scale, time_unit, space_unit


def _available_timepoints(zarr_path: Path, n_time: int) -> list[int]:
    chunk_root = zarr_path / "0" / "c"
    if not chunk_root.exists():
        return list(range(n_time))
    found = sorted(int(p.name) for p in chunk_root.iterdir() if p.name.isdigit())
    return found if found else list(range(n_time))


def open_volume(zarr_path: str | Path) -> tuple[Any, VolumeMeta]:
    zarr_path = Path(zarr_path)
    group = zarr.open_group(zarr_path, mode="r")
    array = group["0"]
    attrs = dict(group.attrs)
    scale, time_unit, space_unit = _parse_scale(attrs)
    quantiles = ((attrs.get("image_statistics") or {}).get("quantiles")) or {}
    quantiles = {str(k): float(v) for k, v in quantiles.items()}
    shape = tuple(int(x) for x in array.shape)
    meta = VolumeMeta(
        path=zarr_path,
        name=zarr_path.name.removesuffix(".zarr"),
        shape=shape,
        dtype=str(array.dtype),
        scale_tzyx=scale,
        time_unit=time_unit,
        space_unit=space_unit,
        quantiles=quantiles,
        available_timepoints=_available_timepoints(zarr_path, shape[0]),
        attrs=attrs,
    )
    return array, meta


def iter_timepoints(
    array: Any,
    meta: VolumeMeta,
    start: int = 0,
    max_timepoints: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    times = [t for t in meta.available_timepoints if t >= start]
    if max_timepoints is not None:
        times = times[: int(max_timepoints)]
    for t in times:
        yield t, np.asarray(array[t])
