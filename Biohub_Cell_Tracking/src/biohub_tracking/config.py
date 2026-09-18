from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SCALE_ZYX = (1.625, 0.40625, 0.40625)
DEFAULT_MATCH_DISTANCE_UM = 7.0
DEFAULT_LINK_DISTANCE_UM = 15.0


@dataclass(frozen=True)
class DetectionConfig:
    method: str = "blob_dog"
    min_sigma: float = 2.0
    max_sigma: float = 5.0
    threshold: float = 0.05
    normalize: str = "minmax"
    min_distance_um: float = 4.0
    suppress_duplicates: bool = False
    anisotropic: bool = False


@dataclass(frozen=True)
class LinkingConfig:
    method: str = "greedy"
    max_distance_um: float = DEFAULT_LINK_DISTANCE_UM
    allow_divisions: bool = False
    division_gap_um: float = 8.0


@dataclass(frozen=True)
class PipelineConfig:
    scale_zyx: tuple[float, float, float] = DEFAULT_SCALE_ZYX
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    linking: LinkingConfig = field(default_factory=LinkingConfig)
    seed: int = 0
    max_timepoints: int | None = None
    timepoint_start: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale_zyx": list(self.scale_zyx),
            "seed": self.seed,
            "max_timepoints": self.max_timepoints,
            "timepoint_start": self.timepoint_start,
            "detection": {
                "method": self.detection.method,
                "min_sigma": self.detection.min_sigma,
                "max_sigma": self.detection.max_sigma,
                "threshold": self.detection.threshold,
                "normalize": self.detection.normalize,
                "min_distance_um": self.detection.min_distance_um,
                "suppress_duplicates": self.detection.suppress_duplicates,
                "anisotropic": self.detection.anisotropic,
            },
            "linking": {
                "method": self.linking.method,
                "max_distance_um": self.linking.max_distance_um,
                "allow_divisions": self.linking.allow_divisions,
                "division_gap_um": self.linking.division_gap_um,
            },
        }


def _as_tuple3(value: Any) -> tuple[float, float, float]:
    seq = list(value)
    if len(seq) != 3:
        raise ValueError(f"scale_zyx must have 3 values, got {value!r}")
    return float(seq[0]), float(seq[1]), float(seq[2])


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> PipelineConfig:
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)
    det = raw.get("detection") or {}
    link = raw.get("linking") or {}
    return PipelineConfig(
        scale_zyx=_as_tuple3(raw.get("scale_zyx", DEFAULT_SCALE_ZYX)),
        seed=int(raw.get("seed", 0)),
        max_timepoints=raw.get("max_timepoints"),
        timepoint_start=int(raw.get("timepoint_start", 0)),
        detection=DetectionConfig(
            method=str(det.get("method", "blob_dog")),
            min_sigma=float(det.get("min_sigma", 2.0)),
            max_sigma=float(det.get("max_sigma", 5.0)),
            threshold=float(det.get("threshold", 0.05)),
            normalize=str(det.get("normalize", "minmax")),
            min_distance_um=float(det.get("min_distance_um", 4.0)),
            suppress_duplicates=bool(det.get("suppress_duplicates", False)),
            anisotropic=bool(det.get("anisotropic", False)),
        ),
        linking=LinkingConfig(
            method=str(link.get("method", "greedy")),
            max_distance_um=float(link.get("max_distance_um", DEFAULT_LINK_DISTANCE_UM)),
            allow_divisions=bool(link.get("allow_divisions", False)),
            division_gap_um=float(link.get("division_gap_um", 8.0)),
        ),
    )


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
