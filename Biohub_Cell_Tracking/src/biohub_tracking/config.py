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
    motion_model: str = "none"
    gate_um: float | None = None
    max_gap_frames: int = 0
    assignment: str = "greedy"
    intensity_weight: float = 0.0
    track_start_penalty_um: float = 0.0
    track_end_penalty_um: float = 0.0
    boundary_margin_um: float = 0.0
    interpolate_gaps: bool = True
    min_child_separation_um: float = 2.0
    max_parent_child_um: float | None = None
    intensity_sum_tolerance: float = 0.6
    max_children: int = 1


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
                "motion_model": self.linking.motion_model,
                "gate_um": self.linking.gate_um,
                "max_gap_frames": self.linking.max_gap_frames,
                "assignment": self.linking.assignment,
                "intensity_weight": self.linking.intensity_weight,
                "track_start_penalty_um": self.linking.track_start_penalty_um,
                "track_end_penalty_um": self.linking.track_end_penalty_um,
                "boundary_margin_um": self.linking.boundary_margin_um,
                "interpolate_gaps": self.linking.interpolate_gaps,
                "min_child_separation_um": self.linking.min_child_separation_um,
                "max_parent_child_um": self.linking.max_parent_child_um,
                "intensity_sum_tolerance": self.linking.intensity_sum_tolerance,
                "max_children": self.linking.max_children,
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
            motion_model=str(link.get("motion_model", "none")),
            gate_um=(None if link.get("gate_um") is None else float(link.get("gate_um"))),
            max_gap_frames=int(link.get("max_gap_frames", 0)),
            assignment=str(link.get("assignment", "greedy")),
            intensity_weight=float(link.get("intensity_weight", 0.0)),
            track_start_penalty_um=float(link.get("track_start_penalty_um", 0.0)),
            track_end_penalty_um=float(link.get("track_end_penalty_um", 0.0)),
            boundary_margin_um=float(link.get("boundary_margin_um", 0.0)),
            interpolate_gaps=bool(link.get("interpolate_gaps", True)),
            min_child_separation_um=float(link.get("min_child_separation_um", 2.0)),
            max_parent_child_um=(
                None if link.get("max_parent_child_um") is None else float(link.get("max_parent_child_um"))
            ),
            intensity_sum_tolerance=float(link.get("intensity_sum_tolerance", 0.6)),
            max_children=int(link.get("max_children", 1)),
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
