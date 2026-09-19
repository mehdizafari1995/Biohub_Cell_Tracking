"""Specimen-separated complete validation against the frozen split.

This module does not retune locked operating points. It inventories both
frozen validation sequences, scores available volumes with the official
scorer, and scores every annotated-division event with the organizer
division metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_MATCH_DISTANCE_UM, DEFAULT_SCALE_ZYX
from .errors import analyse_errors
from .graphs import TrackGraph
from .io_geff import load_geff
from .io_zarr import embryo_id, open_volume
from .official_eval import (
    OfficialSampleResult,
    evaluate_official,
    official_scorer_available,
    score_official_division_events,
    summarise_official,
)
from .splits import load_split_manifest

LOCKED_V1_TUNING_EMBRYOS = ("44b6",)
LOCKED_V1_VALIDATION_EMBRYOS = ("6bba",)
LOCKED_V1_TUNING = ("44b6_12dfb391", "44b6_267148e4")
LOCKED_V1_VALIDATION = ("6bba_05db0fb1", "6bba_c328f2fd")


class FrozenSplitError(ValueError):
    """The frozen split was rewritten or is not specimen-separated."""


@dataclass(frozen=True)
class DatasetInventory:
    dataset: str
    embryo: str
    zarr_exists: bool
    geff_exists: bool
    available_timepoints: list[int]
    n_timepoints: int | None
    complete_sequence: bool
    n_gt_nodes: int | None
    n_gt_edges: int | None
    n_gt_divisions: int | None
    status: str
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "embryo": self.embryo,
            "zarr_exists": self.zarr_exists,
            "geff_exists": self.geff_exists,
            "available_timepoints": list(self.available_timepoints),
            "n_available_timepoints": len(self.available_timepoints),
            "n_timepoints": self.n_timepoints,
            "complete_sequence": self.complete_sequence,
            "n_gt_nodes": self.n_gt_nodes,
            "n_gt_edges": self.n_gt_edges,
            "n_gt_divisions": self.n_gt_divisions,
            "status": self.status,
            "error": self.error,
        }


def assert_specimen_separated(manifest: dict[str, Any]) -> None:
    """Fail if tuning and validation share a specimen or a dataset."""
    tuning = [str(x) for x in manifest.get("tuning") or []]
    validation = [str(x) for x in manifest.get("validation") or []]
    overlap = sorted(set(tuning) & set(validation))
    if overlap:
        raise FrozenSplitError(f"tuning and validation share datasets: {overlap}")
    tuning_embryos = {embryo_id(name) for name in tuning}
    validation_embryos = {embryo_id(name) for name in validation}
    shared = sorted(tuning_embryos & validation_embryos)
    if shared:
        raise FrozenSplitError(f"tuning and validation share specimen prefixes: {shared}")
    declared_tune = {str(x) for x in manifest.get("tuning_embryos") or []}
    declared_val = {str(x) for x in manifest.get("validation_embryos") or []}
    if declared_tune and declared_tune != tuning_embryos:
        raise FrozenSplitError(
            f"tuning_embryos {sorted(declared_tune)} do not match dataset prefixes {sorted(tuning_embryos)}"
        )
    if declared_val and declared_val != validation_embryos:
        raise FrozenSplitError(
            f"validation_embryos {sorted(declared_val)} do not match dataset prefixes {sorted(validation_embryos)}"
        )


def assert_frozen_v1_identity(manifest: dict[str, Any]) -> None:
    """Refuse a rewritten v1 split. Do not retune after seeing holdout."""
    if str(manifest.get("version")) != "v1":
        return
    checks = {
        "tuning_embryos": LOCKED_V1_TUNING_EMBRYOS,
        "validation_embryos": LOCKED_V1_VALIDATION_EMBRYOS,
        "tuning": LOCKED_V1_TUNING,
        "validation": LOCKED_V1_VALIDATION,
    }
    for key, expected in checks.items():
        got = tuple(sorted(str(x) for x in manifest.get(key) or []))
        want = tuple(sorted(expected))
        if got != want:
            raise FrozenSplitError(f"frozen v1 {key} changed: {got} != {want}")


def expected_annotated_divisions(manifest: dict[str, Any], role: str = "validation") -> list[dict[str, Any]]:
    """Division counts declared in the frozen split GEFF summaries."""
    summaries = manifest.get("geff_summaries") or {}
    rows = []
    for name in manifest.get(role) or []:
        summary = summaries.get(name) or {}
        rows.append(
            {
                "dataset": name,
                "embryo": embryo_id(name),
                "n_divisions": int(summary.get("n_divisions") or 0),
                "n_nodes": summary.get("n_nodes"),
                "n_edges": summary.get("n_edges"),
                "t_min": summary.get("t_min"),
                "t_max": summary.get("t_max"),
            }
        )
    return rows


def plan_complete_validation(manifest: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Build the locked validation plan. Never searches hyperparameters."""
    if not isinstance(manifest, dict):
        manifest = load_split_manifest(manifest)
    assert_specimen_separated(manifest)
    assert_frozen_v1_identity(manifest)
    expected = expected_annotated_divisions(manifest, "validation")
    return {
        "split_version": manifest.get("version"),
        "role": "validation",
        "tuning_embryos": list(manifest.get("tuning_embryos") or []),
        "validation_embryos": list(manifest.get("validation_embryos") or []),
        "validation_datasets": list(manifest.get("validation") or []),
        "expected_events": expected,
        "expected_annotated_divisions": int(sum(row["n_divisions"] for row in expected)),
        "retune_after_holdout": False,
        "scorer": "royerlab/tracking_cellmot.metrics",
        "note": (
            "Score both frozen validation sequences and every annotated-division "
            "event with the official scorer. Do not retune v1 after seeing holdout."
        ),
    }


def inventory_dataset(data_dir: str | Path, name: str) -> DatasetInventory:
    """Report whether a frozen-split dataset is present and complete."""
    data_dir = Path(data_dir)
    zarr_path = data_dir / f"{name}.zarr"
    geff_path = data_dir / f"{name}.geff"
    available: list[int] = []
    n_time: int | None = None
    complete = False
    n_nodes = n_edges = n_divs = None
    status = "missing"
    error = None
    if zarr_path.exists():
        try:
            _, meta = open_volume(zarr_path)
            available = [int(t) for t in meta.available_timepoints]
            n_time = int(meta.shape[0])
            complete = len(available) >= n_time
            status = "complete" if complete else "partial"
        except Exception as exc:  # noqa: BLE001 — inventory must not crash the harness
            status = "unreadable"
            error = str(exc)
    if geff_path.exists():
        try:
            gt = load_geff(geff_path)
            n_nodes = gt.n_nodes
            n_edges = gt.n_edges
            n_divs = int(sum(len(children) >= 2 for children in gt.out_edges().values()))
            if status == "missing":
                status = "geff_only"
        except Exception as exc:  # noqa: BLE001
            error = f"{error + '; ' if error else ''}geff: {exc}"
            if status == "missing":
                status = "unreadable"
    return DatasetInventory(
        dataset=name,
        embryo=embryo_id(name),
        zarr_exists=zarr_path.exists(),
        geff_exists=geff_path.exists(),
        available_timepoints=available,
        n_timepoints=n_time,
        complete_sequence=complete,
        n_gt_nodes=n_nodes,
        n_gt_edges=n_edges,
        n_gt_divisions=n_divs,
        status=status,
        error=error,
    )


def inventory_validation(data_dir: str | Path, datasets: list[str]) -> list[DatasetInventory]:
    return [inventory_dataset(data_dir, name) for name in datasets]


def extract_annotated_divisions(
    gt: TrackGraph,
    scale_zyx: tuple[float, float, float] = DEFAULT_SCALE_ZYX,
) -> list[dict[str, Any]]:
    """List GT nodes with two or more outgoing edges."""
    index = gt.id_to_index()
    events: list[dict[str, Any]] = []
    for parent, children in gt.out_edges().items():
        if len(children) < 2:
            continue
        pi = index[int(parent)]
        child_rows = []
        for cid in children:
            ci = index[int(cid)]
            dz = (float(gt.z[ci]) - float(gt.z[pi])) * scale_zyx[0]
            dy = (float(gt.y[ci]) - float(gt.y[pi])) * scale_zyx[1]
            dx = (float(gt.x[ci]) - float(gt.x[pi])) * scale_zyx[2]
            child_rows.append(
                {
                    "node_id": int(cid),
                    "t": int(gt.t[ci]),
                    "z": int(gt.z[ci]),
                    "y": int(gt.y[ci]),
                    "x": int(gt.x[ci]),
                    "parent_distance_um": float((dz * dz + dy * dy + dx * dx) ** 0.5),
                }
            )
        events.append(
            {
                "dataset": gt.dataset,
                "parent_id": int(parent),
                "t": int(gt.t[pi]),
                "z": int(gt.z[pi]),
                "y": int(gt.y[pi]),
                "x": int(gt.x[pi]),
                "n_children": len(children),
                "children": child_rows,
            }
        )
    events.sort(key=lambda row: (row["dataset"], row["t"], row["parent_id"]))
    return events


def score_graphs_official(
    pred: TrackGraph,
    gt: TrackGraph,
    scale_zyx: tuple[float, float, float] = DEFAULT_SCALE_ZYX,
    max_distance_um: float = DEFAULT_MATCH_DISTANCE_UM,
) -> dict[str, Any]:
    """Official sequence score plus per-event official division scores."""
    official = evaluate_official(pred, gt, scale_zyx=scale_zyx, max_distance_um=max_distance_um)
    divisions = score_official_division_events(
        pred, gt, scale_zyx=scale_zyx, max_distance_um=max_distance_um
    )
    annotated = extract_annotated_divisions(gt, scale_zyx=scale_zyx)
    recovered = {int(row["parent_id"]): bool(row["official_recovered"]) for row in divisions["events"]}
    events = []
    for event in annotated:
        events.append({**event, "official_recovered": recovered.get(int(event["parent_id"]), False)})
    analysis = analyse_errors(pred, gt, scale_zyx)
    return {
        "dataset": pred.dataset,
        "status": "scored",
        "official": official.as_dict(),
        "division_events": events,
        "official_division_summary": {
            "division_tp": divisions["division_tp"],
            "division_fp": divisions["division_fp"],
            "division_fn": divisions["division_fn"],
            "tp_forks": divisions["tp_forks"],
            "fp_forks": divisions["fp_forks"],
        },
        "errors": analysis.summary(),
    }


def assemble_complete_validation(
    plan: dict[str, Any],
    inventory: list[DatasetInventory],
    scored_samples: list[dict[str, Any]],
    config_name: str,
) -> dict[str, Any]:
    """Combine inventory + official scores. Does not invent missing scores."""
    by_dataset = {row["dataset"]: row for row in scored_samples}
    samples = []
    official_rows: list[OfficialSampleResult] = []
    all_events: list[dict[str, Any]] = []
    for item in inventory:
        scored = by_dataset.get(item.dataset)
        if scored is None:
            samples.append({**item.as_dict(), "official": None, "division_events": []})
            continue
        samples.append({**item.as_dict(), **scored})
        if scored.get("official"):
            official_rows.append(OfficialSampleResult(**scored["official"]))
        all_events.extend(scored.get("division_events") or [])
    complete = bool(inventory) and all(item.complete_sequence and item.geff_exists for item in inventory)
    missing = [item.dataset for item in inventory if item.status in {"missing", "unreadable", "geff_only"}]
    partial = [item.dataset for item in inventory if item.status == "partial"]
    summary = summarise_official(official_rows) if official_rows and official_scorer_available() else None
    event_tp = int(sum(1 for ev in all_events if ev.get("official_recovered")))
    return {
        "plan": plan,
        "config_name": config_name,
        "complete": complete,
        "missing_datasets": missing,
        "partial_datasets": partial,
        "n_scored_samples": len(official_rows),
        "n_expected_samples": len(inventory),
        "official_summary": summary,
        "annotated_division_events": all_events,
        "n_annotated_division_events": len(all_events),
        "n_official_recovered_divisions": event_tp,
        "n_official_missed_divisions": len(all_events) - event_tp,
        "samples": samples,
        "retune_after_holdout": False,
        "score_source": "official" if summary else "unavailable",
        "blocked_reason": _blocked_reason(complete, missing, partial, official_rows),
    }


def _blocked_reason(
    complete: bool,
    missing: list[str],
    partial: list[str],
    official_rows: list[OfficialSampleResult],
) -> str | None:
    if complete and official_rows:
        return None
    parts = []
    if missing:
        parts.append(f"missing {missing}")
    if partial:
        parts.append(f"partial {partial}")
    if not official_rows:
        parts.append("no official scores")
    return "; ".join(parts) if parts else "incomplete"


def load_frozen_validation_plan(split_path: str | Path) -> dict[str, Any]:
    return plan_complete_validation(load_split_manifest(split_path))
