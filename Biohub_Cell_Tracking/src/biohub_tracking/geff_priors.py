from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .graphs import TrackGraph
from .io_geff import load_geff
from .link import physical_coords

SCALE_ZYX = (1.625, 0.40625, 0.40625)
TUNING_SPECIMEN = "44b6"
VALIDATION_SPECIMEN = "6bba"

# Historical five-sequence set kept for comparison. Never use validation
# (6bba_*) rows to lock operating points.
DIVISION_DATASETS = [
    "44b6_12dfb391",
    "44b6_267148e4",
    "44b6_2a2eff9f",
    "6bba_05db0fb1",
    "6bba_c328f2fd",
]
TUNING_DIVISION_DATASETS = [
    "44b6_12dfb391",
    "44b6_267148e4",
    "44b6_2a2eff9f",
]


def specimen_of(dataset: str) -> str:
    return str(dataset).split("_", 1)[0]


REQUIRED_GEFF_ARRAYS = (
    "nodes/ids",
    "nodes/props/t/values",
    "nodes/props/z/values",
    "nodes/props/y/values",
    "nodes/props/x/values",
    "edges/ids",
)


def geff_store_complete(path: Path) -> bool:
    """True when the GEFF directory has the node/edge arrays we read."""
    if not path.is_dir():
        return False
    return all((path / rel).exists() for rel in REQUIRED_GEFF_ARRAYS)


def iter_specimen_geffs(data_dir: str | Path, specimen: str = TUNING_SPECIMEN) -> list[Path]:
    """Return complete `{specimen}_*.geff` stores. Does not open image volumes."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    found: list[Path] = []
    for path in sorted(data_dir.iterdir()):
        if path.name.startswith(f"{specimen}_") and path.name.endswith(".geff") and geff_store_complete(path):
            found.append(path)
    return found


def division_events(graph: TrackGraph) -> list[dict]:
    out = graph.out_edges()
    index = graph.id_to_index()
    coords = graph.coords_zyx()
    events = []
    for parent, children in out.items():
        if len(children) < 2:
            continue
        if parent not in index:
            continue
        pxyz = physical_coords(coords[index[parent] : index[parent] + 1], SCALE_ZYX)[0]
        child_recs = []
        dts = []
        displacements = []
        for child in children:
            if child not in index:
                continue
            cxyz = physical_coords(coords[index[child] : index[child] + 1], SCALE_ZYX)[0]
            dt = int(graph.t[index[child]]) - int(graph.t[index[parent]])
            dts.append(dt)
            disp = cxyz - pxyz
            displacements.append(disp)
            child_recs.append(
                {
                    "child_id": int(child),
                    "dt": dt,
                    "parent_child_um": float(np.linalg.norm(disp)),
                    "coord_zyx": coords[index[child]].tolist(),
                }
            )
        if len(child_recs) < 2:
            continue
        sib = []
        cosines = []
        for a in range(len(child_recs)):
            for b in range(a + 1, len(child_recs)):
                ca = physical_coords(np.array(child_recs[a]["coord_zyx"]), SCALE_ZYX)[0]
                cb = physical_coords(np.array(child_recs[b]["coord_zyx"]), SCALE_ZYX)[0]
                sib.append(float(np.linalg.norm(ca - cb)))
                n1 = float(np.linalg.norm(displacements[a]))
                n2 = float(np.linalg.norm(displacements[b]))
                if n1 >= 1.0 and n2 >= 1.0:
                    cosines.append(float(np.dot(displacements[a], displacements[b]) / (n1 * n2)))
        events.append(
            {
                "dataset": graph.dataset,
                "specimen": specimen_of(graph.dataset),
                "parent_id": int(parent),
                "parent_t": int(graph.t[index[parent]]),
                "parent_coord_zyx": coords[index[parent]].tolist(),
                "n_children": len(child_recs),
                "dt": dts,
                "parent_child_um": [c["parent_child_um"] for c in child_recs],
                "sibling_um": sib,
                "sibling_alignment": cosines,
            }
        )
    return events


def summarise_priors(events: list[dict], label: str) -> dict:
    pc = np.array([d for e in events for d in e["parent_child_um"]], dtype=float)
    sib = np.array([d for e in events for d in e["sibling_um"]], dtype=float)
    dt = np.array([d for e in events for d in e["dt"]], dtype=float)
    align = np.array([d for e in events for d in e.get("sibling_alignment") or []], dtype=float)

    def stats(arr: np.ndarray) -> dict:
        if len(arr) == 0:
            return {"n": 0}
        return {
            "n": int(len(arr)),
            "min": float(arr.min()),
            "p10": float(np.quantile(arr, 0.10)),
            "p25": float(np.quantile(arr, 0.25)),
            "median": float(np.median(arr)),
            "p75": float(np.quantile(arr, 0.75)),
            "p90": float(np.quantile(arr, 0.90)),
            "max": float(arr.max()),
        }

    suggested_max_parent = float(np.quantile(pc, 0.9)) if len(pc) else 8.0
    suggested_min_sep = float(max(np.quantile(sib, 0.10), 1.5)) if len(sib) else 2.0
    if len(align):
        # Observed 44b6 forks are anti-aligned (negative cosine). The lock is an
        # upper bound that still rejects same-direction pairs.
        suggested_align = 0.05 if float(align.max()) < 0.05 else float(min(max(float(align.max()), 0.05), 0.25))
    else:
        suggested_align = 0.15
    return {
        "label": label,
        "n_events": len(events),
        "n_datasets": len({e["dataset"] for e in events}),
        "parent_child_um": stats(pc),
        "sibling_um": stats(sib),
        "dt_frames": stats(dt),
        "sibling_alignment": stats(align),
        "suggested_max_parent_child_um": suggested_max_parent,
        "suggested_min_child_separation_um": suggested_min_sep,
        "suggested_max_sibling_alignment": suggested_align,
    }


def lock_operating_point(tuning_summary: dict) -> dict:
    """Round conservative gates from tuning-specimen GEFF stats only."""
    pc = tuning_summary.get("parent_child_um") or {}
    sib = tuning_summary.get("sibling_um") or {}
    align = tuning_summary.get("sibling_alignment") or {}
    max_pc = float(tuning_summary.get("suggested_max_parent_child_um") or 8.0)
    min_sep = float(tuning_summary.get("suggested_min_child_separation_um") or 8.5)
    max_align = float(tuning_summary.get("suggested_max_sibling_alignment") or 0.15)
    observed_max_pc = float(pc.get("max") or max_pc)
    observed_min_sib = float(sib.get("min") or min_sep)
    # Cover every observed tuning event, then round. Never widen from validation.
    locked_max_pc = float(min(round(max(max_pc, observed_max_pc) + 0.1, 1), 10.0))
    locked_min_sep = float(max(round(min(min_sep, observed_min_sib) - 0.1, 1), 8.0))
    locked_align = float(min(max(round(max_align, 2), 0.05), 0.25))
    return {
        "source": tuning_summary.get("label"),
        "n_events": tuning_summary.get("n_events"),
        "n_datasets": tuning_summary.get("n_datasets"),
        "max_parent_child_um": locked_max_pc,
        "min_child_separation_um": locked_min_sep,
        "min_parent_child_um": 0.0,
        "max_sibling_alignment": locked_align,
        "require_parent_predecessor": True,
        "require_no_alternative_parent": True,
        "alternative_parent_gate_um": 12.0,
        "intensity_sum_tolerance": 0.45,
        "max_children": 2,
        "note": (
            "Locked from tuning-specimen (44b6) GEFF geometry only. "
            f"parent_child max={pc.get('max')} p90={pc.get('p90')} "
            f"sibling min={sib.get('min')} p10={sib.get('p10')} "
            f"alignment_max={align.get('max')}. Validation 6bba events were not used."
        ),
    }


def _load_dataset_events(path: Path) -> dict:
    if not path.exists():
        return {"missing": True}
    if path.is_dir() and not geff_store_complete(path):
        return {"missing": True, "incomplete": True}
    graph = load_geff(path)
    events = division_events(graph)
    return {"n_events": len(events), "events": events}


def collect_geff_division_priors(data_dir: str | Path) -> dict:
    """GEFF-only geometry. Does not open image volumes.

    Suggested / locked thresholds use the tuning specimen (`44b6`) only.
    Validation (`6bba`) graphs may be inventoried for completeness but are
    excluded from ``tuning_specimen`` and ``locked_operating_point``.
    """
    data_dir = Path(data_dir)
    by_dataset: dict[str, dict] = {}
    historical_all: list[dict] = []
    historical_tune: list[dict] = []
    for name in DIVISION_DATASETS:
        rec = _load_dataset_events(data_dir / f"{name}.geff")
        by_dataset[name] = rec
        events = rec.get("events") or []
        historical_all.extend(events)
        if name in TUNING_DIVISION_DATASETS:
            historical_tune.extend(events)

    specimen_events: list[dict] = []
    scanned = []
    for path in iter_specimen_geffs(data_dir, TUNING_SPECIMEN):
        name = path.name.removesuffix(".geff")
        scanned.append(name)
        if name in by_dataset and "events" in by_dataset[name]:
            events = by_dataset[name]["events"]
        else:
            rec = _load_dataset_events(path)
            by_dataset[name] = rec
            events = rec.get("events") or []
        if specimen_of(name) != TUNING_SPECIMEN:
            continue
        specimen_events.extend(events)

    val_scanned = [p.name.removesuffix(".geff") for p in iter_specimen_geffs(data_dir, VALIDATION_SPECIMEN)]
    tuning_summary = summarise_priors(specimen_events, "tuning_44b6_specimen_geffs")
    return {
        "note": (
            "Priors use GEFF node/edge coordinates only. Validation image outcomes "
            "were not inspected while estimating these numbers. Thresholds are "
            "locked on the tuning specimen (44b6) only, then scored once on frozen validation."
        ),
        "tuning_specimen": TUNING_SPECIMEN,
        "validation_specimen_excluded": VALIDATION_SPECIMEN,
        "n_tuning_geffs_scanned": len(scanned),
        "tuning_geffs_scanned": scanned,
        "n_validation_geffs_present_unused": len(val_scanned),
        "datasets": by_dataset,
        "all_five": summarise_priors(historical_all, "all_five_division_geffs"),
        "tuning_three": summarise_priors(historical_tune, "tuning_44b6_division_geffs"),
        "tuning_specimen_summary": tuning_summary,
        "locked_operating_point": lock_operating_point(tuning_summary),
    }


def write_priors(data_dir: str | Path, dest: str | Path) -> dict:
    payload = collect_geff_division_priors(data_dir)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text(json.dumps(payload, indent=2))
    return payload
