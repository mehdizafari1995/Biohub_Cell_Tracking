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


# Directory markers used by older completeness checks. Prefer REQUIRED_GEFF_FILES.
REQUIRED_GEFF_ARRAYS = (
    "nodes/ids",
    "nodes/props/t/values",
    "nodes/props/z/values",
    "nodes/props/y/values",
    "nodes/props/x/values",
    "edges/ids",
)

# Arrays actually opened by io_geff.load_geff. Skip optional group metadata.
REQUIRED_GEFF_FILES = (
    "zarr.json",
    "nodes/ids/zarr.json",
    "nodes/ids/c/0",
    "edges/ids/zarr.json",
    "edges/ids/c/0/0",
    "nodes/props/t/values/zarr.json",
    "nodes/props/t/values/c/0",
    "nodes/props/z/values/zarr.json",
    "nodes/props/z/values/c/0",
    "nodes/props/y/values/zarr.json",
    "nodes/props/y/values/c/0",
    "nodes/props/x/values/zarr.json",
    "nodes/props/x/values/c/0",
)

OPTIONAL_GEFF_METADATA_FILES = (
    "nodes/zarr.json",
    "nodes/props/zarr.json",
    "nodes/ids/zarr.json",
    "edges/zarr.json",
    "edges/props/zarr.json",
    "nodes/props/t/zarr.json",
    "nodes/props/z/zarr.json",
    "nodes/props/y/zarr.json",
    "nodes/props/x/zarr.json",
)

# Conservative lock currently in configs/division_v2.yaml. New typical 44b6
# events may widen max_parent_child_um up to 10.0; they must not lower
# min_child_separation_um or raise max_sibling_alignment.
CURRENT_LOCKED_OPERATING_POINT = {
    "source": "tuning_44b6_specimen_geffs",
    "max_parent_child_um": 8.5,
    "min_child_separation_um": 8.7,
    "min_parent_child_um": 0.0,
    "max_sibling_alignment": 0.05,
    "require_parent_predecessor": True,
    "require_no_alternative_parent": True,
    "alternative_parent_gate_um": 12.0,
    "intensity_sum_tolerance": 0.45,
    "max_children": 2,
}

TYPICAL_MIN_DISPLACEMENT_UM = 1.0
TYPICAL_MIN_SIBLING_UM = 8.0
TYPICAL_MAX_PARENT_CHILD_UM = 10.0
TYPICAL_MAX_ALIGNMENT = 0.05


def all_geff_files() -> tuple[str, ...]:
    """Full 21-file GEFF layout. Prefer REQUIRED_GEFF_FILES for downloads."""
    files = [
        "zarr.json",
        "nodes/zarr.json",
        "nodes/props/zarr.json",
        "nodes/ids/zarr.json",
        "nodes/ids/c/0",
        "edges/zarr.json",
        "edges/ids/zarr.json",
        "edges/ids/c/0/0",
        "edges/props/zarr.json",
    ]
    for prop in ("t", "z", "y", "x"):
        files.extend(
            [
                f"nodes/props/{prop}/zarr.json",
                f"nodes/props/{prop}/values/zarr.json",
                f"nodes/props/{prop}/values/c/0",
            ]
        )
    return tuple(files)


def geff_store_complete(path: Path) -> bool:
    """True when required node/edge arrays (and chunks) are present and non-empty."""
    if not path.is_dir():
        return False
    for rel in REQUIRED_GEFF_FILES:
        dest = path / rel
        if not dest.exists() or dest.stat().st_size <= 0:
            return False
    return True


def inventory_geff_names(
    inventory_path: str | Path,
    specimen: str | None = None,
    split: str = "train",
) -> list[str]:
    """Return unique GEFF dataset stems from docs/dataset_inventory.json."""
    payload = json.loads(Path(inventory_path).read_text())
    names: list[str] = []
    seen: set[str] = set()
    for row in payload.get("datasets") or []:
        if row.get("kind") != "geff" or row.get("split") != split:
            continue
        name = str(row.get("dataset") or "")
        if not name or name in seen:
            continue
        if specimen is not None and str(row.get("embryo") or specimen_of(name)) != specimen:
            continue
        seen.add(name)
        names.append(name)
    return names


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


def atypical_reasons(event: dict) -> list[str]:
    """Why a GEFF fork is excluded from typical-mitosis lock updates.

    Typical 44b6 mitoses used for locking are bipolar, dt=1, both daughters
    displaced, sibling separation at or above the 8.0 µm floor, parent–child
    at or below the 10.0 µm cap, and anti-aligned when the angle is defined.
    Close twins, long-range attachments, and near-coincident daughters stay
    inventoried but must not relax the conservative gates.
    """
    reasons: list[str] = []
    n_children = int(event.get("n_children") or len(event.get("parent_child_um") or []))
    if n_children != 2:
        reasons.append("n_children")
    dts = [int(dt) for dt in (event.get("dt") or [])]
    if not dts or any(dt != 1 for dt in dts):
        reasons.append("dt")
    pcs = [float(d) for d in (event.get("parent_child_um") or [])]
    sibs = [float(d) for d in (event.get("sibling_um") or [])]
    aligns = [float(d) for d in (event.get("sibling_alignment") or [])]
    if not pcs or min(pcs) < TYPICAL_MIN_DISPLACEMENT_UM:
        reasons.append("near_coincident_daughter")
    if pcs and max(pcs) > TYPICAL_MAX_PARENT_CHILD_UM:
        reasons.append("parent_child_above_cap")
    if not sibs or min(sibs) < TYPICAL_MIN_SIBLING_UM:
        reasons.append("close_twin")
    if not aligns:
        reasons.append("alignment_undefined")
    elif max(aligns) > TYPICAL_MAX_ALIGNMENT:
        reasons.append("same_direction")
    return reasons


def is_typical_fork(event: dict) -> bool:
    return not atypical_reasons(event)


def retain_or_update_lock(
    current: dict | None,
    typical_events: list[dict],
) -> dict:
    """Keep the locked gates unless a typical 44b6 event needs a wider parent–child.

    Never lower ``min_child_separation_um`` (close twins are atypical).
    Never raise ``max_sibling_alignment``. Cap ``max_parent_child_um`` at 10.0.
    """
    locked = dict(current or CURRENT_LOCKED_OPERATING_POINT)
    for key, value in CURRENT_LOCKED_OPERATING_POINT.items():
        locked.setdefault(key, value)
    action = "retain"
    reasons: list[str] = []
    pc_max = None
    if typical_events:
        pc_max = max(float(d) for event in typical_events for d in event.get("parent_child_um") or [0.0])
        current_pc = float(locked["max_parent_child_um"])
        if pc_max > current_pc:
            new_pc = float(min(round(pc_max + 0.1, 1), TYPICAL_MAX_PARENT_CHILD_UM))
            if new_pc > current_pc:
                locked["max_parent_child_um"] = new_pc
                action = "update_typical"
                reasons.append(
                    f"typical parent_child max {pc_max:.3f} µm exceeds locked {current_pc}; "
                    f"widen to {new_pc} (cap {TYPICAL_MAX_PARENT_CHILD_UM})"
                )
    locked["min_child_separation_um"] = float(CURRENT_LOCKED_OPERATING_POINT["min_child_separation_um"])
    locked["max_sibling_alignment"] = float(CURRENT_LOCKED_OPERATING_POINT["max_sibling_alignment"])
    locked["require_parent_predecessor"] = True
    locked["require_no_alternative_parent"] = True
    locked["n_typical_events"] = len(typical_events)
    locked["n_typical_datasets"] = len({e.get("dataset") for e in typical_events})
    locked["typical_parent_child_max_um"] = pc_max
    locked["action"] = action
    locked["update_reasons"] = reasons
    locked["note"] = (
        "Locked from typical tuning-specimen (44b6) GEFF geometry only. "
        "Atypical forks (close twins, near-coincident daughters, parent–child "
        f"above {TYPICAL_MAX_PARENT_CHILD_UM} µm) were inventoried and not used "
        "to relax min_child_separation_um or max_sibling_alignment. "
        "Validation 6bba events were not used."
    )
    return locked


def apply_operating_point_to_yaml(path: str | Path, lock: dict) -> None:
    """Patch gate scalars in division_v2.yaml without dropping comments."""
    import re

    text = Path(path).read_text()
    for key in ("max_parent_child_um", "min_child_separation_um", "max_sibling_alignment"):
        text, n = re.subn(
            rf"^(\s*{key}:\s*)[-0-9.]+",
            rf"\g<1>{lock[key]}",
            text,
            count=1,
            flags=re.M,
        )
        if n != 1:
            raise ValueError(f"failed to patch {key} in {path}")
    Path(path).write_text(text)


def lock_operating_point(tuning_summary: dict) -> dict:
    """Round conservative gates from tuning-specimen GEFF stats only.

    Do not use this on the unfiltered event list: a close-twin sibling can
    pull ``min_child_separation_um`` down to the 8.0 floor. Prefer
    ``retain_or_update_lock`` on ``is_typical_fork`` events.
    """
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
    typical_events = [event for event in specimen_events if is_typical_fork(event)]
    atypical_events = []
    for event in specimen_events:
        reasons = atypical_reasons(event)
        if reasons:
            atypical_events.append({**event, "atypical_reasons": reasons})
    typical_summary = summarise_priors(typical_events, "typical_44b6_specimen_geffs")
    tuning_summary = summarise_priors(specimen_events, "tuning_44b6_specimen_geffs")
    proposed = retain_or_update_lock(CURRENT_LOCKED_OPERATING_POINT, typical_events)
    proposed["n_events"] = len(typical_events)
    proposed["n_datasets"] = proposed.get("n_typical_datasets")
    unfiltered = lock_operating_point(tuning_summary)
    unfiltered["do_not_use"] = True
    unfiltered["reason"] = (
        "Unfiltered lock_operating_point on every 44b6 fork, including close twins. "
        "Kept for audit only. division_v2 uses locked_operating_point from typical forks."
    )
    return {
        "note": (
            "Priors use GEFF node/edge coordinates only. Validation image outcomes "
            "were not inspected while estimating these numbers. Thresholds are "
            "locked on typical tuning-specimen (44b6) forks only, then scored once "
            "on frozen validation. Close twins and near-coincident daughters are "
            "inventoried as atypical and must not lower min_child_separation_um."
        ),
        "tuning_specimen": TUNING_SPECIMEN,
        "validation_specimen_excluded": VALIDATION_SPECIMEN,
        "n_tuning_geffs_scanned": len(scanned),
        "tuning_geffs_scanned": scanned,
        "n_validation_geffs_present_unused": len(val_scanned),
        "n_typical_events": len(typical_events),
        "n_atypical_events": len(atypical_events),
        "typical_event_datasets": sorted({e.get("dataset") for e in typical_events if e.get("dataset")}),
        "atypical_event_datasets": sorted({e.get("dataset") for e in atypical_events if e.get("dataset")}),
        "datasets": by_dataset,
        "all_five": summarise_priors(historical_all, "all_five_division_geffs"),
        "tuning_three": summarise_priors(historical_tune, "tuning_44b6_division_geffs"),
        "tuning_specimen_summary": tuning_summary,
        "typical_tuning_summary": typical_summary,
        "atypical_tuning_events": atypical_events,
        "locked_operating_point": proposed,
        "unfiltered_lock_rejected": unfiltered,
    }


def write_priors(data_dir: str | Path, dest: str | Path) -> dict:
    payload = collect_geff_division_priors(data_dir)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text(json.dumps(payload, indent=2))
    return payload
