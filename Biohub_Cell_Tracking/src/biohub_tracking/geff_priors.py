from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .graphs import TrackGraph
from .io_geff import load_geff
from .link import physical_coords

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
        pxyz = physical_coords(coords[index[parent] : index[parent] + 1], (1.625, 0.40625, 0.40625))[0]
        child_recs = []
        dts = []
        for child in children:
            if child not in index:
                continue
            cxyz = physical_coords(coords[index[child] : index[child] + 1], (1.625, 0.40625, 0.40625))[0]
            dt = int(graph.t[index[child]]) - int(graph.t[index[parent]])
            dts.append(dt)
            child_recs.append(
                {
                    "child_id": int(child),
                    "dt": dt,
                    "parent_child_um": float(np.linalg.norm(cxyz - pxyz)),
                    "coord_zyx": coords[index[child]].tolist(),
                }
            )
        if len(child_recs) < 2:
            continue
        sib = []
        for a in range(len(child_recs)):
            for b in range(a + 1, len(child_recs)):
                ca = physical_coords(np.array(child_recs[a]["coord_zyx"]), (1.625, 0.40625, 0.40625))[0]
                cb = physical_coords(np.array(child_recs[b]["coord_zyx"]), (1.625, 0.40625, 0.40625))[0]
                sib.append(float(np.linalg.norm(ca - cb)))
        events.append(
            {
                "dataset": graph.dataset,
                "parent_id": int(parent),
                "parent_t": int(graph.t[index[parent]]),
                "parent_coord_zyx": coords[index[parent]].tolist(),
                "n_children": len(child_recs),
                "dt": dts,
                "parent_child_um": [c["parent_child_um"] for c in child_recs],
                "sibling_um": sib,
            }
        )
    return events


def summarise_priors(events: list[dict], label: str) -> dict:
    pc = np.array([d for e in events for d in e["parent_child_um"]], dtype=float)
    sib = np.array([d for e in events for d in e["sibling_um"]], dtype=float)
    dt = np.array([d for e in events for d in e["dt"]], dtype=float)
    def stats(arr: np.ndarray) -> dict:
        if len(arr) == 0:
            return {"n": 0}
        return {
            "n": int(len(arr)),
            "min": float(arr.min()),
            "p25": float(np.quantile(arr, 0.25)),
            "median": float(np.median(arr)),
            "p75": float(np.quantile(arr, 0.75)),
            "p90": float(np.quantile(arr, 0.90)),
            "max": float(arr.max()),
        }
    return {
        "label": label,
        "n_events": len(events),
        "parent_child_um": stats(pc),
        "sibling_um": stats(sib),
        "dt_frames": stats(dt),
        "suggested_max_parent_child_um": float(np.quantile(pc, 0.9)) if len(pc) else 8.0,
        "suggested_min_child_separation_um": float(max(np.quantile(sib, 0.1), 1.5)) if len(sib) else 2.0,
    }


def collect_geff_division_priors(data_dir: str | Path) -> dict:
    """GEFF-only geometry. Does not open image volumes."""
    data_dir = Path(data_dir)
    by_dataset = {}
    all_events = []
    tuning_events = []
    for name in DIVISION_DATASETS:
        path = data_dir / f"{name}.geff"
        if not path.exists():
            by_dataset[name] = {"missing": True}
            continue
        graph = load_geff(path)
        events = division_events(graph)
        by_dataset[name] = {"n_events": len(events), "events": events}
        all_events.extend(events)
        if name in TUNING_DIVISION_DATASETS:
            tuning_events.extend(events)
    return {
        "note": (
            "Priors use GEFF node/edge coordinates only. Validation image outcomes "
            "were not inspected while estimating these numbers. Thresholds should be "
            "locked on tuning GEFFs/volumes, then scored once on frozen validation."
        ),
        "datasets": by_dataset,
        "all_five": summarise_priors(all_events, "all_five_division_geffs"),
        "tuning_three": summarise_priors(tuning_events, "tuning_44b6_division_geffs"),
    }


def write_priors(data_dir: str | Path, dest: str | Path) -> dict:
    payload = collect_geff_division_priors(data_dir)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text(json.dumps(payload, indent=2))
    return payload
