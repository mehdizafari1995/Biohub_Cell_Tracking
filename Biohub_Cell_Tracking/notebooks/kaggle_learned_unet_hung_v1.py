"""
Kaggle helper kernel: learned_unet_hung_v1 CPU relink.

Loads locked learned_unet_infer_v1 JSON graphs and replaces consecutive
edges with Hungarian 1-1 assignment on last-position distance plus the
locked 12 µm gate. Never opens 6bba. Never writes submission.csv. Not a
competition submission. Do not overwrite rejected learned_unet_link_v1 /
link_v2 / motion_v1.

Attach competition `biohub-cell-tracking-during-development` and the
private infer helper `mahdimzafari/biohub-learned-unet-infer-v1`.
Internet OFF. GPU OFF. Do not overwrite locked train/infer/holdout/
test/gap-test/link slugs.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

CONFIG_LABEL = "learned_unet_hung_v1"
TRAIN_LABEL = "learned_unet_v1"
INFER_LABEL = "learned_unet_infer_v1"
TRAIN_CONFIG_HASH = "4c38035d6657f2ad"
INFER_CONFIG_HASH = "d7578b5f64ddac08"
REJECTED_LINK_V1_HASH = "3a7423c6858ace41"
REJECTED_LINK_V2_HASH = "1d4015dc499bd604"
REJECTED_MOTION_HASH = "66853b6f1445c18b"
TUNING_SPECIMEN = "44b6"
HOLDOUT_SPECIMENS = ("6bba",)
INFER_STEMS = ("44b6_12dfb391", "44b6_267148e4")
FROZEN_HOLDOUT_STEMS = ("6bba_05db0fb1", "6bba_c328f2fd")
KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-hung-v1"
REJECTED_LINK_V1_SLUG = "mahdimzafari/biohub-learned-unet-link-v1"
REJECTED_LINK_V2_SLUG = "mahdimzafari/biohub-learned-unet-link-v2"
REJECTED_MOTION_SLUG = "mahdimzafari/biohub-learned-unet-motion-v1"
TRAIN_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-v1"
INFER_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-infer-v1"
SCALE_ZYX = (1.625, 0.40625, 0.40625)
MAX_LINK_UM = 12.0
INF = 1.0e6
COMPETITION = "biohub-cell-tracking-during-development"


class HoldoutLeakError(ValueError):
    """Raised when a frozen-holdout specimen would enter inference."""


def hung_config_payload() -> dict[str, Any]:
    return {
        "label": CONFIG_LABEL,
        "train_label": TRAIN_LABEL,
        "infer_label": INFER_LABEL,
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": INFER_CONFIG_HASH,
        "rejected_link_v1_hash": REJECTED_LINK_V1_HASH,
        "rejected_link_v2_hash": REJECTED_LINK_V2_HASH,
        "rejected_motion_hash": REJECTED_MOTION_HASH,
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "infer_stems": list(INFER_STEMS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "source_graphs": "locked_learned_unet_infer_graphs",
        "retunes_detection": False,
        "uses_intensity": False,
        "uses_transformer": False,
        "uses_velocity": False,
        "linker": "last_position_hungarian_1to1",
        "motion_model": "last_position",
        "gate_um": 12.0,
        "max_link_um": 12.0,
        "max_gap_frames": 0,
        "assignment": "hungarian",
        "interpolate_gaps": False,
        "max_parents_per_node": 1,
        "max_children_per_node": 1,
        "intensity_weight": 0.0,
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": False,
        "kernel_slug": KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
        "rejected_link_v1_slug": REJECTED_LINK_V1_SLUG,
        "rejected_link_v2_slug": REJECTED_LINK_V2_SLUG,
        "rejected_motion_slug": REJECTED_MOTION_SLUG,
    }


def hung_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or hung_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_HUNG_HASH = hung_config_hash()
assert LOCKED_HUNG_HASH == "f67bc97d0521dc78"


def specimen_of(stem: str) -> str:
    name = Path(stem).name
    for suffix in (".zarr", ".geff", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.split("_", 1)[0]


def refuse_non_infer_stems(stems: Iterable[str]) -> None:
    cleaned: list[str] = []
    for raw in stems:
        stem = Path(raw).name
        for suffix in (".zarr", ".geff", ".json"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        if stem not in cleaned:
            cleaned.append(stem)
    leaked = [s for s in cleaned if specimen_of(s) in HOLDOUT_SPECIMENS or s in FROZEN_HOLDOUT_STEMS]
    if leaked:
        raise HoldoutLeakError("learned_unet_hung_v1 refuses holdout stems: " + ", ".join(leaked))
    extra = [s for s in cleaned if s not in INFER_STEMS]
    if extra:
        raise HoldoutLeakError("learned_unet_hung_v1 only opens frozen-tuning stems; got " + ", ".join(extra))


def assert_no_submission_csv(working: str | Path) -> None:
    banned = Path(working) / "submission.csv"
    if banned.exists():
        banned.unlink()
        raise RuntimeError("learned_unet_hung_v1 must not write submission.csv")


def physical(coords: np.ndarray, scale_zyx: Sequence[float] = SCALE_ZYX) -> np.ndarray:
    return np.asarray(coords, dtype=np.float64).reshape(-1, 3) * np.asarray(scale_zyx, dtype=np.float64)


def hungarian_assign(cost: np.ndarray, allowed: np.ndarray) -> list[tuple[int, int]]:
    if cost.size == 0:
        return []
    mask = np.asarray(allowed, dtype=bool)
    if not mask.any():
        return []
    work = np.full(np.asarray(cost).shape, INF, dtype=np.float64)
    work[mask] = np.asarray(cost, dtype=np.float64)[mask]
    rows, cols = linear_sum_assignment(work)
    pairs: list[tuple[int, int]] = []
    for i, j in zip(rows, cols):
        ii, jj = int(i), int(j)
        if mask[ii, jj] and work[ii, jj] < INF / 2:
            pairs.append((ii, jj))
    return pairs


def assign_hung_links(
    last_zyx: np.ndarray,
    det_zyx: np.ndarray,
    max_link_um: float = MAX_LINK_UM,
    scale_zyx: Sequence[float] = SCALE_ZYX,
) -> list[tuple[int, int]]:
    last = np.asarray(last_zyx, dtype=np.float64).reshape(-1, 3)
    det = np.asarray(det_zyx, dtype=np.float64).reshape(-1, 3)
    if last.size == 0 or det.size == 0:
        return []
    dist = np.linalg.norm(physical(last, scale_zyx)[:, None, :] - physical(det, scale_zyx)[None, :, :], axis=2)
    return hungarian_assign(dist, dist <= float(max_link_um))


def apply_hung_relink_arrays(
    node_id: Sequence[int],
    t: Sequence[int],
    z: Sequence[int],
    y: Sequence[int],
    x: Sequence[int],
) -> list[tuple[int, int]]:
    """Standalone Hungarian last-position 1-1 relink. Returns (src, dst) node ids."""
    nids = [int(v) for v in node_id]
    ts = [int(v) for v in t]
    zs = [int(v) for v in z]
    ys = [int(v) for v in y]
    xs = [int(v) for v in x]
    by_t: dict[int, list[int]] = defaultdict(list)
    for i, tk in enumerate(ts):
        by_t[tk].append(i)
    tracks: list[dict[str, Any]] = []
    edges: list[tuple[int, int]] = []
    for tk in sorted(by_t):
        idxs = by_t[tk]
        coords = np.stack(
            [
                np.asarray([zs[i] for i in idxs], dtype=np.float64),
                np.asarray([ys[i] for i in idxs], dtype=np.float64),
                np.asarray([xs[i] for i in idxs], dtype=np.float64),
            ],
            axis=1,
        )
        live = [tr for tr in tracks if tk - tr["times"][-1] == 1]
        used_dets: set[int] = set()
        if live and len(coords):
            last = np.stack([tr["positions"][-1] for tr in live])
            for li, j in assign_hung_links(last, coords):
                tr = live[li]
                edges.append((int(tr["node_ids"][-1]), int(nids[idxs[j]])))
                tr["node_ids"].append(int(nids[idxs[j]]))
                tr["times"].append(tk)
                tr["positions"].append(coords[j])
                used_dets.add(j)
        for j, i in enumerate(idxs):
            if j in used_dets:
                continue
            tracks.append(
                {
                    "node_ids": [int(nids[i])],
                    "times": [tk],
                    "positions": [coords[j]],
                }
            )
    return edges


def graph_payload(dataset: str, coords: np.ndarray, edges: Sequence[tuple[int, int]]) -> dict[str, Any]:
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 4)
    node_id = np.arange(1, len(coords) + 1, dtype=np.int64)
    refuse_non_infer_stems([dataset])
    return {
        "dataset": dataset,
        "node_id": node_id.tolist(),
        "t": coords[:, 0].astype(np.int64).tolist() if len(coords) else [],
        "z": coords[:, 1].astype(np.int64).tolist() if len(coords) else [],
        "y": coords[:, 2].astype(np.int64).tolist() if len(coords) else [],
        "x": coords[:, 3].astype(np.int64).tolist() if len(coords) else [],
        "edges": [[int(s), int(d)] for s, d in edges],
    }


def write_graph_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    refuse_non_infer_stems([str(payload["dataset"])])
    dest.write_text(json.dumps(payload, indent=2))
    return dest


def find_infer_graphs(roots: Iterable[str | Path] | None = None) -> dict[str, Path]:
    if roots is None:
        roots = [
            Path("/kaggle/input/notebooks"),
            Path("/kaggle/input"),
            Path("artifacts/kaggle/learned_unet_infer_v1"),
        ]
        extra = os.environ.get("BIOHUB_INFER_GRAPH_DIR")
        if extra:
            roots.insert(0, Path(extra))
    found: dict[str, Path] = {}
    for raw in roots:
        root = Path(raw)
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if any(part.endswith((".zarr", ".geff")) for part in path.parts):
                continue
            stem = path.name[: -len(".json")] if path.name.endswith(".json") else path.name
            if stem in INFER_STEMS and stem not in found:
                found[stem] = path
            if len(found) == len(INFER_STEMS):
                return found
    return found


def peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(usage) / 1024.0


def main() -> dict[str, Any]:
    working = Path(os.environ.get("BIOHUB_WORKING", "/kaggle/working" if Path("/kaggle/working").exists() else "."))
    working.mkdir(parents=True, exist_ok=True)
    cfg = hung_config_payload()
    digest = hung_config_hash(cfg)
    if digest != LOCKED_HUNG_HASH:
        raise RuntimeError("hungarian config hash drifted; refusing to retune")
    assert_no_submission_csv(working)
    report: dict[str, Any] = {
        "status": "blocked_no_graphs",
        "label": CONFIG_LABEL,
        "config_hash": digest,
        "kernel_slug": KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
        "writes_submission_csv": False,
        "is_competition_submission": False,
        "is_leaderboard_score": False,
        "opened_holdout": False,
    }
    if os.environ.get("BIOHUB_LEARNED_DRY_RUN") == "1":
        report["status"] = "dry_run"
        (working / "infer_manifest.json").write_text(json.dumps(report, indent=2))
        assert_no_submission_csv(working)
        return report
    found = find_infer_graphs()
    missing = [stem for stem in INFER_STEMS if stem not in found]
    if missing:
        report["missing_graphs"] = missing
        (working / "infer_manifest.json").write_text(json.dumps(report, indent=2))
        assert_no_submission_csv(working)
        return report
    refuse_non_infer_stems(found)
    started = time.time()
    graphs = []
    for stem in INFER_STEMS:
        payload = json.loads(found[stem].read_text())
        refuse_non_infer_stems([payload["dataset"]])
        edges = apply_hung_relink_arrays(
            payload["node_id"],
            payload["t"],
            payload["z"],
            payload["y"],
            payload["x"],
        )
        out = {
            "dataset": stem,
            "node_id": payload["node_id"],
            "t": payload["t"],
            "z": payload["z"],
            "y": payload["y"],
            "x": payload["x"],
            "edges": [[int(s), int(d)] for s, d in edges],
        }
        write_graph_json(working / f"{stem}.json", out)
        graphs.append({"dataset": stem, "n_nodes": len(out["node_id"]), "n_edges": len(out["edges"])})
    report.update(
        {
            "status": "inferred",
            "graphs": graphs,
            "runtime_sec": time.time() - started,
            "peak_memory_mb": peak_memory_mb(),
        }
    )
    (working / "infer_manifest.json").write_text(json.dumps(report, indent=2))
    assert_no_submission_csv(working)
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
