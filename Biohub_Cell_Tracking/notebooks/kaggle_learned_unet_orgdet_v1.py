"""
Kaggle helper kernel: learned_unet_orgdet_v1 CPU relink.

Loads locked learned_unet_orgop_v1 JSON detections and replaces edges
with the locked infer greedy 12 µm last-position linker. Never opens
6bba. Never writes submission.csv. Not a competition submission.
Do not overwrite rejected learned_unet_orgop_v1.

Attach competition `biohub-cell-tracking-during-development` and the
private orgop helper `mahdimzafari/biohub-learned-unet-orgop-v1`.
Internet OFF. GPU OFF. Do not overwrite locked train/infer/holdout/
test/gap-test/prune-test/orgop slugs.
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

CONFIG_LABEL = "learned_unet_orgdet_v1"
TRAIN_LABEL = "learned_unet_v1"
INFER_LABEL = "learned_unet_infer_v1"
ORGOP_LABEL = "learned_unet_orgop_v1"
TRAIN_CONFIG_HASH = "4c38035d6657f2ad"
INFER_CONFIG_HASH = "d7578b5f64ddac08"
REJECTED_ORGOP_HASH = "250c5bfacd4a8198"
TUNING_SPECIMEN = "44b6"
HOLDOUT_SPECIMENS = ("6bba",)
INFER_STEMS = ("44b6_12dfb391", "44b6_267148e4")
FROZEN_HOLDOUT_STEMS = ("6bba_05db0fb1", "6bba_c328f2fd")
KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-v1"
ORGOP_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgop-v1"
TRAIN_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-v1"
INFER_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-infer-v1"
SCALE_ZYX = (1.625, 0.40625, 0.40625)
MAX_LINK_UM = 12.0
COMPETITION = "biohub-cell-tracking-during-development"


class HoldoutLeakError(ValueError):
    """Raised when a frozen-holdout specimen would enter inference."""


def orgdet_config_payload() -> dict[str, Any]:
    return {
        "label": CONFIG_LABEL,
        "train_label": TRAIN_LABEL,
        "infer_label": INFER_LABEL,
        "orgop_label": ORGOP_LABEL,
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": INFER_CONFIG_HASH,
        "rejected_orgop_hash": REJECTED_ORGOP_HASH,
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "infer_stems": list(INFER_STEMS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "source_graphs": "locked_learned_unet_orgop_graphs",
        "retunes_detection": False,
        "uses_intensity": False,
        "uses_transformer": False,
        "uses_velocity": False,
        "uses_ilp": False,
        "linker": "greedy_gate_12um",
        "motion_model": "last_position",
        "gate_um": 12.0,
        "max_link_um": 12.0,
        "max_gap_frames": 0,
        "assignment": "greedy",
        "interpolate_gaps": False,
        "max_parents_per_node": 1,
        "max_children_per_node": 1,
        "intensity_weight": 0.0,
        "det_threshold": 0.99,
        "det_tta": True,
        "pool_kernel_um": 3.0,
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": False,
        "kernel_slug": KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
        "orgop_kernel_slug": ORGOP_KERNEL_SLUG,
    }


def orgdet_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or orgdet_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGDET_HASH = orgdet_config_hash()
assert LOCKED_ORGDET_HASH == "80b20589198148b7"


def specimen_of(stem: str) -> str:
    return str(stem).split("_", 1)[0]


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
        raise HoldoutLeakError("learned_unet_orgdet_v1 refuses holdout stems: " + ", ".join(leaked))
    extra = [s for s in cleaned if s not in INFER_STEMS]
    if extra:
        raise HoldoutLeakError(
            "learned_unet_orgdet_v1 only opens frozen-tuning stems; got " + ", ".join(extra)
        )


def assert_no_submission_csv(working: str | Path) -> None:
    banned = Path(working) / "submission.csv"
    if banned.exists():
        banned.unlink()
        raise RuntimeError("learned_unet_orgdet_v1 must not write submission.csv")


def greedy_gate_links(
    coords_t: np.ndarray,
    coords_t1: np.ndarray,
    scale_zyx: Sequence[float] = SCALE_ZYX,
    gate_um: float = MAX_LINK_UM,
) -> list[tuple[int, int, float]]:
    a = np.asarray(coords_t, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(coords_t1, dtype=np.float64).reshape(-1, 3)
    if a.size == 0 or b.size == 0:
        return []
    scale = np.asarray(scale_zyx, dtype=np.float64)
    pa = a * scale
    pb = b * scale
    delta = pa[:, None, :] - pb[None, :, :]
    dist = np.sqrt(np.sum(delta * delta, axis=-1))
    candidates = [
        (float(dist[i, j]), int(i), int(j))
        for i in range(a.shape[0])
        for j in range(b.shape[0])
        if float(dist[i, j]) <= float(gate_um)
    ]
    candidates.sort()
    used_i: set[int] = set()
    used_j: set[int] = set()
    edges: list[tuple[int, int, float]] = []
    for d_um, i, j in candidates:
        if i in used_i or j in used_j:
            continue
        edges.append((i, j, d_um))
        used_i.add(i)
        used_j.add(j)
    return edges


def apply_orgdet_relink_arrays(
    node_id: Sequence[int],
    t: Sequence[int],
    z: Sequence[int],
    y: Sequence[int],
    x: Sequence[int],
) -> list[tuple[int, int]]:
    nids = [int(v) for v in node_id]
    ts = [int(v) for v in t]
    zs = [int(v) for v in z]
    ys = [int(v) for v in y]
    xs = [int(v) for v in x]
    by_t: dict[int, list[int]] = defaultdict(list)
    for i, tk in enumerate(ts):
        by_t[tk].append(i)
    times = sorted(by_t)
    edges: list[tuple[int, int]] = []
    for prev, cur in zip(times, times[1:]):
        if cur - prev != 1:
            continue
        idx_a = by_t[prev]
        idx_b = by_t[cur]
        coords_a = np.stack(
            [
                np.asarray([zs[i] for i in idx_a], dtype=np.float64),
                np.asarray([ys[i] for i in idx_a], dtype=np.float64),
                np.asarray([xs[i] for i in idx_a], dtype=np.float64),
            ],
            axis=1,
        )
        coords_b = np.stack(
            [
                np.asarray([zs[i] for i in idx_b], dtype=np.float64),
                np.asarray([ys[i] for i in idx_b], dtype=np.float64),
                np.asarray([xs[i] for i in idx_b], dtype=np.float64),
            ],
            axis=1,
        )
        for i, j, _dist in greedy_gate_links(coords_a, coords_b, SCALE_ZYX, MAX_LINK_UM):
            edges.append((int(nids[idx_a[i]]), int(nids[idx_b[j]])))
    return edges


def write_graph_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    refuse_non_infer_stems([str(payload["dataset"])])
    dest.write_text(json.dumps(payload, indent=2))
    return dest


def find_orgop_graphs(roots: Iterable[str | Path] | None = None) -> dict[str, Path]:
    if roots is None:
        roots = [
            Path("/kaggle/input/notebooks"),
            Path("/kaggle/input"),
            Path("artifacts/kaggle/learned_unet_orgop_v1"),
        ]
        extra = os.environ.get("BIOHUB_ORGOP_GRAPH_DIR")
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
    cfg = orgdet_config_payload()
    digest = orgdet_config_hash(cfg)
    if digest != LOCKED_ORGDET_HASH:
        raise RuntimeError("orgdet config hash drifted; refusing to retune")
    assert_no_submission_csv(working)
    report: dict[str, Any] = {
        "status": "blocked_no_graphs",
        "label": CONFIG_LABEL,
        "config_hash": digest,
        "kernel_slug": KERNEL_SLUG,
        "orgop_kernel_slug": ORGOP_KERNEL_SLUG,
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
    found = find_orgop_graphs()
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
        edges = apply_orgdet_relink_arrays(
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
