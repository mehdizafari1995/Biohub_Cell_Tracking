"""Test-time inference helpers for learned_unet_test_v1.

Uses locked ``learned_unet_holdout_v1`` / ``learned_unet_infer_v1`` operating
point and specimen-separated ``learned_unet_v1`` weights. Discovers every
``test/*.zarr`` at runtime. Never opens ``train/`` volumes. Writes
``submission.csv`` + ``run_manifest.json``. Not a competition submission
until a human submits from the Kaggle UI. Do not retune. Do not score
frozen ``6bba`` holdout again.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .learned_holdout import LOCKED_HOLDOUT_HASH
from .learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from .learned_split import KERNEL_SLUG as TRAIN_KERNEL_SLUG

TEST_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-test-v1"
INFER_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-infer-v1"
HOLDOUT_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-holdout-v1"
WEIGHTS_NAME = "learned_unet_v1_best.pt"
SCALE_ZYX = (1.625, 0.40625, 0.40625)
HOLDOUT_SEC_PER_VOLUME = 92.2
HOLDOUT_PEAK_MEMORY_MB = 1177.0
ESTIMATE_HIDDEN_VOLUMES = 199
KAGGLE_GPU_LIMIT_HOURS = 12.0
COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]


def test_config_payload() -> dict[str, Any]:
    """Locked infer operating point applied to discovered test volumes."""
    return {
        "label": "learned_unet_test_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "holdout_label": "learned_unet_holdout_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "holdout_config_hash": LOCKED_HOLDOUT_HASH,
        "discovers_test_zarrs": True,
        "hardcoded_test_stems": [],
        "opens_train_volumes": False,
        "det_threshold": 0.5,
        "det_tta": True,
        "pool_kernel_um": 5.0,
        "linker": "greedy_gate_12um",
        "gate_um": 12.0,
        "max_gap_frames": 0,
        "max_parents_per_node": 1,
        "max_children_per_node": 1,
        "downsample": [1, 4, 4],
        "normalize": "window_minmax",
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": True,
        "writes_run_manifest": True,
        "one_shot_holdout": False,
        "kernel_slug": TEST_KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
        "holdout_kernel_slug": HOLDOUT_KERNEL_SLUG,
        "weights_name": WEIGHTS_NAME,
    }


def test_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or test_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_TEST_HASH = test_config_hash(test_config_payload())


def runtime_estimate(
    n_volumes: int,
    sec_per_volume: float = HOLDOUT_SEC_PER_VOLUME,
    limit_hours: float = KAGGLE_GPU_LIMIT_HOURS,
    peak_memory_mb: float = HOLDOUT_PEAK_MEMORY_MB,
) -> dict[str, Any]:
    """Estimate GPU time from holdout v1 (~92 s/volume) before a push."""
    n = int(n_volumes)
    sec = float(n) * float(sec_per_volume)
    hours = sec / 3600.0
    return {
        "n_volumes": n,
        "sec_per_volume": float(sec_per_volume),
        "estimate_sec": sec,
        "estimate_hours": hours,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_observed": float(peak_memory_mb),
        "source": "learned_unet_holdout_v1 v1 184.4s / 2 volumes",
    }


def list_test_zarrs(test_dir: str | Path) -> list[Path]:
    root = Path(test_dir)
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name.endswith(".zarr"))


def discovered_stems(zarrs: Sequence[str | Path]) -> list[str]:
    stems: list[str] = []
    seen: set[str] = set()
    for raw in zarrs:
        stem = Path(raw).name
        if stem.endswith(".zarr"):
            stem = stem[: -len(".zarr")]
        if stem and stem not in seen:
            stems.append(stem)
            seen.add(stem)
    return stems


def refuse_train_volume(path: str | Path) -> Path:
    """Test-time inference may open test volumes only."""
    dest = Path(path)
    parts = {part.lower() for part in dest.parts}
    if "train" in parts and dest.name.endswith(".zarr"):
        raise ValueError(f"learned_unet_test_v1 refuses train volumes: {dest}")
    return dest


def refuse_hardcoded_public_stems(stems: Iterable[str]) -> None:
    if list(stems):
        raise ValueError(
            "learned_unet_test_v1 must discover test/*.zarr; hardcoded stems="
            + ",".join(str(s) for s in stems)
        )


def sample_submission_datasets(root: str | Path) -> list[str]:
    path = Path(root) / "sample_submission.csv"
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        names = {str(row.get("dataset") or "") for row in reader}
    return sorted(name for name in names if name)


def graph_rows(
    dataset: str,
    coords: Sequence[Sequence[int]],
    edges: Sequence[Sequence[int]],
    start_id: int,
) -> tuple[list[list[Any]], int]:
    rows: list[list[Any]] = []
    next_id = int(start_id)
    for index, (t, z, y, x) in enumerate(coords, start=1):
        rows.append([next_id, dataset, "node", int(index), int(t), int(z), int(y), int(x), -1, -1])
        next_id += 1
    for source_id, target_id in edges:
        rows.append([next_id, dataset, "edge", -1, -1, -1, -1, -1, int(source_id), int(target_id)])
        next_id += 1
    return rows, next_id


def write_submission_header(path: str | Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as fh:
        csv.writer(fh).writerow(COLUMNS)
    return dest


def append_submission_rows(path: str | Path, rows: Sequence[Sequence[Any]]) -> None:
    with Path(path).open("a", newline="") as fh:
        writer = csv.writer(fh)
        for row in rows:
            writer.writerow(row)


def audit_graph(
    dataset: str,
    coords: Sequence[Sequence[int]],
    edges: Sequence[Sequence[int]],
    shape: Sequence[int] | None = None,
) -> list[str]:
    errors: list[str] = []
    node_ids = list(range(1, len(coords) + 1))
    if len(set(node_ids)) != len(node_ids):
        errors.append(f"{dataset}: duplicate node_id")
    t_by_id = {i + 1: int(row[0]) for i, row in enumerate(coords)}
    out_count: dict[int, int] = {}
    for source_id, target_id in edges:
        src, dst = int(source_id), int(target_id)
        if src not in t_by_id or dst not in t_by_id:
            errors.append(f"{dataset}: edge {src}->{dst} references missing node_id")
            break
        if t_by_id[dst] <= t_by_id[src]:
            errors.append(f"{dataset}: source time does not precede target time ({src}->{dst})")
            break
        if t_by_id[dst] - t_by_id[src] != 1:
            errors.append(f"{dataset}: non-consecutive edge dt={t_by_id[dst] - t_by_id[src]}")
            break
        out_count[src] = out_count.get(src, 0) + 1
    if any(count >= 2 for count in out_count.values()):
        errors.append(f"{dataset}: predicted a division; learned_unet_test_v1 is 1-1")
    if shape is not None and coords:
        n_time, n_z, n_y, n_x = (int(v) for v in shape[:4])
        for t, z, y, x in coords:
            if not (0 <= int(t) < n_time):
                errors.append(f"{dataset}: t outside [0, {n_time - 1}]")
                break
            if not (0 <= int(z) < n_z and 0 <= int(y) < n_y and 0 <= int(x) < n_x):
                errors.append(f"{dataset}: centroid outside volume bounds")
                break
    return errors


def audit_discovered_coverage(discovered: Sequence[str], present: Sequence[str], sample_names: Sequence[str]) -> list[str]:
    errors: list[str] = []
    missing = [name for name in discovered if name not in set(present)]
    extra = sorted(set(present) - set(discovered))
    if missing:
        errors.append(f"missing discovered datasets: {missing}")
    if extra:
        errors.append(f"unexpected datasets: {extra}")
    sample_in_test = [name for name in sample_names if name in set(discovered)]
    sample_missing = [name for name in sample_in_test if name not in set(present)]
    if sample_missing:
        errors.append(
            "sample_submission datasets present in test/ but missing from CSV: " + ",".join(sample_missing)
        )
    return errors
