"""Collect private learned_unet helper outputs after a COMPLETE Kaggle run.

Downloads stay under ignored ``artifacts/kaggle/``. Weights are never written
to git. Helper kernels must not produce ``submission.csv``. A local extract is
not a Kaggle run and not a leaderboard score.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

from .learned_infer import (
    INFER_KERNEL_SLUG,
    INFER_STEMS,
    TRAIN_KERNEL_SLUG,
    find_weights,
    refuse_non_infer_stems,
    verify_train_manifest,
)
from .learned_split import HoldoutLeakError, refuse_holdout_stems

TRAIN_SLUG = TRAIN_KERNEL_SLUG
INFER_SLUG = INFER_KERNEL_SLUG
BANNED_OUTPUT_NAMES = frozenset({"submission.csv"})
KAGGLE_NOISE_NAMES = frozenset(
    {
        "__notebook__.ipynb",
        "__notebook_source__.ipynb",
        "__output__.json",
        "custom.css",
    }
)
TRAIN_ALLOWED_NAMES = frozenset(
    {
        "train_manifest.json",
        "dataset_splits.json",
        "learned_unet_v1_config.json",
        "learned_unet_v1_best.pt",
        "learned_unet_v1_last.pt",
    }
)
INFER_ALLOWED_NAMES = frozenset(
    {
        "infer_manifest.json",
        "44b6_12dfb391.json",
        "44b6_267148e4.json",
    }
)


def assert_helper_dest(dest: str | Path) -> Path:
    dest = Path(dest)
    posix = dest.resolve().as_posix()
    if "/artifacts/kaggle/" not in posix and not posix.endswith("/artifacts/kaggle"):
        raise ValueError("refusing to write learned helper output outside artifacts/kaggle")
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _zip_files(archive: Path) -> list[str]:
    with zipfile.ZipFile(archive) as zf:
        return [info.filename for info in zf.infolist() if not info.is_dir()]


def _basenames(names: Iterable[str]) -> set[str]:
    return {Path(name).name for name in names}


def _is_noise(name: str) -> bool:
    base = Path(name).name
    if base in KAGGLE_NOISE_NAMES or base.startswith("__"):
        return True
    return base.endswith((".css", ".html", ".log"))


def refuse_submission_outputs(names: Iterable[str]) -> None:
    banned = sorted(_basenames(names) & BANNED_OUTPUT_NAMES)
    if banned:
        raise ValueError(
            "learned helper kernel must not write " + ", ".join(banned)
        )
    leaked = [name for name in _basenames(names) if name.endswith((".zarr", ".geff"))]
    if leaked:
        raise ValueError("refusing competition volume/GEFF members in helper zip: " + ", ".join(sorted(leaked)))


def _extract_allowed(
    archive: Path,
    dest_dir: Path,
    allowed: set[str],
    *,
    keep_graphs_subdir: bool = False,
) -> dict[str, Path]:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    names = _zip_files(archive)
    refuse_submission_outputs(names)
    basenames = _basenames(names)
    unexpected = sorted(name for name in basenames if name not in allowed and not _is_noise(name))
    if unexpected:
        raise ValueError(f"refusing unexpected learned helper output files: {unexpected}")
    written: dict[str, Path] = {}
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if name not in allowed:
                continue
            if keep_graphs_subdir and name.endswith(".json") and name != "infer_manifest.json":
                refuse_non_infer_stems([name])
                dest = dest_dir / "graphs" / name
            else:
                dest = dest_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(info.filename))
            written[name] = dest
    return written


def extract_learned_train_output(archive: Path, dest_dir: Path) -> dict[str, Path]:
    """Extract training-helper weights and manifests. Refuses submission.csv."""
    written = _extract_allowed(Path(archive), Path(dest_dir), set(TRAIN_ALLOWED_NAMES))
    manifest = written.get("train_manifest.json")
    if manifest is not None:
        verify_train_manifest(manifest)
    splits_path = written.get("dataset_splits.json")
    if splits_path is not None:
        splits = json.loads(splits_path.read_text())
        stems = list(splits.get("train") or []) + list(splits.get("test") or [])
        if stems:
            refuse_holdout_stems(stems)
    if find_weights([dest_dir]) is None and not written:
        raise FileNotFoundError("learned train helper zip had no allowed files")
    return written


def extract_learned_infer_output(archive: Path, dest_dir: Path) -> dict[str, Path]:
    """Extract 44b6-only infer graphs. Refuses holdout stems and submission.csv."""
    refuse_non_infer_stems(
        name
        for name in _basenames(_zip_files(Path(archive)))
        if name.endswith(".json") and name != "infer_manifest.json"
    )
    written = _extract_allowed(
        Path(archive),
        Path(dest_dir),
        set(INFER_ALLOWED_NAMES),
        keep_graphs_subdir=True,
    )
    graph_names = [name for name in written if name.endswith(".json") and name != "infer_manifest.json"]
    if graph_names:
        refuse_non_infer_stems(graph_names)
    missing = [stem for stem in INFER_STEMS if f"{stem}.json" not in written]
    if missing and written and "infer_manifest.json" in written:
        # Graphs may be absent on a blocked/dry run; do not invent them.
        pass
    return written


def safe_kernel_status(raw: dict[str, Any], slug: str) -> dict[str, Any]:
    """Return non-secret kernel status fields only."""
    status = str(raw.get("status") or "")
    return {
        "slug": slug,
        "status": status,
        "failureMessage": raw.get("failureMessage") if raw.get("hasFailureMessage") else None,
        "hasFailureMessage": bool(raw.get("hasFailureMessage")),
        "versionNumber": raw.get("versionNumber") or raw.get("currentVersionNumber"),
        "is_complete": status.lower() == "complete",
        "is_error": status.lower() in {"error", "failed"} or bool(raw.get("hasFailureMessage")),
        "is_running": status.lower() in {"running", "queued", "pending", "canceling"},
        "is_competition_submission": False,
        "is_leaderboard_score": False,
    }


def collect_decision(status: dict[str, Any]) -> str:
    """Decide whether a COMPLETE helper zip may be downloaded."""
    if status.get("is_error"):
        return "record_train_error"
    if status.get("is_complete"):
        return "download_complete_helper"
    if status.get("is_running"):
        return "wait_running"
    return "wait_unknown"


def summarize_collected_train(dest_dir: Path) -> dict[str, Any]:
    dest_dir = Path(dest_dir)
    manifest_path = dest_dir / "train_manifest.json"
    payload: dict[str, Any] = {}
    if manifest_path.exists():
        payload = verify_train_manifest(manifest_path)
    weights = find_weights([dest_dir])
    return {
        "status": payload.get("status") or ("weights_present" if weights else "no_weights"),
        "config_hash": payload.get("config_hash"),
        "splits": payload.get("splits"),
        "wrote_best": (dest_dir / "learned_unet_v1_best.pt").is_file(),
        "wrote_last": (dest_dir / "learned_unet_v1_last.pt").is_file(),
        "weights_name": weights.name if weights else None,
        "has_submission_csv": (dest_dir / "submission.csv").exists(),
        "is_kaggle_run": bool(payload.get("is_kaggle_run")),
        "is_competition_submission": False,
        "is_leaderboard_score": False,
    }
