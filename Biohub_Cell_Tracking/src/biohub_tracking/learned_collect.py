"""Collect private learned_unet helper and test-time outputs after COMPLETE.

Downloads stay under ignored ``artifacts/kaggle/``. Weights are never written
to git. Train/infer/holdout helpers must not produce ``submission.csv``. The
test-time kernel is allowed to write ``submission.csv`` only into the ignored
artifact directory. A local extract is not a Kaggle run and not a leaderboard
score.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

from .learned_split import (
    FROZEN_HOLDOUT_STEMS,
    INTERNAL_VAL_STEMS,
    KERNEL_SLUG as TRAIN_KERNEL_SLUG,
    HoldoutLeakError,
    find_weights,
    refuse_holdout_stems,
    specimen_of,
    verify_train_manifest,
)

TRAIN_SLUG = TRAIN_KERNEL_SLUG
INFER_SLUG = "mahdimzafari/biohub-learned-unet-infer-v1"
HOLDOUT_SLUG = "mahdimzafari/biohub-learned-unet-holdout-v1"
TEST_SLUG = "mahdimzafari/biohub-learned-unet-test-v1"
GAP_TEST_SLUG = "mahdimzafari/biohub-learned-unet-gap-test-v1"
PRUNE_TEST_SLUG = "mahdimzafari/biohub-learned-unet-prune-test-v1"
ORGOP_SLUG = "mahdimzafari/biohub-learned-unet-orgop-v1"
ORGDET_HOLDOUT_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-holdout-v1"
ORGDET_TEST_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-test-v1"
ORGDET_GAP_TEST_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1"
ORGDET_GAP_PRUNE_TEST_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-gap-prune-test-v1"
ORGDET_ELF_TEST_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-elf-test-v1"
LINK_SLUG = "mahdimzafari/biohub-learned-unet-link-v1"
LINK_V2_SLUG = "mahdimzafari/biohub-learned-unet-link-v2"
INFER_STEMS = INTERNAL_VAL_STEMS
HOLDOUT_STEMS = FROZEN_HOLDOUT_STEMS
HOLDOUT_CONFIG_HASH = "d374a21f03dddc04"
TEST_CONFIG_HASH = "7a8725b818fc609b"
GAP_TEST_CONFIG_HASH = "85379c198ebc7241"
PRUNE_TEST_CONFIG_HASH = "074a71dec68304e8"
ORGOP_CONFIG_HASH = "250c5bfacd4a8198"
ORGDET_HOLDOUT_CONFIG_HASH = "912f0c19b572a054"
ORGDET_TEST_CONFIG_HASH = "c45c859610d29e90"
ORGDET_GAP_TEST_CONFIG_HASH = "1ce4837ae16bee52"
ORGDET_GAP_PRUNE_TEST_CONFIG_HASH = "52a30a5843e0a880"
ORGDET_ELF_TEST_CONFIG_HASH = "ad375d39e13098d1"
LINK_V2_CONFIG_HASH = "1d4015dc499bd604"
RETRAIN_SLUG = "mahdimzafari/biohub-learned-unet-retrain-v1"
RETRAIN_CONFIG_HASH = "92a419926fa7e40b"
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
RETRAIN_ALLOWED_NAMES = frozenset(
    {
        "retrain_manifest.json",
        "dataset_splits.json",
        "learned_unet_retrain_v1_config.json",
        "learned_unet_retrain_v1_best.pt",
        "learned_unet_retrain_v1_last.pt",
    }
)
INFER_ALLOWED_NAMES = frozenset(
    {
        "infer_manifest.json",
        "44b6_12dfb391.json",
        "44b6_267148e4.json",
    }
)
HOLDOUT_ALLOWED_NAMES = frozenset(
    {
        "infer_manifest.json",
        "6bba_05db0fb1.json",
        "6bba_c328f2fd.json",
    }
)
TEST_ALLOWED_NAMES = frozenset(
    {
        "submission.csv",
        "run_manifest.json",
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
    graph_guard: str = "infer",
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
                if graph_guard == "holdout":
                    _refuse_non_holdout_stems([name])
                else:
                    _refuse_non_infer_stems([name])
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


def extract_learned_retrain_output(archive: Path, dest_dir: Path) -> dict[str, Path]:
    """Extract continuation weights and manifests. Refuses submission.csv."""
    from .learned_retrain import verify_retrain_manifest

    written = _extract_allowed(Path(archive), Path(dest_dir), set(RETRAIN_ALLOWED_NAMES))
    manifest = written.get("retrain_manifest.json")
    if manifest is not None:
        verify_retrain_manifest(manifest)
    splits_path = written.get("dataset_splits.json")
    if splits_path is not None:
        splits = json.loads(splits_path.read_text())
        stems = list(splits.get("train") or []) + list(splits.get("test") or [])
        if stems:
            refuse_holdout_stems(stems)
    best = dest_dir / "learned_unet_retrain_v1_best.pt"
    last = dest_dir / "learned_unet_retrain_v1_last.pt"
    if not written and not best.exists() and not last.exists():
        raise FileNotFoundError("learned retrain helper zip had no allowed files")
    return written


def _refuse_non_infer_stems(stems: Iterable[str]) -> None:
    """Allow only the two frozen-tuning sequences. Holdout is always refused."""
    cleaned: list[str] = []
    for raw in stems:
        stem = Path(raw).name
        for suffix in (".zarr", ".geff", ".json"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        if stem not in cleaned:
            cleaned.append(stem)
    refuse_holdout_stems(cleaned)
    extra = [s for s in cleaned if s not in INFER_STEMS]
    if extra:
        raise HoldoutLeakError(
            "learned_unet_infer_v1 only opens frozen-tuning stems; got " + ", ".join(extra)
        )


def extract_learned_link_output(archive: Path, dest_dir: Path) -> dict[str, Path]:
    """Extract 44b6-only transformer-linker graphs. Same files as infer."""
    return extract_learned_infer_output(archive, dest_dir)


def extract_learned_infer_output(archive: Path, dest_dir: Path) -> dict[str, Path]:
    """Extract 44b6-only infer graphs. Refuses holdout stems and submission.csv."""
    _refuse_non_infer_stems(
        name
        for name in _basenames(_zip_files(Path(archive)))
        if name.endswith(".json") and name != "infer_manifest.json"
    )
    written = _extract_allowed(
        Path(archive),
        Path(dest_dir),
        set(INFER_ALLOWED_NAMES),
        keep_graphs_subdir=True,
        graph_guard="infer",
    )
    graph_names = [name for name in written if name.endswith(".json") and name != "infer_manifest.json"]
    if graph_names:
        _refuse_non_infer_stems(graph_names)
    missing = [stem for stem in INFER_STEMS if f"{stem}.json" not in written]
    if missing and written and "infer_manifest.json" in written:
        # Graphs may be absent on a blocked/dry run; do not invent them.
        pass
    return written


def _refuse_non_holdout_stems(stems: Iterable[str]) -> None:
    """Allow only the two frozen-holdout sequences. Tuning graphs are refused."""
    cleaned: list[str] = []
    for raw in stems:
        stem = Path(raw).name
        for suffix in (".zarr", ".geff", ".json"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        if stem not in cleaned:
            cleaned.append(stem)
    leaked = [s for s in cleaned if s in INFER_STEMS or specimen_of(s) == "44b6"]
    if leaked:
        raise HoldoutLeakError(
            "learned_unet_holdout_v1 refuses 44b6 tuning graphs: " + ", ".join(leaked)
        )
    extra = [s for s in cleaned if s not in HOLDOUT_STEMS]
    if extra:
        raise HoldoutLeakError(
            "learned_unet_holdout_v1 only collects frozen-holdout stems; got " + ", ".join(extra)
        )


def extract_learned_test_output(archive: Path, dest_dir: Path) -> dict[str, Path]:
    """Extract test-time submission.csv + run_manifest.json into artifacts/kaggle.

    Refuses weights and competition volumes. A local extract is not a
    competition submission and not a leaderboard score.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    names = _zip_files(Path(archive))
    leaked = [name for name in _basenames(names) if name.endswith((".zarr", ".geff", ".pt", ".pth"))]
    if leaked:
        raise ValueError("refusing weights/volume members in test-time zip: " + ", ".join(sorted(leaked)))
    unexpected = sorted(
        name for name in _basenames(names) if name not in TEST_ALLOWED_NAMES and not _is_noise(name)
    )
    if unexpected:
        raise ValueError(f"refusing unexpected learned test output files: {unexpected}")
    written: dict[str, Path] = {}
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if name not in TEST_ALLOWED_NAMES:
                continue
            dest = dest_dir / name
            dest.write_bytes(zf.read(info.filename))
            written[name] = dest
    return written


def extract_learned_holdout_output(archive: Path, dest_dir: Path) -> dict[str, Path]:
    """Extract 6bba-only holdout graphs. Refuses 44b6 graphs and submission.csv."""
    _refuse_non_holdout_stems(
        name
        for name in _basenames(_zip_files(Path(archive)))
        if name.endswith(".json") and name != "infer_manifest.json"
    )
    written = _extract_allowed(
        Path(archive),
        Path(dest_dir),
        set(HOLDOUT_ALLOWED_NAMES),
        keep_graphs_subdir=True,
        graph_guard="holdout",
    )
    graph_names = [name for name in written if name.endswith(".json") and name != "infer_manifest.json"]
    if graph_names:
        _refuse_non_holdout_stems(graph_names)
    return written


def safe_kernel_status(raw: dict[str, Any], slug: str) -> dict[str, Any]:
    """Return non-secret kernel status fields only."""
    status = str(raw.get("status") or "")
    last_run = raw.get("lastRunTime") or raw.get("lastRun") or raw.get("startedOn")
    return {
        "slug": slug,
        "status": status,
        "failureMessage": raw.get("failureMessage") if raw.get("hasFailureMessage") else None,
        "hasFailureMessage": bool(raw.get("hasFailureMessage")),
        "versionNumber": raw.get("versionNumber") or raw.get("currentVersionNumber"),
        "lastRunTime": last_run,
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


def summarize_collected_retrain(dest_dir: Path) -> dict[str, Any]:
    from .learned_retrain import verify_retrain_manifest

    dest_dir = Path(dest_dir)
    manifest_path = dest_dir / "retrain_manifest.json"
    payload: dict[str, Any] = {}
    if manifest_path.exists():
        payload = verify_retrain_manifest(manifest_path)
    history = payload.get("history") or []
    return {
        "status": payload.get("status") or "no_manifest",
        "config_hash": payload.get("config_hash") or RETRAIN_CONFIG_HASH,
        "splits": payload.get("splits"),
        "n_epochs": len(history),
        "best_train_loss": payload.get("best_train_loss"),
        "last_edge_loss": history[-1].get("edge_loss") if history else None,
        "last_transformer_grad_norm": (
            history[-1].get("transformer_grad_norm") if history else None
        ),
        "last_n_candidate_edges": history[-1].get("n_candidate_edges") if history else None,
        "wrote_best": (dest_dir / "learned_unet_retrain_v1_best.pt").is_file(),
        "wrote_last": (dest_dir / "learned_unet_retrain_v1_last.pt").is_file(),
        "best_sha16": payload.get("best_sha16"),
        "last_sha16": payload.get("last_sha16"),
        "checkpoint_reload_exact": payload.get("checkpoint_reload_exact"),
        "runtime_sec": payload.get("runtime_sec"),
        "peak_memory_mb": payload.get("peak_memory_mb"),
        "resumed_from": Path(str(payload.get("resumed_from") or "")).name or None,
        "has_submission_csv": (dest_dir / "submission.csv").exists(),
        "is_kaggle_run": bool(payload.get("is_kaggle_run")),
        "is_competition_submission": False,
        "is_leaderboard_score": False,
    }
