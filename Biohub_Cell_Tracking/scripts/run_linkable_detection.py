#!/usr/bin/env python3
"""Official-score linkable_v1 vs locked association_v2.

Tune on 44b6, then score frozen 6bba once after the lock file records an
operating point. Does not overwrite baseline_v1, association_v2,
division_v2, or complementary_v1. A local official score is not a Kaggle
run and not a leaderboard number. A second holdout score is refused.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from biohub_tracking.association import FrameDetections  # noqa: E402
from biohub_tracking.config import PipelineConfig, load_config, pipeline_config_hash  # noqa: E402
from biohub_tracking.detect import detect_volume_with_appearance  # noqa: E402
from biohub_tracking.errors import analyse_errors  # noqa: E402
from biohub_tracking.experiments import (  # noqa: E402
    append_experiment,
    git_version,
    new_experiment_id,
    peak_memory_mb,
    write_run_json,
)
from biohub_tracking.io_geff import load_geff  # noqa: E402
from biohub_tracking.io_zarr import open_volume  # noqa: E402
from biohub_tracking.official_eval import (  # noqa: E402
    evaluate_official,
    official_dependency_versions,
    official_scorer_available,
    summarise_official,
)
from biohub_tracking.track import graph_from_cached_frames  # noqa: E402
from biohub_tracking.validation import (  # noqa: E402
    FrozenSplitError,
    LOCKED_V1_TUNING,
    LOCKED_V1_VALIDATION,
    assert_specimen_separated,
    inventory_validation,
    load_split_manifest,
)

TUNING_DATASETS = LOCKED_V1_TUNING
VALIDATION_DATASETS = LOCKED_V1_VALIDATION
CACHE = ROOT / "artifacts" / "cache" / "detections"
PRIORS = ROOT / "docs" / "linkable_priors.json"
ASSOC_LOCKED_TUNE = 0.6324600072329115
ASSOC_LOCKED_HOLDOUT = 0.5915697467955445


def _lock_payload(lock_path: Path | None) -> dict:
    if lock_path is None or not lock_path.exists():
        return {}
    return json.loads(lock_path.read_text()) or {}


def assert_tuning_only(role: str, lock_path: Path | None = None) -> None:
    """Refuse holdout until locked, and refuse a second holdout score."""
    if role == "tuning":
        return
    payload = _lock_payload(lock_path)
    if not payload.get("locked_operating_point"):
        raise SystemExit("refusing to score holdout before linkable_v1 is locked on tuning")
    if payload.get("holdout_scored"):
        raise SystemExit("refusing to re-score frozen holdout; linkable_v1 already scored once")


def cache_path(dataset: str, label: str) -> Path:
    return CACHE / f"{dataset}_{label}.npz"


def load_or_detect(
    zarr_path: Path,
    config: PipelineConfig,
    label: str,
) -> tuple[list[FrameDetections], tuple[int, int, int]]:
    array, meta = open_volume(zarr_path)
    dest = cache_path(meta.name, label)
    shape = (int(meta.shape[1]), int(meta.shape[2]), int(meta.shape[3]))
    if dest.exists():
        blob = np.load(dest, allow_pickle=True)
        frames = []
        for t, coords, inten, app, ids in zip(
            blob["times"], blob["coords"], blob["intensity"], blob["appearance"], blob["node_ids"]
        ):
            frames.append(
                FrameDetections(
                    t=int(t),
                    coords=np.asarray(coords, dtype=np.float64),
                    intensity=np.asarray(inten, dtype=np.float64),
                    appearance=np.asarray(app, dtype=np.float64),
                    node_ids=np.asarray(ids, dtype=np.int64),
                )
            )
        return frames, shape
    raw = [(int(t), np.asarray(array[t])) for t in meta.available_timepoints]
    detected = detect_volume_with_appearance(raw, config.detection, config.scale_zyx)
    frames: list[FrameDetections] = []
    next_id = 1
    times, coords_l, inten_l, app_l, ids_l = [], [], [], [], []
    for t, coords, intensity, appearance in detected:
        n = len(coords)
        ids = np.arange(next_id, next_id + n, dtype=np.int64)
        next_id += n
        frames.append(FrameDetections(int(t), coords, intensity, appearance, ids))
        times.append(int(t))
        coords_l.append(coords)
        inten_l.append(intensity)
        app_l.append(appearance)
        ids_l.append(ids)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dest,
        times=np.array(times, dtype=np.int64),
        coords=np.array(coords_l, dtype=object),
        intensity=np.array(inten_l, dtype=object),
        appearance=np.array(app_l, dtype=object),
        node_ids=np.array(ids_l, dtype=object),
    )
    return frames, shape


def _score_one(
    name: str,
    datasets: list[str],
    data_dir: Path,
    det_cfg: PipelineConfig,
    link_cfg: PipelineConfig,
    det_label: str,
) -> dict:
    experiment_id = new_experiment_id(name)
    official_rows = []
    error_rows = []
    per_sample = []
    n_pred_nodes = 0
    n_pred_edges = 0
    for dataset in datasets:
        zarr_path = data_dir / f"{dataset}.zarr"
        geff_path = data_dir / f"{dataset}.geff"
        if not zarr_path.exists() or not geff_path.exists():
            continue
        frames, shape = load_or_detect(zarr_path, det_cfg, det_label)
        _, meta = open_volume(zarr_path)
        gt = load_geff(geff_path).filter_timepoints(meta.available_timepoints)
        pred = graph_from_cached_frames(dataset, frames, link_cfg, shape, gt.estimated_n_nodes)
        off = evaluate_official(pred, gt, scale_zyx=link_cfg.scale_zyx)
        analysis = analyse_errors(pred, gt, link_cfg.scale_zyx)
        official_rows.append(off)
        error_rows.append(analysis.summary())
        n_nodes = int(sum(len(f.coords) for f in frames))
        n_pred_nodes += int(pred.n_nodes)
        n_pred_edges += int(pred.n_edges)
        per_sample.append(
            {
                "dataset": dataset,
                "official": off.as_dict(),
                "errors": analysis.summary(),
                "n_det_nodes": n_nodes,
                "n_pred_nodes": pred.n_nodes,
                "n_pred_edges": pred.n_edges,
            }
        )
    summary = summarise_official(official_rows) if official_rows else {}
    switches = sum(int(r.get("identity_switches") or 0) for r in error_rows)
    fragments = sum(int(r.get("fragmented_tracks") or 0) for r in error_rows)
    hashed = pipeline_config_hash(link_cfg if det_label == "baseline_v1" else det_cfg)
    return {
        "experiment_id": experiment_id,
        "variant": name,
        "detection_label": det_label,
        "config_hash": hashed,
        "official_summary": summary,
        "per_sample": per_sample,
        "identity_switches": switches,
        "fragmented_tracks": fragments,
        "n_pred_nodes": n_pred_nodes,
        "n_pred_edges": n_pred_edges,
        "n_scored": len(official_rows),
    }


def _record(registry: Path, payload: dict, extra_note: str) -> None:
    summary = payload.get("official_summary") or {}
    append_experiment(
        registry,
        {
            "experiment_id": payload["experiment_id"],
            "timestamp_utc": payload["experiment_id"].split("_")[0],
            "code_version": payload["code_version"],
            "config_name": payload["variant"],
            "seed": 0,
            "split": payload["split"],
            "n_samples": summary.get("n", 0),
            "score": summary.get("score"),
            "edge_jaccard": summary.get("edge_jaccard"),
            "adj_edge_jaccard": summary.get("adj_edge_jaccard"),
            "division_jaccard": summary.get("division_jaccard"),
            "node_recall": summary.get("node_recall"),
            "identity_switches": payload.get("identity_switches"),
            "broken_tracks": payload.get("fragmented_tracks"),
            "division_tp": summary.get("division_tp"),
            "division_fp": summary.get("division_fp"),
            "division_fn": summary.get("division_fn"),
            "runtime_sec": round(payload["runtime_sec"], 3),
            "peak_memory_mb": round(payload["peak_memory_mb"], 1)
            if payload["peak_memory_mb"] == payload["peak_memory_mb"]
            else "",
            "notes": extra_note,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--split", type=Path, default=ROOT / "docs" / "splits" / "v1.json")
    parser.add_argument("--role", choices=["tuning", "validation"], default="tuning")
    parser.add_argument("--lock-path", type=Path, default=PRIORS)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    assert_tuning_only(args.role, args.lock_path)
    manifest = load_split_manifest(args.split)
    try:
        assert_specimen_separated(manifest)
    except FrozenSplitError as exc:
        raise SystemExit(f"frozen split error: {exc}") from exc
    datasets = list(manifest[args.role])
    if args.role == "tuning" and tuple(datasets) != TUNING_DATASETS:
        raise SystemExit(f"refusing rewritten tuning set: {datasets}")
    if args.role == "validation" and tuple(datasets) != VALIDATION_DATASETS:
        raise SystemExit(f"refusing rewritten validation set: {datasets}")
    assoc_cfg = load_config(ROOT / "configs" / "association_v2.yaml")
    link_cfg = load_config(ROOT / "configs" / "linkable_v1.yaml")
    lock_payload = _lock_payload(args.lock_path)
    locked = bool(lock_payload.get("locked_operating_point"))
    inventory = inventory_validation(args.data_dir, datasets)
    t0 = time.perf_counter()
    assoc = _score_one("association_v2", datasets, args.data_dir, assoc_cfg, assoc_cfg, "baseline_v1")
    cand = _score_one("linkable_v1", datasets, args.data_dir, link_cfg, link_cfg, "linkable_v1")
    runtime = time.perf_counter() - t0
    memory = peak_memory_mb()
    code = git_version(ROOT.parent)
    assoc_score = (assoc.get("official_summary") or {}).get("score")
    cand_score = (cand.get("official_summary") or {}).get("score")
    decision = "pending_no_volumes"
    if assoc_score is not None and cand_score is not None:
        if float(cand_score) < float(assoc_score):
            decision = "reject_linkable_v1"
        elif args.role == "validation":
            decision = "keep_linkable_v1"
        else:
            decision = "keep_candidate_linkable_v1"
    complete = all(item.zarr_exists and item.geff_exists for item in inventory) and len(inventory) == len(datasets)
    report = {
        "decision": decision,
        "complete": complete,
        "role": args.role,
        "datasets": datasets,
        "inventory": [item.as_dict() for item in inventory],
        "association_v2": assoc,
        "linkable_v1": cand,
        "association_v2_locked_tune_score": ASSOC_LOCKED_TUNE,
        "association_v2_locked_holdout_score": ASSOC_LOCKED_HOLDOUT,
        "runtime_sec": runtime,
        "peak_memory_mb": memory,
        "code_version": code,
        "official_versions": official_dependency_versions(),
        "official_available": official_scorer_available(),
        "config_hash_association_v2": pipeline_config_hash(assoc_cfg),
        "config_hash_linkable_v1": pipeline_config_hash(link_cfg),
        "score_source": "local_official",
        "not_a_kaggle_run": True,
        "not_a_leaderboard_score": True,
        "retune_after_holdout": False,
        "locked": locked,
    }
    assoc.update({"code_version": code, "runtime_sec": runtime, "peak_memory_mb": memory, "split": f"{args.role}:v1.json"})
    cand.update({"code_version": code, "runtime_sec": runtime, "peak_memory_mb": memory, "split": f"{args.role}:v1.json"})
    dest = ROOT / "artifacts" / "runs" / f"linkable_{args.role}_pair.json"
    write_run_json(dest, report)
    if args.record and complete:
        note = (
            f"linkable_v1 vs association_v2 on {args.role} only; "
            f"decision={decision}; hashes assoc={report['config_hash_association_v2']} "
            f"linkable={report['config_hash_linkable_v1']}; not a leaderboard score"
        )
        _record(ROOT / "experiments" / "registry.csv", assoc, note)
        _record(ROOT / "experiments" / "registry.csv", cand, note)
    print(
        json.dumps(
            {
                "decision": decision,
                "complete": complete,
                "association_v2": assoc.get("official_summary"),
                "linkable_v1": cand.get("official_summary"),
                "n_inventory": len(inventory),
                "runtime_sec": runtime,
                "config_hash_linkable_v1": report["config_hash_linkable_v1"],
                "not_a_kaggle_run": True,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
