#!/usr/bin/env python3
"""Ablate association_v2 on frozen split data. Does not overwrite previous runs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.association import FrameDetections  # noqa: E402
from biohub_tracking.config import PipelineConfig, load_config  # noqa: E402
from biohub_tracking.detect import detect_cells_with_appearance  # noqa: E402
from biohub_tracking.errors import analyse_errors  # noqa: E402
from biohub_tracking.experiments import append_experiment, git_version, new_experiment_id, peak_memory_mb, write_run_json  # noqa: E402
from biohub_tracking.io_geff import load_geff  # noqa: E402
from biohub_tracking.io_zarr import open_volume  # noqa: E402
from biohub_tracking.official_eval import evaluate_official, official_scorer_available, summarise_official  # noqa: E402
from biohub_tracking.splits import load_split_manifest  # noqa: E402
from biohub_tracking.track import graph_from_cached_frames  # noqa: E402

CACHE = ROOT / "artifacts" / "cache" / "detections"


def variants() -> list[dict]:
    """Independent components first; combined recipe last."""
    base = {"detection": "baseline_v1"}
    return [
        {"name": "greedy_baseline", **base, "linking": {"method": "greedy", "max_distance_um": 15.0}},
        {
            "name": "cv_motion",
            **base,
            "linking": {"method": "association_v2", "motion_model": "constant_velocity", "max_distance_um": 15.0},
        },
        {
            "name": "gate_12um",
            **base,
            "linking": {"method": "association_v2", "gate_um": 12.0, "max_distance_um": 12.0},
        },
        {
            "name": "gate_20um",
            **base,
            "linking": {"method": "association_v2", "gate_um": 20.0, "max_distance_um": 20.0},
        },
        {
            "name": "gap_close_1",
            **base,
            "linking": {"method": "association_v2", "max_gap_frames": 1, "interpolate_gaps": True, "max_distance_um": 15.0},
        },
        {
            "name": "gap_close_2",
            **base,
            "linking": {"method": "association_v2", "max_gap_frames": 2, "interpolate_gaps": True, "max_distance_um": 15.0},
        },
        {
            "name": "gated_hungarian",
            **base,
            "linking": {
                "method": "association_v2",
                "assignment": "gated_hungarian",
                "max_distance_um": 15.0,
            },
        },
        {
            "name": "intensity_cost",
            **base,
            "linking": {"method": "association_v2", "intensity_weight": 0.5, "max_distance_um": 15.0},
        },
        {
            "name": "start_end_penalties",
            **base,
            "linking": {
                "method": "association_v2",
                "assignment": "gated_hungarian",
                "track_start_penalty_um": 12.0,
                "track_end_penalty_um": 12.0,
                "max_distance_um": 15.0,
            },
        },
        {
            "name": "boundary_aware",
            **base,
            "linking": {
                "method": "association_v2",
                "assignment": "gated_hungarian",
                "track_start_penalty_um": 12.0,
                "track_end_penalty_um": 12.0,
                "boundary_margin_um": 6.0,
                "max_distance_um": 15.0,
            },
        },
    ]


def combined_v2(selected: dict) -> dict:
    linking = {
        "method": "association_v2",
        "motion_model": "constant_velocity",
        "max_distance_um": 15.0,
        "gate_um": 15.0,
        "max_gap_frames": 1,
        "interpolate_gaps": True,
        "assignment": "gated_hungarian",
        "intensity_weight": 0.35,
        "track_start_penalty_um": 10.0,
        "track_end_penalty_um": 10.0,
        "boundary_margin_um": 6.0,
    }
    linking.update(selected)
    return {"name": "association_v2", "detection": "baseline_v1", "linking": linking}


def cache_path(dataset: str) -> Path:
    return CACHE / f"{dataset}_baseline_v1.npz"


def load_or_detect(zarr_path: Path, config: PipelineConfig) -> tuple[list[FrameDetections], tuple[int, int, int]]:
    array, meta = open_volume(zarr_path)
    dest = cache_path(meta.name)
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
    frames: list[FrameDetections] = []
    next_id = 1
    times, coords_l, inten_l, app_l, ids_l = [], [], [], [], []
    for t in meta.available_timepoints:
        image = np.asarray(array[t])
        coords, intensity, appearance = detect_cells_with_appearance(image, config.detection, config.scale_zyx)
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


def score_variant(name: str, datasets: list[str], data_dir: Path, linking: dict, split_role: str) -> dict:
    det_cfg = load_config(ROOT / "configs" / "baseline_v1.yaml")
    pipe = load_config(ROOT / "configs" / "baseline_v1.yaml", overrides={"linking": linking})
    experiment_id = new_experiment_id(name)
    run_dir = ROOT / "artifacts" / "runs" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    official_rows = []
    error_rows = []
    per_sample = []
    for dataset in datasets:
        zarr_path = data_dir / f"{dataset}.zarr"
        geff_path = data_dir / f"{dataset}.geff"
        if not zarr_path.exists() or not geff_path.exists():
            continue
        frames, shape = load_or_detect(zarr_path, det_cfg)
        _, meta = open_volume(zarr_path)
        gt_full = load_geff(geff_path)
        gt = gt_full.filter_timepoints(meta.available_timepoints)
        t_s = time.perf_counter()
        pred = graph_from_cached_frames(dataset, frames, pipe, shape, gt.estimated_n_nodes)
        link_sec = time.perf_counter() - t_s
        off = evaluate_official(pred, gt, scale_zyx=pipe.scale_zyx)
        analysis = analyse_errors(pred, gt, pipe.scale_zyx)
        official_rows.append(off)
        error_rows.append(analysis.summary())
        per_sample.append(
            {
                "dataset": dataset,
                "official": off.as_dict(),
                "errors": analysis.summary(),
                "n_pred_nodes": pred.n_nodes,
                "n_pred_edges": pred.n_edges,
                "n_interpolated": (pred.metadata.get("association") or {}).get("n_interpolated"),
                "link_runtime_sec": link_sec,
            }
        )
    runtime = time.perf_counter() - t0
    memory = peak_memory_mb()
    summary = summarise_official(official_rows) if official_rows else {}
    switches = sum(r["identity_switches"] for r in error_rows)
    fragments = sum(r["fragmented_tracks"] for r in error_rows)
    payload = {
        "experiment_id": experiment_id,
        "variant": name,
        "split_role": split_role,
        "linking": pipe.linking.__dict__,
        "detection": "baseline_v1",
        "official_summary": summary,
        "per_sample": per_sample,
        "identity_switches": switches,
        "fragmented_tracks": fragments,
        "runtime_sec": runtime,
        "peak_memory_mb": memory,
        "code_version": git_version(ROOT.parent),
        "official_available": official_scorer_available(),
    }
    write_run_json(run_dir / "metrics.json", payload)
    append_experiment(
        ROOT / "experiments" / "registry.csv",
        {
            "experiment_id": experiment_id,
            "timestamp_utc": experiment_id.split("_")[0],
            "code_version": payload["code_version"],
            "config_name": name,
            "seed": 0,
            "split": f"{split_role}:v1.json",
            "n_samples": summary.get("n", 0),
            "score": summary.get("score"),
            "edge_jaccard": summary.get("edge_jaccard"),
            "adj_edge_jaccard": summary.get("adj_edge_jaccard"),
            "division_jaccard": summary.get("division_jaccard"),
            "node_recall": summary.get("node_recall"),
            "identity_switches": switches,
            "broken_tracks": fragments,
            "division_tp": summary.get("division_tp"),
            "division_fp": summary.get("division_fp"),
            "division_fn": summary.get("division_fn"),
            "runtime_sec": round(runtime, 3),
            "peak_memory_mb": round(memory, 1) if memory == memory else "",
            "notes": f"association ablation; detections=baseline_v1 cached; switches={switches}",
        },
    )
    print(json.dumps({"experiment_id": experiment_id, "variant": name, "summary": summary, "switches": switches, "fragments": fragments, "runtime_sec": runtime, "peak_memory_mb": memory}, indent=2, default=str))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["tuning", "validation"], default="tuning")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--include-combined", action="store_true")
    args = parser.parse_args()
    split = load_split_manifest(ROOT / "docs" / "splits" / "v1.json")
    names = args.datasets or list(split[args.role])
    data_dir = ROOT / "data" / "train"
    chosen = variants()
    if args.only:
        chosen = [v for v in chosen if v["name"] in args.only]
    results = []
    for variant in chosen:
        results.append(score_variant(variant["name"], names, data_dir, variant["linking"], args.role))
    if args.include_combined:
        results.append(score_variant("association_v2", names, data_dir, combined_v2({})["linking"], args.role))
    dest = ROOT / "artifacts" / "runs" / f"association_{args.role}_summary.json"
    dest.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
