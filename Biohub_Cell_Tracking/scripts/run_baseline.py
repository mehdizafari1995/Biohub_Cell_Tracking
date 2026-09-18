#!/usr/bin/env python3
"""Run the baseline on available local samples and record metrics/runtime."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.config import load_config  # noqa: E402
from biohub_tracking.evaluate import evaluate_graph, summarise  # noqa: E402
from biohub_tracking.experiments import append_experiment, git_version, new_experiment_id, peak_memory_mb, write_run_json  # noqa: E402
from biohub_tracking.io_geff import load_geff  # noqa: E402
from biohub_tracking.io_zarr import list_zarr_datasets, open_volume  # noqa: E402
from biohub_tracking.splits import embryo_holdout_splits  # noqa: E402
from biohub_tracking.submit import graphs_to_submission, validate_submission, write_submission  # noqa: E402
from biohub_tracking.track import track_volume  # noqa: E402
from biohub_tracking.visualize import save_detection_overlay, save_track_plot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.yaml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/train")
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--split", default="available_local")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_id = new_experiment_id(args.name)
    run_dir = ROOT / "artifacts" / "runs" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    zarrs = list_zarr_datasets(args.data_dir)
    if not zarrs:
        raise SystemExit(f"No .zarr datasets in {args.data_dir}")

    t0 = time.perf_counter()
    graphs = []
    rows = []
    linking_only = []
    for zarr_path in zarrs:
        array, meta = open_volume(zarr_path)
        pred = track_volume(array, meta, config)
        graphs.append(pred)
        geff_path = zarr_path.with_suffix(".geff")
        if geff_path.exists():
            gt = load_geff(geff_path).filter_timepoints(meta.available_timepoints)
            metrics = evaluate_graph(pred, gt, scale_zyx=config.scale_zyx)
            rows.append(metrics)
            # Diagnostic: linking quality if GT detections were used as nodes.
            from biohub_tracking.graphs import TrackGraph
            from biohub_tracking.link import link_frames

            det = []
            for t in sorted(set(int(x) for x in gt.t)):
                mask = gt.t == t
                det.append((t, gt.coords_zyx()[mask]))
            times = [t for t, _ in det]
            coords = {t: c for t, c in det}
            next_id = 1
            ids = {}
            for t, c in det:
                ids[t] = list(range(next_id, next_id + len(c)))
                next_id += len(c)
            edges = []
            for t_prev, t_curr in zip(times, times[1:]):
                if t_curr != t_prev + 1:
                    continue
                for i, j in link_frames(coords[t_prev], coords[t_curr], config.linking, config.scale_zyx):
                    edges.append((ids[t_prev][i], ids[t_curr][j]))
            oracle = TrackGraph(
                dataset=meta.name + "_gt_detections",
                node_id=list(range(1, next_id)),
                t=gt.t,
                z=gt.z,
                y=gt.y,
                x=gt.x,
                edges=edges,
                estimated_n_nodes=gt.estimated_n_nodes,
            )
            linking_only.append(evaluate_graph(oracle, gt, scale_zyx=config.scale_zyx).as_dict())
            t_vis = meta.available_timepoints[0]
            save_detection_overlay(
                array[t_vis],
                run_dir / f"{meta.name}_overlay.png",
                pred=pred,
                gt=gt,
                timepoint=t_vis,
                title=f"{meta.name} detections vs GT",
            )
        save_track_plot(pred, run_dir / f"{meta.name}_pred_tracks.png", title=f"{meta.name} predicted tracks")

    runtime = time.perf_counter() - t0
    memory = peak_memory_mb()
    summary = summarise(rows) if rows else {"n": 0, "score": float("nan")}
    sub = graphs_to_submission(graphs)
    required = [p.name.removesuffix(".zarr") for p in zarrs]
    schema_errors = validate_submission(sub, required_datasets=required)
    write_submission(sub, run_dir / "submission.csv")

    payload = {
        "experiment_id": experiment_id,
        "config": config.to_dict(),
        "split": args.split,
        "datasets": [p.name for p in zarrs],
        "embryo_splits": embryo_holdout_splits(args.data_dir),
        "summary": summary,
        "per_sample": [r.as_dict() for r in rows],
        "linking_on_gt_detections": linking_only,
        "schema_errors": schema_errors,
        "runtime_sec": runtime,
        "peak_memory_mb": memory,
        "code_version": git_version(ROOT.parent),
    }
    write_run_json(run_dir / "metrics.json", payload)
    append_experiment(
        ROOT / "experiments" / "registry.csv",
        {
            "experiment_id": experiment_id,
            "timestamp_utc": experiment_id.split("_")[0],
            "code_version": payload["code_version"],
            "config_name": args.config.name,
            "seed": config.seed,
            "split": args.split,
            "n_samples": summary.get("n", 0),
            "score": summary.get("score"),
            "edge_jaccard": summary.get("edge_jaccard"),
            "adj_edge_jaccard": summary.get("adj_edge_jaccard"),
            "division_jaccard": summary.get("division_jaccard"),
            "node_recall": summary.get("node_recall"),
            "missed_detections": summary.get("missed_detections"),
            "identity_switches": summary.get("identity_switches"),
            "broken_tracks": summary.get("broken_tracks"),
            "division_tp": summary.get("division_tp"),
            "division_fp": summary.get("division_fp"),
            "division_fn": summary.get("division_fn"),
            "runtime_sec": round(runtime, 3),
            "peak_memory_mb": round(memory, 1) if memory == memory else "",
            "notes": "local official-style edge metric + division proxy; partial volumes if only some timepoints were downloaded",
        },
    )
    print(json.dumps({"experiment_id": experiment_id, "summary": summary, "schema_errors": schema_errors, "runtime_sec": runtime, "peak_memory_mb": memory}, indent=2, default=str))


if __name__ == "__main__":
    main()
