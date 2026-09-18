#!/usr/bin/env python3
"""Evaluate division_v1 on top of a frozen association graph/config."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.config import load_config  # noqa: E402
from biohub_tracking.errors import analyse_errors  # noqa: E402
from biohub_tracking.experiments import append_experiment, git_version, new_experiment_id, peak_memory_mb, write_run_json  # noqa: E402
from biohub_tracking.io_geff import load_geff  # noqa: E402
from biohub_tracking.io_zarr import open_volume  # noqa: E402
from biohub_tracking.official_eval import evaluate_official, summarise_official  # noqa: E402
from biohub_tracking.splits import load_split_manifest  # noqa: E402
from biohub_tracking.track import graph_from_cached_frames  # noqa: E402
from biohub_tracking.visualize import save_error_overlay  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_association_ablation import load_or_detect  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "division_v1.yaml")
    parser.add_argument("--role", choices=["tuning", "validation"], default="tuning")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--name", default="division_v1")
    args = parser.parse_args()
    pipe = load_config(args.config)
    split = load_split_manifest(ROOT / "docs" / "splits" / "v1.json")
    names = args.datasets or list(split[args.role])
    data_dir = ROOT / "data" / "train"
    experiment_id = new_experiment_id(args.name)
    run_dir = ROOT / "artifacts" / "runs" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    official_rows = []
    error_rows = []
    per_sample = []
    det_cfg = load_config(ROOT / "configs" / "baseline_v1.yaml")
    for dataset in names:
        zarr_path = data_dir / f"{dataset}.zarr"
        geff_path = data_dir / f"{dataset}.geff"
        if not zarr_path.exists():
            continue
        frames, shape = load_or_detect(zarr_path, det_cfg)
        array, meta = open_volume(zarr_path)
        gt = load_geff(geff_path).filter_timepoints(meta.available_timepoints)
        pred = graph_from_cached_frames(dataset, frames, pipe, shape, gt.estimated_n_nodes)
        off = evaluate_official(pred, gt, scale_zyx=pipe.scale_zyx)
        analysis = analyse_errors(pred, gt, pipe.scale_zyx)
        official_rows.append(off)
        error_rows.append(analysis.summary())
        t_vis = meta.available_timepoints[len(meta.available_timepoints) // 2]
        save_error_overlay(
            array[t_vis],
            run_dir / f"{dataset}_errors_t{t_vis}.png",
            pred=pred,
            gt=gt,
            missed_gt_ids=analysis.missed_gt_ids,
            fp_edges=analysis.fp_edges,
            fn_edges=analysis.fn_edges,
            recovered_divisions=analysis.recovered_divisions,
            missed_divisions=analysis.missed_divisions,
            timepoint=t_vis,
            title=f"{dataset} t={t_vis} division_v1",
        )
        for nid in analysis.gt_divisions:
            idx = {int(i): k for k, i in enumerate(gt.node_id)}
            t_div = int(gt.t[idx[int(nid)]])
            save_error_overlay(
                array[t_div],
                run_dir / f"{dataset}_gt_division_{nid}_t{t_div}.png",
                pred=pred,
                gt=gt,
                missed_gt_ids=analysis.missed_gt_ids,
                fp_edges=analysis.fp_edges,
                fn_edges=analysis.fn_edges,
                recovered_divisions=analysis.recovered_divisions,
                missed_divisions=analysis.missed_divisions,
                timepoint=t_div,
                title=f"{dataset} GT division parent={nid} t={t_div}",
            )
        per_sample.append({"dataset": dataset, "official": off.as_dict(), "errors": analysis.summary(), "gt_divisions": analysis.gt_divisions, "pred_divisions": analysis.pred_divisions, "recovered": analysis.recovered_divisions, "missed": analysis.missed_divisions})
    runtime = time.perf_counter() - t0
    memory = peak_memory_mb()
    summary = summarise_official(official_rows) if official_rows else {}
    payload = {
        "experiment_id": experiment_id,
        "config": pipe.to_dict(),
        "split_role": args.role,
        "official_summary": summary,
        "per_sample": per_sample,
        "runtime_sec": runtime,
        "peak_memory_mb": memory,
        "code_version": git_version(ROOT.parent),
        "uncertainty_note": "Validation currently has three annotated divisions. Do not claim generalization from n=3.",
    }
    write_run_json(run_dir / "metrics.json", payload)
    append_experiment(
        ROOT / "experiments" / "registry.csv",
        {
            "experiment_id": experiment_id,
            "timestamp_utc": experiment_id.split("_")[0],
            "code_version": payload["code_version"],
            "config_name": args.config.name,
            "seed": 0,
            "split": f"{args.role}:v1.json",
            "n_samples": summary.get("n", 0),
            "score": summary.get("score"),
            "edge_jaccard": summary.get("edge_jaccard"),
            "adj_edge_jaccard": summary.get("adj_edge_jaccard"),
            "division_jaccard": summary.get("division_jaccard"),
            "node_recall": summary.get("node_recall"),
            "division_tp": summary.get("division_tp"),
            "division_fp": summary.get("division_fp"),
            "division_fn": summary.get("division_fn"),
            "runtime_sec": round(runtime, 3),
            "peak_memory_mb": round(memory, 1) if memory == memory else "",
            "notes": "division_v1; n_val_divisions=3; do not claim generalization",
        },
    )
    print(json.dumps({"experiment_id": experiment_id, "summary": summary, "runtime_sec": runtime, "peak_memory_mb": memory, "run_dir": str(run_dir)}, indent=2, default=str))


if __name__ == "__main__":
    main()
