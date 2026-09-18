#!/usr/bin/env python3
"""Evaluate detection vs tracking with the official scorer and the diagnostic evaluator."""

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
from biohub_tracking.evaluate import evaluate_graph, summarise  # noqa: E402
from biohub_tracking.experiments import append_experiment, git_version, new_experiment_id, peak_memory_mb, write_run_json  # noqa: E402
from biohub_tracking.io_geff import load_geff  # noqa: E402
from biohub_tracking.io_zarr import open_volume  # noqa: E402
from biohub_tracking.official_eval import (  # noqa: E402
    compare_official_and_diagnostic,
    official_dependency_versions,
    official_scorer_available,
    summarise_official,
)
from biohub_tracking.oracle import link_gt_centroids  # noqa: E402
from biohub_tracking.splits import load_split_manifest  # noqa: E402
from biohub_tracking.submit import graphs_to_submission, validate_submission, write_submission  # noqa: E402
from biohub_tracking.track import track_volume  # noqa: E402
from biohub_tracking.visualize import save_detection_overlay, save_error_overlay, save_error_panel, save_track_plot  # noqa: E402


def _datasets_for_role(data_dir: Path, names: list[str]) -> list[Path]:
    found = []
    for name in names:
        path = data_dir / f"{name}.zarr"
        if path.exists():
            found.append(path)
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.yaml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/train")
    parser.add_argument("--split", type=Path, default=ROOT / "docs/splits/v1.json")
    parser.add_argument("--role", choices=["validation", "tuning", "all_local"], default="validation")
    parser.add_argument("--datasets", nargs="*", default=None, help="Override split role with explicit dataset names")
    parser.add_argument("--name", default="eval")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_id = new_experiment_id(args.name)
    run_dir = ROOT / "artifacts" / "runs" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    split_info: dict = {}
    if args.datasets:
        names = list(args.datasets)
        split_info = {"role": "explicit", "datasets": names, "note": "explicit --datasets override"}
    elif args.role == "all_local":
        names = [p.name.removesuffix(".zarr") for p in sorted(args.data_dir.glob("*.zarr"))]
        split_info = {"role": "all_local", "note": "no frozen split; using whatever zarrs are on disk"}
    elif args.split.exists():
        split_info = load_split_manifest(args.split)
        names = list(split_info.get(args.role) or [])
    else:
        names = [p.name.removesuffix(".zarr") for p in sorted(args.data_dir.glob("*.zarr"))]
        split_info = {"role": args.role, "note": "split manifest missing; using local zarrs"}

    zarrs = _datasets_for_role(args.data_dir, names) if names else list(sorted(args.data_dir.glob("*.zarr")))
    if not zarrs:
        raise SystemExit(f"No .zarr datasets to evaluate in {args.data_dir} for role={args.role}")

    official_ok = official_scorer_available()
    t0 = time.perf_counter()
    graphs = []
    diagnostic_rows = []
    official_rows = []
    comparisons = []
    oracle_rows = []
    error_rows = []
    per_sample_runtime = []

    for zarr_path in zarrs:
        t_sample = time.perf_counter()
        array, meta = open_volume(zarr_path)
        pred = track_volume(array, meta, config)
        graphs.append(pred)
        geff_path = zarr_path.with_suffix(".geff")
        sample_rec = {
            "dataset": meta.name,
            "available_timepoints": meta.available_timepoints,
            "n_pred_nodes": pred.n_nodes,
            "n_pred_edges": pred.n_edges,
            "complete_sequence": len(meta.available_timepoints) >= int(meta.shape[0]),
            "shape_TZYX": list(meta.shape),
            "time_unit": meta.time_unit,
            "scale_tzyx": list(meta.scale_tzyx),
        }
        if geff_path.exists():
            gt_full = load_geff(geff_path)
            gt = gt_full.filter_timepoints(meta.available_timepoints)
            sample_rec["n_gt_nodes"] = gt.n_nodes
            sample_rec["n_gt_edges"] = gt.n_edges
            sample_rec["n_gt_divisions"] = int(sum(len(v) >= 2 for v in gt.out_edges().values()))
            sample_rec["estimated_n_nodes_eval"] = gt.estimated_n_nodes
            diag = evaluate_graph(pred, gt, scale_zyx=config.scale_zyx)
            diagnostic_rows.append(diag)
            if official_ok:
                comparison = compare_official_and_diagnostic(pred, gt, scale_zyx=config.scale_zyx)
                comparisons.append(comparison)
                from biohub_tracking.official_eval import OfficialSampleResult

                official_rows.append(OfficialSampleResult(**comparison["official"]))
            oracle = link_gt_centroids(gt, config.linking, config.scale_zyx)
            oracle_diag = evaluate_graph(oracle, gt, scale_zyx=config.scale_zyx)
            oracle_rec = oracle_diag.as_dict()
            oracle_rec["note"] = "oracle: GT centroids as detections; linking only"
            if official_ok:
                from biohub_tracking.official_eval import evaluate_official

                oracle_off = evaluate_official(oracle, gt, scale_zyx=config.scale_zyx)
                oracle_rec["official"] = oracle_off.as_dict()
            oracle_rows.append(oracle_rec)
            analysis = analyse_errors(pred, gt, config.scale_zyx)
            error_rows.append(analysis.summary())
            t_vis = meta.available_timepoints[len(meta.available_timepoints) // 2]
            save_detection_overlay(
                array[t_vis],
                run_dir / f"{meta.name}_overlay.png",
                pred=pred,
                gt=gt,
                timepoint=t_vis,
                title=f"{meta.name} detections vs GT t={t_vis}",
            )
            save_error_overlay(
                array[t_vis],
                run_dir / f"{meta.name}_errors_t{t_vis}.png",
                pred=pred,
                gt=gt,
                missed_gt_ids=analysis.missed_gt_ids,
                fp_edges=analysis.fp_edges,
                fn_edges=analysis.fn_edges,
                recovered_divisions=analysis.recovered_divisions,
                missed_divisions=analysis.missed_divisions,
                timepoint=t_vis,
                title=f"{meta.name} t={t_vis} missed cells / broken links / divisions",
            )
            save_error_panel(analysis.summary(), run_dir / f"{meta.name}_error_counts.png")
            (run_dir / f"{meta.name}_errors.json").write_text(json.dumps(analysis.as_dict(), indent=2))
        save_track_plot(pred, run_dir / f"{meta.name}_pred_tracks.png", title=f"{meta.name} predicted tracks")
        sample_rec["runtime_sec"] = time.perf_counter() - t_sample
        per_sample_runtime.append(sample_rec)

    runtime = time.perf_counter() - t0
    memory = peak_memory_mb()
    diag_summary = summarise(diagnostic_rows) if diagnostic_rows else {"n": 0, "score": float("nan")}
    official_summary = summarise_official(official_rows) if official_rows else None
    sub = graphs_to_submission(graphs)
    required = [p.name.removesuffix(".zarr") for p in zarrs]
    schema_errors = validate_submission(sub, required_datasets=required)
    write_submission(sub, run_dir / "submission.csv")

    score_label = "official" if official_summary else "diagnostic_only"
    payload = {
        "experiment_id": experiment_id,
        "config": config.to_dict(),
        "split_role": args.role,
        "split": split_info,
        "scorer_status": {
            "official_available": official_ok,
            "versions": official_dependency_versions() if official_ok else {},
            "note": (
                "Official scores use tracking_cellmot.metrics.summarise "
                "(adjusted edge Jaccard + 0.1 * division Jaccard). "
                "biohub_tracking.evaluate is a labelled diagnostic."
            ),
        },
        "datasets": [p.name for p in zarrs],
        "per_sample_runtime": per_sample_runtime,
        "diagnostic_summary": diag_summary,
        "official_summary": official_summary,
        "diagnostic_per_sample": [r.as_dict() for r in diagnostic_rows],
        "official_per_sample": [r.as_dict() for r in official_rows],
        "official_vs_diagnostic": comparisons,
        "oracle_gt_centroid_linking": oracle_rows,
        "error_analysis": error_rows,
        "schema_errors": schema_errors,
        "runtime_sec": runtime,
        "peak_memory_mb": memory,
        "code_version": git_version(ROOT.parent),
        "score_source": score_label,
    }
    write_run_json(run_dir / "metrics.json", payload)

    use_summary = official_summary or diag_summary
    append_experiment(
        ROOT / "experiments" / "registry.csv",
        {
            "experiment_id": experiment_id,
            "timestamp_utc": experiment_id.split("_")[0],
            "code_version": payload["code_version"],
            "config_name": args.config.name,
            "seed": config.seed,
            "split": f"{args.role}:{args.split.name if args.split else 'local'}",
            "n_samples": use_summary.get("n", 0),
            "score": use_summary.get("score"),
            "edge_jaccard": use_summary.get("edge_jaccard"),
            "adj_edge_jaccard": use_summary.get("adj_edge_jaccard"),
            "division_jaccard": use_summary.get("division_jaccard"),
            "node_recall": use_summary.get("node_recall"),
            "missed_detections": diag_summary.get("missed_detections"),
            "identity_switches": diag_summary.get("identity_switches"),
            "broken_tracks": diag_summary.get("broken_tracks"),
            "division_tp": use_summary.get("division_tp"),
            "division_fp": use_summary.get("division_fp"),
            "division_fn": use_summary.get("division_fn"),
            "runtime_sec": round(runtime, 3),
            "peak_memory_mb": round(memory, 1) if memory == memory else "",
            "notes": (
                f"{score_label}; official={official_ok}; "
                "unmatched preds are not node FPs (sparse GT)"
            ),
        },
    )
    print(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "official_available": official_ok,
                "official_summary": official_summary,
                "diagnostic_summary": diag_summary,
                "schema_errors": schema_errors,
                "runtime_sec": runtime,
                "peak_memory_mb": memory,
                "run_dir": str(run_dir),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
