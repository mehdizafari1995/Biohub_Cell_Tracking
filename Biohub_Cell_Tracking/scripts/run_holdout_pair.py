#!/usr/bin/env python3
"""Official-score association_v2 vs division_v2 once on frozen validation.

Uses cached baseline_v1 detections (same path as the locked association_v2
holdout). Does not retune. A local official score is not a Kaggle run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from biohub_tracking.config import load_config  # noqa: E402
from biohub_tracking.experiments import (  # noqa: E402
    append_experiment,
    git_version,
    new_experiment_id,
    peak_memory_mb,
    write_run_json,
)
from biohub_tracking.io_geff import load_geff  # noqa: E402
from biohub_tracking.io_zarr import open_volume  # noqa: E402
from biohub_tracking.official_eval import official_dependency_versions  # noqa: E402
from biohub_tracking.track import graph_from_cached_frames  # noqa: E402
from biohub_tracking.validation import (  # noqa: E402
    FrozenSplitError,
    assemble_complete_validation,
    inventory_validation,
    load_frozen_validation_plan,
    score_graphs_official,
)
from run_association_ablation import load_or_detect  # noqa: E402


def _record(registry: Path, experiment_id: str, config_name: str, report: dict, extra_note: str) -> None:
    summary = report.get("official_summary") or {}
    append_experiment(
        registry,
        {
            "experiment_id": experiment_id,
            "timestamp_utc": experiment_id.split("_")[0],
            "code_version": report["code_version"],
            "config_name": config_name,
            "seed": 0,
            "split": "validation:v1.json",
            "n_samples": summary.get("n", 0),
            "score": summary.get("score"),
            "edge_jaccard": summary.get("edge_jaccard"),
            "adj_edge_jaccard": summary.get("adj_edge_jaccard"),
            "division_jaccard": summary.get("division_jaccard"),
            "node_recall": summary.get("node_recall"),
            "division_tp": summary.get("division_tp"),
            "division_fp": summary.get("division_fp"),
            "division_fn": summary.get("division_fn"),
            "runtime_sec": round(report["runtime_sec"], 3),
            "peak_memory_mb": round(report["peak_memory_mb"], 1)
            if report["peak_memory_mb"] == report["peak_memory_mb"]
            else "",
            "notes": extra_note,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--split", type=Path, default=ROOT / "docs" / "splits" / "v1.json")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    try:
        plan = load_frozen_validation_plan(args.split)
    except FrozenSplitError as exc:
        raise SystemExit(f"frozen split error: {exc}") from exc

    assoc_cfg = load_config(ROOT / "configs" / "association_v2.yaml")
    div_cfg = load_config(ROOT / "configs" / "division_v2.yaml")
    det_cfg = load_config(ROOT / "configs" / "baseline_v1.yaml")
    inventory = inventory_validation(args.data_dir, plan["validation_datasets"])
    t0 = time.perf_counter()
    assoc_scored = []
    div_scored = []
    for item in inventory:
        if not item.zarr_exists or not item.geff_exists:
            continue
        zarr_path = args.data_dir / f"{item.dataset}.zarr"
        geff_path = args.data_dir / f"{item.dataset}.geff"
        frames, shape = load_or_detect(zarr_path, det_cfg)
        array, meta = open_volume(zarr_path)
        gt = load_geff(geff_path).filter_timepoints(meta.available_timepoints)
        pred_assoc = graph_from_cached_frames(
            item.dataset, frames, assoc_cfg, shape, gt.estimated_n_nodes
        )
        pred_div = graph_from_cached_frames(
            item.dataset, frames, div_cfg, shape, gt.estimated_n_nodes
        )
        assoc_scored.append(score_graphs_official(pred_assoc, gt, scale_zyx=assoc_cfg.scale_zyx))
        div_scored.append(score_graphs_official(pred_div, gt, scale_zyx=div_cfg.scale_zyx))
        del array

    runtime = time.perf_counter() - t0
    memory = peak_memory_mb()
    code = git_version(ROOT.parent)
    assoc_report = assemble_complete_validation(plan, inventory, assoc_scored, "association_v2")
    div_report = assemble_complete_validation(plan, inventory, div_scored, "division_v2")
    assoc_id = new_experiment_id("complete_val_association_v2")
    div_id = new_experiment_id("complete_val_division_v2")
    shared = {
        "runtime_sec": runtime,
        "peak_memory_mb": memory,
        "code_version": code,
        "official_versions": official_dependency_versions(),
        "data_dir": str(args.data_dir),
        "score_source": "local_official",
        "not_a_kaggle_run": True,
        "not_a_leaderboard_score": True,
        "retune_after_holdout": False,
        "shared_detections": "baseline_v1 cached",
    }
    assoc_report.update({"experiment_id": assoc_id, **shared})
    div_report.update({"experiment_id": div_id, **shared})

    assoc_score = (assoc_report.get("official_summary") or {}).get("score")
    div_score = (div_report.get("official_summary") or {}).get("score")
    div_fp = (div_report.get("official_summary") or {}).get("division_fp")
    decision = "pending"
    if assoc_score is not None and div_score is not None:
        if float(div_score) < float(assoc_score) or int(div_fp or 0) > 0:
            decision = "reject_division_v2"
        else:
            decision = "keep_division_v2"
    pair = {
        "decision": decision,
        "association_v2": {
            "experiment_id": assoc_id,
            "official_summary": assoc_report.get("official_summary"),
            "n_official_recovered_divisions": assoc_report.get("n_official_recovered_divisions"),
            "n_official_missed_divisions": assoc_report.get("n_official_missed_divisions"),
            "complete": assoc_report.get("complete"),
            "per_sequence": [
                {
                    "dataset": row["dataset"],
                    "official": row.get("official"),
                    "official_division_summary": row.get("official_division_summary"),
                }
                for row in assoc_report.get("samples") or []
            ],
        },
        "division_v2": {
            "experiment_id": div_id,
            "official_summary": div_report.get("official_summary"),
            "n_official_recovered_divisions": div_report.get("n_official_recovered_divisions"),
            "n_official_missed_divisions": div_report.get("n_official_missed_divisions"),
            "complete": div_report.get("complete"),
            "per_sequence": [
                {
                    "dataset": row["dataset"],
                    "official": row.get("official"),
                    "official_division_summary": row.get("official_division_summary"),
                }
                for row in div_report.get("samples") or []
            ],
        },
        "locked_association_hash": "2c6966b5204ba542",
        "locked_division_hash": "0fa756963e5d42d1",
        "not_a_kaggle_run": True,
        "not_a_leaderboard_score": True,
        "runtime_sec": runtime,
        "peak_memory_mb": memory,
        "code_version": code,
    }
    run_dir = ROOT / "artifacts" / "runs" / f"{assoc_id}_pair"
    write_run_json(run_dir / "association_v2.json", assoc_report)
    write_run_json(run_dir / "division_v2.json", div_report)
    write_run_json(run_dir / "pair.json", pair)
    if args.record and assoc_report.get("official_summary") and div_report.get("official_summary"):
        note = (
            "complete specimen-separated official holdout after zarr-export helper; "
            f"events={assoc_report['n_annotated_division_events']}; "
            "shared baseline_v1 detections; do not retune v1; not a leaderboard score"
        )
        _record(
            ROOT / "experiments" / "registry.csv",
            assoc_id,
            "association_v2.yaml",
            assoc_report,
            note + f"; pair_decision={decision}",
        )
        _record(
            ROOT / "experiments" / "registry.csv",
            div_id,
            "division_v2.yaml",
            div_report,
            note + f"; pair_decision={decision}",
        )
    print(json.dumps(pair, indent=2, default=str))


if __name__ == "__main__":
    main()
