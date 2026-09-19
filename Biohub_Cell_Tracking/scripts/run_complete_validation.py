#!/usr/bin/env python3
"""Score both frozen validation sequences and annotated-division events.

Does not retune locked configs. Does not overwrite registry rows unless
``--record`` is passed. Missing volumes are reported; they are not invented
as scores.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.config import load_config  # noqa: E402
from biohub_tracking.experiments import git_version, new_experiment_id, peak_memory_mb, write_run_json  # noqa: E402
from biohub_tracking.io_geff import load_geff  # noqa: E402
from biohub_tracking.io_zarr import open_volume  # noqa: E402
from biohub_tracking.official_eval import official_dependency_versions, official_scorer_available  # noqa: E402
from biohub_tracking.track import track_volume  # noqa: E402
from biohub_tracking.validation import (  # noqa: E402
    FrozenSplitError,
    assemble_complete_validation,
    inventory_validation,
    load_frozen_validation_plan,
    score_graphs_official,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline.yaml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--split", type=Path, default=ROOT / "docs" / "splits" / "v1.json")
    parser.add_argument("--name", default="complete_val")
    parser.add_argument(
        "--record",
        action="store_true",
        help="Append one registry row. Default is off so missing-data runs stay out of the ledger.",
    )
    args = parser.parse_args()

    try:
        plan = load_frozen_validation_plan(args.split)
    except FrozenSplitError as exc:
        raise SystemExit(f"frozen split error: {exc}") from exc

    config = load_config(args.config)
    inventory = inventory_validation(args.data_dir, plan["validation_datasets"])
    official_ok = official_scorer_available()
    t0 = time.perf_counter()
    scored = []
    for item in inventory:
        if not item.zarr_exists or not item.geff_exists:
            continue
        if not official_ok:
            continue
        zarr_path = args.data_dir / f"{item.dataset}.zarr"
        geff_path = args.data_dir / f"{item.dataset}.geff"
        array, meta = open_volume(zarr_path)
        pred = track_volume(array, meta, config)
        gt = load_geff(geff_path).filter_timepoints(meta.available_timepoints)
        scored.append(score_graphs_official(pred, gt, scale_zyx=config.scale_zyx))

    report = assemble_complete_validation(plan, inventory, scored, args.config.name)
    report.update(
        {
            "experiment_id": new_experiment_id(args.name),
            "runtime_sec": time.perf_counter() - t0,
            "peak_memory_mb": peak_memory_mb(),
            "code_version": git_version(ROOT.parent),
            "official_available": official_ok,
            "official_versions": official_dependency_versions() if official_ok else {},
            "data_dir": str(args.data_dir),
        }
    )
    run_dir = ROOT / "artifacts" / "runs" / report["experiment_id"]
    write_run_json(run_dir / "complete_validation.json", report)
    if args.record and report["official_summary"]:
        from biohub_tracking.experiments import append_experiment

        summary = report["official_summary"]
        append_experiment(
            ROOT / "experiments" / "registry.csv",
            {
                "experiment_id": report["experiment_id"],
                "timestamp_utc": report["experiment_id"].split("_")[0],
                "code_version": report["code_version"],
                "config_name": args.config.name,
                "seed": config.seed,
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
                "notes": (
                    "complete specimen-separated validation; official scorer; "
                    f"events={report['n_annotated_division_events']}; "
                    f"recovered={report['n_official_recovered_divisions']}; "
                    "do not retune v1"
                ),
            },
        )
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "complete": report["complete"],
                "blocked_reason": report["blocked_reason"],
                "official_summary": report["official_summary"],
                "n_annotated_division_events": report["n_annotated_division_events"],
                "n_official_recovered_divisions": report["n_official_recovered_divisions"],
                "n_official_missed_divisions": report["n_official_missed_divisions"],
                "missing_datasets": report["missing_datasets"],
                "partial_datasets": report["partial_datasets"],
                "run_dir": str(run_dir),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
