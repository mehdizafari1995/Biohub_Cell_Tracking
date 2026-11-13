#!/usr/bin/env python3
"""Diagnose 44b6 ACF dt=1 speed-change vs official matches.

Rebuilds locked ``learned_unet_orgdet_fnn_acf_v1`` on frozen-tuning orgop
graphs, walks unique-predecessor dt=1 two-hops, and classifies the
continuation edge with the official 7 µm sparse-GT rule. Leftover
unmatched edges are not negatives. Proposes at most one 0.1 µm/frame
round-up gate strictly above TP max |Δv| if conflict FPs sit above it.
Does not sweep thresholds. Does not official-score an overlay.
Never opens ``6bba``. A local diagnostic is not a Kaggle run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.experiments import (  # noqa: E402
    append_experiment,
    git_version,
    new_experiment_id,
    peak_memory_mb,
    write_run_json,
)
from biohub_tracking.io_geff import load_geff  # noqa: E402
from biohub_tracking.learned_gap import graph_from_payload  # noqa: E402
from biohub_tracking.learned_orgdet_fnn_acf import (  # noqa: E402
    apply_orgdet_fnn_acf_overlay,
    resolve_sequence_zarr,
)
from biohub_tracking.learned_orgdet_fnn_acf_scf import (  # noqa: E402
    INFER_STEMS,
    LOCKED_ORGDET_FNN_ACF_SCF_HASH,
    MAX_SPEED_CHANGE_UM,
    ORGDET_FNN_ACF_LOCKED_TUNE,
    TUNE_KEEP_THRESHOLD,
    classify_dt1_speed_changes,
    diagnostic_decision,
    ensure_orgdet_tuning_graph,
    extract_dt1_speed_changes,
    load_role_graphs,
    orgdet_fnn_acf_scf_config_hash,
    orgdet_fnn_acf_scf_config_payload,
    propose_speed_change_gate,
    refuse_non_orgdet_node_count,
    refuse_non_tuning_stems,
    runtime_estimate,
)

DIAG_REPORT = ROOT / "reports" / "learned_unet_orgdet_fnn_acf_scf_v1_diag.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    cfg = orgdet_fnn_acf_scf_config_payload()
    if orgdet_fnn_acf_scf_config_hash(cfg) != LOCKED_ORGDET_FNN_ACF_SCF_HASH:
        raise SystemExit("learned_unet_orgdet_fnn_acf_scf_v1 hash drifted; do not retune")
    refuse_non_tuning_stems(INFER_STEMS)
    try:
        payloads = load_role_graphs(args.graph_dir, INFER_STEMS)
        refuse_non_tuning_stems(p["dataset"] for p in payloads)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    started = time.time()
    classified: list[dict] = []
    overlays: list[dict] = []
    for payload in payloads:
        stem = payload["dataset"]
        refuse_non_orgdet_node_count(payload, "tuning")
        geff = args.data_dir / f"{stem}.geff"
        nested = args.data_dir / "train" / f"{stem}.geff"
        geff = nested if nested.exists() else geff
        if not geff.exists():
            raise SystemExit(f"missing GEFF {geff}; cannot diagnose")
        zarr_path = resolve_sequence_zarr(args.data_dir, stem)
        before = graph_from_payload(payload)
        before = ensure_orgdet_tuning_graph(before)
        acf = apply_orgdet_fnn_acf_overlay(before, zarr_path=zarr_path)
        hops = extract_dt1_speed_changes(acf)
        gt = load_geff(geff)
        rows = classify_dt1_speed_changes(acf, gt, hops)
        classified.extend(rows)
        overlays.append(
            {
                "dataset": stem,
                "n_nodes_acf": acf.n_nodes,
                "n_edges_acf": acf.n_edges,
                "n_dt1_speed_changes": len(rows),
                "n_tp": sum(1 for r in rows if r["edge_class"] == "tp"),
                "n_conflict_fp": sum(1 for r in rows if r["edge_class"] == "conflict_fp"),
                "n_leftover": sum(1 for r in rows if r["edge_class"] == "leftover"),
            }
        )

    proposal = propose_speed_change_gate(classified)
    decision = diagnostic_decision(proposal)
    report = {
        "status": "diagnostic",
        "decision": decision,
        "role": "tuning",
        "split": "tuning:v1.json",
        "config_hash": LOCKED_ORGDET_FNN_ACF_SCF_HASH,
        "orgdet_fnn_acf_locked_reference": ORGDET_FNN_ACF_LOCKED_TUNE,
        "tune_keep_threshold": TUNE_KEEP_THRESHOLD,
        "max_speed_change_um_locked": MAX_SPEED_CHANGE_UM,
        "proposal": proposal,
        "overlays": overlays,
        "n_classified_hops": len(classified),
        "runtime_estimate": runtime_estimate(),
        "code_version": git_version(ROOT),
        "runtime_sec": time.time() - started,
        "peak_memory_mb": peak_memory_mb(),
        "opened_holdout": False,
        "official_overlay_scored": False,
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
        "notes": (
            "44b6 ACF dt=1 unique-predecessor speed-change / acceleration "
            "diagnostic. |Δv| vs official matched TP vs leftover unmatched. "
            "Leftovers are not negatives. Relative |Δv| is recorded, not "
            "gated. No threshold sweep. Not a leaderboard score. Do not "
            "retune max_dt1_edge_um or max_rel_appearance. Do not use "
            "public 0.602 / 0.612 / 0.553."
        ),
    }
    write_run_json(DIAG_REPORT, report)
    print(
        json.dumps(
            {
                "decision": decision,
                "supported": proposal["supported"],
                "reason": proposal["reason"],
                "max_speed_change_um": proposal["max_speed_change_um"],
                "n_tp": proposal["n_tp"],
                "n_conflict_fp": proposal["n_conflict_fp"],
                "n_leftover": proposal["n_leftover"],
                "tp_max_um": proposal["tp_max_um"],
                "conflict_fp_max_um": proposal["conflict_fp_max_um"],
                "n_conflict_fp_above_gate": proposal["n_conflict_fp_above_gate"],
                "opened_holdout": False,
                "is_leaderboard_score": False,
            },
            indent=2,
        )
    )
    if args.record:
        experiment_id = new_experiment_id("learned_unet_orgdet_fnn_acf_scf_v1_diag")
        append_experiment(
            ROOT / "experiments" / "registry.csv",
            {
                "experiment_id": experiment_id,
                "timestamp_utc": experiment_id.split("_")[0],
                "code_version": report["code_version"],
                "config_name": "learned_unet_orgdet_fnn_acf_scf_v1",
                "seed": 0,
                "split": "tuning:v1.json",
                "n_samples": len(overlays),
                "score": "",
                "edge_jaccard": "",
                "adj_edge_jaccard": "",
                "division_jaccard": "",
                "node_recall": "",
                "division_tp": "",
                "division_fp": "",
                "division_fn": "",
                "runtime_sec": round(report["runtime_sec"], 3),
                "peak_memory_mb": report["peak_memory_mb"],
                "notes": (
                    f"{decision}; hash {LOCKED_ORGDET_FNN_ACF_SCF_HASH}; "
                    f"reason={proposal['reason']}; tp={proposal['n_tp']}; "
                    f"conflict_fp={proposal['n_conflict_fp']}; leftover="
                    f"{proposal['n_leftover']}; not a leaderboard score"
                ),
            },
        )


if __name__ == "__main__":
    main()
