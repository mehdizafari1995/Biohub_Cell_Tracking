#!/usr/bin/env python3
"""Official-score learned_unet_orgop_v1 graphs on frozen-tuning 44b6 only.

Applies tracksdata ILP locally if the helper left candidates pending.
Refuses frozen holdout ``6bba``. Does not write submission.csv.
A local official score is not a Kaggle run and not a leaderboard number.
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
from biohub_tracking.learned_infer import graph_from_payload, load_graph_json  # noqa: E402
from biohub_tracking.learned_orgop import (  # noqa: E402
    INFER_LOCKED_TUNE,
    INFER_STEMS,
    LOCKED_ORGOP_HASH,
    apply_tracksdata_ilp,
    assert_tuning_only,
    orgop_config_hash,
    orgop_config_payload,
    refuse_holdout_design,
    tuning_decision,
)
from biohub_tracking.learned_split import HoldoutLeakError  # noqa: E402
from biohub_tracking.official_eval import (  # noqa: E402
    evaluate_official,
    official_dependency_versions,
    official_scorer_available,
    summarise_official,
)


def load_graphs(graph_dir: Path) -> list[dict]:
    payloads = []
    for stem in INFER_STEMS:
        path = graph_dir / f"{stem}.json"
        if not path.exists():
            raise SystemExit(f"missing orgop graph {path}; do not invent a score")
        payload = load_graph_json(path)
        refuse_holdout_design([payload["dataset"]])
        status = str(payload.get("solver_status") or "")
        if status not in {"tracksdata_ilp", "ilp_skipped_no_candidates"}:
            payload = apply_tracksdata_ilp(payload)
        payloads.append(payload)
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--role", default="tuning")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    assert_tuning_only(args.role)
    cfg = orgop_config_payload()
    if orgop_config_hash(cfg) != LOCKED_ORGOP_HASH:
        raise SystemExit("orgop config hash drifted; do not retune from holdout")
    try:
        payloads = load_graphs(args.graph_dir)
    except HoldoutLeakError as exc:
        raise SystemExit(str(exc)) from exc
    refuse_holdout_design(p["dataset"] for p in payloads)

    started = time.time()
    results = []
    if not official_scorer_available():
        raise SystemExit("official scorer source missing")
    for payload in payloads:
        stem = payload["dataset"]
        geff = args.data_dir / f"{stem}.geff"
        if not geff.exists():
            raise SystemExit(f"missing GEFF {geff}; cannot official-score")
        pred = graph_from_payload(payload)
        scored = evaluate_official(pred, geff)
        results.append(scored)
    summary = summarise_official(results)
    decision = tuning_decision(float(summary["score"]))
    report = {
        "status": "tuning_scored",
        "decision": decision,
        "config_hash": LOCKED_ORGOP_HASH,
        "learned_infer_locked_tune": INFER_LOCKED_TUNE,
        "learned_combined": summary["score"],
        "official_summary": summary,
        "per_sequence": [row.as_dict() for row in results],
        "solver_status": [p.get("solver_status") for p in payloads],
        "n_nodes": [len(p.get("node_id") or []) for p in payloads],
        "n_edges": [len(p.get("edges") or []) for p in payloads],
        "n_candidate_edges": [p.get("n_candidate_edges") for p in payloads],
        "code_version": git_version(ROOT),
        "runtime_sec": time.time() - started,
        "peak_memory_mb": peak_memory_mb(),
        "official_deps": official_dependency_versions(),
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
        "notes": "Local official score on frozen-tuning 44b6 only. Not a leaderboard score.",
    }
    dest = ROOT / "reports" / "learned_unet_orgop_v1_tune.json"
    write_run_json(dest, report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "decision",
                    "learned_combined",
                    "learned_infer_locked_tune",
                    "solver_status",
                )
            },
            indent=2,
        )
    )
    if args.record:
        append_experiment(
            ROOT / "experiments" / "registry.csv",
            {
                "experiment_id": new_experiment_id("learned_unet_orgop_v1_tune"),
                "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                "code_version": report["code_version"],
                "config_name": "learned_unet_orgop_v1",
                "seed": 0,
                "split": "tuning:v1.json",
                "n_samples": summary.get("n"),
                "score": summary.get("score"),
                "edge_jaccard": summary.get("edge_jaccard"),
                "adj_edge_jaccard": summary.get("adj_edge_jaccard"),
                "division_jaccard": summary.get("division_jaccard"),
                "node_recall": summary.get("node_recall"),
                "division_tp": summary.get("division_tp"),
                "division_fp": summary.get("division_fp"),
                "division_fn": summary.get("division_fn"),
                "runtime_sec": round(report["runtime_sec"], 3),
                "peak_memory_mb": report["peak_memory_mb"],
                "notes": f"{decision}; hash {LOCKED_ORGOP_HASH}; not a leaderboard score",
            },
        )
    if decision.startswith("reject"):
        raise SystemExit(f"{decision}: combined {summary['score']} is not > {INFER_LOCKED_TUNE}")


if __name__ == "__main__":
    main()
