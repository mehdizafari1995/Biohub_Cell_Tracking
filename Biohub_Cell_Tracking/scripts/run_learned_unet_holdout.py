#!/usr/bin/env python3
"""Official-score learned_unet_holdout_v1 graphs once on frozen 6bba.

Refuses 44b6 graphs and a second holdout score. Does not write submission.csv.
A local official score is not a Kaggle run and not a leaderboard number.
Do not retune after seeing this result.
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
from biohub_tracking.learned_holdout import (  # noqa: E402
    ASSOC_LOCKED_HOLDOUT,
    HOLDOUT_STEMS,
    LOCKED_HOLDOUT_HASH,
    assert_holdout_only,
    graph_from_payload,
    holdout_config_hash,
    holdout_config_payload,
    load_graph_json,
    refuse_existing_holdout_score,
    refuse_non_holdout_stems,
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
    for stem in HOLDOUT_STEMS:
        path = graph_dir / f"{stem}.json"
        if not path.exists():
            raise SystemExit(f"missing inferred graph {path}; do not invent a score")
        payload = load_graph_json(path)
        refuse_non_holdout_stems([payload["dataset"]])
        payloads.append(payload)
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True, help="Directory of inferred {stem}.json graphs")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--role", default="validation", help="Must stay 'validation' for this family")
    parser.add_argument("--record", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "learned_unet_holdout_v1_holdout.json",
    )
    args = parser.parse_args()
    assert_holdout_only(args.role)
    refuse_existing_holdout_score(args.report)
    cfg = holdout_config_payload()
    if holdout_config_hash(cfg) != LOCKED_HOLDOUT_HASH:
        raise SystemExit("holdout config hash drifted; do not retune from holdout")
    try:
        payloads = load_graphs(args.graph_dir)
    except HoldoutLeakError as exc:
        raise SystemExit(str(exc)) from exc
    refuse_non_holdout_stems(p["dataset"] for p in payloads)

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
    learned = float(summary["score"])
    decision = "keep_candidate_learned_unet_holdout_v1"
    if learned < ASSOC_LOCKED_HOLDOUT:
        decision = "reject_learned_unet_holdout_v1"
    report = {
        "status": "holdout_scored",
        "decision": decision,
        "config_hash": LOCKED_HOLDOUT_HASH,
        "association_v2_locked_holdout": ASSOC_LOCKED_HOLDOUT,
        "learned_combined": learned,
        "official_summary": summary,
        "per_sequence": [row.as_dict() for row in results],
        "code_version": git_version(ROOT),
        "runtime_sec": time.time() - started,
        "peak_memory_mb": peak_memory_mb(),
        "official_deps": official_dependency_versions(),
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
        "retune_after_holdout": False,
        "notes": (
            "Local official score on frozen-holdout 6bba once. "
            "Not a leaderboard score. Do not retune. Do not score again."
        ),
    }
    write_run_json(args.report, report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("decision", "learned_combined", "association_v2_locked_holdout")
            },
            indent=2,
        )
    )
    if args.record:
        append_experiment(
            ROOT / "experiments" / "registry.csv",
            {
                "experiment_id": new_experiment_id("learned_unet_holdout_v1"),
                "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                "code_version": report["code_version"],
                "config_name": "learned_unet_holdout_v1",
                "seed": 0,
                "split": "validation:v1.json",
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
                "notes": f"{decision}; hash {LOCKED_HOLDOUT_HASH}; not a leaderboard score",
            },
        )


if __name__ == "__main__":
    main()
