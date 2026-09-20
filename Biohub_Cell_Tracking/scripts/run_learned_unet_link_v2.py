#!/usr/bin/env python3
"""Official-score learned_unet_link_v2 graphs on frozen-tuning 44b6 only.

Refuses frozen holdout ``6bba``. Does not write submission.csv.
A local official score is not a Kaggle run and not a leaderboard number.
Do not overwrite rejected ``learned_unet_link_v1``.
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
from biohub_tracking.learned_link_v2 import (  # noqa: E402
    INFER_LOCKED_TUNE,
    INFER_STEMS,
    LOCKED_LINK_V2_HASH,
    REJECTED_LINK_V1_HASH,
    assert_tuning_only,
    graph_from_payload,
    link_config_hash,
    link_config_payload,
    load_graph_json,
    refuse_non_infer_stems,
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
            raise SystemExit(f"missing inferred graph {path}; do not invent a score")
        payload = load_graph_json(path)
        refuse_non_infer_stems([payload["dataset"]])
        payloads.append(payload)
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True, help="Directory of inferred {stem}.json graphs")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--role", default="tuning", help="Must stay 'tuning' for this family")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    assert_tuning_only(args.role)
    cfg = link_config_payload()
    if link_config_hash(cfg) != LOCKED_LINK_V2_HASH:
        raise SystemExit("link_v2 config hash drifted; do not retune from holdout")
    if cfg["rejected_link_v1_hash"] != REJECTED_LINK_V1_HASH:
        raise SystemExit("rejected link_v1 hash drifted; do not overwrite v1")
    try:
        payloads = load_graphs(args.graph_dir)
    except HoldoutLeakError as exc:
        raise SystemExit(str(exc)) from exc
    refuse_non_infer_stems(p["dataset"] for p in payloads)

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
    decision = "keep_candidate_learned_unet_link_v2"
    if float(summary["score"]) < INFER_LOCKED_TUNE:
        decision = "reject_learned_unet_link_v2"
    report = {
        "status": "tuning_scored",
        "decision": decision,
        "config_hash": LOCKED_LINK_V2_HASH,
        "rejected_link_v1_hash": REJECTED_LINK_V1_HASH,
        "learned_unet_infer_v1_locked_tune": INFER_LOCKED_TUNE,
        "learned_combined": summary["score"],
        "official_summary": summary,
        "per_sequence": [row.as_dict() for row in results],
        "code_version": git_version(ROOT),
        "runtime_sec": time.time() - started,
        "peak_memory_mb": peak_memory_mb(),
        "official_deps": official_dependency_versions(),
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
        "notes": (
            "Local official score on frozen-tuning 44b6 only. "
            "Compared to locked learned_unet_infer_v1. Not a leaderboard score. "
            "Do not score 6bba unless decision keeps the candidate."
        ),
    }
    dest = ROOT / "reports" / "learned_unet_link_v2_tune.json"
    write_run_json(dest, report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("decision", "learned_combined", "learned_unet_infer_v1_locked_tune")
            },
            indent=2,
        )
    )
    if args.record:
        append_experiment(
            ROOT / "experiments" / "registry.csv",
            {
                "experiment_id": new_experiment_id("learned_unet_link_v2_tune"),
                "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                "code_version": report["code_version"],
                "config_name": "learned_unet_link_v2",
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
                "notes": f"{decision}; hash {LOCKED_LINK_V2_HASH}; not a leaderboard score",
            },
        )


if __name__ == "__main__":
    main()
