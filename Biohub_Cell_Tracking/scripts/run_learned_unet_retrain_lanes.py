#!/usr/bin/env python3
"""Official-score preregistered learned_unet_retrain_v1 44b6 lanes A/B/C.

Refuses frozen holdout ``6bba``. Refuses an incomplete retrain helper.
Does not write submission.csv. A local official score is not a Kaggle
run and not a leaderboard number. Do not invent a score when graphs
are missing. Do not open 6bba unless a later locked winner exists.
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
from biohub_tracking.learned_infer import (  # noqa: E402
    graph_from_payload,
    load_graph_json,
)
from biohub_tracking.learned_retrain import LOCKED_RETRAIN_HASH  # noqa: E402
from biohub_tracking.learned_retrain_lanes import (  # noqa: E402
    LANE_A_HASH,
    LANE_B_HASH,
    LANE_C_HASH,
    LANE_LABELS,
    assert_tuning_only,
    decide_all_lanes,
    refuse_incomplete_retrain,
    require_lane_graph_paths,
)
from biohub_tracking.learned_split import HoldoutLeakError  # noqa: E402
from biohub_tracking.official_eval import (  # noqa: E402
    evaluate_official,
    official_dependency_versions,
    official_scorer_available,
    summarise_official,
)

TUNING_REPORT = ROOT / "reports" / "learned_unet_retrain_lanes_v1_tune.json"
POLL_REPORT = ROOT / "reports" / "learned_unet_retrain_v1_kernel_poll.json"


def _score_dir(graph_dir: Path, data_dir: Path) -> tuple[list, dict]:
    payloads = [load_graph_json(path) for path in require_lane_graph_paths(graph_dir)]
    results = []
    for payload in payloads:
        stem = payload["dataset"]
        geff = data_dir / f"{stem}.geff"
        if not geff.exists():
            raise SystemExit(f"missing GEFF {geff}; cannot official-score")
        results.append(evaluate_official(graph_from_payload(payload), geff))
    return results, summarise_official(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir-a", type=Path, required=True)
    parser.add_argument("--graph-dir-b", type=Path, required=True)
    parser.add_argument("--graph-dir-c", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--role", default="tuning", help="Must stay 'tuning' until a locked winner")
    parser.add_argument("--record", action="store_true")
    parser.add_argument(
        "--poll-report",
        type=Path,
        default=POLL_REPORT,
        help="Retrain helper poll JSON; refuse unless complete",
    )
    args = parser.parse_args()
    assert_tuning_only(args.role)
    if args.poll_report.exists():
        poll = json.loads(args.poll_report.read_text())
        kernel = poll.get("kernel") or {}
        refuse_incomplete_retrain(
            kernel.get("status"),
            [item.get("name") for item in (poll.get("output_files") or [])],
        )
    if not official_scorer_available():
        raise SystemExit("official scorer source missing")
    started = time.time()
    try:
        scored = {
            "A": _score_dir(args.graph_dir_a, args.data_dir),
            "B": _score_dir(args.graph_dir_b, args.data_dir),
            "C": _score_dir(args.graph_dir_c, args.data_dir),
        }
    except (FileNotFoundError, HoldoutLeakError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    scores = {name: float(summary["score"]) for name, (_rows, summary) in scored.items()}
    decision = decide_all_lanes(scores)
    report = {
        "status": "tuning_scored",
        "role": args.role,
        "split": "tuning:v1.json",
        "parent_retrain_hash": LOCKED_RETRAIN_HASH,
        "lane_hashes": {"A": LANE_A_HASH, "B": LANE_B_HASH, "C": LANE_C_HASH},
        "scores": scores,
        **decision,
        "official_summaries": {name: summary for name, (_rows, summary) in scored.items()},
        "per_sequence": {
            name: [row.as_dict() for row in rows] for name, (rows, _summary) in scored.items()
        },
        "code_version": git_version(ROOT),
        "runtime_sec": time.time() - started,
        "peak_memory_mb": peak_memory_mb(),
        "official_deps": official_dependency_versions(),
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
        "notes": (
            "Local official score on frozen-tuning 44b6 only. "
            "Lanes A/B/C were locked before scores. Not a leaderboard score. "
            "Do not open 6bba unless a lane strictly wins and is locked."
        ),
    }
    write_run_json(TUNING_REPORT, report)
    print(
        json.dumps(
            {
                "winner": report["winner"],
                "scores": scores,
                "decisions": report["decisions"],
                "may_open_6bba": report["may_open_6bba"],
                "opens_6bba": report["opens_6bba"],
            },
            indent=2,
        )
    )
    if args.record:
        append_experiment(
            ROOT / "experiments" / "registry.csv",
            {
                "experiment_id": new_experiment_id("learned_unet_retrain_lanes_v1_tune"),
                "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                "code_version": report["code_version"],
                "config_name": "learned_unet_retrain_v1",
                "seed": 0,
                "split": "tuning:v1.json",
                "n_samples": 2,
                "score": scores.get(report["winner"] or "A"),
                "notes": (
                    f"winner={report['winner']}; decisions={report['decisions']}; "
                    f"hashes A/B/C {LANE_A_HASH}/{LANE_B_HASH}/{LANE_C_HASH}; "
                    "not a leaderboard score"
                ),
            },
        )
    _ = LANE_LABELS


if __name__ == "__main__":
    main()
