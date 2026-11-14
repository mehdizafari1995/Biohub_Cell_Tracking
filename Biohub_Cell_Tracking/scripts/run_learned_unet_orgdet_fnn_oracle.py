#!/usr/bin/env python3
"""44b6-only official FNN detector oracle / feasibility audit.

Restores nothing into git. Rebuilds locked ``learned_unet_orgdet_fnn_v1``
from orgop detections, official-scores that overlay as a restore check,
then official-scores detection-oracle and candidate-oracle graphs.
Frozen ``6bba`` is refused. A local official score is not a Kaggle run
and not a leaderboard number. Do not retune locked hashes. Do not invent
scores. Do not use public 0.602 / 0.612 / 0.553.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

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
from biohub_tracking.learned_orgdet_fnn import (  # noqa: E402
    INFER_STEMS,
    LEARNED_ORGDET_FNN_LOCKED_TUNE,
    LOCKED_ORGDET_FNN_HASH,
)
from biohub_tracking.learned_orgdet_fnn_oracle import (  # noqa: E402
    TUNE_KEEP_THRESHOLD,
    TUNE_MARGIN,
    aggregate_audits,
    audit_sequence,
    load_fnn_tuning_graphs,
    recommend_hypothesis,
    refuse_oracle_holdout,
)
from biohub_tracking.official_eval import (  # noqa: E402
    evaluate_official,
    official_dependency_versions,
    official_scorer_available,
    score_official_division_events,
    summarise_official,
)

REPORT = ROOT / "reports" / "learned_unet_orgdet_fnn_oracle_44b6.json"
LOCKED_FNN_MATCH_ATOL = 1e-9


def _resolve_geff(data_dir: Path, stem: str) -> Path:
    geff = data_dir / f"{stem}.geff"
    if not geff.exists():
        nested = data_dir / "train" / f"{stem}.geff"
        geff = nested if nested.exists() else geff
    if not geff.exists():
        raise SystemExit(f"missing GEFF {geff}; cannot official-score")
    refuse_oracle_holdout(paths=[geff], stems=[stem])
    return geff


def _score_graph(pred, geff, gt) -> tuple[Any, dict]:
    scored = evaluate_official(pred, geff)
    try:
        events = score_official_division_events(pred, gt)
    except Exception as exc:  # noqa: BLE001 — keep the audit going
        events = {"dataset": pred.dataset, "error": f"{type(exc).__name__}: {exc}"[:240]}
    return scored, events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=ROOT / "artifacts" / "kaggle" / "learned_unet_orgop_v1",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    if not official_scorer_available():
        raise SystemExit("official scorer source missing")
    refuse_oracle_holdout(stems=INFER_STEMS)
    refuse_oracle_holdout(paths=list(args.graph_dir.glob("**/*6bba*")))

    started = time.time()
    fnn_graphs = load_fnn_tuning_graphs(args.graph_dir, INFER_STEMS)
    rows: list[dict[str, Any]] = []
    fnn_official = []
    det_official = []
    cand_official = []
    fnn_events = []
    det_events = []
    cand_events = []
    public_rows = []

    for pred in fnn_graphs:
        geff = _resolve_geff(args.data_dir, pred.dataset)
        gt = load_geff(geff)
        scored, events = _score_graph(pred, geff, gt)
        fnn_official.append(scored)
        fnn_events.append(events)
        row = audit_sequence(pred, gt, official=scored.as_dict())
        det_scored, det_ev = _score_graph(row["oracles"]["detection"], geff, gt)
        cand_scored, cand_ev = _score_graph(row["oracles"]["candidate"], geff, gt)
        det_official.append(det_scored)
        cand_official.append(cand_scored)
        det_events.append(det_ev)
        cand_events.append(cand_ev)
        row["detection_oracle_official"] = det_scored.as_dict()
        row["candidate_oracle_official"] = cand_scored.as_dict()
        public = dict(row)
        public.pop("oracles", None)
        public.pop("edge_rows", None)
        public_rows.append(public)
        rows.append(row)

    fnn_summary = summarise_official(fnn_official)
    det_summary = summarise_official(det_official)
    cand_summary = summarise_official(cand_official)
    fnn_combined = float(fnn_summary["score"])
    det_combined = float(det_summary["score"])
    cand_combined = float(cand_summary["score"])
    matches_locked = abs(fnn_combined - float(LEARNED_ORGDET_FNN_LOCKED_TUNE)) < LOCKED_FNN_MATCH_ATOL
    if not matches_locked:
        raise SystemExit(
            f"fnn restore check {fnn_combined} != locked {LEARNED_ORGDET_FNN_LOCKED_TUNE}; "
            "refusing to invent an oracle on a drifted overlay"
        )

    agg = aggregate_audits(public_rows)
    hypothesis = recommend_hypothesis(
        agg,
        candidate_official_score=cand_combined,
        detection_official_score=det_combined,
        fnn_official_score=fnn_combined,
    )
    report = {
        "status": "tuning_fnn_oracle",
        "role": "tuning",
        "split": "tuning:v1.json",
        "stems": list(INFER_STEMS),
        "parent_hash": LOCKED_ORGDET_FNN_HASH,
        "fnn_restore": {
            "official_summary": fnn_summary,
            "combined": fnn_combined,
            "locked_combined": LEARNED_ORGDET_FNN_LOCKED_TUNE,
            "matches_locked": matches_locked,
            "per_sequence": [row.as_dict() for row in fnn_official],
            "division_events": fnn_events,
        },
        "detection_oracle": {
            "official_summary": det_summary,
            "combined": det_combined,
            "per_sequence": [row.as_dict() for row in det_official],
            "division_events": det_events,
        },
        "candidate_oracle": {
            "official_summary": cand_summary,
            "combined": cand_combined,
            "per_sequence": [row.as_dict() for row in cand_official],
            "division_events": cand_events,
        },
        "per_sequence": public_rows,
        "aggregate": agg,
        "preregistered_hypothesis": hypothesis,
        "tune_keep_threshold": TUNE_KEEP_THRESHOLD,
        "tune_margin": TUNE_MARGIN,
        "opened_holdout": False,
        "code_version": git_version(ROOT),
        "runtime_sec": time.time() - started,
        "peak_memory_mb": peak_memory_mb(),
        "official_deps": official_dependency_versions(),
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
        "notes": (
            "Local official-scorer oracle of locked FNN detections on "
            "frozen-tuning 44b6 only. Same nodes as FNN; oracle edges are "
            "GT-true pairs whose endpoints are detected (detection) or also "
            "inside the locked 12 µm / sqrt(dt) envelope (candidate). "
            "Not a Kaggle run. Not a leaderboard score. Frozen 6bba was not "
            "opened. Do not retune locked hashes. Do not use public "
            "0.602 / 0.612 / 0.553."
        ),
    }
    write_run_json(args.report, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "fnn_combined": fnn_combined,
                "matches_locked_fnn": matches_locked,
                "detection_oracle_combined": det_combined,
                "candidate_oracle_combined": cand_combined,
                "candidate_edge_recall": agg["candidate_edge_recall"],
                "detection_fn_share": agg["detection_fn_share"],
                "preregistered_label": hypothesis["label"],
                "preregistered_family": hypothesis["family"],
                "preregistered_hash": hypothesis["config_hash"],
                "tune_keep_threshold": TUNE_KEEP_THRESHOLD,
                "opened_holdout": False,
                "is_leaderboard_score": False,
            },
            indent=2,
        )
    )
    if args.record:
        append_experiment(
            ROOT / "experiments" / "registry.csv",
            {
                "experiment_id": new_experiment_id("learned_unet_orgdet_fnn_oracle_44b6"),
                "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                "code_version": report["code_version"],
                "config_name": "learned_unet_orgdet_fnn_oracle",
                "seed": 0,
                "split": "tuning:v1.json",
                "n_samples": fnn_summary.get("n", 0),
                "score": cand_combined,
                "edge_jaccard": (cand_summary.get("edge_jaccard")),
                "adj_edge_jaccard": cand_summary.get("adj_edge_jaccard"),
                "division_jaccard": cand_summary.get("division_jaccard"),
                "node_recall": cand_summary.get("node_recall"),
                "division_tp": cand_summary.get("division_tp"),
                "division_fp": cand_summary.get("division_fp"),
                "division_fn": cand_summary.get("division_fn"),
                "runtime_sec": round(report["runtime_sec"], 3),
                "peak_memory_mb": report["peak_memory_mb"],
                "notes": (
                    f"fnn_oracle; fnn={fnn_combined}; det_ub={det_combined}; "
                    f"cand_ub={cand_combined}; hyp={hypothesis['label']}; "
                    "not a leaderboard score"
                ),
            },
        )


if __name__ == "__main__":
    main()
