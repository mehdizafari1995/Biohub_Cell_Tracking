#!/usr/bin/env python3
"""Official-score locked orgdet overlays and decompose 44b6 errors.

Rebuilds orgdet / orgdet-gap / orgdet-gap-prune from locked orgop
detections, scores them with the official scorer, and writes a 44b6-only
error decomposition. Frozen ``6bba`` is refused. A local official score
is not a Kaggle run and not a leaderboard number. Do not retune locked
hashes. Do not invent scores.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.experiments import git_version, peak_memory_mb, write_run_json  # noqa: E402
from biohub_tracking.io_geff import load_geff  # noqa: E402
from biohub_tracking.learned_orgdet_error_decomp import (  # noqa: E402
    INFER_STEMS,
    LOCKED_COMBINED,
    LOCKED_HASHES,
    OVERLAY_LABELS,
    TUNE_KEEP_THRESHOLD,
    TUNE_MARGIN,
    aggregate_error_rows,
    build_locked_overlays,
    classify_sequence,
    dominant_error_mode,
    load_orgop_graphs,
    recommend_hypothesis,
    refuse_holdout_paths,
    refuse_non_tuning_stems,
)
from biohub_tracking.learned_orgdet_gap_prune import LOCKED_ORGDET_GAP_PRUNE_HASH  # noqa: E402
from biohub_tracking.official_eval import (  # noqa: E402
    evaluate_official,
    official_dependency_versions,
    official_scorer_available,
    score_official_division_events,
    summarise_official,
)

REPORT = ROOT / "reports" / "learned_unet_orgdet_error_decomp_44b6.json"


def _resolve_geff(data_dir: Path, stem: str) -> Path:
    geff = data_dir / f"{stem}.geff"
    if not geff.exists():
        nested = data_dir / "train" / f"{stem}.geff"
        geff = nested if nested.exists() else geff
    if not geff.exists():
        raise SystemExit(f"missing GEFF {geff}; cannot official-score")
    refuse_holdout_paths([geff])
    return geff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=ROOT / "artifacts" / "kaggle" / "learned_unet_orgop_v1",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    if not official_scorer_available():
        raise SystemExit("official scorer source missing")
    refuse_non_tuning_stems(INFER_STEMS)
    refuse_holdout_paths(args.data_dir.glob("6bba*"))
    refuse_holdout_paths(args.graph_dir.glob("**/*6bba*"))

    started = time.time()
    graphs = load_orgop_graphs(args.graph_dir, INFER_STEMS)
    overlay_rows: dict[str, list] = {label: [] for label in OVERLAY_LABELS}
    overlay_official: dict[str, list] = {label: [] for label in OVERLAY_LABELS}
    overlay_events: dict[str, list] = {label: [] for label in OVERLAY_LABELS}
    overlay_counts: dict[str, list] = {label: [] for label in OVERLAY_LABELS}

    for orgop in graphs:
        geff = _resolve_geff(args.data_dir, orgop.dataset)
        gt = load_geff(geff)
        overlays = build_locked_overlays(orgop)
        for label, pred in overlays.items():
            scored = evaluate_official(pred, geff)
            overlay_official[label].append(scored)
            overlay_rows[label].append(classify_sequence(pred, gt, scored.as_dict()))
            overlay_counts[label].append(
                {"dataset": pred.dataset, "n_nodes": pred.n_nodes, "n_edges": pred.n_edges}
            )
            try:
                overlay_events[label].append(score_official_division_events(pred, gt))
            except Exception as exc:
                overlay_events[label].append(
                    {"dataset": pred.dataset, "error": f"{type(exc).__name__}: {exc}"[:240]}
                )

    overlays_out: dict[str, Any] = {}
    for label in OVERLAY_LABELS:
        summary = summarise_official(overlay_official[label])
        combined = float(summary["score"])
        locked = LOCKED_COMBINED[label]
        overlays_out[label] = {
            "config_hash": LOCKED_HASHES[label],
            "official_summary": summary,
            "combined": combined,
            "locked_combined": locked,
            "matches_locked": abs(combined - locked) < 1e-9,
            "per_sequence": [row.as_dict() for row in overlay_official[label]],
            "error_rows": overlay_rows[label],
            "aggregate": aggregate_error_rows(overlay_rows[label]),
            "counts": overlay_counts[label],
            "division_events": overlay_events[label],
        }

    prune_agg = overlays_out["orgdet_gap_prune"]["aggregate"]
    mode = dominant_error_mode(prune_agg)
    hypothesis = recommend_hypothesis(mode)
    report = {
        "status": "tuning_error_decomposition",
        "role": "tuning",
        "split": "tuning:v1.json",
        "stems": list(INFER_STEMS),
        "parent_hash": LOCKED_ORGDET_GAP_PRUNE_HASH,
        "overlays": overlays_out,
        "dominant_error": mode,
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
            "Local official-scorer error decomposition of locked orgdet / "
            "orgdet-gap / orgdet-gap-prune on frozen-tuning 44b6 only. "
            "Not a Kaggle run. Not a leaderboard score. Frozen 6bba was not opened. "
            "Do not retune locked hashes. Do not use public 0.602 / 0.612 / 0.553."
        ),
    }
    write_run_json(args.report, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "dominant_mode": mode["mode"],
                "preferred_hypothesis_family": mode["preferred_hypothesis_family"],
                "preregistered_label": hypothesis["label"],
                "tune_keep_threshold": TUNE_KEEP_THRESHOLD,
                "opened_holdout": False,
                "is_leaderboard_score": False,
                "prune_combined": overlays_out["orgdet_gap_prune"]["combined"],
                "matches_locked_prune": overlays_out["orgdet_gap_prune"]["matches_locked"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
