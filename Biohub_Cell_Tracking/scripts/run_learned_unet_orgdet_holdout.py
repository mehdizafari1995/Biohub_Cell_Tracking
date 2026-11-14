#!/usr/bin/env python3
"""Official-score learned_unet_orgdet_holdout_v1 graphs once on frozen 6bba.

Requires the 44b6 orgdet keep report. Refuses 44b6 graphs and a second
holdout score. Compares against locked prune champion 0.701378. Does not
write submission.csv. A local official score is not a Kaggle run and not
a leaderboard number. Do not retune hash 80b20589198148b7.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from io import BytesIO
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
from biohub_tracking.learned_orgdet_holdout import (  # noqa: E402
    HOLDOUT_STEMS,
    LOCKED_ORGDET_HASH,
    LOCKED_ORGDET_HOLDOUT_HASH,
    PRUNE_LOCKED_HOLDOUT,
    REJECTED_ORGOP_HASH,
    assert_holdout_only,
    graph_from_payload,
    holdout_decision,
    load_graph_json,
    orgdet_holdout_config_hash,
    orgdet_holdout_config_payload,
    refuse_existing_holdout_score,
    refuse_non_holdout_stems,
    require_orgdet_tuning_keep,
    runtime_estimate,
)
from biohub_tracking.learned_split import HoldoutLeakError  # noqa: E402
from biohub_tracking.official_eval import (  # noqa: E402
    evaluate_official,
    official_dependency_versions,
    official_scorer_available,
    score_official_division_events,
    summarise_official,
)

GEFF_INNER_ZIP = "train_geffs_44b6.zip"
ORGDET_TUNE_REPORT = ROOT / "reports" / "learned_unet_orgdet_v1_tune.json"
HOLDOUT_REPORT = ROOT / "reports" / "learned_unet_orgdet_holdout_v1_holdout.json"


def extract_selected_geffs(archive: Path, dest_root: Path, stems: tuple[str, ...]) -> list[str]:
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    wanted = {f"{stem}.geff" for stem in stems}
    written: list[str] = []

    def _extract(zf: zipfile.ZipFile) -> None:
        for name in zf.namelist():
            if name.endswith("/") or ".zarr/" in name or name.endswith(".zarr"):
                continue
            if not name.startswith("train/") or ".geff/" not in name:
                continue
            store = name.split("/")[1]
            if store not in wanted:
                continue
            target = dest_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))
            if store not in written:
                written.append(store)

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        inner_name = None
        if GEFF_INNER_ZIP in names:
            inner_name = GEFF_INNER_ZIP
        else:
            for name in names:
                if name.endswith("/" + GEFF_INNER_ZIP) or name.endswith(GEFF_INNER_ZIP):
                    inner_name = name
                    break
        if inner_name:
            with zipfile.ZipFile(BytesIO(zf.read(inner_name))) as inner:
                _extract(inner)
        else:
            _extract(zf)
    return written


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
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--role", default="validation", help="Must stay 'validation' for this family")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--geff-archive", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=HOLDOUT_REPORT)
    args = parser.parse_args()
    assert_holdout_only(args.role)
    refuse_existing_holdout_score(args.report)
    require_orgdet_tuning_keep(ORGDET_TUNE_REPORT)
    cfg = orgdet_holdout_config_payload()
    if orgdet_holdout_config_hash(cfg) != LOCKED_ORGDET_HOLDOUT_HASH:
        raise SystemExit("orgdet holdout config hash drifted; do not retune")
    if cfg["orgdet_config_hash"] != LOCKED_ORGDET_HASH:
        raise SystemExit("parent orgdet hash drifted; do not retune 80b20589198148b7")
    if cfg["rejected_orgop_hash"] != REJECTED_ORGOP_HASH:
        raise SystemExit("rejected orgop hash drifted; do not overwrite orgop")
    if not official_scorer_available():
        raise SystemExit("official scorer source missing")
    if args.geff_archive is not None:
        extract_selected_geffs(args.geff_archive, args.data_dir.parent, HOLDOUT_STEMS)
    try:
        payloads = load_graphs(args.graph_dir)
    except HoldoutLeakError as exc:
        raise SystemExit(str(exc)) from exc
    refuse_non_holdout_stems(p["dataset"] for p in payloads)

    started = time.time()
    results = []
    events = []
    from biohub_tracking.io_geff import load_geff

    for payload in payloads:
        stem = payload["dataset"]
        geff = args.data_dir / f"{stem}.geff"
        if not geff.exists():
            raise SystemExit(f"missing GEFF {geff}; cannot official-score")
        pred = graph_from_payload(payload)
        scored = evaluate_official(pred, geff)
        results.append(scored)
        try:
            gt = load_geff(geff)
            events.append(score_official_division_events(pred, gt))
        except Exception as exc:
            events.append({"dataset": stem, "error": f"{type(exc).__name__}: {exc}"[:240]})
    summary = summarise_official(results)
    learned = float(summary["score"])
    decision = holdout_decision(learned)
    report = {
        "status": "holdout_scored",
        "decision": decision,
        "role": args.role,
        "split": "validation:v1.json",
        "config_hash": LOCKED_ORGDET_HOLDOUT_HASH,
        "orgdet_config_hash": LOCKED_ORGDET_HASH,
        "rejected_orgop_hash": REJECTED_ORGOP_HASH,
        "learned_prune_locked_holdout": PRUNE_LOCKED_HOLDOUT,
        "learned_combined": learned,
        "runtime_estimate": runtime_estimate(2),
        "division_events": events,
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
            "Local official score on frozen-holdout 6bba once at locked orgdet "
            "0.99/pool-3.0/greedy-12. Compared to learned_unet_prune_v1 0.701378. "
            "Not a leaderboard score. Do not retune. Do not score again."
        ),
    }
    write_run_json(args.report, report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "decision",
                    "learned_combined",
                    "learned_prune_locked_holdout",
                    "config_hash",
                )
            },
            indent=2,
        )
    )
    if args.record:
        append_experiment(
            ROOT / "experiments" / "registry.csv",
            {
                "experiment_id": new_experiment_id("learned_unet_orgdet_holdout_v1"),
                "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                "code_version": report["code_version"],
                "config_name": "learned_unet_orgdet_holdout_v1",
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
                "notes": f"{decision}; hash {LOCKED_ORGDET_HOLDOUT_HASH}; not a leaderboard score",
            },
        )


if __name__ == "__main__":
    main()
