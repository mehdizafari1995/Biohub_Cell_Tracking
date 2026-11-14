#!/usr/bin/env python3
"""Official-score learned_unet_orgdet_v1 on frozen-tuning 44b6 only.

Relinks locked ``learned_unet_orgop_v1`` detections with the locked
infer greedy 12 µm last-position linker. Refuses frozen holdout
``6bba`` until a keep_* tuning report exists. Does not write
submission.csv. A local official score is not a Kaggle run and not a
leaderboard number. Do not overwrite rejected orgop or locked slugs.
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
from biohub_tracking.learned_orgdet import (  # noqa: E402
    INFER_LOCKED_TUNE,
    INFER_STEMS,
    LOCKED_ORGDET_HASH,
    REJECTED_ORGOP_HASH,
    apply_orgdet_relink,
    assert_tuning_only,
    edge_geometry,
    graph_from_payload,
    load_role_graphs,
    orgdet_config_hash,
    orgdet_config_payload,
    refuse_existing_holdout_score,
    refuse_non_infer_stems,
    relink_summary,
    require_tuning_keep,
    tuning_decision,
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
TUNING_REPORT = ROOT / "reports" / "learned_unet_orgdet_v1_tune.json"
HOLDOUT_REPORT = ROOT / "reports" / "learned_unet_orgdet_v1_holdout.json"


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


def _record(registry: Path, experiment_id: str, report: dict, extra_note: str) -> None:
    summary = report.get("official_summary") or {}
    append_experiment(
        registry,
        {
            "experiment_id": experiment_id,
            "timestamp_utc": experiment_id.split("_")[0],
            "code_version": report["code_version"],
            "config_name": "learned_unet_orgdet_v1",
            "seed": 0,
            "split": report["split"],
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
            "peak_memory_mb": report["peak_memory_mb"],
            "notes": extra_note,
        },
    )


def score_payloads(payloads: list[dict], data_dir: Path) -> tuple[list, list[dict], list[dict], list[dict]]:
    from biohub_tracking.io_geff import load_geff

    results = []
    summaries = []
    geometries = []
    events = []
    for payload in payloads:
        stem = payload["dataset"]
        geff = data_dir / f"{stem}.geff"
        if not geff.exists():
            raise SystemExit(f"missing GEFF {geff}; cannot official-score")
        before = graph_from_payload(payload)
        after = apply_orgdet_relink(before)
        summaries.append(relink_summary(before, after))
        geometries.append(edge_geometry(after))
        scored = evaluate_official(after, geff)
        results.append(scored)
        try:
            gt = load_geff(geff)
            events.append(score_official_division_events(after, gt))
        except Exception as exc:
            events.append({"dataset": stem, "error": f"{type(exc).__name__}: {exc}"[:240]})
    return results, summaries, geometries, events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--role", default="tuning", help="Must stay 'tuning' for this family until a keep")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--geff-archive", type=Path, default=None)
    args = parser.parse_args()
    assert_tuning_only(args.role)
    cfg = orgdet_config_payload()
    if orgdet_config_hash(cfg) != LOCKED_ORGDET_HASH:
        raise SystemExit("orgdet config hash drifted; do not retune from holdout")
    if cfg["rejected_orgop_hash"] != REJECTED_ORGOP_HASH:
        raise SystemExit("rejected orgop hash drifted; do not overwrite orgop")
    if not official_scorer_available():
        raise SystemExit("official scorer source missing")
    if args.geff_archive is not None:
        extract_selected_geffs(args.geff_archive, args.data_dir.parent, INFER_STEMS)
    try:
        payloads = load_role_graphs(args.graph_dir, INFER_STEMS)
        refuse_non_infer_stems(p["dataset"] for p in payloads)
    except (FileNotFoundError, ValueError, HoldoutLeakError) as exc:
        raise SystemExit(str(exc)) from exc

    started = time.time()
    results, summaries, geometries, events = score_payloads(payloads, args.data_dir)
    summary = summarise_official(results)
    combined = float(summary["score"])
    decision = tuning_decision(combined)
    report = {
        "status": "tuning_scored",
        "decision": decision,
        "role": args.role,
        "split": "tuning:v1.json",
        "config_hash": LOCKED_ORGDET_HASH,
        "rejected_orgop_hash": REJECTED_ORGOP_HASH,
        "learned_unet_infer_v1_locked_tune": INFER_LOCKED_TUNE,
        "learned_combined": combined,
        "relink_summaries": summaries,
        "graph_geometry": geometries,
        "division_events": events,
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
    write_run_json(TUNING_REPORT, report)
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
        _record(
            ROOT / "experiments" / "registry.csv",
            new_experiment_id("learned_unet_orgdet_v1_tune"),
            report,
            f"{decision}; hash {LOCKED_ORGDET_HASH}; not a leaderboard score",
        )
    if HOLDOUT_REPORT.exists():
        refuse_existing_holdout_score(HOLDOUT_REPORT)
    if decision.startswith("keep_"):
        require_tuning_keep(TUNING_REPORT)


if __name__ == "__main__":
    main()
