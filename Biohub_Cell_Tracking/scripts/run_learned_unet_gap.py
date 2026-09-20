#!/usr/bin/env python3
"""Official-score learned_unet_gap_v1 overlay on locked learned graphs.

Tuning (``--role tuning``) uses frozen-tuning ``44b6`` graphs only.
Holdout (``--role holdout``) is refused until a keep_* tuning report exists,
then scored once on frozen ``6bba``. Do not retune. Do not invent scores.
A local official score is not a Kaggle run and not a leaderboard number.
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
from biohub_tracking.learned_gap import (  # noqa: E402
    HOLDOUT_STEMS,
    INFER_STEMS,
    LEARNED_LOCKED_HOLDOUT,
    LEARNED_LOCKED_TUNE,
    LOCKED_GAP_HASH,
    apply_locked_overlay,
    gap_config_hash,
    gap_config_payload,
    graph_from_payload,
    holdout_decision,
    load_role_graphs,
    overlay_summary,
    refuse_existing_holdout_score,
    refuse_non_holdout_stems,
    refuse_non_tuning_stems,
    require_tuning_keep,
    tuning_decision,
)
from biohub_tracking.official_eval import (  # noqa: E402
    evaluate_official,
    official_dependency_versions,
    official_scorer_available,
    score_official_division_events,
    summarise_official,
)

GEFF_INNER_ZIP = "train_geffs_44b6.zip"
TUNING_REPORT = ROOT / "reports" / "learned_unet_gap_v1_tune.json"
HOLDOUT_REPORT = ROOT / "reports" / "learned_unet_gap_v1_holdout.json"


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


def _record(registry: Path, experiment_id: str, config_name: str, report: dict, extra_note: str) -> None:
    summary = report.get("official_summary") or {}
    append_experiment(
        registry,
        {
            "experiment_id": experiment_id,
            "timestamp_utc": experiment_id.split("_")[0],
            "code_version": report["code_version"],
            "config_name": config_name,
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


def score_payloads(payloads: list[dict], data_dir: Path) -> tuple[list, list[dict], list[dict], int]:
    from biohub_tracking.io_geff import load_geff

    results = []
    overlays = []
    events = []
    n_extra = 0
    for payload in payloads:
        stem = payload["dataset"]
        geff = data_dir / f"{stem}.geff"
        if not geff.exists():
            raise SystemExit(f"missing GEFF {geff}; cannot official-score")
        before = graph_from_payload(payload)
        after = apply_locked_overlay(before)
        summary = overlay_summary(before, after)
        overlays.append(summary)
        n_extra += int(summary["n_extra_edges"])
        scored = evaluate_official(after, geff)
        results.append(scored)
        try:
            gt = load_geff(geff)
            events.append(score_official_division_events(after, gt))
        except Exception as exc:
            events.append({"dataset": stem, "error": f"{type(exc).__name__}: {exc}"[:240]})
    return results, overlays, events, n_extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--role", choices=("tuning", "holdout"), required=True)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    cfg = gap_config_payload()
    if gap_config_hash(cfg) != LOCKED_GAP_HASH:
        raise SystemExit("learned_unet_gap_v1 hash drifted; do not retune")
    if not official_scorer_available():
        raise SystemExit("official scorer source missing")

    if args.role == "tuning":
        stems = INFER_STEMS
        refuse_fn = refuse_non_tuning_stems
        dest = TUNING_REPORT
        locked_ref = LEARNED_LOCKED_TUNE
        split = "tuning:v1.json"
    else:
        require_tuning_keep(TUNING_REPORT)
        refuse_existing_holdout_score(HOLDOUT_REPORT)
        stems = HOLDOUT_STEMS
        refuse_fn = refuse_non_holdout_stems
        dest = HOLDOUT_REPORT
        locked_ref = LEARNED_LOCKED_HOLDOUT
        split = "validation:v1.json"

    try:
        payloads = load_role_graphs(args.graph_dir, stems)
        refuse_fn(p["dataset"] for p in payloads)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    started = time.time()
    results, overlays, events, n_extra = score_payloads(payloads, args.data_dir)
    summary = summarise_official(results)
    combined = float(summary["score"])
    if args.role == "tuning":
        decision = tuning_decision(combined, n_extra)
        status = "tuning_scored"
    else:
        decision = holdout_decision(combined)
        status = "holdout_scored"

    report = {
        "status": status,
        "decision": decision,
        "role": args.role,
        "split": split,
        "config_hash": LOCKED_GAP_HASH,
        "learned_locked_reference": locked_ref,
        "learned_gap_combined": combined,
        "n_extra_edges": n_extra,
        "overlays": overlays,
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
            "Local official score of locked association_v2 gap-close overlay "
            "on locked learned graphs. Not a leaderboard score. Do not retune."
        ),
    }
    write_run_json(dest, report)
    print(
        json.dumps(
            {
                "decision": decision,
                "learned_gap_combined": combined,
                "learned_locked_reference": locked_ref,
                "n_extra_edges": n_extra,
            },
            indent=2,
        )
    )
    if args.record:
        _record(
            ROOT / "experiments" / "registry.csv",
            new_experiment_id(f"learned_unet_gap_v1_{args.role}"),
            "learned_unet_gap_v1",
            report,
            f"{decision}; hash {LOCKED_GAP_HASH}; extra_edges={n_extra}; not a leaderboard score",
        )


if __name__ == "__main__":
    main()
