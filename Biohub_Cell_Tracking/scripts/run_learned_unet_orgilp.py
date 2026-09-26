#!/usr/bin/env python3
"""Official-score learned_unet_orgilp_v1 on locked orgop detections.

Tuning (``--role tuning``) uses frozen-tuning ``44b6`` orgop graphs only.
A fail-closed preflight must see nonzero detections, spatial candidates,
and a nonempty ILP graph before scoring. Holdout is refused until a
keep_* tuning report exists, then scored once on frozen ``6bba``. Do not
retune. Do not invent scores. A local official score is not a Kaggle run
and not a leaderboard number.
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
from biohub_tracking.learned_orgilp import (  # noqa: E402
    INFER_STEMS,
    LOCKED_ORGILP_HASH,
    ORGDET_GAP_LOCKED_TUNE,
    PreflightError,
    apply_orgilp,
    holdout_decision,
    load_role_graphs,
    orgilp_config_hash,
    orgilp_config_payload,
    overlay_summary,
    refuse_existing_holdout_score,
    refuse_non_tuning_stems,
    require_tuning_keep,
    runtime_estimate,
    tuning_decision,
)
from biohub_tracking.learned_orgop import LOCKED_ORGOP_HASH as REJECTED_ORGOP_HASH  # noqa: E402
from biohub_tracking.learned_split import HoldoutLeakError  # noqa: E402
from biohub_tracking.official_eval import (  # noqa: E402
    evaluate_official,
    official_dependency_versions,
    official_scorer_available,
    score_official_division_events,
    summarise_official,
)

GEFF_INNER_ZIP = "train_geffs_44b6.zip"
TUNING_REPORT = ROOT / "reports" / "learned_unet_orgilp_v1_tune.json"
HOLDOUT_REPORT = ROOT / "reports" / "learned_unet_orgilp_v1_holdout.json"


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
            "config_name": "learned_unet_orgilp_v1",
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


def score_payloads(payloads: list[dict], data_dir: Path) -> tuple[list, list[dict], list[dict]]:
    from biohub_tracking.io_geff import load_geff
    from biohub_tracking.learned_infer import graph_from_payload

    results = []
    summaries = []
    events = []
    for payload in payloads:
        stem = payload["dataset"]
        geff = data_dir / f"{stem}.geff"
        if not geff.exists():
            nested = data_dir / "train" / f"{stem}.geff"
            geff = nested if nested.exists() else geff
        if not geff.exists():
            raise SystemExit(f"missing GEFF {geff}; cannot official-score")
        before = graph_from_payload(payload)
        try:
            after = apply_orgilp(before)
        except PreflightError as exc:
            raise SystemExit(f"preflight failed on {stem}: {exc}") from exc
        summaries.append(overlay_summary(before, after))
        scored = evaluate_official(after, geff)
        results.append(scored)
        try:
            gt = load_geff(geff)
            events.append(score_official_division_events(after, gt))
        except Exception as exc:
            events.append({"dataset": stem, "error": f"{type(exc).__name__}: {exc}"[:240]})
    return results, summaries, events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--role", default="tuning")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--geff-archive", type=Path, default=None)
    args = parser.parse_args()
    if args.role != "tuning":
        raise SystemExit("learned_unet_orgilp_v1 is 44b6-only until a documented keep")
    cfg = orgilp_config_payload()
    if orgilp_config_hash(cfg) != LOCKED_ORGILP_HASH:
        raise SystemExit("orgilp config hash drifted; do not retune from holdout")
    if cfg["rejected_orgop_hash"] != REJECTED_ORGOP_HASH:
        raise SystemExit("rejected orgop hash drifted; do not overwrite orgop")
    if not official_scorer_available():
        raise SystemExit("official scorer source missing")
    estimate = runtime_estimate()
    if not estimate["under_limit"]:
        raise SystemExit("orgilp estimate exceeds the 12 h Kaggle limit; refusing")
    if args.geff_archive is not None:
        extract_selected_geffs(args.geff_archive, args.data_dir.parent, INFER_STEMS)
    try:
        payloads = load_role_graphs(args.graph_dir, INFER_STEMS)
        refuse_non_tuning_stems(p["dataset"] for p in payloads)
    except (FileNotFoundError, ValueError, HoldoutLeakError) as exc:
        raise SystemExit(str(exc)) from exc

    started = time.time()
    results, summaries, events = score_payloads(payloads, args.data_dir)
    summary = summarise_official(results)
    combined = float(summary["score"])
    decision = tuning_decision(combined)
    report = {
        "status": "tuning_scored",
        "decision": decision,
        "role": args.role,
        "split": "tuning:v1.json",
        "config_hash": LOCKED_ORGILP_HASH,
        "rejected_orgop_hash": REJECTED_ORGOP_HASH,
        "orgdet_gap_locked_tune": ORGDET_GAP_LOCKED_TUNE,
        "learned_combined": combined,
        "overlays": summaries,
        "division_events": events,
        "official_summary": summary,
        "per_sequence": [row.as_dict() for row in results],
        "runtime_estimate": estimate,
        "code_version": git_version(ROOT),
        "runtime_sec": time.time() - started,
        "peak_memory_mb": peak_memory_mb(),
        "official_deps": official_dependency_versions(),
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
        "notes": (
            "Local official score on frozen-tuning 44b6 only. "
            "Compared to locked learned_unet_orgdet_gap_v1 0.687221. "
            "Not a leaderboard score. Do not score 6bba unless decision keeps."
        ),
    }
    write_run_json(TUNING_REPORT, report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("decision", "learned_combined", "orgdet_gap_locked_tune", "config_hash")
            },
            indent=2,
        )
    )
    if args.record:
        _record(
            ROOT / "experiments" / "registry.csv",
            new_experiment_id("learned_unet_orgilp_v1_tune"),
            report,
            f"{decision}; hash {LOCKED_ORGILP_HASH}; not a leaderboard score",
        )
    if HOLDOUT_REPORT.exists():
        refuse_existing_holdout_score(HOLDOUT_REPORT)
    if decision.startswith("keep_"):
        require_tuning_keep(TUNING_REPORT)
    else:
        print("holdout refused; 44b6 did not beat locked orgdet-gap", flush=True)


if __name__ == "__main__":
    main()
