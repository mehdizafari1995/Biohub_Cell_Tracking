#!/usr/bin/python3
"""Official-score learned_unet_orgdet_fnn_rank_v1 on locked 44b6 FNN detections.

Trains the locked pair head on the two frozen-tuning sequences only, applies
constrained 1-in / 2-out selection inside the locked envelope, and
official-scores versus FNN 0.724653 with gate 0.726653. Frozen 6bba is
refused. Does not write submission.csv. A local official score is not a
Kaggle run and not a leaderboard number. Do not retune locked hashes.
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
from biohub_tracking.io_geff import load_geff  # noqa: E402
from biohub_tracking.learned_orgdet_fnn import (  # noqa: E402
    INFER_STEMS,
    LEARNED_ORGDET_FNN_LOCKED_TUNE,
    LOCKED_ORGDET_FNN_HASH,
)
from biohub_tracking.learned_orgdet_fnn_oracle import (  # noqa: E402
    LOCKED_RANK_HASH,
    TUNE_KEEP_THRESHOLD,
    TUNE_MARGIN,
    load_fnn_tuning_graphs,
    rank_config_payload,
)
from biohub_tracking.learned_orgdet_fnn_rank import (  # noqa: E402
    KERNEL_SLUG,
    apply_trained_rank_overlay,
    enumerate_candidates,
    helper_config_payload,
    overlay_summary,
    public_lb_not_a_tune_target,
    refuse_rank_holdout,
    runtime_estimate,
    train_ranker,
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
TUNING_REPORT = ROOT / "reports" / "learned_unet_orgdet_fnn_rank_v1_tune.json"


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


def _resolve_geff(data_dir: Path, stem: str) -> Path:
    geff = data_dir / f"{stem}.geff"
    if not geff.exists():
        nested = data_dir / "train" / f"{stem}.geff"
        geff = nested if nested.exists() else geff
    if not geff.exists():
        raise SystemExit(f"missing GEFF {geff}; cannot official-score")
    refuse_rank_holdout(paths=[geff], stems=[stem])
    return geff


def _record(registry: Path, experiment_id: str, report: dict, extra_note: str) -> None:
    summary = report.get("official_summary") or {}
    append_experiment(
        registry,
        {
            "experiment_id": experiment_id,
            "timestamp_utc": experiment_id.split("_")[0],
            "code_version": report["code_version"],
            "config_name": "learned_unet_orgdet_fnn_rank_v1",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=ROOT / "artifacts" / "kaggle" / "learned_unet_orgop_v1",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--geff-archive", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=TUNING_REPORT)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    cfg = rank_config_payload()
    if cfg["parent_hash"] != LOCKED_ORGDET_FNN_HASH:
        raise SystemExit("ranker is not pinned to locked FNN")
    if LOCKED_RANK_HASH != "2748c4a2b2823c0a":
        raise SystemExit("ranker hash drifted; do not retune")
    if not official_scorer_available():
        raise SystemExit("official scorer source missing")
    public_lb_not_a_tune_target(0.6557507105877639)
    refuse_rank_holdout(stems=INFER_STEMS)
    refuse_rank_holdout(paths=list(Path(args.graph_dir).glob("**/*6bba*")))

    if args.geff_archive is not None:
        extract_selected_geffs(args.geff_archive, args.data_dir.parent, INFER_STEMS)

    started = time.time()
    fnn_graphs = load_fnn_tuning_graphs(args.graph_dir, INFER_STEMS)
    gts = []
    for pred in fnn_graphs:
        geff = _resolve_geff(args.data_dir, pred.dataset)
        gts.append(load_geff(geff))

    trained = train_ranker(fnn_graphs, gts)
    results = []
    overlays = []
    events = []
    n_candidates = 0
    for pred, gt in zip(fnn_graphs, gts):
        cands = enumerate_candidates(pred)
        n_candidates += len(cands)
        after = apply_trained_rank_overlay(pred, trained["model"])
        geff = _resolve_geff(args.data_dir, pred.dataset)
        scored = evaluate_official(after, geff)
        results.append(scored)
        overlays.append(overlay_summary(pred, after, len(cands)))
        try:
            events.append(score_official_division_events(after, gt))
        except Exception as exc:  # noqa: BLE001
            events.append({"dataset": pred.dataset, "error": f"{type(exc).__name__}: {exc}"[:240]})

    summary = summarise_official(results)
    combined = float(summary["score"])
    decision = tuning_decision(combined)
    helper = helper_config_payload()
    report = {
        "status": "tuning_scored",
        "decision": decision,
        "role": "tuning",
        "split": "tuning:v1.json",
        "config_hash": LOCKED_RANK_HASH,
        "parent_fnn_hash": LOCKED_ORGDET_FNN_HASH,
        "fnn_locked_reference": LEARNED_ORGDET_FNN_LOCKED_TUNE,
        "tune_keep_threshold": TUNE_KEEP_THRESHOLD,
        "tune_margin": TUNE_MARGIN,
        "learned_orgdet_fnn_rank_combined": combined,
        "n_candidates": n_candidates,
        "n_true_train": trained["n_true"],
        "n_hard_train": trained["n_hard"],
        "nonzero_edge_grad": trained["nonzero_edge_grad"],
        "finite_losses": trained["finite_losses"],
        "reload_max_abs": trained["reload_max_abs"],
        "used_zarr_appearance": trained["used_zarr_appearance"],
        "used_199_video_weights": trained["used_199_video_weights"],
        "runtime_estimate": runtime_estimate(),
        "helper": helper,
        "kernel_slug": KERNEL_SLUG,
        "overlays": overlays,
        "division_events": events,
        "official_summary": summary,
        "per_sequence": [row.as_dict() for row in results],
        "code_version": git_version(ROOT),
        "runtime_sec": time.time() - started,
        "peak_memory_mb": peak_memory_mb(),
        "official_deps": official_dependency_versions(),
        "opened_holdout": False,
        "wrote_submission_csv": False,
        "kernel_pushed": False,
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
        "notes": (
            "Local official score of learned ranking / constrained 1-in 2-out "
            "selection on locked FNN detections inside the 12 µm / sqrt(dt) "
            "envelope. Coordinate embeddings. Prefer tuning Zarrs; no 69-volume "
            "train. Compared to locked learned_unet_orgdet_fnn_v1 0.724653 with "
            "gate 0.726653. 6bba refused. Not a leaderboard score."
        ),
    }
    write_run_json(args.report, report)
    print(
        json.dumps(
            {
                "decision": decision,
                "learned_orgdet_fnn_rank_combined": combined,
                "fnn_locked_reference": LEARNED_ORGDET_FNN_LOCKED_TUNE,
                "tune_keep_threshold": TUNE_KEEP_THRESHOLD,
                "n_candidates": n_candidates,
                "opened_holdout": False,
                "is_leaderboard_score": False,
            },
            indent=2,
        )
    )
    if args.record:
        _record(
            ROOT / "experiments" / "registry.csv",
            new_experiment_id("learned_unet_orgdet_fnn_rank_v1_tuning"),
            report,
            f"{decision}; hash {LOCKED_RANK_HASH}; candidates={n_candidates}; not a leaderboard score",
        )
    if decision.startswith("keep_"):
        print("KEEP on 44b6; one-shot 6bba is authorized for a later cycle. Not opened.")
    else:
        print("REJECT on 44b6; frozen 6bba stays closed.")


if __name__ == "__main__":
    main()
