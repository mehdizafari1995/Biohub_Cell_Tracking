#!/usr/bin/env python3
"""Poll and collect one preregistered learned_unet_retrain 44b6 lane helper.

Lanes A/B/C only. If the kernel is still running or not yet pushed, record a
poll and do not download. If COMPLETE, extract 44b6-only graphs into ignored
``artifacts/kaggle/``. Never writes ``submission.csv`` into git. Never
submits. Never opens ``6bba``. A local extract is not a leaderboard score.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.experiments import git_version, write_run_json  # noqa: E402
from biohub_tracking.kaggle_api import (  # noqa: E402
    download_kernel_output_zip,
    kernel_output_list,
    kernel_status,
)
from biohub_tracking.learned_collect import (  # noqa: E402
    RETRAIN_SLUG,
    TRAIN_SLUG,
    assert_helper_dest,
    collect_decision,
    extract_learned_infer_output,
    safe_kernel_status,
)
from biohub_tracking.learned_retrain_lanes import (  # noqa: E402
    LANE_A_HASH,
    LANE_A_SLUG,
    LANE_B_HASH,
    LANE_B_SLUG,
    LANE_C_HASH,
    LANE_C_SLUG,
)

LANES = {
    "A": {"slug": LANE_A_SLUG, "hash": LANE_A_HASH, "label": "learned_unet_retrain_infer_v1"},
    "B": {"slug": LANE_B_SLUG, "hash": LANE_B_HASH, "label": "learned_unet_retrain_orgdet_v1"},
    "C": {"slug": LANE_C_SLUG, "hash": LANE_C_HASH, "label": "learned_unet_retrain_orgop_v1"},
}
FORBIDDEN = {TRAIN_SLUG, RETRAIN_SLUG}


def _safe_files(slug: str) -> list[dict]:
    files = kernel_output_list(slug)
    return [
        {
            "name": item.get("fileName") or item.get("name"),
            "size": item.get("fileSize") or item.get("totalBytes") or item.get("size"),
        }
        for item in files
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True, choices=sorted(LANES))
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    spec = LANES[args.lane]
    slug = spec["slug"]
    if slug in FORBIDDEN:
        raise SystemExit(f"refusing locked parent slug {slug}")
    dest = assert_helper_dest(ROOT / "artifacts" / "kaggle" / spec["label"])
    report_path = ROOT / "reports" / f"{spec['label']}_kernel_poll.json"
    try:
        raw = kernel_status(slug)
        status = safe_kernel_status(raw, slug)
        files = _safe_files(slug)
    except Exception as exc:
        report = {
            "slug": slug,
            "lane": args.lane,
            "polled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config_hash": spec["hash"],
            "decision": "not_pushed",
            "error_type": type(exc).__name__,
            "downloaded": False,
            "is_competition_submission": False,
            "is_leaderboard_score": False,
            "code_version": git_version(ROOT),
        }
        write_run_json(report_path, report)
        print(json.dumps({"lane": args.lane, "decision": "not_pushed"}, indent=2))
        return
    names = [str(item.get("name") or "") for item in files]
    if "submission.csv" in names:
        raise SystemExit("retrain lane helper must not write submission.csv")
    decision = "status_only" if args.status_only else collect_decision(status)
    report = {
        "slug": slug,
        "lane": args.lane,
        "polled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_hash": spec["hash"],
        "kernel": status,
        "output_files": files,
        "decision": decision,
        "downloaded": False,
        "extracted": {},
        "code_version": git_version(ROOT),
        "is_competition_submission": False,
        "is_leaderboard_score": False,
        "notes": (
            "Private GPU 44b6 lane poll. Not a competition submission. "
            "Do not open 6bba. Not a leaderboard score."
        ),
    }
    if decision == "download_complete_helper":
        archive = download_kernel_output_zip(slug, dest / "kernel_output.zip")
        written = extract_learned_infer_output(archive, dest)
        report["downloaded"] = True
        report["extracted"] = {name: str(path.name) for name, path in written.items()}
        if (dest / "submission.csv").exists():
            raise SystemExit("retrain lane helper wrote submission.csv; refusing")
    write_run_json(report_path, report)
    print(
        json.dumps(
            {
                "lane": args.lane,
                "slug": slug,
                "decision": decision,
                "status": status.get("status"),
                "n_files": len(files),
                "downloaded": report["downloaded"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
