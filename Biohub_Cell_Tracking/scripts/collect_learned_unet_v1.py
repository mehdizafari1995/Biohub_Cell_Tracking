#!/usr/bin/env python3
"""Poll and collect the private learned_unet_v1 GPU helper.

If the kernel is still running, record a poll and do not download.
If COMPLETE, download the zip into ignored ``artifacts/kaggle/`` and extract
weights + manifests only. Never writes ``submission.csv`` into git. Never
submits to the competition. A local extract is not a leaderboard score.
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
    TRAIN_SLUG,
    assert_helper_dest,
    collect_decision,
    extract_learned_train_output,
    safe_kernel_status,
    summarize_collected_train,
)
from biohub_tracking.learned_split import config_hash  # noqa: E402


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
    parser.add_argument("--slug", default=TRAIN_SLUG)
    parser.add_argument(
        "--dest",
        type=Path,
        default=ROOT / "artifacts" / "kaggle" / "learned_unet_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "learned_unet_v1_kernel_poll.json",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Poll status and list outputs; never download.",
    )
    args = parser.parse_args()
    if args.slug != TRAIN_SLUG:
        raise SystemExit(f"refusing unknown learned train slug {args.slug!r}")
    dest = assert_helper_dest(args.dest)
    raw = kernel_status(args.slug)
    status = safe_kernel_status(raw, args.slug)
    files = _safe_files(args.slug)
    decision = "status_only" if args.status_only else collect_decision(status)
    report = {
        "slug": args.slug,
        "polled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kernel": status,
        "output_files": files,
        "decision": decision,
        "config_hash": config_hash(),
        "code_version": git_version(ROOT),
        "downloaded": False,
        "extracted": {},
        "collected": None,
        "is_kaggle_run": status.get("is_complete"),
        "is_competition_submission": False,
        "is_leaderboard_score": False,
        "notes": (
            "Private GPU helper poll. Not a competition submission. "
            "Weights stay under ignored artifacts/kaggle/. Not a leaderboard score."
        ),
    }
    if decision == "download_complete_helper":
        archive = download_kernel_output_zip(args.slug, dest / "kernel_output.zip")
        written = extract_learned_train_output(archive, dest)
        report["downloaded"] = True
        report["extracted"] = {name: str(path.name) for name, path in written.items()}
        report["collected"] = summarize_collected_train(dest)
        if report["collected"].get("has_submission_csv"):
            raise SystemExit("learned_unet_v1 helper wrote submission.csv; refusing")
    write_run_json(args.report, report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "status": status.get("status"),
                "is_complete": status.get("is_complete"),
                "n_output_files": len(files),
                "downloaded": report["downloaded"],
                "collected_status": (report["collected"] or {}).get("status"),
                "lastRunTime": status.get("lastRunTime"),
                "is_leaderboard_score": False,
            },
            indent=2,
        )
    )
    if decision == "record_train_error":
        raise SystemExit("training helper reported an error; do not invent weights")


if __name__ == "__main__":
    main()
