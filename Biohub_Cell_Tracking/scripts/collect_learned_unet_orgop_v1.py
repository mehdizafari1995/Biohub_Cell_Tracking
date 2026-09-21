#!/usr/bin/env python3
"""Poll and collect the private learned_unet_orgop_v1 GPU helper.

If the kernel is still running or not yet pushed, record a poll and do not
download. If COMPLETE, download the zip into ignored ``artifacts/kaggle/``
and extract 44b6-only graphs. Never writes ``submission.csv`` into git.
Never submits to the competition. A local extract is not a leaderboard score.
Do not score frozen ``6bba`` until a documented 44b6 keep.
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
    INFER_SLUG,
    ORGOP_CONFIG_HASH,
    ORGOP_SLUG,
    TRAIN_SLUG,
    assert_helper_dest,
    collect_decision,
    extract_learned_infer_output,
    safe_kernel_status,
)


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
    parser.add_argument("--slug", default=ORGOP_SLUG)
    parser.add_argument(
        "--dest",
        type=Path,
        default=ROOT / "artifacts" / "kaggle" / "learned_unet_orgop_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "learned_unet_orgop_v1_kernel_poll.json",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Poll status and list outputs; never download.",
    )
    args = parser.parse_args()
    if args.slug != ORGOP_SLUG:
        raise SystemExit(f"refusing unknown learned orgop slug {args.slug!r}")
    if args.slug in {TRAIN_SLUG, INFER_SLUG}:
        raise SystemExit("refusing to collect a locked helper with the orgop collector")
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
        "config_hash": ORGOP_CONFIG_HASH,
        "code_version": git_version(ROOT),
        "downloaded": False,
        "extracted": {},
        "is_kaggle_run": status.get("is_complete"),
        "is_competition_submission": False,
        "is_leaderboard_score": False,
        "notes": (
            "Private GPU orgop-helper poll. Not a competition submission. "
            "Graphs stay under ignored artifacts/kaggle/. Not a leaderboard score. "
            "Do not score frozen 6bba until a documented 44b6 lock."
        ),
    }
    if decision == "download_complete_helper":
        archive = download_kernel_output_zip(args.slug, dest / "kernel_output.zip")
        written = extract_learned_infer_output(archive, dest)
        report["downloaded"] = True
        report["extracted"] = {name: str(path.name) for name, path in written.items()}
        if (dest / "submission.csv").exists():
            raise SystemExit("learned_unet_orgop_v1 helper wrote submission.csv; refusing")
    write_run_json(args.report, report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "status": status.get("status"),
                "is_complete": status.get("is_complete"),
                "n_output_files": len(files),
                "downloaded": report["downloaded"],
                "is_leaderboard_score": False,
            },
            indent=2,
        )
    )
    if decision == "record_train_error":
        raise SystemExit("orgop helper reported an error; do not invent graphs")


if __name__ == "__main__":
    main()
