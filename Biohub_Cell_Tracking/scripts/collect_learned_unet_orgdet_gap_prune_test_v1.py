#!/usr/bin/env python3
"""Poll and collect mahdimzafari/biohub-learned-unet-orgdet-gap-prune-test-v1.

If the kernel is still running or not yet pushed, record a poll and do not
download. If COMPLETE, download the zip into ignored ``artifacts/kaggle/``
and extract ``submission.csv`` + ``run_manifest.json``. Never writes those
files into git. Never submits to the competition. A local extract is not a
leaderboard score. Do not official-score frozen 6bba holdout again. Do not
collect or overwrite locked orgdet-gap-test, orgdet-test, prune-test,
gap-test, or no-gap test slugs.
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
    GAP_TEST_SLUG,
    HOLDOUT_SLUG,
    INFER_SLUG,
    ORGDET_GAP_PRUNE_TEST_CONFIG_HASH,
    ORGDET_GAP_PRUNE_TEST_SLUG,
    ORGDET_GAP_TEST_SLUG,
    ORGDET_HOLDOUT_SLUG,
    ORGDET_TEST_SLUG,
    PRUNE_TEST_SLUG,
    TEST_SLUG,
    TRAIN_SLUG,
    assert_helper_dest,
    collect_decision,
    extract_learned_test_output,
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
    parser.add_argument("--slug", default=ORGDET_GAP_PRUNE_TEST_SLUG)
    parser.add_argument(
        "--dest",
        type=Path,
        default=ROOT / "artifacts" / "kaggle" / "learned_unet_orgdet_gap_prune_test_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "learned_unet_orgdet_gap_prune_test_v1_kernel_poll.json",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Poll status and list outputs; never download.",
    )
    args = parser.parse_args()
    if args.slug != ORGDET_GAP_PRUNE_TEST_SLUG:
        raise SystemExit(f"refusing unknown learned orgdet-gap-prune-test slug {args.slug!r}")
    locked = {
        TRAIN_SLUG,
        INFER_SLUG,
        HOLDOUT_SLUG,
        TEST_SLUG,
        GAP_TEST_SLUG,
        PRUNE_TEST_SLUG,
        ORGDET_HOLDOUT_SLUG,
        ORGDET_TEST_SLUG,
        ORGDET_GAP_TEST_SLUG,
    }
    if args.slug in locked:
        raise SystemExit("refusing to collect a locked train/infer/holdout/test helper")
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
        "orgdet_gap_prune_test_config_hash": ORGDET_GAP_PRUNE_TEST_CONFIG_HASH,
        "code_version": git_version(ROOT),
        "downloaded": False,
        "extracted": {},
        "is_kaggle_run": status.get("is_complete"),
        "is_competition_submission": False,
        "is_leaderboard_score": False,
        "notes": (
            "Private GPU orgdet-gap-prune test-time poll. Not a competition "
            "submission. CSV stays under ignored artifacts/kaggle/. Not a "
            "leaderboard score. Do not score frozen 6bba holdout again. Do not "
            "overwrite mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1."
        ),
    }
    if decision == "download_complete_helper":
        archive = download_kernel_output_zip(args.slug, dest / "kernel_output.zip")
        written = extract_learned_test_output(archive, dest)
        report["downloaded"] = True
        report["extracted"] = {name: str(path.name) for name, path in written.items()}
        if "submission.csv" not in written:
            raise SystemExit("learned_unet_orgdet_gap_prune_test_v1 COMPLETE zip missing submission.csv")
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
        raise SystemExit("orgdet-gap-prune test-time kernel reported an error; do not invent a submission")


if __name__ == "__main__":
    main()
