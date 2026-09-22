#!/usr/bin/env python3
"""Poll and collect the private learned_unet_retrain_v1 GPU helper.

If the kernel is still running, record a poll and do not download.
If COMPLETE, download the zip into ignored ``artifacts/kaggle/``.
Never writes ``submission.csv`` into git. Never submits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.kaggle_api import (  # noqa: E402
    download_kernel_output_zip,
    kernel_output_list,
    kernel_status,
)
from biohub_tracking.learned_collect import (  # noqa: E402
    extract_learned_retrain_output,
    summarize_collected_retrain,
)
from biohub_tracking.learned_retrain import LOCKED_RETRAIN_HASH, RETRAIN_KERNEL_SLUG  # noqa: E402


def _safe_status(raw: dict, slug: str) -> dict:
    return {
        "slug": slug,
        "status": raw.get("status"),
        "failureMessage": raw.get("failureMessage"),
        "hasFailureMessage": raw.get("hasFailureMessage"),
        "versionNumber": raw.get("versionNumber") or raw.get("currentVersionNumber"),
    }


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
    parser.add_argument("--slug", default=RETRAIN_KERNEL_SLUG)
    parser.add_argument(
        "--dest",
        type=Path,
        default=ROOT / "artifacts" / "kaggle" / "learned_unet_retrain_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "learned_unet_retrain_v1_kernel_poll.json",
    )
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    if args.slug != RETRAIN_KERNEL_SLUG:
        raise SystemExit(f"refusing unknown retrain slug {args.slug!r}")
    dest = args.dest.resolve()
    if "artifacts" not in dest.parts or "kaggle" not in dest.parts:
        raise SystemExit("retrain downloads must stay under ignored artifacts/kaggle/")
    raw = kernel_status(args.slug)
    status = _safe_status(raw, args.slug)
    files = _safe_files(args.slug)
    names = [str(item.get("name") or "") for item in files]
    if "submission.csv" in names:
        raise SystemExit("retrain helper must not write submission.csv")
    decision = "status_only"
    extracted: dict[str, str] = {}
    collected = None
    if not args.status_only and str(status.get("status") or "").lower() in {"complete", "completed"}:
        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest / "learned_unet_retrain_v1_output.zip"
        if not zip_path.exists():
            download_kernel_output_zip(args.slug, zip_path)
        written = extract_learned_retrain_output(zip_path, dest)
        extracted = {name: str(path.name) for name, path in written.items()}
        collected = summarize_collected_retrain(dest)
        if collected.get("has_submission_csv"):
            raise SystemExit("retrain helper wrote submission.csv; refusing")
        decision = "downloaded"
    report = {
        "slug": args.slug,
        "polled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_hash": LOCKED_RETRAIN_HASH,
        "kernel": status,
        "output_files": files,
        "decision": decision,
        "extracted": extracted,
        "collected": collected,
        "is_competition_submission": False,
        "is_leaderboard_score": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": status.get("status"), "decision": decision, "n_files": len(files)}))


if __name__ == "__main__":
    main()
