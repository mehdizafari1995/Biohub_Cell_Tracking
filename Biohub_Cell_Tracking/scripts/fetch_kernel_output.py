#!/usr/bin/env python3
"""Download a private kernel output archive into ignored artifacts/kaggle.

Never prints credentials. Never submits. Never writes competition data into git.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.kaggle_api import (  # noqa: E402
    download_kernel_output_zip,
    kernel_output_list,
    kernel_status,
)

ALLOWED_SLUGS = {
    "mahdimzafari/biohub-geff-export",
    "mahdimzafari/biohub-holdout-zarr-export",
    "mahdimzafari/biohub-tuning-zarr-export",
    "mahdimzafari/biohub-association-v2",
    "mahdimzafari/biohub-baseline-v1",
    "mahdimzafari/biohub-learned-unet-v1",
}


def _safe_status(slug: str) -> dict:
    raw = kernel_status(slug)
    return {
        "slug": slug,
        "status": raw.get("status"),
        "failureMessage": raw.get("failureMessage"),
        "hasFailureMessage": raw.get("hasFailureMessage"),
        "versionNumber": raw.get("versionNumber") or raw.get("currentVersionNumber"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", help="user/kernel slug")
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination zip under artifacts/kaggle (ignored).",
    )
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    if args.slug not in ALLOWED_SLUGS:
        raise SystemExit(f"refusing unknown kernel slug {args.slug!r}")
    status = _safe_status(args.slug)
    print(json.dumps(status), flush=True)
    if args.status_only:
        return
    if args.list_only:
        files = kernel_output_list(args.slug)
        safe = [{"name": f.get("fileName") or f.get("name"), "size": f.get("fileSize") or f.get("size")} for f in files]
        print(json.dumps({"n": len(safe), "files": safe}), flush=True)
        return
    dest = args.dest
    if dest is None:
        stem = args.slug.split("/", 1)[1]
        dest = ROOT / "artifacts" / "kaggle" / f"{stem}_output.zip"
    if dest.resolve().as_posix().find("/artifacts/kaggle/") < 0:
        raise SystemExit("refusing to write kernel output outside artifacts/kaggle")
    path = download_kernel_output_zip(args.slug, dest)
    print(json.dumps({"dest": str(path), "bytes": path.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
