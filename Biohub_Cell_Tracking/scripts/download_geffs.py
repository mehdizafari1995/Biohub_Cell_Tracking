#!/usr/bin/env python3
"""Download GEFF graphs only (no image volumes) for inventory and split selection.

Use ``--required-only`` to skip optional group metadata. Previous full-file
crawls at 0.35–1.8 s spacing hit HTTP 429; default interval is 4.0 s with a
process-wide cool-down after 429. Never run two downloaders in parallel.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.geff_priors import (  # noqa: E402
    REQUIRED_GEFF_FILES,
    VALIDATION_SPECIMEN,
    all_geff_files,
    geff_store_complete,
    inventory_geff_names,
)
from biohub_tracking.kaggle_api import download_file  # noqa: E402


def geff_files(required_only: bool) -> tuple[str, ...]:
    return REQUIRED_GEFF_FILES if required_only else all_geff_files()


def download_geff(
    split: str,
    name: str,
    dest_root: Path,
    required_only: bool = True,
    interval: float = 1.8,
) -> dict:
    prefix = f"{split}/{name}.geff"
    dest_dir = dest_root / prefix
    stats = {"name": name, "ok": 0, "skip": 0, "have": 0, "fail": 0, "complete": False}
    for rel in geff_files(required_only):
        file_name = f"{prefix}/{rel}"
        dest = dest_root / file_name
        if dest.exists() and dest.stat().st_size > 0:
            stats["have"] += 1
            stats["ok"] += 1
            continue
        try:
            if download_file(file_name, dest, min_interval_sec=interval):
                stats["ok"] += 1
                print("OK", file_name, flush=True)
            else:
                stats["skip"] += 1
                print("skip", file_name, flush=True)
        except Exception as exc:  # noqa: BLE001 — keep the crawl resumable
            stats["fail"] += 1
            print("fail", file_name, type(exc).__name__, flush=True)
    stats["complete"] = geff_store_complete(dest_dir)
    return stats


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--specimen", default=None, help="Select GEFF stems from the inventory.")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=ROOT / "docs" / "dataset_inventory.json",
    )
    parser.add_argument("--required-only", action="store_true", default=True)
    parser.add_argument("--full-metadata", action="store_true", help="Download all 21 GEFF files.")
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--interval", type=float, default=4.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--also",
        nargs="+",
        default=[],
        help="Extra dataset stems (e.g. held-out 6bba GEFFs, never used to lock).",
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=ROOT / "artifacts" / "tmp" / "geff_download_status.json",
    )
    args = parser.parse_args()
    required_only = not args.full_metadata
    names: list[str] = []
    if args.datasets:
        names.extend(args.datasets)
    elif args.specimen:
        names.extend(inventory_geff_names(args.inventory, specimen=args.specimen, split=args.split))
    else:
        parser.error("pass --datasets or --specimen")
    for extra in args.also:
        if extra not in names:
            names.append(extra)
    if args.limit:
        names = names[: args.limit]
    started = time.time()
    complete_n = 0
    rows = []
    for i, name in enumerate(names, start=1):
        dest_dir = args.out / args.split / f"{name}.geff"
        if args.skip_complete and geff_store_complete(dest_dir):
            complete_n += 1
            print(f"HAVE {complete_n}/{len(names)} {name}", flush=True)
            rows.append({"name": name, "complete": True, "skipped": True})
            continue
        stats = download_geff(args.split, name, args.out, required_only=required_only, interval=args.interval)
        rows.append(stats)
        if stats["complete"]:
            complete_n += 1
        print(
            f"{'COMPLETE' if stats['complete'] else 'PARTIAL'} {complete_n}/{len(names)} {name} "
            f"ok={stats['ok']} have={stats['have']} skip={stats['skip']} fail={stats['fail']}",
            flush=True,
        )
        _write_status(
            args.status,
            {
                "started_unix": started,
                "elapsed_sec": round(time.time() - started, 1),
                "required_only": required_only,
                "interval": args.interval,
                "n_targets": len(names),
                "index": i,
                "complete": complete_n,
                "current": name,
                "rows": rows,
            },
        )
    n_val = sum(1 for name in names if name.startswith(f"{VALIDATION_SPECIMEN}_"))
    print(
        json.dumps(
            {
                "n_targets": len(names),
                "n_complete": complete_n,
                "n_validation_geffs": n_val,
                "elapsed_sec": round(time.time() - started, 1),
                "required_only": required_only,
                "status": str(args.status),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
