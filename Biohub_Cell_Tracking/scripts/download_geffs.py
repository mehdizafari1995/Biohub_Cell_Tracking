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
import zipfile
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

HOLDOUT_ZARRS = ("6bba_05db0fb1", "6bba_c328f2fd")
HOLDOUT_ZARR_INNER_ZIP = "holdout_zarrs_6bba.zip"


def extract_geff_export_zip(zip_path: Path, dest_root: Path) -> dict:
    """Extract a private kernel GEFF zip into ignored data/train. Never copies Zarrs."""
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    n_files = 0
    stores: set[str] = set()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("/") or ".zarr/" in name or name.endswith(".zarr"):
                continue
            if not name.startswith("train/") or ".geff/" not in name:
                continue
            target = dest_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))
            n_files += 1
            stores.add(name.split("/")[1])
    return {"n_files": n_files, "n_stores": len(stores), "stores": sorted(stores)}


def _zarr_member_store(name: str) -> str | None:
    """Return the holdout store directory name, or None if the member is refused."""
    if name.endswith("/") or ".geff/" in name or name.endswith(".geff"):
        return None
    if not name.startswith("train/") or ".zarr/" not in name:
        return None
    store = name.split("/", 1)[1].split("/", 1)[0]
    stem = store[: -len(".zarr")] if store.endswith(".zarr") else store
    if stem not in HOLDOUT_ZARRS:
        return None
    return store


def _extract_zarr_members(zf: zipfile.ZipFile, dest_root: Path) -> dict:
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    n_files = 0
    stores: set[str] = set()
    for name in zf.namelist():
        store = _zarr_member_store(name)
        if store is None:
            continue
        target = dest_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(zf.read(name))
        n_files += 1
        stores.add(store)
    return {"n_files": n_files, "n_stores": len(stores), "stores": sorted(stores)}


def extract_holdout_zarr_zip(zip_path: Path, dest_root: Path) -> dict:
    """Extract only the two frozen-validation Zarrs. Never copies other volumes or GEFFs.

    Accepts either the inner helper zip or a Kaggle worker archive that wraps it.
    """
    dest_root = Path(dest_root)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if HOLDOUT_ZARR_INNER_ZIP in names or any(n.endswith("/" + HOLDOUT_ZARR_INNER_ZIP) for n in names):
            inner_name = HOLDOUT_ZARR_INNER_ZIP if HOLDOUT_ZARR_INNER_ZIP in names else next(
                n for n in names if n.endswith("/" + HOLDOUT_ZARR_INNER_ZIP) or n.endswith(HOLDOUT_ZARR_INNER_ZIP)
            )
            inner_bytes = zf.read(inner_name)
            from io import BytesIO

            with zipfile.ZipFile(BytesIO(inner_bytes)) as inner:
                return _extract_zarr_members(inner, dest_root)
        return _extract_zarr_members(zf, dest_root)


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
    parser.add_argument(
        "--extract-zip",
        type=Path,
        default=None,
        help="Extract a private kernel GEFF zip into --out. Skips Zarr members.",
    )
    parser.add_argument(
        "--extract-zarr-zip",
        type=Path,
        default=None,
        help="Extract only the two frozen-validation Zarrs into --out. Skips other volumes and GEFFs.",
    )
    args = parser.parse_args()
    if args.extract_zip is not None:
        stats = extract_geff_export_zip(args.extract_zip, args.out)
        print(json.dumps(stats), flush=True)
        return
    if args.extract_zarr_zip is not None:
        stats = extract_holdout_zarr_zip(args.extract_zarr_zip, args.out)
        print(json.dumps(stats), flush=True)
        return
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
