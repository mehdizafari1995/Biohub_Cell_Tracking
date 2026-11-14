#!/usr/bin/env python3
"""Resume competition file downloads with backoff/jitter. Never hammer after HTTP 429."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.kaggle_api import download_file, iter_competition_files  # noqa: E402

GEFF_FILES = [
    "zarr.json",
    "nodes/zarr.json",
    "nodes/props/zarr.json",
    "nodes/ids/zarr.json",
    "nodes/ids/c/0",
    "edges/zarr.json",
    "edges/ids/zarr.json",
    "edges/ids/c/0/0",
    "edges/props/zarr.json",
]
for _prop in ("t", "z", "y", "x"):
    GEFF_FILES.extend(
        [
            f"nodes/props/{_prop}/zarr.json",
            f"nodes/props/{_prop}/values/zarr.json",
            f"nodes/props/{_prop}/values/c/0",
        ]
    )


def download_geff(split: str, name: str, dest_root: Path) -> None:
    prefix = f"{split}/{name}.geff"
    for rel in GEFF_FILES:
        file_name = f"{prefix}/{rel}"
        ok = download_file(file_name, dest_root / file_name)
        print(("OK" if ok else "skip"), file_name, flush=True)


def download_zarr(split: str, name: str, dest_root: Path, timepoints: list[int]) -> None:
    prefix = f"{split}/{name}.zarr"
    for rel in ("zarr.json", "0/zarr.json"):
        ok = download_file(f"{prefix}/{rel}", dest_root / prefix / rel)
        print(("OK" if ok else "skip"), f"{prefix}/{rel}", flush=True)
    for t in timepoints:
        file_name = f"{prefix}/0/c/{t}/0/0/0"
        dest = dest_root / file_name
        if dest.exists() and dest.stat().st_size > 0:
            print("have", file_name, flush=True)
            continue
        ok = download_file(file_name, dest)
        print(("OK" if ok else "skip"), file_name, flush=True)


def resume_inventory(dest: Path, checkpoint: Path) -> None:
    datasets = {}
    extras = []
    n_files = 0
    total = 0
    if dest.exists():
        prev = json.loads(dest.read_text())
        n_files = int(prev.get("n_files") or 0)
        total = int(prev.get("total_bytes") or 0)
        extras = list(prev.get("extras") or [])
        for row in prev.get("datasets") or []:
            datasets[row["key"]] = row
    for item in iter_competition_files(checkpoint_path=checkpoint, page_sleep=1.2):
        name = item.get("name") or ""
        size = int(item.get("totalBytes") or 0)
        n_files += 1
        total += size
        parts = name.split("/")
        if len(parts) >= 2 and (parts[1].endswith(".zarr") or parts[1].endswith(".geff")):
            key = f"{parts[0]}/{parts[1]}"
            rec = datasets.setdefault(key, {"bytes": 0, "files": 0, "kind": None, "key": key})
            rec["bytes"] = int(rec.get("bytes") or 0) + size
            rec["files"] = int(rec.get("files") or 0) + 1
            rec["kind"] = "zarr" if parts[1].endswith(".zarr") else "geff"
            rec["split"] = parts[0]
            rec["dataset"] = parts[1].rsplit(".", 1)[0]
            rec["mb"] = round(rec["bytes"] / 1e6, 2)
        else:
            extras.append({"name": name, "bytes": size})
        if n_files % 50 == 0:
            dest.write_text(
                json.dumps(
                    {
                        "n_files": n_files,
                        "total_bytes": total,
                        "total_gb": round(total / 1e9, 3),
                        "n_dataset_stores": len(datasets),
                        "extras": extras,
                        "datasets": list(datasets.values()),
                        "partial": True,
                    },
                    indent=2,
                )
            )
            print(f"inventory checkpoint files={n_files} stores={len(datasets)}", flush=True)
    dest.write_text(
        json.dumps(
            {
                "n_files": n_files,
                "total_bytes": total,
                "total_gb": round(total / 1e9, 3),
                "n_dataset_stores": len(datasets),
                "extras": extras,
                "datasets": list(datasets.values()),
                "partial": False,
            },
            indent=2,
        )
    )
    print(f"inventory complete files={n_files} stores={len(datasets)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--split", default="train")
    parser.add_argument("--timepoints", default="0-99")
    parser.add_argument("--skip-inventory", action="store_true")
    args = parser.parse_args()
    times: list[int] = []
    for part in args.timepoints.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            times.extend(range(int(a), int(b) + 1))
        elif part:
            times.append(int(part))
    if not args.skip_inventory:
        resume_inventory(ROOT / "docs" / "dataset_inventory.json", ROOT / "artifacts" / "tmp" / "inventory_checkpoint.json")
        if args.inventory_only:
            return
    targets = args.dataset or ["44b6_2a2eff9f"]
    for name in targets:
        print(f"download {args.split}/{name}", flush=True)
        download_geff(args.split, name, args.out)
        download_zarr(args.split, name, args.out, times)


if __name__ == "__main__":
    main()
