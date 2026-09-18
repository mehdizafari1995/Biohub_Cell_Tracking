#!/usr/bin/env python3
"""Download a small official sample: GEFF + selected timepoints, never credentials."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.kaggle_api import download_file  # noqa: E402

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
        print(("OK" if ok else "skip"), file_name)


def download_zarr(split: str, name: str, dest_root: Path, timepoints: list[int]) -> None:
    prefix = f"{split}/{name}.zarr"
    for rel in ("zarr.json", "0/zarr.json"):
        file_name = f"{prefix}/{rel}"
        ok = download_file(file_name, dest_root / file_name)
        print(("OK" if ok else "skip"), file_name)
    for t in timepoints:
        file_name = f"{prefix}/0/c/{t}/0/0/0"
        ok = download_file(file_name, dest_root / file_name)
        print(("OK" if ok else "skip"), file_name, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--dataset", default="44b6_0db75fae")
    parser.add_argument("--split", default="train")
    parser.add_argument("--timepoints", default="0-9,50,99")
    parser.add_argument("--with-sample-submission", action="store_true", default=True)
    args = parser.parse_args()
    times: list[int] = []
    for part in args.timepoints.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            times.extend(range(int(a), int(b) + 1))
        elif part:
            times.append(int(part))
    if args.with_sample_submission:
        ok = download_file("sample_submission.csv", args.out / "sample_submission.csv")
        print(("OK" if ok else "skip"), "sample_submission.csv")
    download_geff(args.split, args.dataset, args.out)
    download_zarr(args.split, args.dataset, args.out, times)


if __name__ == "__main__":
    main()
