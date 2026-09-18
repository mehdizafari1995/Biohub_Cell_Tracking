#!/usr/bin/env python3
"""Download GEFF graphs only (no image volumes) for inventory and split selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.kaggle_api import download_file  # noqa: E402


def _geff_files() -> list[str]:
    files = [
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
    for prop in ("t", "z", "y", "x"):
        files.extend(
            [
                f"nodes/props/{prop}/zarr.json",
                f"nodes/props/{prop}/values/zarr.json",
                f"nodes/props/{prop}/values/c/0",
            ]
        )
    return files


def download_geff(split: str, name: str, dest_root: Path) -> int:
    prefix = f"{split}/{name}.geff"
    ok_n = 0
    for rel in _geff_files():
        file_name = f"{prefix}/{rel}"
        dest = dest_root / file_name
        if dest.exists() and dest.stat().st_size > 0:
            ok_n += 1
            continue
        if download_file(file_name, dest):
            ok_n += 1
            print("OK", file_name, flush=True)
        else:
            print("skip", file_name, flush=True)
    return ok_n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()
    for name in args.datasets:
        n = download_geff(args.split, name, args.out)
        print(f"{name}: {n} files", flush=True)


if __name__ == "__main__":
    main()
