#!/usr/bin/env python3
"""Local simulation of Kaggle inference with outbound TCP blocked. Not a Kaggle submission."""

from __future__ import annotations

import argparse
import os
import runpy
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_OrigSocket = socket.socket


class _NetworkDisabled(_OrigSocket):
    def __init__(self, family=-1, type=-1, proto=-1, fileno=None):
        super().__init__(family, type, proto, fileno)

    def connect(self, address):
        raise RuntimeError("outbound network disabled for local Kaggle simulation")

    def connect_ex(self, address):
        raise RuntimeError("outbound network disabled for local Kaggle simulation")


def _prepare_sim_root() -> Path:
    sim = ROOT / "artifacts" / "tmp" / "kaggle_sim"
    test = sim / "test"
    test.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data" / "train" / "6bba_05db0fb1.zarr"
    if not source.exists():
        raise SystemExit(f"Need a complete local volume at {source} for the simulation")
    dest = test / "6bba_05db0fb1.zarr"
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    os.symlink(source.resolve(), dest)
    return sim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        default="notebooks/kaggle_inference.py",
        help="Kernel source to simulate. A local run is not a Kaggle execution.",
    )
    args = parser.parse_args()
    notebook = (ROOT / args.notebook).resolve()
    if not notebook.exists():
        raise SystemExit(f"notebook not found: {notebook}")
    sim_root = _prepare_sim_root()
    os.chdir(ROOT)
    os.environ["BIOHUB_DATA_ROOT"] = str(sim_root)
    sys.path.insert(0, str(ROOT / "notebooks"))
    socket.socket = _NetworkDisabled  # type: ignore[misc, assignment]
    print("LOCAL SIMULATION of Kaggle inference (outbound TCP blocked in-process).")
    print("This is not an official Kaggle execution and reports no leaderboard score.")
    print(f"sim_root={sim_root}")
    print(f"notebook={notebook}")
    runpy.run_path(str(notebook), run_name="__main__")


if __name__ == "__main__":
    main()
