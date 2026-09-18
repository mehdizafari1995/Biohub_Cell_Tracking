#!/usr/bin/env python3
"""Local simulation of Kaggle inference with network blocked. Not a Kaggle submission."""

from __future__ import annotations

import os
import runpy
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_inference.py"


class _NetworkDisabled(socket.socket):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("network disabled for local Kaggle simulation")


def main() -> None:
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "notebooks"))
    socket.socket = _NetworkDisabled  # type: ignore[misc, assignment]
    print("LOCAL SIMULATION of Kaggle inference (internet blocked in-process).")
    print("This is not an official Kaggle execution and reports no leaderboard score.")
    runpy.run_path(str(NOTEBOOK), run_name="__main__")


if __name__ == "__main__":
    main()
