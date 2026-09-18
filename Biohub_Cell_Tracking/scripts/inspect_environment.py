#!/usr/bin/env python3
"""Record local hardware and Python package versions. Never prints secrets."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def main() -> None:
    mem = None
    try:
        import psutil as _ps

        mem = _ps.virtual_memory()
    except Exception:
        mem = None

    nvidia = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    disk = shutil.disk_usage(Path.cwd())
    packages = {}
    try:
        import importlib.metadata as md

        for name in [
            "numpy",
            "scipy",
            "scikit-image",
            "pandas",
            "matplotlib",
            "zarr",
            "numcodecs",
            "pyyaml",
            "pillow",
            "pytest",
        ]:
            try:
                packages[name] = md.version(name)
            except md.PackageNotFoundError:
                packages[name] = None
    except Exception:
        packages = {}

    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": round(mem.total / 1e9, 2) if mem else None,
        "ram_available_gb": round(mem.available / 1e9, 2) if mem else None,
        "disk_total_gb": round(disk.total / 1e9, 2),
        "disk_free_gb": round(disk.free / 1e9, 2),
        "gpu": nvidia or "none detected",
        "kaggle_token_present": bool(os.environ.get("KAGGLE_API_TOKEN") or (Path.home() / ".kaggle" / "kaggle.json").exists()),
        "packages": packages,
    }
    out = Path("docs/environment_snapshot.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Fallback RAM from /proc if psutil missing
    if report["ram_total_gb"] is None and Path("/proc/meminfo").exists():
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, val = line.split(":")
            info[key] = int(val.strip().split()[0])
        report["ram_total_gb"] = round(info.get("MemTotal", 0) / 1e6, 2)
        report["ram_available_gb"] = round(info.get("MemAvailable", 0) / 1e6, 2)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
