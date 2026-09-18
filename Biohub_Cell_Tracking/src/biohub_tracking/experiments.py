from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REGISTRY_COLUMNS = [
    "experiment_id",
    "timestamp_utc",
    "code_version",
    "config_name",
    "seed",
    "split",
    "n_samples",
    "score",
    "edge_jaccard",
    "adj_edge_jaccard",
    "division_jaccard",
    "node_recall",
    "missed_detections",
    "identity_switches",
    "broken_tracks",
    "division_tp",
    "division_fp",
    "division_fn",
    "runtime_sec",
    "peak_memory_mb",
    "notes",
]


def git_version(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def append_experiment(registry_path: str | Path, row: dict[str, Any]) -> None:
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    exists = registry_path.exists()
    with registry_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REGISTRY_COLUMNS)
        if not exists:
            writer.writeheader()
        out = {k: row.get(k, "") for k in REGISTRY_COLUMNS}
        writer.writerow(out)


def new_experiment_id(name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{name}"


def write_run_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def peak_memory_mb() -> float:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB; macOS reports bytes.
        if usage > 10_000_000:
            return usage / (1024 * 1024)
        return usage / 1024
    except Exception:
        return float("nan")
