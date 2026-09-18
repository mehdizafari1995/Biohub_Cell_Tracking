#!/usr/bin/env python3
"""List unique competition datasets and sizes before downloading volumes."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.kaggle_api import iter_competition_files  # noqa: E402


def main() -> None:
    datasets = defaultdict(lambda: {"bytes": 0, "files": 0, "kind": None})
    extras = []
    n_files = 0
    total = 0
    for item in iter_competition_files():
        name = item.get("name") or ""
        size = int(item.get("totalBytes") or 0)
        n_files += 1
        total += size
        if name == "sample_submission.csv":
            extras.append({"name": name, "bytes": size})
            continue
        parts = name.split("/")
        if len(parts) >= 2 and (parts[1].endswith(".zarr") or parts[1].endswith(".geff")):
            key = f"{parts[0]}/{parts[1]}"
            datasets[key]["bytes"] += size
            datasets[key]["files"] += 1
            datasets[key]["kind"] = "zarr" if parts[1].endswith(".zarr") else "geff"
            datasets[key]["split"] = parts[0]
            datasets[key]["dataset"] = parts[1].rsplit(".", 1)[0]
        else:
            extras.append({"name": name, "bytes": size})

    rows = []
    for key, rec in sorted(datasets.items()):
        rec["key"] = key
        rec["mb"] = round(rec["bytes"] / 1e6, 2)
        rows.append(rec)

    out = {
        "n_files": n_files,
        "total_bytes": total,
        "total_gb": round(total / 1e9, 3),
        "n_dataset_stores": len(rows),
        "extras": extras,
        "datasets": rows,
    }
    dest = ROOT / "docs" / "dataset_inventory.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"files={n_files} stores={len(rows)} total_gb={out['total_gb']}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
