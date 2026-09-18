#!/usr/bin/env python3
"""Inventory competition files, preferring GEFF graphs before downloading images."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.kaggle_api import iter_competition_files  # noqa: E402
from biohub_tracking.io_zarr import embryo_id  # noqa: E402


def main() -> None:
    dest = ROOT / "docs" / "dataset_inventory.json"
    ckpt = ROOT / "artifacts" / "tmp" / "inventory_checkpoint.json"
    datasets = defaultdict(lambda: {"bytes": 0, "files": 0, "kind": None})
    extras = []
    n_files = 0
    total = 0
    if dest.exists():
        existing = json.loads(dest.read_text())
        if existing.get("complete"):
            print(f"inventory already complete: {dest}")
            return
        for rec in existing.get("datasets") or []:
            datasets[rec["key"]].update(rec)
        extras = existing.get("extras") or []
        n_files = int(existing.get("n_files") or 0)
        total = int(existing.get("total_bytes") or 0)

    try:
        for item in iter_competition_files(checkpoint_path=ckpt, page_sleep=0.45):
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
                datasets[key]["embryo"] = embryo_id(datasets[key]["dataset"])
            else:
                extras.append({"name": name, "bytes": size})
            if n_files % 200 == 0:
                _write(dest, datasets, extras, n_files, total, complete=False)
                print(f"progress files={n_files} stores={len(datasets)}", flush=True)
    except Exception as exc:
        _write(dest, datasets, extras, n_files, total, complete=False)
        print(f"STOPPED after {n_files} files: {type(exc).__name__}: {exc}")
        raise

    _write(dest, datasets, extras, n_files, total, complete=True)
    print(f"files={n_files} stores={len(datasets)} total_gb={total / 1e9:.3f}")
    print(f"wrote {dest}")


def _write(dest: Path, datasets, extras, n_files, total, complete: bool) -> None:
    rows = []
    for key, rec in sorted(datasets.items()):
        rec = dict(rec)
        rec["key"] = key
        rec["mb"] = round(rec["bytes"] / 1e6, 2)
        rows.append(rec)
    embryos = sorted({r.get("embryo") for r in rows if r.get("split") == "train" and r.get("kind") == "geff"})
    train_geffs = [r for r in rows if r.get("split") == "train" and r.get("kind") == "geff"]
    train_zarrs = [r for r in rows if r.get("split") == "train" and r.get("kind") == "zarr"]
    test_zarrs = [r for r in rows if r.get("split") == "test" and r.get("kind") == "zarr"]
    out = {
        "n_files": n_files,
        "total_bytes": total,
        "total_gb": round(total / 1e9, 3),
        "n_dataset_stores": len(rows),
        "complete": complete,
        "train_geff_count": len(train_geffs),
        "train_zarr_count": len(train_zarrs),
        "test_zarr_count": len(test_zarrs),
        "train_embryos": embryos,
        "train_geff_bytes": sum(r["bytes"] for r in train_geffs),
        "train_zarr_bytes": sum(r["bytes"] for r in train_zarrs),
        "extras": extras,
        "datasets": rows,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
