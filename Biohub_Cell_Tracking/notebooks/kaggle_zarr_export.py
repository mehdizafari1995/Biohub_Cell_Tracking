#!/usr/bin/env python3
"""Private internet-off helper: zip only the two frozen-validation Zarrs.

Not a competition submission. Does not copy other volumes or GEFFs. Do not submit.
"""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

COMPETITION = "biohub-cell-tracking-during-development"
HOLDOUT_ZARRS = ("6bba_05db0fb1", "6bba_c328f2fd")
INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
ZIP_NAME = "holdout_zarrs_6bba.zip"
MANIFEST_NAME = "zarr_export_manifest.json"
# Already-compressed blosc chunks. STORE avoids a second CPU-heavy deflate.
ZIP_COMPRESSION = zipfile.ZIP_STORED
MAX_STORES = 2


def stem_of(path: Path) -> str:
    name = path.name
    if name.endswith(".zarr"):
        return name[: -len(".zarr")]
    return name


def wanted_zarr(path: Path) -> bool:
    if not path.name.endswith(".zarr"):
        return False
    return stem_of(path) in HOLDOUT_ZARRS


def iter_holdout_zarrs(root: Path) -> list[Path]:
    """Find only the two holdout Zarr directories. Never walk image chunks."""
    if not root.exists():
        return []
    patterns = (
        "*.zarr",
        "train/*.zarr",
        "*/train/*.zarr",
        "competitions/*/train/*.zarr",
        "competitions/*/*.zarr",
    )
    found: dict[str, Path] = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_dir() and wanted_zarr(path):
                found[stem_of(path)] = path
    return [found[key] for key in sorted(found)]


def zip_zarr_stores(stores: list[Path], dest: Path) -> dict:
    if len(stores) > MAX_STORES:
        raise ValueError(f"refusing to zip {len(stores)} stores; max is {MAX_STORES}")
    for store in stores:
        if not wanted_zarr(store):
            raise ValueError(f"refusing non-holdout store {store.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    n_files = 0
    n_bytes = 0
    with zipfile.ZipFile(dest, "w", compression=ZIP_COMPRESSION) as zf:
        for store in stores:
            for file_path in sorted(p for p in store.rglob("*") if p.is_file()):
                rel = file_path.relative_to(store)
                if any(part.endswith(".geff") for part in file_path.parts):
                    continue
                arc = str(Path("train") / store.name / rel)
                zf.write(file_path, arcname=arc)
                n_files += 1
                n_bytes += file_path.stat().st_size
    return {
        "n_stores": len(stores),
        "n_files": n_files,
        "member_bytes": n_bytes,
        "zip_bytes": dest.stat().st_size,
    }


def input_listing(root: Path, limit: int = 40) -> list[str]:
    if not root.exists():
        return []
    names = sorted(p.name for p in root.iterdir())
    return names[:limit]


def main() -> None:
    started = time.time()
    roots = [
        INPUT_ROOT / COMPETITION / "train",
        INPUT_ROOT / COMPETITION,
        INPUT_ROOT,
    ]
    stores: list[Path] = []
    used_root = None
    for root in roots:
        stores = iter_holdout_zarrs(root)
        if stores:
            used_root = str(root)
            break
    dest_zip = WORKING / ZIP_NAME
    manifest: dict = {
        "label": "zarr_export",
        "not_a_competition_submission": True,
        "used_root": used_root,
        "input_children": input_listing(INPUT_ROOT),
        "competition_children": input_listing(INPUT_ROOT / COMPETITION),
        "holdout_stems": list(HOLDOUT_ZARRS),
        "n_stores": 0,
        "stems": [],
        "error": None,
    }
    if not stores:
        manifest["error"] = "no_holdout_zarrs_mounted"
        (WORKING / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest))
        return
    extra = [stem_of(p) for p in stores if stem_of(p) not in HOLDOUT_ZARRS]
    if extra:
        manifest["error"] = "unexpected_store"
        manifest["stems"] = [stem_of(p) for p in stores]
        (WORKING / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest))
        return
    stats = zip_zarr_stores(stores, dest_zip)
    stems = [stem_of(p) for p in stores]
    manifest.update(
        {
            "n_stores": stats["n_stores"],
            "n_files": stats["n_files"],
            "member_bytes": stats["member_bytes"],
            "zip_bytes": stats["zip_bytes"],
            "stems": stems,
            "n_holdout": len(stems),
            "missing_holdout": [s for s in HOLDOUT_ZARRS if s not in stems],
            "elapsed_sec": round(time.time() - started, 2),
        }
    )
    (WORKING / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("n_stores", "n_holdout", "stems", "zip_bytes", "error")
            }
        )
    )


if __name__ == "__main__":
    main()
