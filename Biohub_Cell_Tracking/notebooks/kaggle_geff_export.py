#!/usr/bin/env python3
"""Private internet-off helper: zip train GEFFs for division_v2 priors.

Not a competition submission. Does not open image volumes. Do not submit.
"""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

COMPETITION = "biohub-cell-tracking-during-development"
HOLDOUT_GEFFS = ("6bba_05db0fb1", "6bba_c328f2fd")
TUNING_PREFIX = "44b6_"
INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
ZIP_NAME = "train_geffs_44b6.zip"
MANIFEST_NAME = "geff_export_manifest.json"


def stem_of(path: Path) -> str:
    name = path.name
    if name.endswith(".geff"):
        return name[: -len(".geff")]
    return name


def wanted_geff(path: Path) -> bool:
    if not path.name.endswith(".geff"):
        return False
    stem = stem_of(path)
    return stem.startswith(TUNING_PREFIX) or stem in HOLDOUT_GEFFS


def iter_geff_stores(root: Path) -> list[Path]:
    """Find wanted GEFF directories without walking image-volume chunks."""
    if not root.exists():
        return []
    patterns = (
        "*.geff",
        "train/*.geff",
        "*/train/*.geff",
        "competitions/*/train/*.geff",
        "competitions/*/*.geff",
    )
    found: dict[str, Path] = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_dir() and path.name.endswith(".geff") and wanted_geff(path):
                found[stem_of(path)] = path
    return [found[key] for key in sorted(found)]


def zip_geff_stores(stores: list[Path], dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    n_files = 0
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for store in stores:
            for file_path in sorted(p for p in store.rglob("*") if p.is_file()):
                arc = str(Path("train") / store.name / file_path.relative_to(store))
                zf.write(file_path, arcname=arc)
                n_files += 1
    return {"n_stores": len(stores), "n_files": n_files, "zip_bytes": dest.stat().st_size}


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
        stores = iter_geff_stores(root)
        if stores:
            used_root = str(root)
            break
    dest_zip = WORKING / ZIP_NAME
    manifest: dict = {
        "label": "geff_export",
        "not_a_competition_submission": True,
        "used_root": used_root,
        "input_children": input_listing(INPUT_ROOT),
        "competition_children": input_listing(INPUT_ROOT / COMPETITION),
        "n_stores": 0,
        "stems": [],
        "n_holdout": 0,
        "error": None,
    }
    if not stores:
        manifest["error"] = "no_train_geffs_mounted"
        (WORKING / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest))
        return
    stats = zip_geff_stores(stores, dest_zip)
    stems = [stem_of(p) for p in stores]
    manifest.update(
        {
            "n_stores": stats["n_stores"],
            "n_files": stats["n_files"],
            "zip_bytes": stats["zip_bytes"],
            "stems": stems,
            "n_tuning_44b6": sum(1 for s in stems if s.startswith(TUNING_PREFIX)),
            "n_holdout": sum(1 for s in stems if s in HOLDOUT_GEFFS),
            "elapsed_sec": round(time.time() - started, 2),
        }
    )
    (WORKING / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("n_stores", "n_tuning_44b6", "n_holdout", "zip_bytes", "error")}))


if __name__ == "__main__":
    main()
