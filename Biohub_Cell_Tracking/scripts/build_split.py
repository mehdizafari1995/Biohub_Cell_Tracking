#!/usr/bin/env python3
"""Freeze a specimen-disjoint validation split from GEFF summaries. Run before tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.splits import build_fixed_split, save_split_manifest, summarise_geff_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geff-dir", type=Path, default=ROOT / "data/train")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/splits/v1.json")
    parser.add_argument("--selected", nargs="*", default=None)
    parser.add_argument("--tuning-embryos", nargs="*", default=None)
    parser.add_argument("--max-per-embryo", type=int, default=2)
    args = parser.parse_args()
    summaries = summarise_geff_dir(args.geff_dir)
    manifest = build_fixed_split(
        summaries,
        selected=args.selected,
        tuning_embryos=args.tuning_embryos,
        max_per_embryo=args.max_per_embryo,
    )
    save_split_manifest(manifest, args.out)
    print(json.dumps({k: manifest[k] for k in manifest if k != "geff_summaries"}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
