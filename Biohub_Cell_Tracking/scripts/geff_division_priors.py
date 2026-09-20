#!/usr/bin/env python3
"""GEFF-only division priors. Does not open validation images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.geff_priors import (  # noqa: E402
    apply_operating_point_to_yaml,
    write_priors,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--dest", type=Path, default=ROOT / "docs" / "division_priors.json")
    parser.add_argument(
        "--apply-yaml",
        action="store_true",
        help="Patch configs/division_v2.yaml when typical events require a wider parent–child gate.",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=ROOT / "configs" / "division_v2.yaml",
    )
    args = parser.parse_args()
    payload = write_priors(args.data_dir, args.dest)
    lock = payload["locked_operating_point"]
    applied = False
    if args.apply_yaml and lock.get("action") == "update_typical":
        apply_operating_point_to_yaml(args.yaml, lock)
        applied = True
    print(
        json.dumps(
            {
                "dest": str(args.dest),
                "n_tuning_geffs_scanned": payload["n_tuning_geffs_scanned"],
                "n_typical_events": payload.get("n_typical_events"),
                "n_atypical_events": payload.get("n_atypical_events"),
                "typical_event_datasets": payload.get("typical_event_datasets"),
                "atypical_event_datasets": payload.get("atypical_event_datasets"),
                "typical_tuning_summary": payload.get("typical_tuning_summary"),
                "locked_operating_point": lock,
                "unfiltered_lock_rejected": {
                    "max_parent_child_um": payload["unfiltered_lock_rejected"]["max_parent_child_um"],
                    "min_child_separation_um": payload["unfiltered_lock_rejected"]["min_child_separation_um"],
                    "do_not_use": True,
                },
                "yaml_applied": applied,
                "validation_excluded": payload["validation_specimen_excluded"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
