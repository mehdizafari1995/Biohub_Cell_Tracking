#!/usr/bin/env python3
"""GEFF-only division priors. Does not open validation images."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.geff_priors import write_priors  # noqa: E402


def main() -> None:
    dest = ROOT / "docs" / "division_priors.json"
    payload = write_priors(ROOT / "data" / "train", dest)
    print(json.dumps({"dest": str(dest), "all_five": payload["all_five"], "tuning_three": payload["tuning_three"]}, indent=2))


if __name__ == "__main__":
    main()
