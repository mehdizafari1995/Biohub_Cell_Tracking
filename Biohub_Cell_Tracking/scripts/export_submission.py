#!/usr/bin/env python3
"""Build a submission CSV from local predictions or from a test directory."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.config import load_config  # noqa: E402
from biohub_tracking.experiments import peak_memory_mb  # noqa: E402
from biohub_tracking.io_zarr import list_zarr_datasets, open_volume  # noqa: E402
from biohub_tracking.submit import graphs_to_submission, validate_submission, write_submission  # noqa: E402
from biohub_tracking.track import track_volume  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.yaml")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "data/test")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/submission.csv")
    args = parser.parse_args()
    config = load_config(args.config)
    zarrs = list_zarr_datasets(args.test_dir)
    if not zarrs:
        raise SystemExit(f"No test zarrs in {args.test_dir}")
    t0 = time.perf_counter()
    graphs = []
    for path in zarrs:
        array, meta = open_volume(path)
        graphs.append(track_volume(array, meta, config))
    df = graphs_to_submission(graphs)
    required = [p.name.removesuffix(".zarr") for p in zarrs]
    errors = validate_submission(df, required_datasets=required)
    write_submission(df, args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n_datasets": len(graphs),
                "n_rows": len(df),
                "schema_errors": errors,
                "runtime_sec": time.perf_counter() - t0,
                "peak_memory_mb": peak_memory_mb(),
            },
            indent=2,
        )
    )
    if errors:
        raise SystemExit("submission failed schema validation")


if __name__ == "__main__":
    main()
