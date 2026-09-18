#!/usr/bin/env python3
"""Inspect a small downloaded sample without loading a full 4D volume."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.io_geff import load_geff  # noqa: E402
from biohub_tracking.io_zarr import open_volume  # noqa: E402
from biohub_tracking.visualize import save_detection_overlay, save_track_plot, save_track_time, save_xy_slice  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", type=Path, default=ROOT / "data/train/44b6_0db75fae.zarr")
    parser.add_argument("--geff", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/figures")
    args = parser.parse_args()
    array, meta = open_volume(args.zarr)
    geff_path = args.geff or args.zarr.with_suffix(".geff")
    gt = load_geff(geff_path) if geff_path.exists() else None
    t0 = meta.available_timepoints[0]
    frame = array[t0]
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": meta.name,
        "zarr": str(args.zarr),
        "shape_TZYX": list(meta.shape),
        "dtype": meta.dtype,
        "scale_TZYX": list(meta.scale_tzyx),
        "time_unit": meta.time_unit,
        "space_unit": meta.space_unit,
        "available_timepoints": meta.available_timepoints,
        "loaded_timepoint": int(t0),
        "frame_min": int(frame.min()),
        "frame_max": int(frame.max()),
        "frame_mean": float(frame.mean()),
        "quantiles": meta.quantiles,
    }
    if gt is not None:
        report.update(
            {
                "gt_nodes": gt.n_nodes,
                "gt_edges": gt.n_edges,
                "gt_t_min": int(gt.t.min()) if gt.n_nodes else None,
                "gt_t_max": int(gt.t.max()) if gt.n_nodes else None,
                "gt_z_range": [int(gt.z.min()), int(gt.z.max())] if gt.n_nodes else None,
                "gt_y_range": [int(gt.y.min()), int(gt.y.max())] if gt.n_nodes else None,
                "gt_x_range": [int(gt.x.min()), int(gt.x.max())] if gt.n_nodes else None,
                "estimated_number_of_nodes": gt.estimated_n_nodes,
                "gt_divisions": int(sum(len(v) >= 2 for v in gt.out_edges().values())),
            }
        )
        save_detection_overlay(
            frame,
            args.out / f"{meta.name}_t{t0}_mip_gt.png",
            gt=gt,
            timepoint=t0,
            title=f"{meta.name} t={t0} MIP + ground-truth centroids",
        )
        z_focus = int(gt.z[gt.t == t0][0]) if ((gt.t == t0).any()) else frame.shape[0] // 2
        save_xy_slice(
            frame,
            args.out / f"{meta.name}_t{t0}_z{z_focus}_gt.png",
            z_index=z_focus,
            gt=gt,
            timepoint=t0,
            title=f"{meta.name} t={t0} z={z_focus} + GT",
        )
        save_track_plot(gt, args.out / f"{meta.name}_gt_tracks_xy.png", title=f"{meta.name} GT tracks")
        save_track_time(gt, args.out / f"{meta.name}_gt_tracks_t.png", title=f"{meta.name} GT x vs t")
    dest = ROOT / "artifacts" / f"{meta.name}_inspect.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"figures in {args.out}")


if __name__ == "__main__":
    main()
