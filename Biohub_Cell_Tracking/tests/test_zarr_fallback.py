from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

from kaggle_baseline_v1 import ZarrV3FrameArray, open_volume_array  # noqa: E402


def test_zarr_v3_fallback_matches_zarr_package():
    zarr_path = ROOT / "data" / "train" / "6bba_05db0fb1.zarr"
    if not (zarr_path / "0" / "c" / "0").exists():
        return
    import zarr

    official = np.asarray(zarr.open_group(zarr_path, mode="r")["0"][0])
    fallback = np.asarray(ZarrV3FrameArray(zarr_path / "0")[0])
    assert fallback.shape == official.shape
    assert fallback.dtype == official.dtype
    assert np.array_equal(fallback, official)
    via_open = np.asarray(open_volume_array(zarr_path)[0])
    assert np.array_equal(via_open, official)
