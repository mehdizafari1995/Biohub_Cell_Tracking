from __future__ import annotations

import numpy as np

from biohub_tracking.config import DetectionConfig
from biohub_tracking.detect import (
    anisotropic_sigmas,
    complementary_detections,
    detect_cells,
    nearest_physical_distance,
    select_complementary_coords,
    suppress_close_centroids,
)


SCALE = (1.625, 0.40625, 0.40625)


def test_nms_drops_physical_duplicate():
    coords = np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 12.0]], dtype=float)
    kept = suppress_close_centroids(coords, min_distance_um=4.0, scale_zyx=SCALE)
    assert len(kept) == 1


def test_nms_keeps_far_detections():
    coords = np.array([[10.0, 10.0, 10.0], [10.0, 40.0, 10.0]], dtype=float)
    kept = suppress_close_centroids(coords, min_distance_um=4.0, scale_zyx=SCALE)
    assert len(kept) == 2


def test_anisotropic_sigma_shrinks_z_relative_to_xy():
    mins, maxs = anisotropic_sigmas(2.0, 5.0, SCALE)
    assert mins[0] < mins[1]
    assert mins[1] == mins[2]
    assert maxs[0] < maxs[1]


def test_nearest_physical_distance_is_anisotropic():
    points = np.array([[10.0, 10.0, 10.0]], dtype=float)
    # 8 voxels in z = 13.0 µm; 8 voxels in x = 3.25 µm.
    far_z = np.array([[18.0, 10.0, 10.0]], dtype=float)
    far_x = np.array([[10.0, 10.0, 18.0]], dtype=float)
    dz = nearest_physical_distance(far_z, points, SCALE)
    dx = nearest_physical_distance(far_x, points, SCALE)
    assert dz[0] == 8.0 * 1.625
    assert dx[0] == 8.0 * 0.40625


def test_select_complementary_keeps_isolated_peak():
    primary = np.array([[10.0, 10.0, 10.0]], dtype=float)
    # 20 xy voxels * 0.40625 µm = 8.125 µm, outside the 6.0 µm gate.
    isolated = np.array([[10.0, 10.0, 30.0]], dtype=float)
    kept = select_complementary_coords(primary, isolated, SCALE, min_distance_um=6.0)
    assert len(kept) == 1


def test_select_complementary_drops_near_primary_and_dim_peak():
    primary = np.array([[10.0, 10.0, 10.0]], dtype=float)
    near = np.array([[10.0, 10.0, 16.0]], dtype=float)  # 2.44 µm
    isolated = np.array([[10.0, 10.0, 40.0]], dtype=float)
    candidates = np.vstack([near, isolated])
    kept = select_complementary_coords(
        primary,
        candidates,
        SCALE,
        min_distance_um=6.0,
        candidate_intensity=np.array([80.0, 10.0]),
        primary_intensity=np.array([100.0]),
        intensity_ratio=0.45,
    )
    assert len(kept) == 0
    bright = select_complementary_coords(
        primary,
        isolated,
        SCALE,
        min_distance_um=6.0,
        candidate_intensity=np.array([50.0]),
        primary_intensity=np.array([100.0]),
        intensity_ratio=0.45,
    )
    assert len(bright) == 1


def _gaussian_blob(shape, center, sigma_zyx, amplitude):
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    z0, y0, x0 = center
    sz, sy, sx = sigma_zyx
    return amplitude * np.exp(
        -0.5 * (((zz - z0) / sz) ** 2 + ((yy - y0) / sy) ** 2 + ((xx - x0) / sx) ** 2)
    )


def test_complementary_local_max_recovers_isolated_dimmer_blob():
    image = np.zeros((24, 64, 64), dtype=np.float32)
    image += _gaussian_blob((24, 64, 64), (12, 16, 16), (1.2, 2.5, 2.5), 1.0)
    image += _gaussian_blob((24, 64, 64), (12, 16, 48), (1.2, 2.5, 2.5), 0.55)
    primary_cfg = DetectionConfig(
        method="blob_dog",
        min_sigma=2.0,
        max_sigma=5.0,
        threshold=0.12,
        normalize="minmax",
        min_distance_um=4.0,
        suppress_duplicates=True,
        anisotropic=True,
        complementary_method="none",
    )
    primary = detect_cells(image, primary_cfg, SCALE)
    assert len(primary) >= 1
    extra = complementary_detections(
        image,
        primary,
        DetectionConfig(
            complementary_method="local_max",
            complementary_threshold=0.15,
            complementary_min_sigma=2.0,
            complementary_min_distance_um=6.0,
            complementary_intensity_ratio=0.35,
            complementary_anisotropic=True,
            min_distance_um=4.0,
        ),
        SCALE,
    )
    merged = detect_cells(
        image,
        DetectionConfig(
            method="blob_dog",
            min_sigma=2.0,
            max_sigma=5.0,
            threshold=0.12,
            normalize="minmax",
            min_distance_um=4.0,
            suppress_duplicates=True,
            anisotropic=True,
            complementary_method="local_max",
            complementary_threshold=0.15,
            complementary_min_sigma=2.0,
            complementary_min_distance_um=6.0,
            complementary_intensity_ratio=0.35,
            complementary_anisotropic=True,
        ),
        SCALE,
    )
    assert len(merged) >= len(primary)
    if len(extra):
        dists = nearest_physical_distance(extra, primary, SCALE)
        assert np.all(dists >= 6.0)


def test_complementary_off_matches_primary_only():
    image = np.zeros((16, 32, 32), dtype=np.float32)
    image[8, 16, 16] = 1.0
    cfg = DetectionConfig(
        method="blob_dog",
        min_sigma=1.5,
        max_sigma=4.0,
        threshold=0.05,
        normalize="minmax",
        min_distance_um=4.0,
        suppress_duplicates=True,
        anisotropic=True,
        complementary_method="none",
    )
    a = detect_cells(image, cfg, SCALE)
    b = detect_cells(image, DetectionConfig(**{**cfg.__dict__, "complementary_method": "none"}), SCALE)
    assert a.shape == b.shape
