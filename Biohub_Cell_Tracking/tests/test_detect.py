from __future__ import annotations

import numpy as np

from biohub_tracking.config import DetectionConfig
from biohub_tracking.detect import anisotropic_sigmas, suppress_close_centroids


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
