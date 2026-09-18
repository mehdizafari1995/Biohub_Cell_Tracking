from __future__ import annotations

import numpy as np
from skimage.feature import blob_dog, peak_local_max

from .config import DetectionConfig


def normalize_volume(image: np.ndarray, method: str = "minmax") -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if method == "none":
        return image
    if method == "minmax":
        lo = float(image.min())
        hi = float(image.max())
        if hi <= lo:
            return np.zeros_like(image, dtype=np.float32)
        return (image - lo) / (hi - lo)
    if method == "quantile":
        lo, hi = np.quantile(image.ravel()[::50], [0.001, 0.999])
        return np.clip((image - lo) / (hi - lo + 1e-6), 0.0, None)
    raise ValueError(f"Unknown normalize method: {method}")


def anisotropic_sigmas(
    min_sigma: float,
    max_sigma: float,
    scale_zyx: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Convert xy-voxel sigmas into physically isotropic (z, y, x) voxel sigmas."""
    xy = float(scale_zyx[2])
    mins = tuple(max(float(min_sigma) * xy / float(s), 0.25) for s in scale_zyx)
    maxs = tuple(max(float(max_sigma) * xy / float(s), mins[i]) for i, s in enumerate(scale_zyx))
    return mins, maxs  # type: ignore[return-value]


def suppress_close_centroids(
    coords_zyx: np.ndarray,
    min_distance_um: float,
    scale_zyx: tuple[float, float, float],
) -> np.ndarray:
    """Greedy physical-space NMS. Keeps earlier detections when two are too close."""
    coords = np.asarray(coords_zyx, dtype=np.float64).reshape(-1, 3)
    if len(coords) == 0 or min_distance_um <= 0:
        return coords
    phys = coords * np.asarray(scale_zyx, dtype=np.float64)
    keep: list[int] = []
    for i, point in enumerate(phys):
        if keep:
            dists = np.linalg.norm(phys[keep] - point, axis=1)
            if np.any(dists < min_distance_um):
                continue
        keep.append(i)
    return coords[np.asarray(keep, dtype=np.int64)]


def detect_cells(image_zyx: np.ndarray, config: DetectionConfig, scale_zyx: tuple[float, float, float]) -> np.ndarray:
    """Return (N, 3) array of (z, y, x) centroids in voxels."""
    norm = normalize_volume(image_zyx, config.normalize)
    method = config.method.lower()
    if method == "blob_dog":
        min_sigma: float | tuple[float, float, float] = config.min_sigma
        max_sigma: float | tuple[float, float, float] = config.max_sigma
        if config.anisotropic:
            min_sigma, max_sigma = anisotropic_sigmas(config.min_sigma, config.max_sigma, scale_zyx)
        blobs = blob_dog(
            norm,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold=config.threshold,
        )
        coords = blobs[:, :3] if len(blobs) else np.zeros((0, 3), dtype=np.float64)
        coords = np.asarray(coords, dtype=np.float64)
        if config.suppress_duplicates:
            coords = suppress_close_centroids(coords, config.min_distance_um, scale_zyx)
        return coords
    if method == "local_max":
        sigma_zyx = np.array(
            [
                max(config.min_sigma * scale_zyx[2] / scale_zyx[0], 0.5),
                config.min_sigma,
                config.min_sigma,
            ],
            dtype=np.float64,
        )
        from scipy.ndimage import gaussian_filter

        smooth = gaussian_filter(norm, sigma=sigma_zyx)
        min_dist = max(int(round(config.min_distance_um / min(scale_zyx))), 1)
        peaks = peak_local_max(
            smooth,
            min_distance=min_dist,
            threshold_abs=config.threshold,
            exclude_border=False,
        )
        coords = np.asarray(peaks, dtype=np.float64).reshape(-1, 3)
        if config.suppress_duplicates:
            coords = suppress_close_centroids(coords, config.min_distance_um, scale_zyx)
        return coords
    raise ValueError(f"Unknown detection method: {config.method}")
