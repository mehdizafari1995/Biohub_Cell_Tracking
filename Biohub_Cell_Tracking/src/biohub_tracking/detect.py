from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from skimage.feature import blob_dog, peak_local_max

from .config import DetectionConfig


def _empty_coords() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float64)


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


def nearest_physical_distance(
    points_zyx: np.ndarray,
    references_zyx: np.ndarray,
    scale_zyx: tuple[float, float, float],
) -> np.ndarray:
    """Minimum physical distance (µm) from each point to any reference."""
    points = np.asarray(points_zyx, dtype=np.float64).reshape(-1, 3)
    refs = np.asarray(references_zyx, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(refs) == 0:
        return np.full((len(points),), np.inf, dtype=np.float64)
    scale = np.asarray(scale_zyx, dtype=np.float64)
    delta = points[:, None, :] * scale - refs[None, :, :] * scale
    return np.linalg.norm(delta, axis=2).min(axis=1)


def select_complementary_coords(
    primary_zyx: np.ndarray,
    candidates_zyx: np.ndarray,
    scale_zyx: tuple[float, float, float],
    min_distance_um: float,
    candidate_intensity: np.ndarray | None = None,
    primary_intensity: np.ndarray | None = None,
    intensity_ratio: float = 0.0,
    self_nms_um: float = 0.0,
) -> np.ndarray:
    """Keep complementary peaks that are isolated from the primary set.

    A candidate is dropped when it sits inside ``min_distance_um`` of any
    primary detection, or when its intensity is below
    ``intensity_ratio * median(primary_intensity)``. Remaining peaks may be
    self-NMS'd. This does not replace the primary detector.
    """
    candidates = np.asarray(candidates_zyx, dtype=np.float64).reshape(-1, 3)
    primary = np.asarray(primary_zyx, dtype=np.float64).reshape(-1, 3)
    if len(candidates) == 0:
        return candidates
    keep = nearest_physical_distance(candidates, primary, scale_zyx) >= float(min_distance_um)
    if (
        float(intensity_ratio) > 0
        and candidate_intensity is not None
        and primary_intensity is not None
        and len(np.asarray(primary_intensity)) > 0
    ):
        floor = float(intensity_ratio) * float(np.median(np.asarray(primary_intensity, dtype=np.float64)))
        inten = np.asarray(candidate_intensity, dtype=np.float64).reshape(-1)
        if len(inten) == len(candidates):
            keep &= inten >= floor
    selected = candidates[keep]
    if float(self_nms_um) > 0:
        selected = suppress_close_centroids(selected, self_nms_um, scale_zyx)
    return selected


def _dog_coords(
    norm: np.ndarray,
    min_sigma: float,
    max_sigma: float,
    threshold: float,
    anisotropic: bool,
    scale_zyx: tuple[float, float, float],
) -> np.ndarray:
    sigma_min: float | tuple[float, float, float] = min_sigma
    sigma_max: float | tuple[float, float, float] = max_sigma
    if anisotropic:
        sigma_min, sigma_max = anisotropic_sigmas(min_sigma, max_sigma, scale_zyx)
    blobs = blob_dog(norm, min_sigma=sigma_min, max_sigma=sigma_max, threshold=threshold)
    if len(blobs) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(blobs[:, :3], dtype=np.float64)


def _local_max_coords(
    norm: np.ndarray,
    min_sigma: float,
    threshold: float,
    min_distance_um: float,
    scale_zyx: tuple[float, float, float],
    anisotropic: bool,
) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    if anisotropic:
        sigma_zyx = np.array(
            [
                max(min_sigma * scale_zyx[2] / scale_zyx[0], 0.5),
                min_sigma,
                min_sigma,
            ],
            dtype=np.float64,
        )
    else:
        sigma_zyx = np.full(3, max(min_sigma, 0.5), dtype=np.float64)
    smooth = gaussian_filter(norm, sigma=sigma_zyx)
    min_dist = max(int(round(min_distance_um / min(scale_zyx))), 1)
    peaks = peak_local_max(
        smooth,
        min_distance=min_dist,
        threshold_abs=threshold,
        exclude_border=False,
    )
    return np.asarray(peaks, dtype=np.float64).reshape(-1, 3)


def _run_detector(
    norm: np.ndarray,
    method: str,
    min_sigma: float,
    max_sigma: float,
    threshold: float,
    min_distance_um: float,
    anisotropic: bool,
    scale_zyx: tuple[float, float, float],
    suppress: bool,
) -> np.ndarray:
    name = method.lower()
    if name == "none" or name == "":
        return np.zeros((0, 3), dtype=np.float64)
    if name == "blob_dog":
        coords = _dog_coords(norm, min_sigma, max_sigma, threshold, anisotropic, scale_zyx)
    elif name == "local_max":
        coords = _local_max_coords(norm, min_sigma, threshold, min_distance_um, scale_zyx, anisotropic)
    else:
        raise ValueError(f"Unknown detection method: {method}")
    if suppress:
        coords = suppress_close_centroids(coords, min_distance_um, scale_zyx)
    return coords


def complementary_detections(
    image_zyx: np.ndarray,
    primary_zyx: np.ndarray,
    config: DetectionConfig,
    scale_zyx: tuple[float, float, float],
) -> np.ndarray:
    """Second-pass detections kept only when isolated from the primary set."""
    method = str(config.complementary_method or "none").lower()
    if method in {"none", "", "off"}:
        return _empty_coords()
    norm = normalize_volume(image_zyx, config.normalize)
    candidates = _run_detector(
        norm,
        method,
        config.complementary_min_sigma,
        config.complementary_max_sigma,
        config.complementary_threshold,
        config.complementary_min_distance_um,
        config.complementary_anisotropic,
        scale_zyx,
        suppress=False,
    )
    cand_int, _ = sample_appearance(image_zyx, candidates)
    prim_int, _ = sample_appearance(image_zyx, primary_zyx)
    return select_complementary_coords(
        primary_zyx,
        candidates,
        scale_zyx,
        config.complementary_min_distance_um,
        candidate_intensity=cand_int,
        primary_intensity=prim_int,
        intensity_ratio=config.complementary_intensity_ratio,
        self_nms_um=config.min_distance_um,
    )


def detect_primary_cells(
    image_zyx: np.ndarray, config: DetectionConfig, scale_zyx: tuple[float, float, float]
) -> np.ndarray:
    """Primary detector only. Complementary peaks are not attached."""
    norm = normalize_volume(image_zyx, config.normalize)
    return _run_detector(
        norm,
        config.method,
        config.min_sigma,
        config.max_sigma,
        config.threshold,
        config.min_distance_um,
        config.anisotropic,
        scale_zyx,
        suppress=config.suppress_duplicates,
    )


def select_linkable_coords(
    extra_zyx: np.ndarray,
    neighbor_zyx: np.ndarray,
    scale_zyx: tuple[float, float, float],
    gate_um: float,
) -> np.ndarray:
    """Keep extra peaks that sit inside ``gate_um`` of any neighbor-frame detection."""
    extra = np.asarray(extra_zyx, dtype=np.float64).reshape(-1, 3)
    if len(extra) == 0:
        return extra
    dist = nearest_physical_distance(extra, neighbor_zyx, scale_zyx)
    return extra[dist <= float(gate_um)]


def filter_temporally_linkable(
    extra_by_t: dict[int, np.ndarray],
    primary_by_t: dict[int, np.ndarray],
    scale_zyx: tuple[float, float, float],
    gate_um: float,
    max_dt: int = 1,
) -> dict[int, np.ndarray]:
    """Keep complementary peaks that can attach to a primary detection nearby in time.

    Neighbors are **primary** detections only, at ``t ± dt`` for ``1 <= dt <= max_dt``.
    Complementary-to-complementary chains are not used, so unlinked local-max noise
    cannot keep itself alive. This is the linkable_v1 T_pred control.
    """
    max_dt = max(int(max_dt), 0)
    kept: dict[int, np.ndarray] = {}
    for t, extra in extra_by_t.items():
        extra = np.asarray(extra, dtype=np.float64).reshape(-1, 3)
        if len(extra) == 0 or max_dt <= 0:
            kept[int(t)] = extra
            continue
        neighbors: list[np.ndarray] = []
        for dt in range(1, max_dt + 1):
            for t2 in (int(t) - dt, int(t) + dt):
                refs = primary_by_t.get(t2)
                if refs is None:
                    continue
                refs = np.asarray(refs, dtype=np.float64).reshape(-1, 3)
                if len(refs):
                    neighbors.append(refs)
        if not neighbors:
            kept[int(t)] = _empty_coords()
            continue
        kept[int(t)] = select_linkable_coords(extra, np.vstack(neighbors), scale_zyx, gate_um)
    return kept


def merge_sequence_detections(
    primary_by_t: dict[int, np.ndarray],
    extra_by_t: dict[int, np.ndarray],
    config: DetectionConfig,
    scale_zyx: tuple[float, float, float],
) -> dict[int, np.ndarray]:
    """Stack primary detections with complementary peaks, optionally link-filtered."""
    extras = extra_by_t
    if config.complementary_require_temporal_link:
        extras = filter_temporally_linkable(
            extra_by_t,
            primary_by_t,
            scale_zyx,
            gate_um=float(config.complementary_link_gate_um),
            max_dt=int(config.complementary_link_max_dt),
        )
    times = sorted(set(int(t) for t in primary_by_t) | set(int(t) for t in extras))
    merged: dict[int, np.ndarray] = {}
    for t in times:
        primary = np.asarray(primary_by_t.get(t, _empty_coords()), dtype=np.float64).reshape(-1, 3)
        extra = np.asarray(extras.get(t, _empty_coords()), dtype=np.float64).reshape(-1, 3)
        if len(extra):
            merged[t] = np.vstack([primary, extra])
        else:
            merged[t] = primary
    return merged


def detect_cells(image_zyx: np.ndarray, config: DetectionConfig, scale_zyx: tuple[float, float, float]) -> np.ndarray:
    """Return (N, 3) array of (z, y, x) centroids in voxels.

    When ``complementary_require_temporal_link`` is set, extras are **not**
    attached here; call ``detect_volume_with_appearance`` so adjacent frames
    can be checked against the locked association gate.
    """
    coords = detect_primary_cells(image_zyx, config, scale_zyx)
    if config.complementary_require_temporal_link:
        return coords
    extra = complementary_detections(image_zyx, coords, config, scale_zyx)
    if len(extra):
        coords = np.vstack([coords, extra])
    return coords


def detect_volume_with_appearance(
    frames: Iterable[tuple[int, np.ndarray]],
    config: DetectionConfig,
    scale_zyx: tuple[float, float, float],
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Detect a whole sequence, applying the temporal-link filter when configured."""
    items = [(int(t), np.asarray(image)) for t, image in frames]
    primary_by_t: dict[int, np.ndarray] = {}
    extra_by_t: dict[int, np.ndarray] = {}
    images: dict[int, np.ndarray] = {}
    for t, image in items:
        images[t] = image
        primary = detect_primary_cells(image, config, scale_zyx)
        extra = complementary_detections(image, primary, config, scale_zyx)
        primary_by_t[t] = primary
        extra_by_t[t] = extra
    merged = merge_sequence_detections(primary_by_t, extra_by_t, config, scale_zyx)
    out: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for t, _image in items:
        coords = merged.get(t, _empty_coords())
        intensity, appearance = sample_appearance(images[t], coords)
        out.append((t, coords, intensity, appearance))
    return out


def sample_appearance(image_zyx: np.ndarray, coords_zyx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid intensity and 3x3x3 mean. Empty-safe."""
    coords = np.asarray(coords_zyx, dtype=np.float64).reshape(-1, 3)
    if len(coords) == 0:
        empty = np.zeros((0,), dtype=np.float64)
        return empty, empty
    image = np.asarray(image_zyx)
    z = np.clip(np.round(coords[:, 0]).astype(np.int64), 0, image.shape[0] - 1)
    y = np.clip(np.round(coords[:, 1]).astype(np.int64), 0, image.shape[1] - 1)
    x = np.clip(np.round(coords[:, 2]).astype(np.int64), 0, image.shape[2] - 1)
    intensity = image[z, y, x].astype(np.float64)
    patch = np.zeros(len(coords), dtype=np.float64)
    for i, (zi, yi, xi) in enumerate(zip(z, y, x)):
        sl = image[max(zi - 1, 0) : zi + 2, max(yi - 1, 0) : yi + 2, max(xi - 1, 0) : xi + 2]
        patch[i] = float(np.mean(sl)) if sl.size else float(intensity[i])
    return intensity, patch


def detect_cells_with_appearance(
    image_zyx: np.ndarray, config: DetectionConfig, scale_zyx: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = detect_cells(image_zyx, config, scale_zyx)
    intensity, patch = sample_appearance(image_zyx, coords)
    return coords, intensity, patch
