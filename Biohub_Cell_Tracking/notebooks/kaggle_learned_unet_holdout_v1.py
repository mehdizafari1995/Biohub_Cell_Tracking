"""
Kaggle helper kernel: learned_unet_holdout_v1 GPU holdout inference.

Loads locked learned_unet_v1 weights and predicts only the frozen-holdout
pair 6bba_05db0fb1 + 6bba_c328f2fd. Never opens 44b6 volumes.
Never writes submission.csv. Not a competition submission. One-shot only.

Attach competition `biohub-cell-tracking-during-development` and the
private training helper `mahdimzafari/biohub-learned-unet-v1` after that
helper is COMPLETE. Internet OFF. GPU ON. Do not overwrite locked slugs.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

CONFIG_LABEL = "learned_unet_holdout_v1"
TRAIN_LABEL = "learned_unet_v1"
INFER_LABEL = "learned_unet_infer_v1"
TRAIN_CONFIG_HASH = "4c38035d6657f2ad"
INFER_CONFIG_HASH = "d7578b5f64ddac08"
TUNING_SPECIMEN = "44b6"
HOLDOUT_SPECIMENS = ("6bba",)
INFER_STEMS = ("6bba_05db0fb1", "6bba_c328f2fd")
FROZEN_HOLDOUT_STEMS = ("6bba_05db0fb1", "6bba_c328f2fd")
TUNING_INFER_STEMS = ("44b6_12dfb391", "44b6_267148e4")
KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-holdout-v1"
TRAIN_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-v1"
INFER_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-infer-v1"
WEIGHTS_NAME = "learned_unet_v1_best.pt"
FALLBACK_WEIGHTS_NAME = "learned_unet_v1_last.pt"
SCALE_ZYX = (1.625, 0.40625, 0.40625)
_POS_EMBED_DIM = 8
COMPETITION = "biohub-cell-tracking-during-development"
CANDIDATE_ROOTS = [
    Path("/kaggle/input") / COMPETITION,
    Path("/kaggle/input/competitions") / COMPETITION,
    Path("data"),
]
if os.environ.get("BIOHUB_DATA_ROOT"):
    CANDIDATE_ROOTS.insert(0, Path(os.environ["BIOHUB_DATA_ROOT"]))


class HoldoutLeakError(ValueError):
    """Raised when a stem outside the one-shot 6bba holdout would be opened,
    or when training weights were not 44b6-only.
    """


def infer_config_payload() -> dict[str, Any]:
    return {
        "label": CONFIG_LABEL,
        "train_label": TRAIN_LABEL,
        "infer_label": INFER_LABEL,
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": INFER_CONFIG_HASH,
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "infer_stems": list(INFER_STEMS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "refused_tuning_stems": list(TUNING_INFER_STEMS),
        "det_threshold": 0.5,
        "det_tta": True,
        "pool_kernel_um": 5.0,
        "linker": "greedy_gate_12um",
        "gate_um": 12.0,
        "max_gap_frames": 0,
        "max_parents_per_node": 1,
        "max_children_per_node": 1,
        "downsample": [1, 4, 4],
        "normalize": "window_minmax",
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": False,
        "one_shot_holdout": True,
        "kernel_slug": KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
        "weights_name": WEIGHTS_NAME,
    }


def infer_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or infer_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def specimen_of(stem: str) -> str:
    name = Path(stem).name
    for suffix in (".zarr", ".geff", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.split("_", 1)[0]


def holdout_stems(stems: Iterable[str]) -> list[str]:
    leaked: list[str] = []
    seen: set[str] = set()
    for raw in stems:
        stem = Path(raw).name
        for suffix in (".zarr", ".geff", ".json"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        if stem in seen:
            continue
        if specimen_of(stem) in HOLDOUT_SPECIMENS or stem in FROZEN_HOLDOUT_STEMS:
            leaked.append(stem)
            seen.add(stem)
    return leaked


def refuse_holdout_stems(stems: Iterable[str]) -> None:
    leaked = holdout_stems(stems)
    if leaked:
        raise HoldoutLeakError(
            "learned_unet_holdout_v1 refuses holdout stems in training weights: " + ", ".join(leaked)
        )


def refuse_tuning_stems(stems: Iterable[str]) -> None:
    leaked: list[str] = []
    seen: set[str] = set()
    for raw in stems:
        stem = Path(raw).name
        for suffix in (".zarr", ".geff", ".json"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        if stem in seen:
            continue
        if specimen_of(stem) == TUNING_SPECIMEN or stem in TUNING_INFER_STEMS:
            leaked.append(stem)
            seen.add(stem)
    if leaked:
        raise HoldoutLeakError(
            "learned_unet_holdout_v1 refuses 44b6 tuning stems: " + ", ".join(leaked)
        )


def refuse_non_infer_stems(stems: Iterable[str]) -> None:
    cleaned: list[str] = []
    for raw in stems:
        stem = Path(raw).name
        for suffix in (".zarr", ".geff", ".json"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        if stem not in cleaned:
            cleaned.append(stem)
    refuse_tuning_stems(cleaned)
    extra = [s for s in cleaned if s not in INFER_STEMS]
    if extra:
        raise HoldoutLeakError(
            "learned_unet_holdout_v1 only opens frozen-holdout stems; got " + ", ".join(extra)
        )


def assert_no_submission_csv(working: Path) -> None:
    banned = working / "submission.csv"
    if banned.exists():
        banned.unlink()
        raise RuntimeError("learned_unet_holdout_v1 must not write submission.csv")


def find_root() -> Path:
    for path in CANDIDATE_ROOTS:
        if (path / "train").is_dir() or (path / "test").is_dir():
            return path
    raise FileNotFoundError("Could not find the competition train directory")


def environment_label(root: Path) -> str:
    return "kaggle_input" if str(root).startswith("/kaggle/input") else "local_simulation"


def output_dir() -> Path:
    kaggle = Path("/kaggle/working")
    return kaggle if kaggle.is_dir() else Path(".")


def peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage > 10_000_000:
        return usage / (1024 * 1024)
    return usage / 1024.0


def decode_blosc(payload: bytes) -> bytes:
    errors: list[str] = []
    try:
        import imagecodecs

        return bytes(imagecodecs.blosc_decode(payload))
    except Exception as exc:
        errors.append(f"imagecodecs:{type(exc).__name__}")
    try:
        from numcodecs import Blosc

        return bytes(Blosc().decode(payload))
    except Exception as extra:
        errors.append(f"numcodecs:{type(extra).__name__}")
    try:
        import blosc2

        return bytes(blosc2.decompress(payload))
    except Exception as extra:
        errors.append(f"blosc2:{type(extra).__name__}")
    raise ModuleNotFoundError(
        "Cannot decode blosc zarr chunks. " + ",".join(errors)
    )


def decode_zstd(payload: bytes) -> bytes:
    errors: list[str] = []
    try:
        import imagecodecs

        return bytes(imagecodecs.zstd_decode(payload))
    except Exception as extra:
        errors.append(f"imagecodecs:{type(extra).__name__}")
    try:
        from numcodecs.zstd import Zstd

        return bytes(Zstd().decode(payload))
    except Exception as extra:
        errors.append(f"numcodecs:{type(extra).__name__}")
    try:
        import zstandard

        return bytes(zstandard.ZstdDecompressor().decompress(payload))
    except Exception as extra:
        errors.append(f"zstandard:{type(extra).__name__}")
    try:
        return decode_blosc(payload)
    except Exception as extra:
        errors.append(f"blosc_fallback:{type(extra).__name__}")
    raise ModuleNotFoundError("Cannot decode zstd chunks. " + ",".join(errors))


def _decode_payload(raw: bytes, codecs: list[dict[str, Any]]) -> bytes:
    payload = raw
    for codec in reversed(list(codecs or [])):
        if isinstance(codec, str):
            name, cfg = codec, {}
        else:
            name = str(codec.get("name") or "")
            cfg = codec.get("configuration") or {}
        if name == "blosc":
            payload = decode_blosc(payload)
        elif name in {"zstd", "zstandard"}:
            payload = decode_zstd(payload)
        elif name in {"bytes", "crc32c", "crc32", ""}:
            continue
        elif name == "transpose":
            continue
        else:
            raise ValueError(f"unsupported zarr codec {name!r} cfg={cfg!r}")
    return payload


class ZarrV3FrameArray:
    def __init__(self, array_dir: str | Path):
        self.array_dir = Path(array_dir)
        meta = json.loads((self.array_dir / "zarr.json").read_text())
        self.shape = tuple(int(x) for x in meta["shape"])
        self.dtype = np.dtype(meta["data_type"])
        self.chunk_shape = tuple(int(x) for x in meta["chunk_grid"]["configuration"]["chunk_shape"])
        self.codecs = list(meta.get("codecs") or [])
        encoding = meta.get("chunk_key_encoding") or {}
        self.separator = str((encoding.get("configuration") or {}).get("separator") or "/")

    def __getitem__(self, t: int) -> np.ndarray:
        key = self.separator.join(["c", str(int(t)), "0", "0", "0"])
        path = self.array_dir / key
        if not path.exists():
            raise FileNotFoundError(path)
        raw = _decode_payload(path.read_bytes(), self.codecs)
        arr = np.frombuffer(raw, dtype=self.dtype)
        expected = int(np.prod(self.chunk_shape))
        if arr.size != expected:
            raise ValueError(f"{path}: decoded {arr.size} values, expected {expected}")
        frame = arr.reshape(self.chunk_shape)
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]
        return np.asarray(frame)


def open_volume_array(zarr_path: str | Path):
    path = Path(zarr_path)
    custom = ZarrV3FrameArray(path / "0") if (path / "0" / "zarr.json").exists() else None
    try:
        import zarr

        arr = zarr.open_group(path, mode="r")["0"]
        _ = arr[0]
        return arr
    except Exception:
        if custom is not None:
            return custom
        raise


def downsample_frame(frame: np.ndarray, downsample: Sequence[int]) -> np.ndarray:
    dz, dy, dx = (int(x) for x in downsample)
    return np.asarray(frame)[::dz, ::dy, ::dx]


def pool_kernel_from_um(um: float, voxel_size: Sequence[float]) -> tuple[int, ...]:
    kernel: list[int] = []
    for size in voxel_size:
        k = max(1, round(float(um) / float(size)))
        if k % 2 == 0:
            k += 1
        kernel.append(k)
    return tuple(kernel)


def detect_cells_pooled(
    logits: np.ndarray,
    t: int,
    det_threshold: float,
    pool_kernel: Sequence[int],
) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    pad = tuple(int(k) // 2 for k in pool_kernel)
    padded = np.pad(logits, [(p, p) for p in pad], mode="constant", constant_values=-np.inf)
    windows = np.lib.stride_tricks.sliding_window_view(padded, tuple(int(k) for k in pool_kernel))
    pooled = windows.max(axis=(-3, -2, -1))
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    is_peak = (logits == pooled) & (probs > float(det_threshold))
    zyx = np.argwhere(is_peak)
    if zyx.size == 0:
        return np.empty((0, 4), dtype=np.int16)
    t_col = np.full((len(zyx), 1), int(t), dtype=np.int16)
    return np.concatenate([t_col, zyx.astype(np.int16)], axis=1)


def scale_coords_to_original(coords: np.ndarray, downsample: Sequence[int]) -> np.ndarray:
    out = np.asarray(coords, dtype=np.float32).copy()
    if out.size == 0:
        return np.empty((0, 4), dtype=np.int16)
    ds = np.asarray(downsample, dtype=np.float32)
    out[:, 1:] *= ds
    return np.rint(out).astype(np.int16)


def greedy_gate_links(
    coords_t: np.ndarray,
    coords_t1: np.ndarray,
    scale_zyx: Sequence[float] = SCALE_ZYX,
    gate_um: float = 12.0,
) -> list[tuple[int, int, float]]:
    a = np.asarray(coords_t, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(coords_t1, dtype=np.float64).reshape(-1, 3)
    if a.size == 0 or b.size == 0:
        return []
    scale = np.asarray(scale_zyx, dtype=np.float64)
    pa = a * scale
    pb = b * scale
    delta = pa[:, None, :] - pb[None, :, :]
    dist = np.sqrt(np.sum(delta * delta, axis=-1))
    candidates = [
        (float(dist[i, j]), int(i), int(j))
        for i in range(a.shape[0])
        for j in range(b.shape[0])
        if float(dist[i, j]) <= float(gate_um)
    ]
    candidates.sort()
    used_i: set[int] = set()
    used_j: set[int] = set()
    edges: list[tuple[int, int, float]] = []
    for d_um, i, j in candidates:
        if i in used_i or j in used_j:
            continue
        edges.append((i, j, d_um))
        used_i.add(i)
        used_j.add(j)
    return edges


def graph_payload(dataset: str, coords: np.ndarray, edges: Sequence[tuple[int, int]]) -> dict[str, Any]:
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 4)
    node_id = np.arange(1, len(coords) + 1, dtype=np.int64)
    return {
        "dataset": dataset,
        "node_id": node_id.tolist(),
        "t": coords[:, 0].astype(np.int64).tolist() if len(coords) else [],
        "z": coords[:, 1].astype(np.int64).tolist() if len(coords) else [],
        "y": coords[:, 2].astype(np.int64).tolist() if len(coords) else [],
        "x": coords[:, 3].astype(np.int64).tolist() if len(coords) else [],
        "edges": [[int(s), int(d)] for s, d in edges],
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


KAGGLE_INPUT = Path("/kaggle/input")
SKIP_INPUT_DIRS = frozenset(
    {
        "biohub-cell-tracking-during-development",
        "competitions",
    }
)


def _looks_like_volume(path: Path) -> bool:
    if path.suffix in {".zarr", ".geff"}:
        return True
    return (path / "zarr.json").exists() or (path / "0" / "zarr.json").exists()


def default_weight_roots() -> list[Path]:
    roots: list[Path] = []
    extra = os.environ.get("BIOHUB_LEARNED_WEIGHTS_DIR")
    if extra:
        roots.append(Path(extra))
    roots.extend(
        [
            Path("/kaggle/input/biohub-learned-unet-v1"),
            Path("/kaggle/input/biohub-learned-unet-v1/biohub-learned-unet-v1"),
            Path("/kaggle/input/mahdimzafari-biohub-learned-unet-v1"),
            Path("/kaggle/input/mahdimzafari/biohub-learned-unet-v1"),
            Path("/kaggle/input/notebooks/mahdimzafari/biohub-learned-unet-v1"),
            Path("/kaggle/input/notebooks/biohub-learned-unet-v1"),
            Path("/kaggle/input/notebooks/mahdimzafari-biohub-learned-unet-v1"),
            Path("/kaggle/input/notebooks"),
            Path("artifacts/kaggle/learned_unet_v1"),
            Path("artifacts/kaggle"),
            Path("."),
        ]
    )
    if KAGGLE_INPUT.is_dir():
        for child in sorted(KAGGLE_INPUT.iterdir()):
            if child.name in SKIP_INPUT_DIRS or _looks_like_volume(child):
                continue
            roots.append(child)
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def describe_weight_search(roots: Sequence[Path] | None = None) -> dict[str, Any]:
    roots = list(roots) if roots is not None else default_weight_roots()
    listing: list[dict[str, Any]] = []
    if KAGGLE_INPUT.is_dir():
        for child in sorted(KAGGLE_INPUT.iterdir()):
            entry: dict[str, Any] = {"name": child.name, "is_dir": child.is_dir()}
            if child.is_dir() and child.name not in SKIP_INPUT_DIRS and not _looks_like_volume(child):
                try:
                    kids = list(child.iterdir())
                    entry["files"] = sorted(p.name for p in kids if p.is_file())[:20]
                    entry["dirs"] = sorted(p.name for p in kids if p.is_dir())[:20]
                except OSError:
                    entry["files"] = []
                    entry["dirs"] = []
            listing.append(entry)
    return {"roots": [str(path) for path in roots], "kaggle_input": listing}


def find_weights(search_roots: Sequence[str | Path] | None = None) -> Path | None:
    roots = [Path(p) for p in search_roots if p] if search_roots else default_weight_roots()
    names = (WEIGHTS_NAME, FALLBACK_WEIGHTS_NAME)
    for root in roots:
        found = _first_weight(root, names)
        if found is not None:
            return found
    return None


def _first_weight(root: Path, names: tuple[str, ...], max_depth: int = 5) -> Path | None:
    if not root.exists() or _looks_like_volume(root):
        return None
    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > max_depth or _looks_like_volume(current):
            continue
        try:
            children = list(current.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_file() and child.name in names:
                return child
        if depth == max_depth:
            continue
        for child in children:
            if child.is_dir() and not _looks_like_volume(child) and child.name not in SKIP_INPUT_DIRS:
                stack.append((child, depth + 1))
    return None


def verify_train_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    splits = payload.get("splits") or {}
    stems = list(splits.get("train") or []) + list(splits.get("test") or [])
    if stems:
        refuse_holdout_stems(stems)
        other = [s for s in stems if specimen_of(s) != TUNING_SPECIMEN]
        if other:
            raise HoldoutLeakError("training manifest is not 44b6-only: " + ", ".join(other))
    return payload


def probe_runtime() -> dict[str, Any]:
    info: dict[str, Any] = {"python": sys.version.split()[0], "cuda_available": False, "torch": None}
    try:
        import torch

        info["torch"] = getattr(torch, "__version__", "present")
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count()) if info["cuda_available"] else 0
        if info["cuda_available"]:
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["torch_error"] = type(exc).__name__
    return info


def _build_models(cfg: dict[str, Any]):
    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    class TemporalAttention(nn.Module):
        def __init__(self, channels: int, n_heads: int = 4) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(channels)
            self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, t, c = x.shape[:3]
            spatial = x.shape[3:]
            s = math.prod(spatial)
            h = x.reshape(b, t, c, s).permute(0, 3, 1, 2).reshape(b * s, t, c)
            h = self.norm(h)
            h, _ = self.attn(h, h, h, need_weights=False)
            h = h.reshape(b, s, t, c).permute(0, 2, 3, 1).reshape(b, t, c, *spatial)
            return x + h

    class TemporalUNet3D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = list(cfg["unet_layers"])
            self.encoder_blocks = nn.ModuleList()
            self.temporal_blocks = nn.ModuleList()
            prev = 1
            for i, ch in enumerate(layers):
                self.encoder_blocks.append(_conv_block(prev, ch))
                if cfg["skip_fullres_temporal"] and i == 0:
                    self.temporal_blocks.append(nn.Identity())
                else:
                    self.temporal_blocks.append(TemporalAttention(ch))
                prev = ch
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.upsamples = nn.ModuleList()
            self.decoder_blocks = nn.ModuleList()
            for i in range(len(layers) - 1, 0, -1):
                self.upsamples.append(nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False))
                self.decoder_blocks.append(_conv_block(layers[i] + layers[i - 1], layers[i - 1]))
            self.head = nn.Conv3d(layers[0], int(cfg["unet_out_channels"]), kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, t = x.shape[:2]
            x = x.reshape(b * t, *x.shape[2:])
            skips: list[torch.Tensor] = []
            for i, (block, temporal) in enumerate(zip(self.encoder_blocks, self.temporal_blocks)):
                if i > 0:
                    x = self.pool(x)
                x = block(x)
                x = temporal(x.reshape(b, t, *x.shape[1:])).reshape(b * t, *x.shape[1:])
                if i < len(self.encoder_blocks) - 1:
                    skips.append(x)
            for up, block, skip in zip(self.upsamples, self.decoder_blocks, skips[::-1]):
                x = up(x)
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
                x = torch.cat([x, skip], dim=1)
                x = block(x)
            x = self.head(x)
            return x.reshape(b, t, *x.shape[1:])

    class CrossAttentionBlock(nn.Module):
        def __init__(self, hidden_dim: int = 128, n_heads: int = 4, dropout: float = 0.3):
            super().__init__()
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=dropout)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout),
            )

        def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_mask: torch.Tensor | None = None) -> torch.Tensor:
            key_padding_mask = ~kv_mask if kv_mask is not None else None
            attn_out, _ = self.cross_attn(self.norm1(q), self.norm1(kv), self.norm1(kv), key_padding_mask=key_padding_mask)
            q = q + attn_out
            return q + self.mlp(self.norm2(q))

    class SimpleNodeTransformer(nn.Module):
        def __init__(self, feat_dim: int):
            super().__init__()
            hidden_dim = 128
            self.proj = nn.Linear(feat_dim, hidden_dim)
            self.norm_in = nn.LayerNorm(hidden_dim)
            self.blocks = nn.ModuleList([CrossAttentionBlock(hidden_dim) for _ in range(4)])
            self.norm_out = nn.LayerNorm(hidden_dim)
            self.pair_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, feat_t, feat_t1, coords_t, coords_t1, mask_t=None, mask_t1=None):
            q = self.norm_in(self.proj(feat_t))
            k = self.norm_in(self.proj(feat_t1))
            for block in self.blocks:
                q = block(q, k, kv_mask=mask_t1)
                k = block(k, q, kv_mask=mask_t)
            q = self.norm_out(q)
            k = self.norm_out(k)
            n1 = k.shape[1]
            qe = q.unsqueeze(2).expand(-1, -1, n1, -1)
            ke = k.unsqueeze(1).expand(-1, q.shape[1], -1, -1)
            rel = (coords_t.unsqueeze(2) - coords_t1.unsqueeze(1)) / 100.0
            return self.pair_mlp(torch.cat([qe, ke, rel], dim=-1)).squeeze(-1)

    class UNetNodeTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.unet = TemporalUNet3D()
            self.detect_head = nn.Conv3d(int(cfg["unet_out_channels"]), 1, kernel_size=1)
            self.transformer = SimpleNodeTransformer(
                feat_dim=int(cfg["unet_out_channels"]) + 4 * _POS_EMBED_DIM
            )

        def encode(self, imgs: torch.Tensor):
            window = imgs.unsqueeze(2)
            unet_out = self.unet(window)
            det = [self.detect_head(unet_out[:, i]) for i in range(unet_out.shape[1])]
            return unet_out, det

    return UNetNodeTransformer


def load_checkpoint(weights_path: Path, device):
    import torch

    blob = torch.load(weights_path, map_location=device, weights_only=False)
    cfg = dict(blob.get("config") or {})
    if blob.get("config_hash") and blob.get("config_hash") != TRAIN_CONFIG_HASH:
        raise HoldoutLeakError("checkpoint config_hash is not locked learned_unet_v1")
    splits = blob.get("splits") or {}
    stems = list(splits.get("train") or []) + list(splits.get("test") or [])
    if stems:
        refuse_holdout_stems(stems)
    needed = {
        "unet_layers": [32, 64, 128],
        "unet_out_channels": 32,
        "skip_fullres_temporal": True,
        "downsample": [1, 4, 4],
    }
    for key, value in needed.items():
        cfg.setdefault(key, value)
    UNetNodeTransformer = _build_models(cfg)
    model = UNetNodeTransformer()
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, cfg, blob


def predict_volume(model, volume, cfg: dict[str, Any], infer_cfg: dict[str, Any], device) -> tuple[np.ndarray, list[tuple[int, int]]]:
    import torch

    downsample = list(infer_cfg["downsample"])
    t_count = int(getattr(volume, "shape", [100])[0])
    voxel_down = tuple(s * d for s, d in zip(SCALE_ZYX, downsample))
    pool_k = pool_kernel_from_um(float(infer_cfg["pool_kernel_um"]), voxel_down)
    seen: dict[int, np.ndarray] = {}
    stride = 1
    window_starts = list(range(0, max(t_count - 1, 0), stride))
    if t_count >= 2 and (not window_starts or window_starts[-1] != t_count - 2):
        window_starts.append(t_count - 2)

    for t in window_starts:
        frame0 = downsample_frame(volume[t], downsample).astype(np.float32)
        frame1 = downsample_frame(volume[t + 1], downsample).astype(np.float32)
        lo = float(min(frame0.min(), frame1.min()))
        hi = float(max(frame0.max(), frame1.max()))
        frame0 = (frame0 - lo) / (hi - lo + 1e-6)
        frame1 = (frame1 - lo) / (hi - lo + 1e-6)
        imgs = torch.from_numpy(np.stack([frame0, frame1], axis=0)).unsqueeze(0).to(device)
        with torch.no_grad():
            _, det_logits = model.encode(imgs)
            if infer_cfg["det_tta"]:
                tta = [(-1,), (-2,), (-2, -1)]
                for dims in tta:
                    _, det_flip = model.encode(imgs.flip(dims))
                    for f in range(2):
                        det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                for f in range(2):
                    det_logits[f] = det_logits[f] / 4.0
        for offset in (0, 1):
            frame_t = t + offset
            if frame_t in seen:
                continue
            logits = det_logits[offset][0, 0].detach().float().cpu().numpy()
            seen[frame_t] = detect_cells_pooled(
                logits, frame_t, float(infer_cfg["det_threshold"]), pool_k
            )

    ordered_t = sorted(seen)
    coord_lists = [scale_coords_to_original(seen[t], downsample) for t in ordered_t]
    coords = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
    offsets: dict[int, tuple[int, int]] = {}
    cursor = 0
    for t, arr in zip(ordered_t, coord_lists):
        offsets[t] = (cursor, cursor + len(arr))
        cursor += len(arr)
    edges: list[tuple[int, int]] = []
    for t in ordered_t:
        if t + 1 not in offsets:
            continue
        s0, e0 = offsets[t]
        s1, e1 = offsets[t + 1]
        local = greedy_gate_links(
            coords[s0:e0, 1:],
            coords[s1:e1, 1:],
            SCALE_ZYX,
            float(infer_cfg["gate_um"]),
        )
        for i, j, _dist in local:
            edges.append((s0 + i + 1, s1 + j + 1))
    return coords, edges


def infer_holdout_volumes(
    train_dir: Path,
    model,
    train_cfg: dict[str, Any],
    infer_cfg: dict[str, Any],
    working: Path,
    device,
) -> list[dict[str, Any]]:
    refuse_non_infer_stems(INFER_STEMS)
    graphs_dir = working / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for stem in INFER_STEMS:
        refuse_non_infer_stems([stem])
        zarr_path = train_dir / f"{stem}.zarr"
        if not zarr_path.exists():
            summaries.append({"stem": stem, "status": "missing_zarr"})
            continue
        volume = open_volume_array(zarr_path)
        coords, edges = predict_volume(model, volume, train_cfg, infer_cfg, device)
        payload = graph_payload(stem, coords, edges)
        (graphs_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2))
        summaries.append(
            {
                "stem": stem,
                "status": "predicted",
                "n_nodes": len(payload["node_id"]),
                "n_edges": len(payload["edges"]),
            }
        )
    return summaries


def main() -> dict[str, Any]:
    started = time.time()
    cfg = infer_config_payload()
    working = output_dir()
    working.mkdir(parents=True, exist_ok=True)
    assert_no_submission_csv(working)
    runtime = probe_runtime()
    manifest: dict[str, Any] = {
        "status": "starting",
        "label": CONFIG_LABEL,
        "config_hash": infer_config_hash(cfg),
        "train_config_hash": TRAIN_CONFIG_HASH,
        "config": cfg,
        "environment": None,
        "is_kaggle_run": False,
        "is_competition_submission": False,
        "runtime": runtime,
        "notes": "Not a competition submission. Not a leaderboard score. Frozen 6bba holdout one-shot. 44b6 volumes refused.",
    }
    write_manifest(working / "infer_manifest.json", manifest)
    try:
        root = find_root()
    except FileNotFoundError as exc:
        manifest.update({"status": "blocked_no_train_dir", "error": str(exc)})
        write_manifest(working / "infer_manifest.json", manifest)
        return manifest
    manifest["environment"] = environment_label(root)
    manifest["is_kaggle_run"] = manifest["environment"] == "kaggle_input"
    refuse_non_infer_stems(INFER_STEMS)
    if os.environ.get("BIOHUB_LEARNED_DRY_RUN") == "1":
        manifest.update({"status": "dry_run", "runtime_sec": time.time() - started})
        write_manifest(working / "infer_manifest.json", manifest)
        assert_no_submission_csv(working)
        return manifest
    weights = find_weights()
    if weights is None:
        manifest.update(
            {
                "status": "blocked_no_weights",
                "weight_search": describe_weight_search(),
                "runtime_sec": time.time() - started,
            }
        )
        write_manifest(working / "infer_manifest.json", manifest)
        assert_no_submission_csv(working)
        return manifest
    manifest["weights_path"] = str(weights)
    train_manifest = weights.parent / "train_manifest.json"
    if train_manifest.exists():
        verify_train_manifest(train_manifest)
        manifest["train_manifest_status"] = json.loads(train_manifest.read_text()).get("status")
    if not runtime.get("torch"):
        manifest.update({"status": "blocked_no_torch", "runtime_sec": time.time() - started})
        write_manifest(working / "infer_manifest.json", manifest)
        return manifest
    if not runtime.get("cuda_available"):
        manifest.update({"status": "blocked_no_gpu", "runtime_sec": time.time() - started})
        write_manifest(working / "infer_manifest.json", manifest)
        return manifest
    import torch

    device = torch.device("cuda")
    try:
        model, train_cfg, blob = load_checkpoint(weights, device)
        summaries = infer_holdout_volumes(root / "train", model, train_cfg, cfg, working, device)
    except Exception as exc:
        manifest.update(
            {
                "status": "infer_error",
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
                "runtime_sec": time.time() - started,
                "peak_memory_mb": peak_memory_mb(),
            }
        )
        write_manifest(working / "infer_manifest.json", manifest)
        assert_no_submission_csv(working)
        raise
    manifest.update(
        {
            "status": "inferred",
            "checkpoint_hash": blob.get("config_hash"),
            "graphs": summaries,
            "runtime_sec": time.time() - started,
            "peak_memory_mb": peak_memory_mb(),
        }
    )
    write_manifest(working / "infer_manifest.json", manifest)
    assert_no_submission_csv(working)
    return manifest


if __name__ == "__main__":
    result = main()
    print(
        json.dumps(
            {k: result.get(k) for k in ("status", "config_hash", "environment", "graphs")},
            indent=2,
            default=str,
        )
    )
    print("NOTE: private helper kernel, not a competition submission.")
