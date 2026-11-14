"""
Kaggle helper kernel: learned_unet_v1 GPU training.

Specimen-separated organizer TemporalUNet3D + SimpleNodeTransformer.
Trains on frozen-tuning specimen 44b6 only. Internal val is the two
frozen-tuning sequences. Never opens 6bba volumes. Never writes
submission.csv. Not a competition submission.

Attach competition `biohub-cell-tracking-during-development`.
Internet OFF. GPU ON. Do not overwrite locked helper/prediction slugs.
Weights stay in /kaggle/working (not git). A local run is not a Kaggle execution.
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

CONFIG_LABEL = "learned_unet_v1"
TUNING_SPECIMEN = "44b6"
HOLDOUT_SPECIMENS = ("6bba",)
INTERNAL_VAL_STEMS = ("44b6_12dfb391", "44b6_267148e4")
FROZEN_HOLDOUT_STEMS = ("6bba_05db0fb1", "6bba_c328f2fd")
KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-v1"
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
    """Raised when a frozen-holdout specimen would enter training."""


def config_payload() -> dict[str, Any]:
    return {
        "label": CONFIG_LABEL,
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "internal_val": list(INTERNAL_VAL_STEMS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "epochs": 3,
        "batch_size": 1,
        "grad_accum": 8,
        "lr": 0.0001,
        "downsample": [1, 4, 4],
        "unet_layers": [32, 64, 128],
        "unet_out_channels": 32,
        "window_size": 2,
        "pool_kernel_um": 5.0,
        "det_loss_weight": 1.0,
        "det_neg_weight": 0.01,
        "skip_fullres_temporal": True,
        "seed": 0,
        "max_hours": 11.0,
        "writes_submission_csv": False,
        "edge_supervision": "gt_node_index",
        "kernel_slug": KERNEL_SLUG,
    }


def config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def specimen_of(stem: str) -> str:
    name = Path(stem).name
    for suffix in (".zarr", ".geff"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.split("_", 1)[0]


def holdout_stems(stems: Iterable[str]) -> list[str]:
    leaked: list[str] = []
    seen: set[str] = set()
    for raw in stems:
        stem = Path(raw).name
        for suffix in (".zarr", ".geff"):
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
            "learned_unet_v1 refuses holdout stems in train/val: " + ", ".join(leaked)
        )


def build_44b6_splits(stems: Sequence[str]) -> dict[str, Any]:
    cleaned: list[str] = []
    for raw in stems:
        stem = Path(raw).name
        for suffix in (".zarr", ".geff"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        if stem not in cleaned:
            cleaned.append(stem)
    refuse_holdout_stems(cleaned)
    other = [s for s in cleaned if specimen_of(s) != TUNING_SPECIMEN]
    if other:
        raise HoldoutLeakError(
            "learned_unet_v1 train split is 44b6-only; got " + ", ".join(other)
        )
    train = sorted(s for s in cleaned if s not in INTERNAL_VAL_STEMS)
    val = sorted(s for s in cleaned if s in INTERNAL_VAL_STEMS)
    if not train:
        raise ValueError("learned_unet_v1 train split is empty after removing internal val")
    refuse_holdout_stems(train + val)
    return {
        "split": 0,
        "train": train,
        "test": val,
        "tuning_specimen": TUNING_SPECIMEN,
        "internal_val": list(INTERNAL_VAL_STEMS),
        "n_train": len(train),
        "n_test": len(val),
        "config_hash": config_hash(),
    }


def list_tuning_pairs(train_dir: str | Path) -> list[dict[str, str]]:
    root = Path(train_dir)
    if not root.exists():
        return []
    pairs: list[dict[str, str]] = []
    for zarr_path in sorted(root.glob("*.zarr")):
        if not zarr_path.is_dir():
            continue
        stem = zarr_path.name[: -len(".zarr")]
        if specimen_of(stem) != TUNING_SPECIMEN:
            continue
        geff_path = root / f"{stem}.geff"
        if not geff_path.exists():
            continue
        pairs.append({"stem": stem, "zarr": str(zarr_path), "geff": str(geff_path)})
    refuse_holdout_stems(p["stem"] for p in pairs)
    return pairs


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
        "Cannot decode blosc zarr chunks (zarr/imagecodecs/numcodecs/blosc2 missing). "
        + ",".join(errors)
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


class ZarrV3Array:
    def __init__(self, array_dir: str | Path):
        self.array_dir = Path(array_dir)
        meta = json.loads((self.array_dir / "zarr.json").read_text())
        self.shape = tuple(int(x) for x in meta["shape"])
        self.dtype = np.dtype(meta.get("data_type") or meta.get("dtype"))
        grid = meta.get("chunk_grid") or {}
        cfg = grid.get("configuration") or {}
        self.chunk_shape = tuple(int(x) for x in (cfg.get("chunk_shape") or self.shape))
        self.codecs = list(meta.get("codecs") or [])
        encoding = meta.get("chunk_key_encoding") or {}
        self.separator = str((encoding.get("configuration") or {}).get("separator") or "/")

    def _chunk_path(self, indices: tuple[int, ...]) -> Path:
        key = self.separator.join(["c", *[str(int(i)) for i in indices]])
        return self.array_dir / key

    def read(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=self.dtype)
        n_chunks = tuple(int(np.ceil(s / c)) for s, c in zip(self.shape, self.chunk_shape))
        for chunk_idx in np.ndindex(n_chunks):
            path = self._chunk_path(tuple(int(i) for i in chunk_idx))
            if not path.exists():
                continue
            raw = _decode_payload(path.read_bytes(), self.codecs)
            arr = np.frombuffer(raw, dtype=self.dtype)
            starts = tuple(int(i) * c for i, c in zip(chunk_idx, self.chunk_shape))
            stops = tuple(min(st + c, s) for st, c, s in zip(starts, self.chunk_shape, self.shape))
            sl = tuple(slice(a, b) for a, b in zip(starts, stops))
            expected = int(np.prod([b - a for a, b in zip(starts, stops)]))
            if arr.size != expected:
                chunk = arr.reshape(self.chunk_shape)
                local = tuple(slice(0, b - a) for a, b in zip(starts, stops))
                out[sl] = chunk[local]
            else:
                out[sl] = arr.reshape([b - a for a, b in zip(starts, stops)])
        return out


class ZarrV2Array:
    def __init__(self, array_dir: str | Path):
        self.array_dir = Path(array_dir)
        meta = json.loads((self.array_dir / ".zarray").read_text())
        self.shape = tuple(int(x) for x in meta["shape"])
        self.dtype = np.dtype(meta["dtype"])
        chunks = meta.get("chunks") or self.shape
        self.chunk_shape = tuple(int(x) for x in chunks)
        self.compressor = meta.get("compressor")
        self.separator = str(meta.get("dimension_separator") or ".")

    def _chunk_path(self, indices: tuple[int, ...]) -> Path:
        candidates = [
            self.array_dir / self.separator.join(str(int(i)) for i in indices),
            self.array_dir / ".".join(str(int(i)) for i in indices),
            self.array_dir / "/".join(str(int(i)) for i in indices),
        ]
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    def read(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=self.dtype)
        n_chunks = tuple(int(np.ceil(s / c)) for s, c in zip(self.shape, self.chunk_shape))
        for chunk_idx in np.ndindex(n_chunks):
            path = self._chunk_path(tuple(int(i) for i in chunk_idx))
            if not path.exists():
                continue
            raw = path.read_bytes()
            if self.compressor:
                cid = str(self.compressor.get("id") or self.compressor.get("name") or "")
                if cid not in {"blosc", "blosc2"}:
                    raise ValueError(f"unsupported zarr v2 compressor {cid!r}")
                raw = decode_blosc(raw)
            arr = np.frombuffer(raw, dtype=self.dtype)
            starts = tuple(int(i) * c for i, c in zip(chunk_idx, self.chunk_shape))
            stops = tuple(min(st + c, s) for st, c, s in zip(starts, self.chunk_shape, self.shape))
            sl = tuple(slice(a, b) for a, b in zip(starts, stops))
            expected = int(np.prod([b - a for a, b in zip(starts, stops)]))
            if arr.size != expected:
                chunk = arr.reshape(self.chunk_shape)
                local = tuple(slice(0, b - a) for a, b in zip(starts, stops))
                out[sl] = chunk[local]
            else:
                out[sl] = arr.reshape([b - a for a, b in zip(starts, stops)])
        return out


def read_array(array_dir: str | Path) -> np.ndarray:
    path = Path(array_dir)
    try:
        import zarr

        return np.asarray(zarr.open_array(path, mode="r")[:])
    except Exception:
        pass
    if (path / "zarr.json").exists():
        return ZarrV3Array(path).read()
    if (path / ".zarray").exists():
        return ZarrV2Array(path).read()
    raise FileNotFoundError(f"no zarr v2/v3 array metadata under {path}")


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


def load_geff_arrays(geff_path: str | Path) -> dict[str, np.ndarray]:
    root = Path(geff_path)
    try:
        import zarr

        group = zarr.open_group(root, mode="r")
        node_ids = np.asarray(group["nodes"]["ids"][:])
        props = {key: np.asarray(group["nodes"]["props"][key]["values"][:]) for key in ("t", "z", "y", "x")}
        edges = np.asarray(group["edges"]["ids"][:])
    except Exception:
        node_ids = read_array(root / "nodes" / "ids")
        props = {key: read_array(root / "nodes" / "props" / key / "values") for key in ("t", "z", "y", "x")}
        edges = read_array(root / "edges" / "ids")
    if edges.ndim == 1:
        edges = edges.reshape(-1, 2)
    return {
        "node_id": np.asarray(node_ids),
        "t": np.asarray(props["t"]),
        "z": np.asarray(props["z"]),
        "y": np.asarray(props["y"]),
        "x": np.asarray(props["x"]),
        "edges": np.asarray(edges, dtype=np.int64),
    }


def nodes_by_time(geff: dict[str, np.ndarray]) -> dict[int, dict[str, np.ndarray]]:
    grouped: dict[int, dict[str, list[Any]]] = {}
    for i, t in enumerate(geff["t"]):
        slot = grouped.setdefault(int(t), {"node_id": [], "z": [], "y": [], "x": []})
        slot["node_id"].append(geff["node_id"][i])
        slot["z"].append(geff["z"][i])
        slot["y"].append(geff["y"][i])
        slot["x"].append(geff["x"][i])
    out: dict[int, dict[str, np.ndarray]] = {}
    for t, slot in grouped.items():
        out[t] = {k: np.asarray(v) for k, v in slot.items()}
    return out


def transition_matrix(ids_t: np.ndarray, ids_t1: np.ndarray, edges: np.ndarray) -> np.ndarray:
    row = {int(n): i for i, n in enumerate(ids_t)}
    col = {int(n): i for i, n in enumerate(ids_t1)}
    matrix = np.zeros((len(ids_t), len(ids_t1)), dtype=np.float32)
    if edges.size == 0:
        return matrix
    for src, tgt in edges:
        if int(src) in row and int(tgt) in col:
            matrix[row[int(src)], col[int(tgt)]] = 1.0
    return matrix


def downsample_frame(frame: np.ndarray, downsample: Sequence[int]) -> np.ndarray:
    dz, dy, dx = (int(x) for x in downsample)
    return np.asarray(frame)[::dz, ::dy, ::dx]


def extract_pos_features(coords: np.ndarray, image_shape: Sequence[int]) -> np.ndarray:
    t, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    norms = [c / max(s, 1) for c, s in zip([t, z, y, x], image_shape)]

    def _embed(vals: np.ndarray) -> np.ndarray:
        freqs = 2 ** np.arange(_POS_EMBED_DIM // 2)
        angles = vals[:, None] * freqs * np.pi
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)

    return np.concatenate([_embed(n) for n in norms], axis=1).astype(np.float32)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def assert_no_submission_csv(working: Path) -> None:
    banned = working / "submission.csv"
    if banned.exists():
        banned.unlink()
        raise RuntimeError("learned_unet_v1 must not write submission.csv")


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
    from torch.utils.checkpoint import checkpoint as grad_ckpt

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

        def index_features(self, feat_maps, coords, mask):
            b, c = feat_maps.shape[:2]
            spatial = feat_maps.shape[2:]
            max_nodes = coords.shape[1]
            out = torch.zeros(b, max_nodes, c, device=feat_maps.device, dtype=feat_maps.dtype)
            for bi in range(b):
                nt = int(mask[bi].sum().item())
                if nt == 0:
                    continue
                z = coords[bi, :nt, 0].long().clamp(0, spatial[0] - 1)
                y = coords[bi, :nt, 1].long().clamp(0, spatial[1] - 1)
                x = coords[bi, :nt, 2].long().clamp(0, spatial[2] - 1)
                out[bi, :nt] = feat_maps[bi, :, z, y, x].T
            return out

    def detection_loss(det_logits, coords, mask, neg_weight: float):
        logits = det_logits[:, 0]
        target = torch.zeros_like(logits)
        spatial = logits.shape[1:]
        for b in range(logits.shape[0]):
            nt = int(mask[b].sum().item())
            if nt <= 0:
                continue
            gt = coords[b, :nt]
            zi = gt[:, 0].long().clamp(0, spatial[0] - 1)
            yi = gt[:, 1].long().clamp(0, spatial[1] - 1)
            xi = gt[:, 2].long().clamp(0, spatial[2] - 1)
            target[b, zi, yi, xi] = 1.0
        n_pos = target.reshape(target.shape[0], -1).sum(dim=1).clamp(min=1)
        n_neg = (target[0].numel() - n_pos).clamp(min=1)
        shape = (target.shape[0],) + (1,) * len(spatial)
        weight = torch.where(target == 1.0, (1.0 / n_pos).reshape(shape), (neg_weight / n_neg).reshape(shape))
        return F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="sum") / target.shape[0]

    def edge_loss(logits, target):
        active_rows = target.sum(dim=1) > 0
        active_cols = target.sum(dim=0) > 0
        mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
        if not mask.any():
            return logits.sum() * 0.0
        probs = torch.softmax(logits, dim=0)
        bce = F.binary_cross_entropy(probs, target, reduction="none")
        p_t = probs * target + (1 - probs) * (1 - target)
        loss = ((1 - p_t) ** 2) * bce
        return loss[mask].mean()

    return UNetNodeTransformer, detection_loss, edge_loss


def iter_windows(geff: dict[str, np.ndarray], downsample: Sequence[int], t_count: int):
    grouped = nodes_by_time(geff)
    ds = np.asarray(downsample, dtype=np.float32)
    for t in range(int(t_count) - 1):
        if t not in grouped or (t + 1) not in grouped:
            continue
        a = grouped[t]
        b = grouped[t + 1]
        coords_t = np.stack([a["z"], a["y"], a["x"]], axis=1).astype(np.float32) / ds
        coords_t1 = np.stack([b["z"], b["y"], b["x"]], axis=1).astype(np.float32) / ds
        target = transition_matrix(a["node_id"], b["node_id"], geff["edges"])
        yield t, coords_t, coords_t1, target


def train_on_splits(
    pairs: dict[str, dict[str, str]],
    splits: dict[str, Any],
    cfg: dict[str, Any],
    working: Path,
    started: float,
) -> dict[str, Any]:
    import torch

    UNetNodeTransformer, detection_loss, edge_loss = _build_models(cfg)
    device = torch.device("cuda")
    torch.manual_seed(int(cfg["seed"]))
    model = UNetNodeTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    downsample = list(cfg["downsample"])
    accum = int(cfg["grad_accum"])
    max_seconds = float(cfg["max_hours"]) * 3600.0
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    stopped_early = False
    last_epoch = -1

    def timed_out() -> bool:
        return (time.time() - started) >= max_seconds

    def save(name: str) -> None:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": cfg,
                "config_hash": config_hash(cfg),
                "splits": {"train": splits["train"], "test": splits["test"]},
            },
            working / name,
        )

    for epoch in range(int(cfg["epochs"])):
        if timed_out():
            stopped_early = True
            break
        model.train()
        optimizer.zero_grad()
        running = 0.0
        n_steps = 0
        n_windows = 0
        for stem in splits["train"]:
            if timed_out():
                stopped_early = True
                break
            refuse_holdout_stems([stem])
            pair = pairs[stem]
            geff = load_geff_arrays(pair["geff"])
            volume = open_volume_array(pair["zarr"])
            t_count = int(getattr(volume, "shape", [100])[0])
            for t, coords_t, coords_t1, target in iter_windows(geff, downsample, t_count):
                if timed_out():
                    stopped_early = True
                    break
                frame0 = downsample_frame(volume[t], downsample).astype(np.float32)
                frame1 = downsample_frame(volume[t + 1], downsample).astype(np.float32)
                lo = float(min(frame0.min(), frame1.min()))
                hi = float(max(frame0.max(), frame1.max()))
                frame0 = (frame0 - lo) / (hi - lo + 1e-6)
                frame1 = (frame1 - lo) / (hi - lo + 1e-6)
                imgs = torch.from_numpy(np.stack([frame0, frame1], axis=0)).unsqueeze(0).to(device)
                nt = coords_t.shape[0]
                nt1 = coords_t1.shape[0]
                coords = torch.zeros(1, 2, max(nt, nt1), 3, device=device)
                masks = torch.zeros(1, 2, max(nt, nt1), dtype=torch.bool, device=device)
                coords[0, 0, :nt] = torch.from_numpy(coords_t).to(device)
                coords[0, 1, :nt1] = torch.from_numpy(coords_t1).to(device)
                masks[0, 0, :nt] = True
                masks[0, 1, :nt1] = True
                unet_out, det_logits = model.encode(imgs)
                dloss = 0.5 * (
                    detection_loss(det_logits[0], coords[:, 0], masks[:, 0], float(cfg["det_neg_weight"]))
                    + detection_loss(det_logits[1], coords[:, 1], masks[:, 1], float(cfg["det_neg_weight"]))
                )
                feat0 = model.index_features(unet_out[:, 0], coords[:, 0], masks[:, 0])[:, :nt]
                feat1 = model.index_features(unet_out[:, 1], coords[:, 1], masks[:, 1])[:, :nt1]
                shape = (t_count, *frame0.shape)
                pos0 = torch.from_numpy(
                    extract_pos_features(
                        np.column_stack([np.full(nt, t, dtype=np.float32), coords_t]),
                        shape,
                    )
                ).to(device).unsqueeze(0)
                pos1 = torch.from_numpy(
                    extract_pos_features(
                        np.column_stack([np.full(nt1, t + 1, dtype=np.float32), coords_t1]),
                        shape,
                    )
                ).to(device).unsqueeze(0)
                logits = model.transformer(
                    torch.cat([feat0, pos0], dim=-1),
                    torch.cat([feat1, pos1], dim=-1),
                    coords[:, 0, :nt],
                    coords[:, 1, :nt1],
                    masks[:, 0, :nt],
                    masks[:, 1, :nt1],
                )[0]
                eloss = edge_loss(logits, torch.from_numpy(target).to(device))
                loss = (eloss + float(cfg["det_loss_weight"]) * dloss) / accum
                loss.backward()
                running += float(loss.item()) * accum
                n_windows += 1
                n_steps += 1
                if n_steps % accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            if stopped_early:
                break
        if n_steps % accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        epoch_loss = running / max(n_windows, 1)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "n_windows": n_windows})
        last_epoch = epoch
        if epoch_loss <= best_loss:
            best_loss = epoch_loss
            save("learned_unet_v1_best.pt")
        save("learned_unet_v1_last.pt")
        print(
            f"epoch {epoch} loss={epoch_loss:.4f} windows={n_windows} params={n_params}",
            flush=True,
        )
        if stopped_early:
            break

    return {
        "n_params": n_params,
        "best_train_loss": None if best_loss == float("inf") else best_loss,
        "history": history,
        "last_epoch": last_epoch,
        "stopped_early_time_budget": stopped_early,
        "wrote_best": (working / "learned_unet_v1_best.pt").exists(),
        "wrote_last": (working / "learned_unet_v1_last.pt").exists(),
    }


def main() -> dict[str, Any]:
    started = time.time()
    cfg = config_payload()
    working = output_dir()
    working.mkdir(parents=True, exist_ok=True)
    assert_no_submission_csv(working)
    runtime = probe_runtime()
    manifest: dict[str, Any] = {
        "status": "starting",
        "label": CONFIG_LABEL,
        "config_hash": config_hash(cfg),
        "config": cfg,
        "environment": None,
        "is_kaggle_run": False,
        "is_competition_submission": False,
        "runtime": runtime,
        "notes": "Not a competition submission. Not a leaderboard score.",
    }
    write_manifest(working / "train_manifest.json", manifest)
    try:
        root = find_root()
    except FileNotFoundError as exc:
        manifest.update({"status": "blocked_no_train_dir", "error": str(exc)})
        write_manifest(working / "train_manifest.json", manifest)
        return manifest
    manifest["environment"] = environment_label(root)
    manifest["is_kaggle_run"] = manifest["environment"] == "kaggle_input"
    train_dir = root / "train"
    pairs_list = list_tuning_pairs(train_dir)
    splits = build_44b6_splits([p["stem"] for p in pairs_list])
    pairs = {p["stem"]: p for p in pairs_list}
    refuse_holdout_stems(splits["train"] + splits["test"])
    (working / "dataset_splits.json").write_text(json.dumps(splits, indent=2))
    (working / "learned_unet_v1_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
    manifest["splits"] = {"n_train": splits["n_train"], "n_test": splits["n_test"], "test": splits["test"]}
    manifest["status"] = "splits_ready"
    write_manifest(working / "train_manifest.json", manifest)
    if os.environ.get("BIOHUB_LEARNED_DRY_RUN") == "1":
        manifest.update({"status": "dry_run", "runtime_sec": time.time() - started})
        write_manifest(working / "train_manifest.json", manifest)
        assert_no_submission_csv(working)
        return manifest
    if not runtime.get("torch"):
        manifest.update({"status": "blocked_no_torch", "runtime_sec": time.time() - started})
        write_manifest(working / "train_manifest.json", manifest)
        return manifest
    if not runtime.get("cuda_available"):
        manifest.update({"status": "blocked_no_gpu", "runtime_sec": time.time() - started})
        write_manifest(working / "train_manifest.json", manifest)
        return manifest
    try:
        train_stats = train_on_splits(pairs, splits, cfg, working, started)
    except Exception as exc:
        manifest.update(
            {
                "status": "train_error",
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
                "runtime_sec": time.time() - started,
                "peak_memory_mb": peak_memory_mb(),
            }
        )
        write_manifest(working / "train_manifest.json", manifest)
        assert_no_submission_csv(working)
        raise
    manifest.update(train_stats)
    manifest["status"] = "trained" if train_stats.get("wrote_last") else "train_incomplete"
    manifest["runtime_sec"] = time.time() - started
    manifest["peak_memory_mb"] = peak_memory_mb()
    write_manifest(working / "train_manifest.json", manifest)
    assert_no_submission_csv(working)
    return manifest


if __name__ == "__main__":
    result = main()
    print(json.dumps({k: result.get(k) for k in ("status", "config_hash", "environment", "splits")}, indent=2, default=str))
    print("NOTE: private helper kernel, not a competition submission.")
