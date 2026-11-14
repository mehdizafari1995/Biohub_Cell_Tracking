"""Minimal Zarr v3 reader/writer that does not require the ``zarr`` package.

Used by the learned_unet_v1 kernel because the Kaggle CPU image has no ``zarr``.
Image volumes and GEFF arrays from this competition use Zarr v3 + bytes/blosc.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


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
    except Exception as exc:
        errors.append(f"imagecodecs:{type(exc).__name__}")
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


def _decode_v2_chunk(raw: bytes, compressor: dict[str, Any] | None) -> bytes:
    if not compressor:
        return raw
    cid = str(compressor.get("id") or compressor.get("name") or "")
    if cid in {"blosc", "blosc2"}:
        return decode_blosc(raw)
    raise ValueError(f"unsupported zarr v2 compressor {cid!r}")


class ZarrV2Array:
    """Read a Zarr v2 array (GEFF default format) without the zarr package."""

    def __init__(self, array_dir: str | Path):
        self.array_dir = Path(array_dir)
        meta = json.loads((self.array_dir / ".zarray").read_text())
        self.meta = meta
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
            raw = _decode_v2_chunk(path.read_bytes(), self.compressor)
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
    """Read a Zarr v2 or v3 array. Prefer the zarr package when present."""
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


def write_array_v2(array_dir: str | Path, data: np.ndarray) -> Path:
    data = np.ascontiguousarray(data)
    array_dir = Path(array_dir)
    array_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "zarr_format": 2,
        "shape": list(int(x) for x in data.shape),
        "chunks": list(int(x) for x in data.shape) if data.shape else [1],
        "dtype": str(data.dtype.str),
        "compressor": None,
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "dimension_separator": "/",
    }
    (array_dir / ".zarray").write_text(json.dumps(meta, indent=2))
    chunk = array_dir.joinpath(*(["0"] * max(data.ndim, 1)))
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk.write_bytes(data.tobytes())
    return array_dir


class ZarrV3Array:
    """Read a Zarr v3 array (any rank) from chunk files."""

    def __init__(self, array_dir: str | Path):
        self.array_dir = Path(array_dir)
        meta = json.loads((self.array_dir / "zarr.json").read_text())
        self.meta = meta
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

    def _n_chunks(self) -> tuple[int, ...]:
        return tuple(int(np.ceil(s / c)) for s, c in zip(self.shape, self.chunk_shape))

    def __getitem__(self, indexer):  # noqa: ANN001
        if isinstance(indexer, int) and len(self.shape) >= 1:
            return self._take_first_axis(int(indexer))
        if indexer in (slice(None), Ellipsis):
            return self.read()
        raise TypeError(f"unsupported indexer {indexer!r}")

    def _take_first_axis(self, index: int) -> np.ndarray:
        if index < 0 or index >= self.shape[0]:
            raise IndexError(index)
        full = self.read()
        return full[index]

    def read(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=self.dtype)
        n_chunks = self._n_chunks()
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
                # Some writers store the full chunk including padding.
                chunk = arr.reshape(self.chunk_shape)
                local = tuple(slice(0, b - a) for a, b in zip(starts, stops))
                out[sl] = chunk[local]
            else:
                out[sl] = arr.reshape([b - a for a, b in zip(starts, stops)])
        return out


class ZarrV3FrameArray(ZarrV3Array):
    """TZYX volume reader: one timepoint chunk, matching association_v2."""

    def _chunk_path_t(self, t: int) -> Path:
        key = self.separator.join(["c", str(int(t)), "0", "0", "0"])
        return self.array_dir / key

    def __getitem__(self, t: int) -> np.ndarray:
        path = self._chunk_path_t(int(t))
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
    try:
        import zarr

        return zarr.open_group(path, mode="r")["0"]
    except Exception:
        return ZarrV3FrameArray(path / "0")


def write_array(array_dir: str | Path, data: np.ndarray) -> Path:
    """Write an uncompressed Zarr v3 array (tests / tiny GEFFs)."""
    data = np.asarray(data)
    array_dir = Path(array_dir)
    array_dir.mkdir(parents=True, exist_ok=True)
    chunk_shape = tuple(int(x) for x in data.shape) if data.shape else (1,)
    if not chunk_shape:
        chunk_shape = (1,)
    meta = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": list(int(x) for x in data.shape),
        "data_type": str(data.dtype),
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(chunk_shape)},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": 0,
    }
    (array_dir / "zarr.json").write_text(json.dumps(meta, indent=2))
    chunk_path = array_dir.joinpath("c", *[ "0" for _ in data.shape] or ["0"])
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_bytes(np.ascontiguousarray(data).tobytes())
    return array_dir


def load_geff_arrays(geff_path: str | Path) -> dict[str, np.ndarray]:
    """Load GEFF node/edge arrays without requiring the ``geff`` package."""
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
    extras: dict[str, Any] = {}
    for meta_name in ("zarr.json", ".zattrs"):
        group_meta = root / meta_name
        if not group_meta.exists():
            continue
        try:
            payload = json.loads(group_meta.read_text())
            geff_meta = (payload.get("attributes") or {}).get("geff") or payload.get("geff") or {}
            extras = (geff_meta.get("extra") or {}) if isinstance(geff_meta, dict) else {}
            if extras:
                break
        except Exception:
            extras = {}
    return {
        "node_id": np.asarray(node_ids),
        "t": np.asarray(props["t"]),
        "z": np.asarray(props["z"]),
        "y": np.asarray(props["y"]),
        "x": np.asarray(props["x"]),
        "edges": np.asarray(edges, dtype=np.int64),
        "estimated_number_of_nodes": extras.get("estimated_number_of_nodes"),
    }


def write_geff_arrays(
    geff_path: str | Path,
    *,
    node_id: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    edges: np.ndarray,
    estimated_number_of_nodes: float | None = None,
) -> Path:
    root = Path(geff_path)
    root.mkdir(parents=True, exist_ok=True)
    write_array(root / "nodes" / "ids", np.asarray(node_id))
    for key, values in {"t": t, "z": z, "y": y, "x": x}.items():
        write_array(root / "nodes" / "props" / key / "values", np.asarray(values))
    write_array(root / "edges" / "ids", np.asarray(edges, dtype=np.int64))
    attrs = {"geff": {"extra": {}}}
    if estimated_number_of_nodes is not None:
        attrs["geff"]["extra"]["estimated_number_of_nodes"] = float(estimated_number_of_nodes)
    (root / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": attrs}, indent=2)
    )
    return root
