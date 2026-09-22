"""
Kaggle helper kernel: learned_unet_retrain_orgop_v1 GPU inference.

Loads specimen-separated learned_unet_retrain_v1 weights and predicts only
the frozen-tuning pair 44b6_12dfb391 + 44b6_267148e4 using organizer
defaults on the continuation weights: det 0.99, pool 3.0 µm, flip TTA,
transformer softmax 0.5, tracksdata ILP (-1.0 / 0.1 / 0.1 / 1.0).
Never opens 6bba. Never writes submission.csv. Not a competition
submission. Do not attach public 199-video weights.

Attach competition `biohub-cell-tracking-during-development` and the
private retrain helper `mahdimzafari/biohub-learned-unet-retrain-v1`.
Internet OFF. GPU ON. Do not overwrite locked slugs.
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

CONFIG_LABEL = "learned_unet_retrain_orgop_v1"
TRAIN_LABEL = "learned_unet_v1"
INFER_LABEL = "learned_unet_infer_v1"
TRAIN_CONFIG_HASH = "4c38035d6657f2ad"
PARENT_RETRAIN_HASH = "92a419926fa7e40b"
INFER_CONFIG_HASH = "d7578b5f64ddac08"
LOCKED_ORGOP_HASH = "250c5bfacd4a8198"
TUNING_SPECIMEN = "44b6"
HOLDOUT_SPECIMENS = ("6bba",)
INFER_STEMS = ("44b6_12dfb391", "44b6_267148e4")
FROZEN_HOLDOUT_STEMS = ("6bba_05db0fb1", "6bba_c328f2fd")
KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-retrain-orgop-v1"
TRAIN_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-retrain-v1"
INFER_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-infer-v1"
WEIGHTS_NAME = "learned_unet_retrain_v1_best.pt"
FALLBACK_WEIGHTS_NAME = "learned_unet_retrain_v1_last.pt"
SCALE_ZYX = (1.625, 0.40625, 0.40625)
_POS_EMBED_DIM = 8
PAIR_CHUNK = 256
LOCKED_SLUGS = {
    "mahdimzafari/biohub-learned-unet-v1",
    "mahdimzafari/biohub-learned-unet-retrain-v1",
    "mahdimzafari/biohub-learned-unet-infer-v1",
    "mahdimzafari/biohub-learned-unet-holdout-v1",
    "mahdimzafari/biohub-learned-unet-test-v1",
    "mahdimzafari/biohub-learned-unet-gap-test-v1",
    "mahdimzafari/biohub-learned-unet-prune-test-v1",
    "mahdimzafari/biohub-learned-unet-link-v1",
    "mahdimzafari/biohub-learned-unet-link-v2",
}
BANNED_PUBLIC_WEIGHT_MARKERS = (
    "pilkwang/biohub-tracking-support-pack",
    "pilkwang/biohub-temporal-unet3d",
    "pilkwang/biohub-deepcenter",
    "reyhanksatria/biohub-tracking-support-pack",
    "reyhanksatria/biohub-deepcenter",
    "reyhanksatria/biohub-temporalunet",
    "thibautgoldsborough/cellmot-baseline-artifacts",
    "giorgosi/biohub-divnet",
    "biohub-tracking-support-pack-50ep",
    "biohub-temporal-unet3d-seed314159",
    "biohub-deepcenter-unet3d-center-prior",
    "edge_predictor_best.pth",
    "checkpoint_last.pth",
)
COMPETITION = "biohub-cell-tracking-during-development"
CANDIDATE_ROOTS = [
    Path("/kaggle/input") / COMPETITION,
    Path("/kaggle/input/competitions") / COMPETITION,
    Path("data"),
]
if os.environ.get("BIOHUB_DATA_ROOT"):
    CANDIDATE_ROOTS.insert(0, Path(os.environ["BIOHUB_DATA_ROOT"]))


class HoldoutLeakError(ValueError):
    """Raised when a frozen-holdout specimen would enter inference."""


def orgop_config_payload() -> dict[str, Any]:
    """Locked lane C payload. Hash f924802e5f716a61. Do not add keys."""
    return {
        "label": CONFIG_LABEL,
        "lane": "C",
        "parent_retrain_hash": PARENT_RETRAIN_HASH,
        "parent_train_hash": TRAIN_CONFIG_HASH,
        "parent_orgop_hash": "250c5bfacd4a8198",
        "comparator": "lane_B_official_combined",
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "infer_stems": list(INFER_STEMS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "weights_name": WEIGHTS_NAME,
        "fallback_weights_name": FALLBACK_WEIGHTS_NAME,
        "det_threshold": 0.99,
        "det_tta": True,
        "pool_kernel_um": 3.0,
        "edge_activation": "softmax",
        "edge_threshold": 0.5,
        "linker": "tracksdata_ilp",
        "use_ilp": True,
        "ilp_edge_weight": -1.0,
        "ilp_appearance_weight": 0.1,
        "ilp_disappearance_weight": 0.1,
        "ilp_division_weight": 1.0,
        "uses_transformer": True,
        "writes_submission_csv": False,
        "opens_6bba": False,
        "kernel_slug": KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
    }


def orgop_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or orgop_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_LANE_HASH = orgop_config_hash()
assert LOCKED_LANE_HASH == "f924802e5f716a61"


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
            "learned_unet_orgop_v1 refuses holdout stems: " + ", ".join(leaked)
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
    refuse_holdout_stems(cleaned)
    extra = [s for s in cleaned if s not in INFER_STEMS]
    if extra:
        raise HoldoutLeakError(
            "learned_unet_orgop_v1 only opens frozen-tuning stems; got " + ", ".join(extra)
        )


def assert_no_submission_csv(working: Path) -> None:
    banned = working / "submission.csv"
    if banned.exists():
        banned.unlink()
        raise RuntimeError("learned_unet_orgop_v1 must not write submission.csv")


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


def extract_pos_features(coords: np.ndarray, image_shape: Sequence[int]) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 4)
    if coords.size == 0:
        return np.empty((0, 4 * _POS_EMBED_DIM), dtype=np.float32)
    t, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    norms = [c / max(float(s), 1.0) for c, s in zip([t, z, y, x], image_shape)]

    def _embed(vals: np.ndarray) -> np.ndarray:
        freqs = 2 ** np.arange(_POS_EMBED_DIM // 2)
        angles = vals[:, None] * freqs * np.pi
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)

    return np.concatenate([_embed(n) for n in norms], axis=1).astype(np.float32)


def pairwise_distances_um(
    src_zyx: np.ndarray,
    tgt_zyx: np.ndarray,
    scale_zyx: Sequence[float] = SCALE_ZYX,
) -> np.ndarray:
    src = np.asarray(src_zyx, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(tgt_zyx, dtype=np.float64).reshape(-1, 3)
    if src.size == 0 or tgt.size == 0:
        return np.empty((src.shape[0], tgt.shape[0]), dtype=np.float64)
    scale = np.asarray(scale_zyx, dtype=np.float64).reshape(1, 3)
    src_um = src * scale
    tgt_um = tgt * scale
    diff = src_um[:, None, :] - tgt_um[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def assign_transformer_links(
    scores: np.ndarray,
    distances_um: np.ndarray | None = None,
    max_link_um: float = 12.0,
    max_parents_per_node: int = 1,
    max_children_per_node: int = 1,
) -> list[tuple[int, int, float]]:
    """Greedy 1-1 on descending scores. No probability cutoff."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        return []
    n_src, n_tgt = values.shape
    if distances_um is None:
        allowed = np.ones((n_src, n_tgt), dtype=bool)
    else:
        dist = np.asarray(distances_um, dtype=np.float64)
        if dist.shape != values.shape:
            raise ValueError("distances_um must match scores shape")
        allowed = dist <= float(max_link_um)
    i_idx, j_idx = np.nonzero(allowed)
    if i_idx.size == 0:
        return []
    order = np.argsort(-values[i_idx, j_idx], kind="stable")
    children_count: dict[int, int] = {}
    parents_count: dict[int, int] = {}
    edges: list[tuple[int, int, float]] = []
    for idx in order:
        i = int(i_idx[idx])
        j = int(j_idx[idx])
        if children_count.get(i, 0) >= int(max_children_per_node):
            continue
        if parents_count.get(j, 0) >= int(max_parents_per_node):
            continue
        edges.append((i, j, float(values[i, j])))
        children_count[i] = children_count.get(i, 0) + 1
        parents_count[j] = parents_count.get(j, 0) + 1
    return edges


def graph_payload(dataset: str, coords: np.ndarray, edges: Sequence[Sequence[float]]) -> dict[str, Any]:
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 4)
    node_id = np.arange(1, len(coords) + 1, dtype=np.int64)
    kept = []
    probs = []
    dists = []
    for edge in edges:
        kept.append([int(edge[0]), int(edge[1])])
        probs.append(float(edge[2]) if len(edge) > 2 else 0.0)
        dists.append(float(edge[3]) if len(edge) > 3 else 0.0)
    return {
        "dataset": dataset,
        "node_id": node_id.tolist(),
        "t": coords[:, 0].astype(np.int64).tolist() if len(coords) else [],
        "z": coords[:, 1].astype(np.int64).tolist() if len(coords) else [],
        "y": coords[:, 2].astype(np.int64).tolist() if len(coords) else [],
        "x": coords[:, 3].astype(np.int64).tolist() if len(coords) else [],
        "edges": kept,
        "edge_prob": probs,
        "edge_dist": dists,
        "candidate_edges": [list(edge) for edge in edges],
    }


def refuse_public_weight_sources(sources: Iterable[str]) -> None:
    cleaned = [str(s).strip().lower() for s in sources if str(s).strip()]
    hits = []
    for src in cleaned:
        for marker in BANNED_PUBLIC_WEIGHT_MARKERS:
            if marker.lower() in src:
                hits.append(src)
                break
    if hits:
        raise HoldoutLeakError(
            "learned_unet_orgop_v1 refuses public 199-video / DeepCenter / "
            "organizer-split / DivNet weights; got " + ", ".join(hits)
        )


def apply_tracksdata_ilp_notebook(payload: dict[str, Any], infer_cfg: Mapping[str, Any]) -> dict[str, Any]:
    import polars as pl
    import tracksdata as td

    refuse_non_infer_stems([str(payload["dataset"])])
    node_id = [int(x) for x in payload.get("node_id") or []]
    t = [int(x) for x in payload.get("t") or []]
    z = [float(x) for x in payload.get("z") or []]
    y = [float(x) for x in payload.get("y") or []]
    x = [float(x) for x in payload.get("x") or []]
    raw_edges = list(payload.get("candidate_edges") or [])
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    assigned = (
        graph.bulk_add_nodes(
            [{"t": ti, "z": zi, "y": yi, "x": xi} for ti, zi, yi, xi in zip(t, z, y, x, strict=True)]
        )
        if node_id
        else []
    )
    id_map = dict(zip(node_id, assigned, strict=True))
    parsed: list[tuple[int, int, float, float]] = []
    for edge in raw_edges:
        src, tgt = int(edge[0]), int(edge[1])
        prob = float(edge[2]) if len(edge) > 2 else 0.0
        dist = float(edge[3]) if len(edge) > 3 else 0.0
        parsed.append((src, tgt, prob, dist))
    if parsed:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        graph.bulk_add_edges(
            [
                {
                    "source_id": id_map[src],
                    "target_id": id_map[tgt],
                    "edge_prob": prob,
                    "edge_dist": dist,
                }
                for src, tgt, prob, dist in parsed
            ]
        )
    if graph.num_edges() == 0:
        payload["solver_status"] = "ilp_skipped_no_candidates"
        payload["n_candidate_edges"] = 0
        return payload
    solver = td.solvers.ILPSolver(
        edge_weight=float(infer_cfg["ilp_edge_weight"]) * td.EdgeAttr("edge_prob"),
        appearance_weight=float(infer_cfg["ilp_appearance_weight"]),
        disappearance_weight=float(infer_cfg["ilp_disappearance_weight"]),
        division_weight=float(infer_cfg["ilp_division_weight"]),
    )
    solved = solver.solve(graph)
    if solved is None:
        raise RuntimeError("tracksdata ILPSolver returned no solution")
    rev = {int(v): int(k) for k, v in id_map.items()}
    edge_df = solved.edge_attrs(attr_keys=["edge_prob", "edge_dist"])
    src_col = "source_id" if "source_id" in edge_df.columns else edge_df.columns[0]
    tgt_col = "target_id" if "target_id" in edge_df.columns else edge_df.columns[1]
    kept = []
    probs = []
    dists = []
    for row in edge_df.iter_rows(named=True):
        kept.append([rev[int(row[src_col])], rev[int(row[tgt_col])]])
        kept_prob = float(row["edge_prob"]) if "edge_prob" in row else 0.0
        kept_dist = float(row["edge_dist"]) if "edge_dist" in row else 0.0
        probs.append(kept_prob)
        dists.append(kept_dist)
    payload["edges"] = kept
    payload["edge_prob"] = probs
    payload["edge_dist"] = dists
    payload["solver_status"] = "tracksdata_ilp"
    payload["n_candidate_edges"] = len(parsed)
    return payload


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
            Path("/kaggle/input/biohub-learned-unet-retrain-v1"),
            Path("/kaggle/input/biohub-learned-unet-retrain-v1/biohub-learned-unet-retrain-v1"),
            Path("/kaggle/input/mahdimzafari-biohub-learned-unet-retrain-v1"),
            Path("/kaggle/input/mahdimzafari/biohub-learned-unet-retrain-v1"),
            Path("/kaggle/input/notebooks/mahdimzafari/biohub-learned-unet-retrain-v1"),
            Path("/kaggle/input/notebooks/biohub-learned-unet-retrain-v1"),
            Path("/kaggle/input/notebooks/mahdimzafari-biohub-learned-unet-retrain-v1"),
            Path("/kaggle/input/notebooks"),
            Path("artifacts/kaggle/learned_unet_retrain_v1"),
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
            refuse_public_weight_sources([str(found), str(root)])
            if found.name not in {WEIGHTS_NAME, FALLBACK_WEIGHTS_NAME}:
                raise HoldoutLeakError(
                    "learned_unet_orgop_v1 only loads locked 44b6 weights; got "
                    + found.name
                )
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



def verify_retrain_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("config_hash") != PARENT_RETRAIN_HASH:
        raise HoldoutLeakError("retrain config hash does not match locked learned_unet_retrain_v1")
    if payload.get("status") != "trained":
        raise HoldoutLeakError(f"retrain helper status is {payload.get('status')!r}")
    if payload.get("is_competition_submission"):
        raise HoldoutLeakError("retrain helper must not be a competition submission")
    splits = payload.get("splits") or {}
    stems = list(splits.get("train") or []) + list(splits.get("test") or [])
    if stems:
        refuse_holdout_stems(stems)
    resumed = str(payload.get("resumed_from") or "")
    if resumed and Path(resumed).name != "learned_unet_v1_last.pt":
        raise HoldoutLeakError("retrain helper must resume locked learned_unet_v1_last.pt")
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

    return UNetNodeTransformer


def load_checkpoint(weights_path: Path, device):
    import torch

    blob = torch.load(weights_path, map_location=device, weights_only=False)
    cfg = dict(blob.get("config") or {})
    allowed_hashes = {TRAIN_CONFIG_HASH, PARENT_RETRAIN_HASH}
    if blob.get("config_hash") and blob.get("config_hash") not in allowed_hashes:
        raise HoldoutLeakError(
            "checkpoint config_hash is not locked learned_unet_v1 or learned_unet_retrain_v1"
        )
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


def _transformer_pair_logits(model, feat_t, feat_t1, coords_t, coords_t1, device, chunk: int = PAIR_CHUNK):
    import torch

    n_src = int(feat_t.shape[1])
    n_tgt = int(feat_t1.shape[1])
    if n_src == 0 or n_tgt == 0:
        return np.empty((n_src, n_tgt), dtype=np.float32)
    q = model.transformer.norm_in(model.transformer.proj(feat_t))
    k = model.transformer.norm_in(model.transformer.proj(feat_t1))
    for block in model.transformer.blocks:
        q = block(q, k)
        k = block(k, q)
    q = model.transformer.norm_out(q)
    k = model.transformer.norm_out(k)
    chunks: list[Any] = []
    for start in range(0, n_src, int(chunk)):
        stop = min(start + int(chunk), n_src)
        qe = q[:, start:stop].unsqueeze(2).expand(-1, -1, n_tgt, -1)
        ke = k.unsqueeze(1).expand(-1, stop - start, -1, -1)
        rel = (coords_t[:, start:stop].unsqueeze(2) - coords_t1.unsqueeze(1)) / 100.0
        logits = model.transformer.pair_mlp(torch.cat([qe, ke, rel], dim=-1)).squeeze(-1)
        chunks.append(logits[0].detach().float().cpu())
        del qe, ke, rel, logits
    return torch.cat(chunks, dim=0).numpy()


def predict_volume(model, volume, cfg: dict[str, Any], infer_cfg: dict[str, Any], device) -> tuple[np.ndarray, list[tuple[int, int]]]:
    import torch

    downsample = list(infer_cfg.get("downsample") or [1, 4, 4])
    t_count = int(getattr(volume, "shape", [100])[0])
    voxel_down = tuple(s * d for s, d in zip(SCALE_ZYX, downsample))
    pool_k = pool_kernel_from_um(float(infer_cfg["pool_kernel_um"]), voxel_down)
    seen: dict[int, np.ndarray] = {}
    edges: list[tuple[int, int]] = []
    offsets: dict[int, tuple[int, int]] = {}
    cursor = 0
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
            unet_out, det_logits = model.encode(imgs)
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
                offsets[frame_t] = (cursor, cursor + len(seen[frame_t]))
                cursor += len(seen[frame_t])
            down0 = seen.get(t)
            down1 = seen.get(t + 1)
            if down0 is None or down1 is None or len(down0) == 0 or len(down1) == 0:
                continue
            n0, n1 = len(down0), len(down1)
            coords0 = torch.from_numpy(down0[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            coords1 = torch.from_numpy(down1[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            mask0 = torch.ones(1, n0, dtype=torch.bool, device=device)
            mask1 = torch.ones(1, n1, dtype=torch.bool, device=device)
            feat0 = model.index_features(unet_out[:, 0], coords0, mask0)
            feat1 = model.index_features(unet_out[:, 1], coords1, mask1)
            shape = (t_count, *frame0.shape)
            pos0 = torch.from_numpy(
                extract_pos_features(
                    np.column_stack([np.full(n0, t, dtype=np.float32), down0[:, 1:]]),
                    shape,
                )
            ).to(device).unsqueeze(0)
            pos1 = torch.from_numpy(
                extract_pos_features(
                    np.column_stack([np.full(n1, t + 1, dtype=np.float32), down1[:, 1:]]),
                    shape,
                )
            ).to(device).unsqueeze(0)
            raw = _transformer_pair_logits(
                model,
                torch.cat([feat0, pos0], dim=-1),
                torch.cat([feat1, pos1], dim=-1),
                coords0,
                coords1,
                device,
            )
            if str(infer_cfg["edge_activation"]) == "softmax":
                raw = raw - raw.max(axis=0, keepdims=True)
                exp = np.exp(np.clip(raw, -40.0, 40.0))
                probs = exp / np.clip(exp.sum(axis=0, keepdims=True), 1e-8, None)
            else:
                probs = 1.0 / (1.0 + np.exp(-np.clip(raw, -40.0, 40.0)))
            orig0 = scale_coords_to_original(down0, downsample)
            orig1 = scale_coords_to_original(down1, downsample)
            dists = pairwise_distances_um(
                orig0[:, 1:], orig1[:, 1:], infer_cfg.get("scale_zyx") or list(SCALE_ZYX)
            )
            threshold = float(infer_cfg["edge_threshold"])
            s0, _e0 = offsets[t]
            s1, _e1 = offsets[t + 1]
            for i in range(n0):
                for j in range(n1):
                    if float(probs[i, j]) > threshold:
                        edges.append(
                            (
                                s0 + i + 1,
                                s1 + j + 1,
                                float(probs[i, j]),
                                float(dists[i, j]),
                            )
                        )

    ordered_t = sorted(seen)
    coord_lists = [scale_coords_to_original(seen[t], downsample) for t in ordered_t]
    coords = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
    return coords, edges


def infer_tuning_volumes(
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
        solver_status = "ilp_unavailable"
        try:
            payload = apply_tracksdata_ilp_notebook(payload, infer_cfg)
            solver_status = str(payload.get("solver_status") or "tracksdata_ilp")
        except Exception as exc:
            payload["solver_status"] = "pending_local_ilp"
            payload["solver_error"] = f"{type(exc).__name__}: {exc}"[:300]
            payload["n_candidate_edges"] = len(payload.get("candidate_edges") or [])
            solver_status = "pending_local_ilp"
        (graphs_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2))
        summaries.append(
            {
                "stem": stem,
                "status": "predicted",
                "n_nodes": len(payload["node_id"]),
                "n_edges": len(payload["edges"]),
                "n_candidate_edges": payload.get("n_candidate_edges", len(payload.get("candidate_edges") or [])),
                "solver_status": solver_status,
            }
        )
    return summaries


def main() -> dict[str, Any]:
    if KERNEL_SLUG in LOCKED_SLUGS:
        raise RuntimeError("refusing to overwrite locked learned slugs")
    started = time.time()
    cfg = orgop_config_payload()
    if orgop_config_hash(cfg) != LOCKED_LANE_HASH:
        raise RuntimeError("lane C config hash drifted; refuse to run")
    working = output_dir()
    working.mkdir(parents=True, exist_ok=True)
    assert_no_submission_csv(working)
    runtime = probe_runtime()
    manifest: dict[str, Any] = {
        "status": "starting",
        "label": CONFIG_LABEL,
        "config_hash": orgop_config_hash(cfg),
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": INFER_CONFIG_HASH,
        "config": cfg,
        "environment": None,
        "is_kaggle_run": False,
        "is_competition_submission": False,
        "runtime": runtime,
        "notes": (
            "Not a competition submission. Not a leaderboard score. "
            "44b6 tuning only. Organizer 0.99 + tracksdata ILP on locked "
            "learned_unet_v1 weights. No public 199-video checkpoints."
        ),
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
    train_manifest = weights.parent / "retrain_manifest.json"
    if train_manifest.exists():
        verify_retrain_manifest(train_manifest)
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
        summaries = infer_tuning_volumes(root / "train", model, train_cfg, cfg, working, device)
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
