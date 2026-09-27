"""
Kaggle test-time kernel: learned_unet_orgdet_fnn_fast_test_v1 GPU inference.

Same locked overlay as learned_unet_orgdet_fnn_test_v1 (hash 1af763e6ffc23188),
with a vectorized forward-NN scan. Do not overwrite fnn-test v1.

Loads locked learned_unet_v1 weights and the locked orgdet+gap+prune
operating point (det_threshold 0.99, pool 3.0 µm, flip TTA, greedy
12 µm last-position, locked association_v2 gap-close, then drop leftover
degree-0 nodes, then drop dt=1 edges longer than 8.0 µm, then keep only forward nearest-neighbor dt=1 edges). Discovers every test/*.zarr at runtime. Never opens
train/ volumes. Writes /kaggle/working/submission.csv and
run_manifest.json. Internet OFF. GPU ON.

Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or orgdet-holdout/orgdet-test/orgdet-gap-test/orgdet-elf-test/orgdet-fnn-test slugs. Do not retune parent prune hash
862e3e7f7ac0330e, elf hash 292c4f1f968f948d, or fnn hash 95e9300eb84ba71c. Do not score frozen 6bba holdout again. Do not
submit from the CLI. A local run is not a Kaggle execution or a
leaderboard score. Refuses public 199-video / DeepCenter weights.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

CONFIG_LABEL = "learned_unet_orgdet_fnn_fast_test_v1"
TRAIN_LABEL = "learned_unet_v1"
INFER_LABEL = "learned_unet_infer_v1"
ORGDET_LABEL = "learned_unet_orgdet_v1"
ORGDET_HOLDOUT_LABEL = "learned_unet_orgdet_holdout_v1"
ORGDET_GAP_LABEL = "learned_unet_orgdet_gap_v1"
ORGDET_GAP_PRUNE_LABEL = "learned_unet_orgdet_gap_prune_v1"
ORGDET_GAP_TEST_LABEL = "learned_unet_orgdet_gap_test_v1"
ORGDET_TEST_LABEL = "learned_unet_orgdet_test_v1"
ORGOP_LABEL = "learned_unet_orgop_v1"
TEST_LABEL = "learned_unet_test_v1"
GAP_TEST_LABEL = "learned_unet_gap_test_v1"
PRUNE_TEST_LABEL = "learned_unet_prune_test_v1"
ORGDET_ELF_LABEL = "learned_unet_orgdet_elf_v1"
ORGDET_FNN_LABEL = "learned_unet_orgdet_fnn_v1"
ORGDET_FNN_TEST_LABEL = "learned_unet_orgdet_fnn_test_v1"
ORGDET_GAP_PRUNE_TEST_LABEL = "learned_unet_orgdet_gap_prune_test_v1"
ORGDET_ELF_TEST_LABEL = "learned_unet_orgdet_elf_test_v1"
TRAIN_CONFIG_HASH = "4c38035d6657f2ad"
INFER_CONFIG_HASH = "d7578b5f64ddac08"
ORGDET_CONFIG_HASH = "80b20589198148b7"
ORGDET_HOLDOUT_CONFIG_HASH = "912f0c19b572a054"
ORGDET_GAP_CONFIG_HASH = "ee31f36151325f9a"
ORGDET_GAP_PRUNE_CONFIG_HASH = "862e3e7f7ac0330e"
ORGDET_GAP_TEST_PARENT_HASH = "1ce4837ae16bee52"
ORGDET_TEST_PARENT_HASH = "c45c859610d29e90"
REJECTED_ORGOP_HASH = "250c5bfacd4a8198"
TEST_PARENT_HASH = "7a8725b818fc609b"
GAP_TEST_PARENT_HASH = "85379c198ebc7241"
PRUNE_TEST_PARENT_HASH = "074a71dec68304e8"
ORGDET_ELF_CONFIG_HASH = "292c4f1f968f948d"
ORGDET_FNN_CONFIG_HASH = "95e9300eb84ba71c"
ORGDET_GAP_PRUNE_TEST_PARENT_HASH = "52a30a5843e0a880"
ORGDET_ELF_TEST_PARENT_HASH = "ad375d39e13098d1"
MAX_DT1_EDGE_UM = 8.0
NN_TIE_EPS = 1e-9
TUNING_SPECIMEN = "44b6"
HOLDOUT_SPECIMENS = ("6bba",)
KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-fnn-fast-test-v1"
ORGDET_FNN_TEST_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-fnn-test-v1"
TRAIN_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-v1"
INFER_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-infer-v1"
ORGDET_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-v1"
ORGDET_HOLDOUT_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-holdout-v1"
ORGDET_GAP_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-gap-v1"
ORGDET_GAP_TEST_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1"
ORGDET_TEST_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-test-v1"
ORGOP_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgop-v1"
TEST_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-test-v1"
GAP_TEST_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-gap-test-v1"
PRUNE_TEST_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-prune-test-v1"
ORGDET_GAP_PRUNE_TEST_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-gap-prune-test-v1"
ORGDET_ELF_TEST_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-elf-test-v1"
# parent fnn-test slug is ORGDET_FNN_TEST_KERNEL_SLUG
WEIGHTS_NAME = "learned_unet_v1_best.pt"
FALLBACK_WEIGHTS_NAME = "learned_unet_v1_last.pt"
SCALE_ZYX = (1.625, 0.40625, 0.40625)
ORGDET_HOLDOUT_MEASURED_SEC = 173.79521799087524
ORGDET_HOLDOUT_MEASURED_VOLUMES = 2
HOLDOUT_SEC_PER_VOLUME = ORGDET_HOLDOUT_MEASURED_SEC / ORGDET_HOLDOUT_MEASURED_VOLUMES
HOLDOUT_PEAK_MEMORY_MB = 1185.26171875
ORGDET_GAP_TEST_MEASURED_PUBLIC_SEC = 312.69036865234375
ORGDET_GAP_TEST_MEASURED_PUBLIC_VOLUMES = 4
ORGDET_GAP_TEST_MEASURED_PEAK_MEMORY_MB = 1176.0
PRUNE_OVERHEAD_SEC_PER_VOLUME = 0.5
ORGDET_GAP_TEST_SEC_PER_VOLUME = (
    ORGDET_GAP_TEST_MEASURED_PUBLIC_SEC / ORGDET_GAP_TEST_MEASURED_PUBLIC_VOLUMES
)
PRUNE_TEST_SEC_PER_VOLUME = ORGDET_GAP_TEST_SEC_PER_VOLUME + PRUNE_OVERHEAD_SEC_PER_VOLUME
ORGDET_ELF_TEST_MEASURED_PUBLIC_SEC = 352.8744547367096
ORGDET_ELF_TEST_MEASURED_PUBLIC_VOLUMES = 4
ORGDET_ELF_TEST_MEASURED_PEAK_MEMORY_MB = 1161.0859375
FNN_FAST_OVERHEAD_SEC_PER_VOLUME = 0.2
ORGDET_ELF_TEST_SEC_PER_VOLUME = (
    ORGDET_ELF_TEST_MEASURED_PUBLIC_SEC / ORGDET_ELF_TEST_MEASURED_PUBLIC_VOLUMES
)
FNN_FAST_TEST_SEC_PER_VOLUME = ORGDET_ELF_TEST_SEC_PER_VOLUME + FNN_FAST_OVERHEAD_SEC_PER_VOLUME
KAGGLE_GPU_LIMIT_HOURS = 12.0
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
_POS_EMBED_DIM = 8
COMPETITION = "biohub-cell-tracking-during-development"
COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]
CANDIDATE_ROOTS = [
    Path("/kaggle/input") / COMPETITION,
    Path("/kaggle/input/competitions") / COMPETITION,
    Path("data"),
]
if os.environ.get("BIOHUB_DATA_ROOT"):
    CANDIDATE_ROOTS.insert(0, Path(os.environ["BIOHUB_DATA_ROOT"]))


class HoldoutLeakError(ValueError):
    """Raised when training weights were not 44b6-only."""


class TrainVolumeError(ValueError):
    """Raised when a train volume would be opened at test time."""


def infer_config_payload() -> dict[str, Any]:
    return {
        "label": CONFIG_LABEL,
        "train_label": TRAIN_LABEL,
        "infer_label": INFER_LABEL,
        "orgdet_label": ORGDET_LABEL,
        "orgdet_holdout_label": ORGDET_HOLDOUT_LABEL,
        "orgdet_gap_label": ORGDET_GAP_LABEL,
        "orgdet_gap_prune_label": ORGDET_GAP_PRUNE_LABEL,
        "orgdet_gap_test_label": ORGDET_GAP_TEST_LABEL,
        "orgdet_test_label": ORGDET_TEST_LABEL,
        "orgop_label": ORGOP_LABEL,
        "test_label": TEST_LABEL,
        "gap_test_label": GAP_TEST_LABEL,
        "prune_test_label": PRUNE_TEST_LABEL,
        "orgdet_elf_label": ORGDET_ELF_LABEL,
        "orgdet_fnn_label": ORGDET_FNN_LABEL,
        "orgdet_fnn_test_label": ORGDET_FNN_TEST_LABEL,
        "orgdet_gap_prune_test_label": ORGDET_GAP_PRUNE_TEST_LABEL,
        "orgdet_elf_test_label": ORGDET_ELF_TEST_LABEL,
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": INFER_CONFIG_HASH,
        "orgdet_config_hash": ORGDET_CONFIG_HASH,
        "orgdet_holdout_config_hash": ORGDET_HOLDOUT_CONFIG_HASH,
        "orgdet_gap_config_hash": ORGDET_GAP_CONFIG_HASH,
        "orgdet_gap_prune_config_hash": ORGDET_GAP_PRUNE_CONFIG_HASH,
        "orgdet_elf_config_hash": ORGDET_ELF_CONFIG_HASH,
        "orgdet_fnn_config_hash": ORGDET_FNN_CONFIG_HASH,
        "orgdet_gap_prune_test_parent_hash": ORGDET_GAP_PRUNE_TEST_PARENT_HASH,
        "orgdet_elf_test_parent_hash": ORGDET_ELF_TEST_PARENT_HASH,
        "orgdet_gap_test_parent_hash": ORGDET_GAP_TEST_PARENT_HASH,
        "orgdet_test_parent_hash": ORGDET_TEST_PARENT_HASH,
        "rejected_orgop_hash": REJECTED_ORGOP_HASH,
        "test_parent_hash": TEST_PARENT_HASH,
        "gap_test_parent_hash": GAP_TEST_PARENT_HASH,
        "prune_test_parent_hash": PRUNE_TEST_PARENT_HASH,
        "orgdet_elf_config_hash": ORGDET_ELF_CONFIG_HASH,
        "orgdet_fnn_config_hash": ORGDET_FNN_CONFIG_HASH,
        "orgdet_fnn_test_parent_hash": "1af763e6ffc23188",
        "orgdet_gap_prune_test_parent_hash": ORGDET_GAP_PRUNE_TEST_PARENT_HASH,
        "orgdet_elf_test_parent_hash": ORGDET_ELF_TEST_PARENT_HASH,
        "discovers_test_zarrs": True,
        "hardcoded_test_stems": [],
        "opens_train_volumes": False,
        "det_threshold": 0.99,
        "det_tta": True,
        "pool_kernel_um": 3.0,
        "linker": "orgdet_elf_then_keep_forward_nn_dt1",
        "motion_model": "last_position",
        "gate_um": 12.0,
        "max_link_um": 12.0,
        "max_gap_frames": 2,
        "gap_scale": "sqrt_dt",
        "assignment": "greedy",
        "interpolate_gaps": True,
        "drop_degree0": True,
        "max_dt1_edge_um": MAX_DT1_EDGE_UM,
        "drop_long_dt1_edges": True,
        "keep_forward_nn_only": True,
        "drop_degree0_after": True,
        "nn_tie_eps": NN_TIE_EPS,
        "nn_impl": "vectorized",
        "max_parents_per_node": 1,
        "max_children_per_node": 1,
        "intensity_weight": 0.0,
        "downsample": [1, 4, 4],
        "normalize": "window_minmax",
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": True,
        "writes_run_manifest": True,
        "one_shot_holdout": False,
        "kernel_slug": KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
        "orgdet_kernel_slug": ORGDET_KERNEL_SLUG,
        "orgdet_holdout_kernel_slug": ORGDET_HOLDOUT_KERNEL_SLUG,
        "orgdet_gap_kernel_slug": ORGDET_GAP_KERNEL_SLUG,
        "orgdet_gap_test_kernel_slug": ORGDET_GAP_TEST_KERNEL_SLUG,
        "orgdet_test_kernel_slug": ORGDET_TEST_KERNEL_SLUG,
        "orgop_kernel_slug": ORGOP_KERNEL_SLUG,
        "test_kernel_slug": TEST_KERNEL_SLUG,
        "gap_test_kernel_slug": GAP_TEST_KERNEL_SLUG,
        "prune_test_kernel_slug": PRUNE_TEST_KERNEL_SLUG,
        "orgdet_gap_prune_test_kernel_slug": ORGDET_GAP_PRUNE_TEST_KERNEL_SLUG,
        "orgdet_elf_test_kernel_slug": ORGDET_ELF_TEST_KERNEL_SLUG,
        "orgdet_fnn_test_kernel_slug": ORGDET_FNN_TEST_KERNEL_SLUG,
        "weights_name": WEIGHTS_NAME,
    }


def infer_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or infer_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGDET_FNN_FAST_TEST_HASH = infer_config_hash()
assert LOCKED_ORGDET_FNN_FAST_TEST_HASH == "9b9763e26bc0f037"
assert ORGDET_ELF_CONFIG_HASH == "292c4f1f968f948d"
assert ORGDET_FNN_CONFIG_HASH == "95e9300eb84ba71c"
assert ORGDET_ELF_TEST_PARENT_HASH == "ad375d39e13098d1"
assert ORGDET_GAP_PRUNE_TEST_PARENT_HASH == "52a30a5843e0a880"
assert ORGDET_GAP_PRUNE_CONFIG_HASH == "862e3e7f7ac0330e"
assert ORGDET_GAP_TEST_PARENT_HASH == "1ce4837ae16bee52"
assert ORGDET_GAP_CONFIG_HASH == "ee31f36151325f9a"
assert ORGDET_HOLDOUT_CONFIG_HASH == "912f0c19b572a054"
assert ORGDET_TEST_PARENT_HASH == "c45c859610d29e90"


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
        if specimen_of(stem) in HOLDOUT_SPECIMENS:
            leaked.append(stem)
            seen.add(stem)
    return leaked


def refuse_holdout_stems(stems: Iterable[str]) -> None:
    leaked = holdout_stems(stems)
    if leaked:
        raise HoldoutLeakError(
            "learned_unet_orgdet_fnn_fast_test_v1 refuses holdout stems in training weights: " + ", ".join(leaked)
        )


def refuse_train_volume(path: str | Path) -> Path:
    dest = Path(path)
    parts = {part.lower() for part in dest.parts}
    if "train" in parts and dest.name.endswith(".zarr"):
        raise TrainVolumeError(f"learned_unet_orgdet_fnn_fast_test_v1 refuses train volumes: {dest}")
    return dest


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
            "learned_unet_orgdet_fnn_fast_test_v1 refuses public 199-video / DeepCenter / "
            "organizer-split / DivNet weights; got " + ", ".join(hits)
        )


def runtime_estimate(
    n_volumes: int,
    sec_per_volume: float = FNN_FAST_TEST_SEC_PER_VOLUME,
    limit_hours: float = KAGGLE_GPU_LIMIT_HOURS,
    peak_memory_mb: float = ORGDET_ELF_TEST_MEASURED_PEAK_MEMORY_MB,
) -> dict[str, Any]:
    n = int(n_volumes)
    sec = float(n) * float(sec_per_volume)
    return {
        "n_volumes": n,
        "sec_per_volume": float(sec_per_volume),
        "orgdet_elf_test_sec_per_volume": ORGDET_ELF_TEST_SEC_PER_VOLUME,
        "fnn_fast_overhead_sec_per_volume": FNN_FAST_OVERHEAD_SEC_PER_VOLUME,
        "estimate_sec": sec,
        "estimate_hours": sec / 3600.0,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_observed": float(peak_memory_mb),
        "source": (
            "learned_unet_orgdet_elf_test_v1 measured 352.87 s / 4 volumes "
            "(88.22 s/volume) + 0.2 s vectorized forward-NN keep"
        ),
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
    }


def list_test_zarrs(test_dir: str | Path) -> list[Path]:
    root = Path(test_dir)
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name.endswith(".zarr"))


def sample_submission_datasets(root: Path) -> list[str]:
    path = root / "sample_submission.csv"
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        names = {str(row.get("dataset") or "") for row in reader}
    return sorted(name for name in names if name)


def graph_rows(
    dataset: str,
    coords: Sequence[Sequence[int]],
    edges: Sequence[Sequence[int]],
    start_id: int,
) -> tuple[list[list[Any]], int]:
    rows: list[list[Any]] = []
    next_id = int(start_id)
    for index, (t, z, y, x) in enumerate(coords, start=1):
        rows.append([next_id, dataset, "node", int(index), int(t), int(z), int(y), int(x), -1, -1])
        next_id += 1
    for source_id, target_id in edges:
        rows.append([next_id, dataset, "edge", -1, -1, -1, -1, -1, int(source_id), int(target_id)])
        next_id += 1
    return rows, next_id


def write_submission_header(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        csv.writer(fh).writerow(COLUMNS)
    return path


def append_submission_rows(path: Path, rows: Sequence[Sequence[Any]]) -> None:
    with path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        for row in rows:
            writer.writerow(row)


def audit_graph(
    dataset: str,
    coords: Sequence[Sequence[int]],
    edges: Sequence[Sequence[int]],
    shape: Sequence[int] | None = None,
) -> list[str]:
    errors: list[str] = []
    t_by_id = {i + 1: int(row[0]) for i, row in enumerate(coords)}
    out_count: dict[int, int] = {}
    for source_id, target_id in edges:
        src, dst = int(source_id), int(target_id)
        if src not in t_by_id or dst not in t_by_id:
            errors.append(f"{dataset}: edge {src}->{dst} references missing node_id")
            break
        if t_by_id[dst] <= t_by_id[src]:
            errors.append(f"{dataset}: source time does not precede target time ({src}->{dst})")
            break
        if t_by_id[dst] - t_by_id[src] != 1:
            errors.append(f"{dataset}: non-consecutive edge dt={t_by_id[dst] - t_by_id[src]}")
            break
        out_count[src] = out_count.get(src, 0) + 1
    if any(count >= 2 for count in out_count.values()):
        errors.append(f"{dataset}: predicted a division; learned_unet_orgdet_fnn_fast_test_v1 is 1-1")
    if shape is not None and coords:
        n_time, n_z, n_y, n_x = (int(v) for v in shape[:4])
        for t, z, y, x in coords:
            if not (0 <= int(t) < n_time):
                errors.append(f"{dataset}: t outside [0, {n_time - 1}]")
                break
            if not (0 <= int(z) < n_z and 0 <= int(y) < n_y and 0 <= int(x) < n_x):
                errors.append(f"{dataset}: centroid outside volume bounds")
                break
    return errors


def find_root() -> Path:
    for path in CANDIDATE_ROOTS:
        if (path / "test").is_dir():
            return path
    raise FileNotFoundError("Could not find the competition test directory")


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


def apply_gap_overlay_arrays(
    coords: np.ndarray,
    edges: Sequence[Sequence[int]],
    gate_um: float = 12.0,
    max_gap_frames: int = 2,
) -> tuple[np.ndarray, list[tuple[int, int]], int, int]:
    """Locked association_v2 gap-close on leftover unmatched orgdet nodes."""
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 4)
    nids = list(range(1, len(coords) + 1))
    ts = [int(row[0]) for row in coords]
    zs = [int(row[1]) for row in coords]
    ys = [int(row[2]) for row in coords]
    xs = [int(row[3]) for row in coords]
    edge_list = [(int(src), int(dst)) for src, dst in edges]
    idx = {nid: i for i, nid in enumerate(nids)}
    has_out = {int(src) for src, _dst in edge_list}
    has_in = {int(dst) for _src, dst in edge_list}
    by_t: dict[int, list[int]] = {}
    for nid in nids:
        by_t.setdefault(ts[idx[nid]], []).append(nid)
    scale = np.asarray(SCALE_ZYX, dtype=np.float64)
    max_dt = int(max_gap_frames) + 1
    candidates: list[tuple[float, int, int, int]] = []
    for parent in nids:
        if parent in has_out:
            continue
        t0 = ts[idx[parent]]
        p0 = np.asarray([zs[idx[parent]], ys[idx[parent]], xs[idx[parent]]], dtype=np.float64) * scale
        for dt in range(2, max_dt + 1):
            for child in by_t.get(t0 + dt, []):
                if child in has_in:
                    continue
                p1 = np.asarray([zs[idx[child]], ys[idx[child]], xs[idx[child]]], dtype=np.float64) * scale
                dist = float(np.linalg.norm(p0 - p1))
                if dist <= float(gate_um) * float(dt) ** 0.5:
                    candidates.append((dist, parent, child, dt))
    candidates.sort()
    used_parent: set[int] = set()
    used_child: set[int] = set()
    accepted: list[tuple[float, int, int, int]] = []
    for dist, parent, child, dt in candidates:
        if parent in used_parent or parent in has_out:
            continue
        if child in used_child or child in has_in:
            continue
        used_parent.add(parent)
        used_child.add(child)
        accepted.append((dist, parent, child, dt))
    next_id = (max(nids) + 1) if nids else 1
    n_interp = 0
    for _dist, parent, child, dt in accepted:
        i0 = idx[parent]
        i1 = idx[child]
        p0 = np.asarray([zs[i0], ys[i0], xs[i0]], dtype=np.float64)
        p1 = np.asarray([zs[i1], ys[i1], xs[i1]], dtype=np.float64)
        prev = parent
        t0 = ts[i0]
        for k in range(1, dt):
            alpha = k / dt
            pos = (1.0 - alpha) * p0 + alpha * p1
            nid = next_id
            next_id += 1
            nids.append(nid)
            ts.append(t0 + k)
            zs.append(int(round(float(pos[0]))))
            ys.append(int(round(float(pos[1]))))
            xs.append(int(round(float(pos[2]))))
            edge_list.append((prev, nid))
            prev = nid
            n_interp += 1
        edge_list.append((prev, child))
    out_coords = (
        np.stack([ts, zs, ys, xs], axis=1).astype(np.int64)
        if nids
        else np.empty((0, 4), dtype=np.int64)
    )
    return out_coords, edge_list, len(accepted), n_interp



def drop_degree0_arrays(
    coords: np.ndarray,
    edges: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, list[tuple[int, int]], int]:
    """Remove leftover isolates after the locked gap overlay. Remap 1..N."""
    if ORGDET_GAP_PRUNE_CONFIG_HASH != "862e3e7f7ac0330e":
        raise RuntimeError("learned_unet_orgdet_gap_prune_v1 hash drifted; refusing to retune")
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 4)
    nids = list(range(1, len(coords) + 1))
    linked: set[int] = set()
    for src, dst in edges:
        linked.add(int(src))
        linked.add(int(dst))
    keep_idx = [i for i, nid in enumerate(nids) if nid in linked]
    old_to_new = {nids[i]: j + 1 for j, i in enumerate(keep_idx)}
    if keep_idx:
        out_coords = coords[np.asarray(keep_idx, dtype=np.int64)]
    else:
        out_coords = np.empty((0, 4), dtype=np.int64)
    out_edges = [(old_to_new[int(src)], old_to_new[int(dst)]) for src, dst in edges]
    leftover = set(old_to_new.values())
    if any(src not in leftover or dst not in leftover for src, dst in out_edges):
        raise RuntimeError("prune dropped a linked node; refusing")
    return out_coords, out_edges, int(len(coords) - len(keep_idx))



def drop_long_dt1_edges_arrays(
    coords: np.ndarray,
    edges: Sequence[tuple[int, int]],
    max_um: float = MAX_DT1_EDGE_UM,
) -> tuple[np.ndarray, list[tuple[int, int]], int]:
    """Drop consecutive dt=1 edges longer than 8.0 µm. Keep node ids."""
    if float(max_um) != float(MAX_DT1_EDGE_UM):
        raise RuntimeError("learned_unet_orgdet_elf_v1 would retune max_dt1_edge_um; refusing")
    if ORGDET_ELF_CONFIG_HASH != "292c4f1f968f948d":
        raise RuntimeError("learned_unet_orgdet_elf_v1 hash drifted; refusing to retune")
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 4)
    nids = list(range(1, len(coords) + 1))
    idx = {nid: i for i, nid in enumerate(nids)}
    scale = np.asarray(SCALE_ZYX, dtype=np.float64)
    kept: list[tuple[int, int]] = []
    dropped = 0
    for src, dst in edges:
        s, d = int(src), int(dst)
        if s not in idx or d not in idx:
            raise RuntimeError("elf saw a dangling edge; refusing")
        i, j = idx[s], idx[d]
        dt = int(coords[j, 0]) - int(coords[i, 0])
        if dt != 1:
            kept.append((s, d))
            continue
        p0 = coords[i, 1:].astype(np.float64) * scale
        p1 = coords[j, 1:].astype(np.float64) * scale
        if float(np.linalg.norm(p0 - p1)) > float(max_um):
            dropped += 1
            continue
        kept.append((s, d))
    return coords, kept, int(dropped)


def drop_non_forward_nn_edges_arrays(
    coords: np.ndarray,
    edges: Sequence[tuple[int, int]],
    tie_eps: float = NN_TIE_EPS,
) -> tuple[np.ndarray, list[tuple[int, int]], int]:
    """Keep dt=1 edges whose destination is a nearest neighbor of the source at t+1.

    Vectorized per-frame pairwise distances. Same keep/drop rule as the
    locked naive scan. Do not retune nn_tie_eps.
    """
    if float(tie_eps) != float(NN_TIE_EPS):
        raise RuntimeError("learned_unet_orgdet_fnn_v1 would retune nn_tie_eps; refusing")
    if ORGDET_FNN_CONFIG_HASH != "95e9300eb84ba71c":
        raise RuntimeError("learned_unet_orgdet_fnn_v1 hash drifted; refusing to retune")
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 4)
    nids = list(range(1, len(coords) + 1))
    idx = {nid: i for i, nid in enumerate(nids)}
    by_t_idx: dict[int, list[int]] = {}
    for nid in nids:
        by_t_idx.setdefault(int(coords[idx[nid], 0]), []).append(idx[nid])
    scale = np.asarray(SCALE_ZYX, dtype=np.float64)
    xyz = coords[:, 1:].astype(np.float64) * scale
    min_fwd: dict[int, float | None] = {}
    for t, idxs in by_t_idx.items():
        nxt = by_t_idx.get(int(t) + 1)
        if not nxt:
            for i in idxs:
                min_fwd[int(i)] = None
            continue
        src_xyz = xyz[np.asarray(idxs, dtype=np.int64)]
        nxt_xyz = xyz[np.asarray(nxt, dtype=np.int64)]
        dists = np.linalg.norm(src_xyz[:, None, :] - nxt_xyz[None, :, :], axis=2)
        for i, m in zip(idxs, dists.min(axis=1)):
            min_fwd[int(i)] = float(m)
    kept: list[tuple[int, int]] = []
    dropped = 0
    for src, dst in edges:
        s, d = int(src), int(dst)
        if s not in idx or d not in idx:
            raise RuntimeError("fnn saw a dangling edge; refusing")
        i, j = idx[s], idx[d]
        dt = int(coords[j, 0]) - int(coords[i, 0])
        if dt != 1:
            kept.append((s, d))
            continue
        min_dist = min_fwd.get(i)
        if min_dist is None:
            dropped += 1
            continue
        dest_dist = float(np.linalg.norm(xyz[j] - xyz[i]))
        if dest_dist <= float(min_dist) + float(tie_eps):
            kept.append((s, d))
        else:
            dropped += 1
    leftover = set(nids)
    if any(src not in leftover or dst not in leftover for src, dst in kept):
        raise RuntimeError("fnn kept a dangling edge; refusing")
    return coords, kept, int(dropped)



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
            refuse_public_weight_sources([str(found), str(root)])
            if found.name not in {WEIGHTS_NAME, FALLBACK_WEIGHTS_NAME}:
                raise HoldoutLeakError(
                    "learned_unet_orgdet_fnn_fast_test_v1 only loads locked 44b6 weights; got "
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
    coords, edges, _n_gap, _n_interp = apply_gap_overlay_arrays(
        coords,
        edges,
        gate_um=float(infer_cfg["gate_um"]),
        max_gap_frames=int(infer_cfg["max_gap_frames"]),
    )
    coords, edges, _n_dropped = drop_degree0_arrays(coords, edges)
    coords, edges, _n_long = drop_long_dt1_edges_arrays(coords, edges)
    coords, edges, _n_dropped_after = drop_degree0_arrays(coords, edges)
    coords, edges, _n_fnn = drop_non_forward_nn_edges_arrays(coords, edges)
    coords, edges, _n_dropped_fnn = drop_degree0_arrays(coords, edges)
    return coords, edges, _n_gap, _n_interp, _n_dropped + _n_dropped_after + _n_dropped_fnn, _n_long, _n_fnn


def infer_test_volumes(
    zarrs: Sequence[Path],
    model,
    train_cfg: dict[str, Any],
    infer_cfg: dict[str, Any],
    submission_path: Path,
    device,
) -> tuple[list[dict[str, Any]], list[str], int, int]:
    write_submission_header(submission_path)
    summaries: list[dict[str, Any]] = []
    errors: list[str] = []
    next_id = 0
    n_nodes = 0
    n_edges = 0
    for zarr_path in zarrs:
        refuse_train_volume(zarr_path)
        stem = zarr_path.name[: -len(".zarr")] if zarr_path.name.endswith(".zarr") else zarr_path.name
        started = time.time()
        volume = open_volume_array(zarr_path)
        shape = tuple(int(x) for x in getattr(volume, "shape", (100, 64, 256, 256)))
        coords, edges, n_gap, n_interp, n_dropped, n_long, n_fnn = predict_volume(model, volume, train_cfg, infer_cfg, device)
        coords_list = np.asarray(coords, dtype=np.int64).reshape(-1, 4).tolist()
        edge_list = [[int(s), int(d)] for s, d in edges]
        errors.extend(audit_graph(stem, coords_list, edge_list, shape))
        rows, next_id = graph_rows(stem, coords_list, edge_list, next_id)
        append_submission_rows(submission_path, rows)
        n_nodes += len(coords_list)
        n_edges += len(edge_list)
        summaries.append(
            {
                "stem": stem,
                "status": "predicted",
                "shape": list(shape),
                "n_nodes": len(coords_list),
                "n_edges": len(edge_list),
                "n_gap_links": int(n_gap),
                "n_interpolated": int(n_interp),
                "n_dropped_isolates": int(n_dropped),
                "n_dropped_long_dt1_edges": int(n_long),
                "n_dropped_non_forward_nn_edges": int(n_fnn),
                "runtime_sec": time.time() - started,
            }
        )
    return summaries, errors, n_nodes, n_edges


def main() -> dict[str, Any]:
    locked = {
        TEST_KERNEL_SLUG,
        GAP_TEST_KERNEL_SLUG,
        PRUNE_TEST_KERNEL_SLUG,
        ORGDET_HOLDOUT_KERNEL_SLUG,
        ORGDET_TEST_KERNEL_SLUG,
        ORGDET_GAP_TEST_KERNEL_SLUG,
        ORGDET_GAP_PRUNE_TEST_KERNEL_SLUG,
        ORGDET_ELF_TEST_KERNEL_SLUG,
    }
    if KERNEL_SLUG in locked:
        raise RuntimeError("refusing to overwrite locked test/gap-test/prune-test/orgdet slugs")
    started = time.time()
    cfg = infer_config_payload()
    working = output_dir()
    working.mkdir(parents=True, exist_ok=True)
    runtime = probe_runtime()
    submission_path = working / "submission.csv"
    manifest: dict[str, Any] = {
        "status": "starting",
        "label": CONFIG_LABEL,
        "config_hash": infer_config_hash(cfg),
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": INFER_CONFIG_HASH,
        "orgdet_config_hash": ORGDET_CONFIG_HASH,
        "orgdet_holdout_config_hash": ORGDET_HOLDOUT_CONFIG_HASH,
        "orgdet_gap_config_hash": ORGDET_GAP_CONFIG_HASH,
        "orgdet_gap_prune_config_hash": ORGDET_GAP_PRUNE_CONFIG_HASH,
        "orgdet_elf_config_hash": ORGDET_ELF_CONFIG_HASH,
        "orgdet_fnn_config_hash": ORGDET_FNN_CONFIG_HASH,
        "orgdet_elf_test_parent_hash": ORGDET_ELF_TEST_PARENT_HASH,
        "orgdet_gap_test_parent_hash": ORGDET_GAP_TEST_PARENT_HASH,
        "orgdet_test_parent_hash": ORGDET_TEST_PARENT_HASH,
        "rejected_orgop_hash": REJECTED_ORGOP_HASH,
        "test_parent_hash": TEST_PARENT_HASH,
        "gap_test_parent_hash": GAP_TEST_PARENT_HASH,
        "prune_test_parent_hash": PRUNE_TEST_PARENT_HASH,
        "kernel_slug": KERNEL_SLUG,
        "config": cfg,
        "environment": None,
        "is_kaggle_run": False,
        "is_competition_submission": False,
        "is_leaderboard_score": False,
        "runtime": runtime,
        "notes": (
            "Private GPU test-time kernel at locked orgdet+gap+prune+elf+fnn operating "
            "point (0.99 / pool 3.0 / TTA + greedy 12 µm + gap-close + drop "
            "degree-0 + drop dt=1 edges > 8.0 µm + keep forward-NN dt=1). Discovers test/*.zarr. Writes "
            "submission.csv. Not a competition submission. Not a leaderboard "
            "score. Do not overwrite orgdet-elf-test, orgdet-gap-prune-test, orgdet-gap-test, "
            "orgdet-test, prune-test, gap-test, no-gap test, or orgdet-holdout. "
            "Do not retune. Do not score 6bba holdout again. "
            "Public 199-video weights refused."
        ),
    }
    write_manifest(working / "run_manifest.json", manifest)
    try:
        root = find_root()
    except FileNotFoundError as exc:
        manifest.update({"status": "blocked_no_test_dir", "error": str(exc)})
        write_manifest(working / "run_manifest.json", manifest)
        return manifest
    test_dir = root / "test"
    zarrs = list_test_zarrs(test_dir)
    discovered = [p.name[: -len(".zarr")] for p in zarrs]
    sample_names = sample_submission_datasets(root)
    estimate = runtime_estimate(len(zarrs))
    manifest.update(
        {
            "environment": environment_label(root),
            "is_kaggle_run": str(root).startswith("/kaggle/input"),
            "datasets": discovered,
            "sample_submission_datasets": sample_names,
            "estimate": estimate,
        }
    )
    write_manifest(working / "run_manifest.json", manifest)
    if not zarrs:
        manifest.update({"status": "blocked_no_test_zarrs", "runtime_sec": time.time() - started})
        write_manifest(working / "run_manifest.json", manifest)
        return manifest
    if os.environ.get("BIOHUB_LEARNED_DRY_RUN") == "1":
        manifest.update({"status": "dry_run", "runtime_sec": time.time() - started})
        write_manifest(working / "run_manifest.json", manifest)
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
        write_manifest(working / "run_manifest.json", manifest)
        return manifest
    manifest["weights_path"] = str(weights)
    train_manifest = weights.parent / "train_manifest.json"
    if train_manifest.exists():
        verify_train_manifest(train_manifest)
        manifest["train_manifest_status"] = json.loads(train_manifest.read_text()).get("status")
    if not runtime.get("torch"):
        manifest.update({"status": "blocked_no_torch", "runtime_sec": time.time() - started})
        write_manifest(working / "run_manifest.json", manifest)
        return manifest
    if not runtime.get("cuda_available"):
        manifest.update({"status": "blocked_no_gpu", "runtime_sec": time.time() - started})
        write_manifest(working / "run_manifest.json", manifest)
        return manifest
    import torch

    device = torch.device("cuda")
    try:
        model, train_cfg, blob = load_checkpoint(weights, device)
        summaries, errors, n_nodes, n_edges = infer_test_volumes(
            zarrs, model, train_cfg, cfg, submission_path, device
        )
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
        write_manifest(working / "run_manifest.json", manifest)
        raise
    present = [row["stem"] for row in summaries if row.get("status") == "predicted"]
    missing = [name for name in discovered if name not in set(present)]
    extra = sorted(set(present) - set(discovered))
    if missing:
        errors.append(f"missing discovered datasets: {missing}")
    if extra:
        errors.append(f"unexpected datasets: {extra}")
    sample_in_test = [name for name in sample_names if name in set(discovered)]
    sample_missing = [name for name in sample_in_test if name not in set(present)]
    if sample_missing:
        errors.append(
            "sample_submission datasets present in test/ but missing from CSV: "
            + ",".join(sample_missing)
        )
    size = submission_path.stat().st_size if submission_path.exists() else 0
    n_frames = sum(int((row.get("shape") or [0])[0]) for row in summaries)
    n_gap_links = sum(int(row.get("n_gap_links") or 0) for row in summaries)
    n_interpolated = sum(int(row.get("n_interpolated") or 0) for row in summaries)
    n_dropped = sum(int(row.get("n_dropped_isolates") or 0) for row in summaries)
    n_long = sum(int(row.get("n_dropped_long_dt1_edges") or 0) for row in summaries)
    n_fnn = sum(int(row.get("n_dropped_non_forward_nn_edges") or 0) for row in summaries)
    elapsed = time.time() - started
    manifest.update(
        {
            "status": "inferred" if not errors else "audit_failed",
            "checkpoint_hash": blob.get("config_hash"),
            "per_dataset": summaries,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "n_gap_links": n_gap_links,
            "n_interpolated": n_interpolated,
            "n_dropped_isolates": n_dropped,
            "n_dropped_long_dt1_edges": n_long,
            "n_dropped_non_forward_nn_edges": n_fnn,
            "n_predicted_divisions": 0,
            "rows": n_nodes + n_edges,
            "submission_path": str(submission_path),
            "submission_bytes": int(size),
            "submission_empty": size == 0,
            "runtime_sec": elapsed,
            "sec_per_frame": elapsed / max(n_frames, 1),
            "peak_memory_mb": peak_memory_mb(),
            "audit_errors": errors,
            "audit_ok": not errors,
        }
    )
    write_manifest(working / "run_manifest.json", manifest)
    if errors:
        raise SystemExit("submission invalid:\n" + "\n".join(errors))
    if size == 0:
        raise SystemExit("submission.csv is empty")
    return manifest


if __name__ == "__main__":
    result = main()
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in (
                    "status",
                    "config_hash",
                    "environment",
                    "datasets",
                    "n_nodes",
                    "n_edges",
                    "audit_ok",
                    "estimate",
                )
            },
            indent=2,
            default=str,
        )
    )
    print("NOTE: private test-time kernel, not a competition submission.")
