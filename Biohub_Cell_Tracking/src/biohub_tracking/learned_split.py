"""Specimen-separated split and config for learned_unet_v1.

Train on frozen-tuning specimen ``44b6`` only. Never put ``6bba`` (or any
frozen-holdout stem) in the train or internal-val lists. Internal val is the
two frozen-tuning sequences from ``docs/splits/v1.json``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

TUNING_SPECIMEN = "44b6"
HOLDOUT_SPECIMENS = ("6bba",)
INTERNAL_VAL_STEMS = ("44b6_12dfb391", "44b6_267148e4")
FROZEN_HOLDOUT_STEMS = ("6bba_05db0fb1", "6bba_c328f2fd")
KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-v1"


class HoldoutLeakError(ValueError):
    """Raised when a frozen-holdout specimen would enter training."""


def specimen_of(stem: str) -> str:
    name = Path(stem).name
    for suffix in (".zarr", ".geff"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.split("_", 1)[0]


def config_payload() -> dict[str, Any]:
    """Canonical learned_unet_v1 operating point. Do not retune from holdout."""
    return {
        "label": "learned_unet_v1",
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
    """Build organizer-style fold 0 from 44b6 stems only.

    ``train`` is every 44b6 stem except the two frozen-tuning sequences, which
    are reserved as internal val (organizer ``test`` key). Holdout embryos never
    appear.
    """
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
    """Discover 44b6 zarr+geff pairs. Never opens holdout volumes."""
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
        pairs.append(
            {
                "stem": stem,
                "zarr": str(zarr_path),
                "geff": str(geff_path),
            }
        )
    refuse_holdout_stems(p["stem"] for p in pairs)
    return pairs
