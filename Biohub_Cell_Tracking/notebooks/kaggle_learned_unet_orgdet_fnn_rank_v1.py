"""
Kaggle helper kernel: learned_unet_orgdet_fnn_rank_v1.

44b6-only learned ranking inside the locked 12 µm / sqrt(dt) envelope
on locked FNN detections. Prefers the already exported tuning Zarrs
44b6_12dfb391 / 44b6_267148e4. Never opens 6bba. Never writes
submission.csv. Not a competition submission. Do not overwrite
fnn-fast-test v1 or any locked slug.

Attach competition biohub-cell-tracking-during-development plus
mahdimzafari/biohub-learned-unet-orgop-v1, biohub-geff-export, and
biohub-tuning-zarr-export. Internet OFF. GPU ON.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

CONFIG_LABEL = "learned_unet_orgdet_fnn_rank_v1"
PARENT_FNN_HASH = "95e9300eb84ba71c"
LOCKED_RANK_HASH = "2748c4a2b2823c0a"
TUNING_SPECIMEN = "44b6"
HOLDOUT_SPECIMENS = ("6bba",)
INFER_STEMS = ("44b6_12dfb391", "44b6_267148e4")
FROZEN_HOLDOUT_STEMS = ("6bba_05db0fb1", "6bba_c328f2fd")
KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-fnn-rank-v1"
FORBIDDEN_PARENTS = (
    "mahdimzafari/biohub-learned-unet-orgdet-fnn-fast-test-v1",
    "mahdimzafari/biohub-learned-unet-orgdet-fnn-test-v1",
    "mahdimzafari/biohub-learned-unet-v1",
)
TUNE_KEEP_THRESHOLD = 0.7266526336456757
WRITES_SUBMISSION_CSV = False


class HoldoutLeakError(ValueError):
    """Raised when a frozen-holdout specimen would enter the ranker."""


def _specimen_of(stem: str) -> str:
    return str(stem).split("_", 1)[0]


def refuse_holdout(stems: list[str]) -> None:
    leaked = [s for s in stems if _specimen_of(s) in HOLDOUT_SPECIMENS or s in FROZEN_HOLDOUT_STEMS]
    if leaked:
        raise HoldoutLeakError("ranker helper refuses 6bba; got " + ", ".join(leaked))


def rank_helper_payload() -> dict[str, Any]:
    return {
        "label": CONFIG_LABEL,
        "parent_hash": PARENT_FNN_HASH,
        "config_hash": LOCKED_RANK_HASH,
        "kernel_slug": KERNEL_SLUG,
        "infer_stems": list(INFER_STEMS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "tune_keep_threshold": TUNE_KEEP_THRESHOLD,
        "writes_submission_csv": WRITES_SUBMISSION_CSV,
        "opens_6bba": False,
        "uses_69_volume_train": False,
        "prefers_tuning_zarrs": True,
    }


def main() -> None:
    refuse_holdout(list(INFER_STEMS))
    payload = rank_helper_payload()
    blob = json.dumps(
        {
            "label": CONFIG_LABEL,
            "parent_label": "learned_unet_orgdet_fnn_v1",
            "parent_hash": PARENT_FNN_HASH,
            "parent_orgdet_gap_hash": "ee31f36151325f9a",
            "source_graphs": "locked_learned_unet_orgdet_fnn_graphs",
            "linker": "fnn_detections_then_learned_rank_inside_locked_candidate_envelope",
            "candidate_gate_um": 12.0,
            "candidate_max_dt": 2,
            "candidate_gap_scale": "sqrt_dt",
            "hard_negatives": True,
            "train_specimen": TUNING_SPECIMEN,
            "verify_nonzero_link_head": True,
            "retunes_detection": False,
            "retunes_consecutive_links": False,
            "retunes_gap_gates": False,
            "retunes_max_dt1_edge_um": False,
            "retunes_nn_tie_eps": False,
            "uses_intensity": False,
            "uses_transformer": True,
            "uses_velocity": False,
            "uses_ilp": False,
            "assignment": "unchanged_locked_candidate_envelope",
            "tune_keep_threshold": TUNE_KEEP_THRESHOLD,
            "tune_margin": 0.002,
            "opened_holdout": False,
            "is_12um_assignment": False,
            "is_rejected_overlay": False,
            "writes_submission_csv": False,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(blob).hexdigest()[:16]
    if digest != LOCKED_RANK_HASH:
        raise SystemExit(f"rank helper hash drifted {digest}; refusing")
    if Path("/kaggle/working/submission.csv").exists():
        raise SystemExit("refusing to keep a submission.csv from the rank helper")
    out = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
    if str(out) == "/kaggle/working":
        out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "packaged_not_scored_until_local_official_44b6",
        "config": payload,
        "config_hash": LOCKED_RANK_HASH,
        "stems": list(INFER_STEMS),
        "opened_holdout": False,
        "wrote_submission_csv": False,
        "is_competition_submission": False,
        "notes": (
            "44b6-only ranker helper. Official scoring is the local runner "
            "scripts/run_learned_unet_orgdet_fnn_rank.py versus FNN 0.724653 "
            "with gate 0.726653. Do not overwrite locked slugs."
        ),
    }
    dest = out / "rank_manifest.json" if out.exists() else Path("rank_manifest.json")
    if dest.parent.exists():
        dest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "config_hash": LOCKED_RANK_HASH, "opened_holdout": False}))


if __name__ == "__main__":
    main()
