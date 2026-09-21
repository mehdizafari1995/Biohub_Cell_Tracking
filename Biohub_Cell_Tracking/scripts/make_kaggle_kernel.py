#!/usr/bin/env python3
"""Build private internet-off Kaggle notebooks from self-contained kernel sources."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KERNELS: dict[str, dict] = {
    "baseline_v1": {
        "source": "notebooks/kaggle_baseline_v1.py",
        "notebook": "notebooks/kaggle_baseline_v1.ipynb",
        "kernel_dir": "kaggle_kernel/baseline_v1",
        "code_file": "kaggle_baseline_v1.ipynb",
        "kernel_id": "mahdimzafari/biohub-baseline-v1",
        "title": "Biohub Baseline V1",
        "markdown": """# Biohub baseline_v1

Private CPU kernel for **Biohub - Cell Tracking During Development**.

- Config: anisotropic DoG + 4 µm NMS + greedy linking (`baseline_v1`)
- Attach the competition dataset. **Internet OFF**.
- Discovers every `test/*.zarr` at runtime (no hard-coded public names).
- Writes `/kaggle/working/submission.csv` and `run_manifest.json`.
- This notebook is the Kaggle source. A local simulation is not a Kaggle execution.
""",
    },
    "association_v2": {
        "source": "notebooks/kaggle_association_v2.py",
        "notebook": "notebooks/kaggle_association_v2.ipynb",
        "kernel_dir": "kaggle_kernel/association_v2",
        "code_file": "kaggle_association_v2.ipynb",
        "kernel_id": "mahdimzafari/biohub-association-v2",
        "title": "Biohub Association V2",
        "markdown": """# Biohub association_v2

Private CPU kernel for **Biohub - Cell Tracking During Development**.

- Config: anisotropic DoG + 4 µm NMS + greedy 12 µm gate + 2-frame gap close + intensity 0.5 (`association_v2`)
- Same detections as `baseline_v1`. Linking interpolates skipped frames into consecutive edges.
- Attach the competition dataset. **Internet OFF**. Do **not** overwrite `mahdimzafari/biohub-baseline-v1`.
- Discovers every `test/*.zarr` at runtime (no hard-coded public names).
- Writes `/kaggle/working/submission.csv` and `run_manifest.json`.
- This notebook is the Kaggle source. A local simulation is not a Kaggle execution.
- Do **not** submit this version to the competition from this packaging step.
""",
    },
    "geff_export": {
        "source": "notebooks/kaggle_geff_export.py",
        "notebook": "notebooks/kaggle_geff_export.ipynb",
        "kernel_dir": "kaggle_kernel/geff_export",
        "code_file": "kaggle_geff_export.ipynb",
        "kernel_id": "mahdimzafari/biohub-geff-export",
        "title": "Biohub GEFF Export",
        "markdown": """# Biohub GEFF export

Private CPU helper kernel. **Not a competition submission.**

- Zips train `44b6_*.geff` stores plus frozen-holdout `6bba_05db0fb1` and `6bba_c328f2fd`.
- Does **not** copy image volumes.
- Attach the competition dataset. **Internet OFF**. Keep private.
- Do **not** overwrite `mahdimzafari/biohub-baseline-v1` or `mahdimzafari/biohub-association-v2`.
- Do **not** Submit to Competition.
""",
    },
    "zarr_export": {
        "source": "notebooks/kaggle_zarr_export.py",
        "notebook": "notebooks/kaggle_zarr_export.ipynb",
        "kernel_dir": "kaggle_kernel/zarr_export",
        "code_file": "kaggle_zarr_export.ipynb",
        "kernel_id": "mahdimzafari/biohub-holdout-zarr-export",
        "title": "Biohub Holdout Zarr Export",
        "markdown": """# Biohub holdout Zarr export

Private CPU helper kernel. **Not a competition submission.**

- Zips **only** frozen-holdout `6bba_05db0fb1.zarr` and `6bba_c328f2fd.zarr`.
- Does **not** copy other volumes or GEFF stores.
- Attach the competition dataset. **Internet OFF**. Keep private.
- Do **not** overwrite `mahdimzafari/biohub-baseline-v1`, `mahdimzafari/biohub-association-v2`, or `mahdimzafari/biohub-geff-export`.
- Do **not** Submit to Competition.
""",
    },
    "tuning_zarr_export": {
        "source": "notebooks/kaggle_tuning_zarr_export.py",
        "notebook": "notebooks/kaggle_tuning_zarr_export.ipynb",
        "kernel_dir": "kaggle_kernel/tuning_zarr_export",
        "code_file": "kaggle_tuning_zarr_export.ipynb",
        "kernel_id": "mahdimzafari/biohub-tuning-zarr-export",
        "title": "Biohub Tuning Zarr Export",
        "markdown": """# Biohub tuning Zarr export

Private CPU helper kernel. **Not a competition submission.**

- Zips **only** frozen-tuning `44b6_12dfb391.zarr` and `44b6_267148e4.zarr`.
- Does **not** copy holdout volumes or GEFF stores.
- Attach the competition dataset. **Internet OFF**. Keep private.
- Do **not** overwrite `mahdimzafari/biohub-baseline-v1`, `mahdimzafari/biohub-association-v2`, `mahdimzafari/biohub-geff-export`, or `mahdimzafari/biohub-holdout-zarr-export`.
- Do **not** Submit to Competition.
""",
    },
    "learned_unet_v1": {
        "source": "notebooks/kaggle_learned_unet_v1.py",
        "notebook": "notebooks/kaggle_learned_unet_v1.ipynb",
        "kernel_dir": "kaggle_kernel/learned_unet_v1",
        "code_file": "kaggle_learned_unet_v1.ipynb",
        "kernel_id": "mahdimzafari/biohub-learned-unet-v1",
        "title": "Biohub Learned UNet V1",
        "enable_gpu": True,
        "markdown": """# Biohub learned_unet_v1

Private **GPU** helper kernel. **Not a competition submission.**

- Trains organizer `TemporalUNet3D` + `SimpleNodeTransformer` on frozen-tuning specimen `44b6` only.
- Internal val: `44b6_12dfb391`, `44b6_267148e4`. Never opens `6bba` volumes.
- Attach the competition dataset. **Internet OFF**. **GPU ON**. Keep private.
- Writes weights + `train_manifest.json`. Does **not** write `submission.csv`.
- Do **not** overwrite `mahdimzafari/biohub-baseline-v1`, `mahdimzafari/biohub-association-v2`, or the helper export kernels.
- Do **not** Submit to Competition. Weights stay out of git.
""",
    },
    "learned_unet_infer_v1": {
        "source": "notebooks/kaggle_learned_unet_infer_v1.py",
        "notebook": "notebooks/kaggle_learned_unet_infer_v1.ipynb",
        "kernel_dir": "kaggle_kernel/learned_unet_infer_v1",
        "code_file": "kaggle_learned_unet_infer_v1.ipynb",
        "kernel_id": "mahdimzafari/biohub-learned-unet-infer-v1",
        "title": "Biohub Learned UNet Infer V1",
        "enable_gpu": True,
        "kernel_sources": ["mahdimzafari/biohub-learned-unet-v1"],
        "markdown": """# Biohub learned_unet_infer_v1

Private **GPU** helper kernel. **Not a competition submission.**

- Loads specimen-separated `learned_unet_v1` weights after that helper is COMPLETE.
- Infers only frozen-tuning `44b6_12dfb391` and `44b6_267148e4`. Never opens `6bba`.
- Writes JSON graphs + `infer_manifest.json`. Does **not** write `submission.csv`.
- Attach the competition dataset and `mahdimzafari/biohub-learned-unet-v1`. **Internet OFF**. **GPU ON**.
- Do **not** overwrite locked prediction/helper slugs or the training kernel.
- Do **not** Submit to Competition. Weights stay out of git.
""",
    },
    "learned_unet_holdout_v1": {
        "source": "notebooks/kaggle_learned_unet_holdout_v1.py",
        "notebook": "notebooks/kaggle_learned_unet_holdout_v1.ipynb",
        "kernel_dir": "kaggle_kernel/learned_unet_holdout_v1",
        "code_file": "kaggle_learned_unet_holdout_v1.ipynb",
        "kernel_id": "mahdimzafari/biohub-learned-unet-holdout-v1",
        "title": "Biohub Learned UNet Holdout V1",
        "enable_gpu": True,
        "kernel_sources": ["mahdimzafari/biohub-learned-unet-v1"],
        "markdown": """# Biohub learned_unet_holdout_v1

Private **GPU** helper kernel. **Not a competition submission.** One-shot only.

- Loads locked `learned_unet_v1` weights. Operating point is locked `learned_unet_infer_v1`.
- Infers only frozen-holdout `6bba_05db0fb1` and `6bba_c328f2fd`. Never opens `44b6`.
- Writes JSON graphs + `infer_manifest.json`. Does **not** write `submission.csv`.
- Attach the competition dataset and `mahdimzafari/biohub-learned-unet-v1`. **Internet OFF**. **GPU ON**.
- Do **not** overwrite locked train/infer slugs. Do **not** retune.
- Do **not** Submit to Competition. Weights stay out of git.
""",
    },
    "learned_unet_test_v1": {
        "source": "notebooks/kaggle_learned_unet_test_v1.py",
        "notebook": "notebooks/kaggle_learned_unet_test_v1.ipynb",
        "kernel_dir": "kaggle_kernel/learned_unet_test_v1",
        "code_file": "kaggle_learned_unet_test_v1.ipynb",
        "kernel_id": "mahdimzafari/biohub-learned-unet-test-v1",
        "title": "Biohub Learned UNet Test V1",
        "enable_gpu": True,
        "kernel_sources": ["mahdimzafari/biohub-learned-unet-v1"],
        "markdown": """# Biohub learned_unet_test_v1

Private **GPU** test-time notebook. Discovers every `test/*.zarr`.

- Loads locked `learned_unet_v1` weights. Operating point is locked `learned_unet_infer_v1` / `learned_unet_holdout_v1`.
- Writes `/kaggle/working/submission.csv` and `run_manifest.json`.
- Attach the competition dataset and `mahdimzafari/biohub-learned-unet-v1`. **Internet OFF**. **GPU ON**.
- Estimate: ~92 s/volume from holdout v1. Hidden test ~199 volumes → ~5.1 h, under the 12 h GPU limit. Peak RAM ~1.2 GB.
- Do **not** overwrite locked train/infer/holdout slugs. Do **not** retune. Do **not** score 6bba again.
- Do **not** Submit to Competition from this packaging step. Weights stay out of git.
""",
    },
    "learned_unet_link_v1": {
        "source": "notebooks/kaggle_learned_unet_link_v1.py",
        "notebook": "notebooks/kaggle_learned_unet_link_v1.ipynb",
        "kernel_dir": "kaggle_kernel/learned_unet_link_v1",
        "code_file": "kaggle_learned_unet_link_v1.ipynb",
        "kernel_id": "mahdimzafari/biohub-learned-unet-link-v1",
        "title": "Biohub Learned UNet Link V1",
        "enable_gpu": True,
        "kernel_sources": ["mahdimzafari/biohub-learned-unet-v1"],
        "markdown": """# Biohub learned_unet_link_v1

Private **GPU** helper kernel. **Not a competition submission.**

- Loads specimen-separated `learned_unet_v1` weights after that helper is COMPLETE.
- Infers only frozen-tuning `44b6_12dfb391` and `44b6_267148e4`. Never opens `6bba`.
- Uses trained `SimpleNodeTransformer` pairwise scores (softmax, threshold 0.5, 1-1) instead of the greedy 12 µm gate.
- Writes JSON graphs + `infer_manifest.json`. Does **not** write `submission.csv`.
- Attach the competition dataset and `mahdimzafari/biohub-learned-unet-v1`. **Internet OFF**. **GPU ON**.
- Do **not** overwrite locked train/infer/holdout/test/gap-test slugs.
- Do **not** Submit to Competition. Weights stay out of git.
""",
    },
    "learned_unet_link_v2": {
        "source": "notebooks/kaggle_learned_unet_link_v2.py",
        "notebook": "notebooks/kaggle_learned_unet_link_v2.ipynb",
        "kernel_dir": "kaggle_kernel/learned_unet_link_v2",
        "code_file": "kaggle_learned_unet_link_v2.ipynb",
        "kernel_id": "mahdimzafari/biohub-learned-unet-link-v2",
        "title": "Biohub Learned UNet Link V2",
        "enable_gpu": True,
        "kernel_sources": ["mahdimzafari/biohub-learned-unet-v1"],
        "markdown": """# Biohub learned_unet_link_v2

Private **GPU** helper kernel. **Not a competition submission.**

- New slug. Do **not** overwrite rejected `mahdimzafari/biohub-learned-unet-link-v1`.
- Loads specimen-separated `learned_unet_v1` weights after that helper is COMPLETE.
- Infers only frozen-tuning `44b6_12dfb391` and `44b6_267148e4`. Never opens `6bba`.
- Keeps locked detections. Replaces rejected `prob > 0.5` with greedy 1-1 on descending softmax scores and the locked association_v2 12 µm spatial gate.
- Writes JSON graphs + `infer_manifest.json`. Does **not** write `submission.csv`.
- Attach the competition dataset and `mahdimzafari/biohub-learned-unet-v1`. **Internet OFF**. **GPU ON**.
- Do **not** overwrite locked train/infer/holdout/test/gap-test/link-v1 slugs.
- Do **not** Submit to Competition. Weights stay out of git.
""",
    },
    "learned_unet_motion_v1": {
        "source": "notebooks/kaggle_learned_unet_motion_v1.py",
        "notebook": "notebooks/kaggle_learned_unet_motion_v1.ipynb",
        "kernel_dir": "kaggle_kernel/learned_unet_motion_v1",
        "code_file": "kaggle_learned_unet_motion_v1.ipynb",
        "kernel_id": "mahdimzafari/biohub-learned-unet-motion-v1",
        "title": "Biohub Learned UNet Motion V1",
        "enable_gpu": False,
        "kernel_sources": ["mahdimzafari/biohub-learned-unet-infer-v1"],
        "markdown": """# Biohub learned_unet_motion_v1

Private **CPU** helper kernel. **Not a competition submission.**

- New slug. Do **not** overwrite rejected `mahdimzafari/biohub-learned-unet-link-v1` or `link-v2`.
- Loads locked `learned_unet_infer_v1` JSON graphs. Detection is not retuned.
- Replaces consecutive edges with constant-velocity greedy 1-1 plus the locked association_v2 12 µm gate.
- Infers only frozen-tuning `44b6_12dfb391` and `44b6_267148e4`. Never opens `6bba`.
- Writes JSON graphs + `infer_manifest.json`. Does **not** write `submission.csv`.
- Attach the competition dataset and `mahdimzafari/biohub-learned-unet-infer-v1`. **Internet OFF**. **GPU OFF**.
- Do **not** overwrite locked train/infer/holdout/test/gap-test/link slugs.
- Do **not** Submit to Competition.
""",
    },
    "learned_unet_conflict_v1": {
        "source": "notebooks/kaggle_learned_unet_conflict_v1.py",
        "notebook": "notebooks/kaggle_learned_unet_conflict_v1.ipynb",
        "kernel_dir": "kaggle_kernel/learned_unet_conflict_v1",
        "code_file": "kaggle_learned_unet_conflict_v1.ipynb",
        "kernel_id": "mahdimzafari/biohub-learned-unet-conflict-v1",
        "title": "Biohub Learned UNet Conflict V1",
        "enable_gpu": False,
        "kernel_sources": ["mahdimzafari/biohub-learned-unet-infer-v1"],
        "markdown": """# Biohub learned_unet_conflict_v1

Private **CPU** helper kernel. **Not a competition submission.**

- New slug. Do **not** overwrite rejected `learned_unet_link_v1` / `link_v2` / `motion_v1` / `hung_v1`.
- Loads locked `learned_unet_infer_v1` JSON graphs. Detection is not retuned.
- Keeps unique last-position nearest-neighbor matches. Applies Hungarian 1-1 only when two tracks share the same nearest neighbor inside the locked association_v2 12 µm gate.
- Infers only frozen-tuning `44b6_12dfb391` and `44b6_267148e4`. Never opens `6bba`.
- Writes JSON graphs + `infer_manifest.json`. Does **not** write `submission.csv`.
- Attach the competition dataset and `mahdimzafari/biohub-learned-unet-infer-v1`. **Internet OFF**. **GPU OFF**.
- Do **not** overwrite locked train/infer/holdout/test/gap-test/link slugs.
- Do **not** Submit to Competition.
""",
    },
    "learned_unet_hung_v1": {
        "source": "notebooks/kaggle_learned_unet_hung_v1.py",
        "notebook": "notebooks/kaggle_learned_unet_hung_v1.ipynb",
        "kernel_dir": "kaggle_kernel/learned_unet_hung_v1",
        "code_file": "kaggle_learned_unet_hung_v1.ipynb",
        "kernel_id": "mahdimzafari/biohub-learned-unet-hung-v1",
        "title": "Biohub Learned UNet Hung V1",
        "enable_gpu": False,
        "kernel_sources": ["mahdimzafari/biohub-learned-unet-infer-v1"],
        "markdown": """# Biohub learned_unet_hung_v1

Private **CPU** helper kernel. **Not a competition submission.**

- New slug. Do **not** overwrite rejected `learned_unet_link_v1` / `link_v2` / `motion_v1`.
- Loads locked `learned_unet_infer_v1` JSON graphs. Detection is not retuned.
- Replaces consecutive edges with Hungarian 1-1 on last-position distance plus the locked association_v2 12 µm gate.
- Infers only frozen-tuning `44b6_12dfb391` and `44b6_267148e4`. Never opens `6bba`.
- Writes JSON graphs + `infer_manifest.json`. Does **not** write `submission.csv`.
- Attach the competition dataset and `mahdimzafari/biohub-learned-unet-infer-v1`. **Internet OFF**. **GPU OFF**.
- Do **not** overwrite locked train/infer/holdout/test/gap-test/link slugs.
- Do **not** Submit to Competition.
""",
    },
    "learned_unet_gap_test_v1": {
        "source": "notebooks/kaggle_learned_unet_gap_test_v1.py",
        "notebook": "notebooks/kaggle_learned_unet_gap_test_v1.ipynb",
        "kernel_dir": "kaggle_kernel/learned_unet_gap_test_v1",
        "code_file": "kaggle_learned_unet_gap_test_v1.ipynb",
        "kernel_id": "mahdimzafari/biohub-learned-unet-gap-test-v1",
        "title": "Biohub Learned UNet Gap Test V1",
        "enable_gpu": True,
        "kernel_sources": ["mahdimzafari/biohub-learned-unet-v1"],
        "markdown": """# Biohub learned_unet_gap_test_v1

Private **GPU** test-time notebook with locked gap overlay. Discovers every `test/*.zarr`.

- Loads locked `learned_unet_v1` weights. Detection/consecutive links are locked `learned_unet_infer_v1` / `learned_unet_holdout_v1`.
- Applies locked `learned_unet_gap_v1` overlay (12 µm, max_gap=2, interpolate) to leftover unmatched nodes.
- Writes `/kaggle/working/submission.csv` and `run_manifest.json`.
- Attach the competition dataset and `mahdimzafari/biohub-learned-unet-v1`. **Internet OFF**. **GPU ON**.
- Estimate: ~97 s/volume (92.2 s infer + 4.8 s overlay). Hidden test ~199 volumes → ~5.4 h, under the 12 h GPU limit.
- Do **not** overwrite locked train/infer/holdout/test slugs, especially `mahdimzafari/biohub-learned-unet-test-v1`.
- Do **not** retune. Do **not** score 6bba again. Do **not** Submit to Competition from this packaging step.
""",
    },
}


def build_notebook(source_text: str, markdown: str, enable_gpu: bool = False) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "kaggle": {
                "accelerator": "gpu" if enable_gpu else "none",
                "isInternetEnabled": False,
            },
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": str(uuid.uuid4()),
                "metadata": {},
                "source": [line + "\n" for line in markdown.strip("\n").split("\n")],
            },
            {
                "cell_type": "code",
                "id": str(uuid.uuid4()),
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [line + "\n" for line in source_text.split("\n")[:-1]]
                + ([source_text.split("\n")[-1]] if source_text.split("\n")[-1] else []),
            },
        ],
    }


def metadata_payload(spec: dict) -> dict:
    return {
        "id": spec["kernel_id"],
        "title": spec["title"],
        "code_file": spec["code_file"],
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": bool(spec.get("enable_gpu", False)),
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": list(spec.get("kernel_sources") or []),
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "model_sources": [],
    }


def build_kernel(name: str) -> list[Path]:
    if name not in KERNELS:
        raise SystemExit(f"unknown kernel {name!r}; choose from {sorted(KERNELS)}")
    spec = KERNELS[name]
    source = ROOT / spec["source"]
    notebook = ROOT / spec["notebook"]
    kernel_dir = ROOT / spec["kernel_dir"]
    metadata = kernel_dir / "kernel-metadata.json"
    source_text = source.read_text()
    nb = build_notebook(source_text, spec["markdown"], enable_gpu=bool(spec.get("enable_gpu", False)))
    notebook.write_text(json.dumps(nb, indent=1) + "\n")
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_nb = kernel_dir / spec["code_file"]
    kernel_nb.write_text(json.dumps(nb, indent=1) + "\n")
    metadata.write_text(json.dumps(metadata_payload(spec), indent=2) + "\n")
    return [notebook, kernel_nb, metadata]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kernels",
        nargs="*",
        default=["association_v2"],
        help="Kernel names to package (default: association_v2). Use 'all' for every known kernel.",
    )
    args = parser.parse_args()
    names = list(KERNELS) if args.kernels == ["all"] else args.kernels
    for name in names:
        written = build_kernel(name)
        for path in written:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
