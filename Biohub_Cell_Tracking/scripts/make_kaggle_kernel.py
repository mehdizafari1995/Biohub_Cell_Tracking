#!/usr/bin/env python3
"""Build a private Kaggle notebook kernel from notebooks/kaggle_baseline_v1.py."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "kaggle_baseline_v1.py"
NOTEBOOK = ROOT / "notebooks" / "kaggle_baseline_v1.ipynb"
KERNEL_DIR = ROOT / "kaggle_kernel" / "baseline_v1"
METADATA = KERNEL_DIR / "kernel-metadata.json"
KERNEL_ID = "mahdimzafari/biohub-baseline-v1"
TITLE = "Biohub Baseline V1"

MARKDOWN = """# Biohub baseline_v1

Private CPU kernel for **Biohub - Cell Tracking During Development**.

- Config: anisotropic DoG + 4 µm NMS + greedy linking (`baseline_v1`)
- Attach the competition dataset. **Internet OFF**.
- Discovers every `test/*.zarr` at runtime (no hard-coded public names).
- Writes `/kaggle/working/submission.csv` and `run_manifest.json`.
- This notebook is the Kaggle source. A local simulation is not a Kaggle execution.
"""


def build_notebook(source_text: str) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "kaggle": {"accelerator": "none", "isInternetEnabled": False},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": str(uuid.uuid4()),
                "metadata": {},
                "source": [line + "\n" for line in MARKDOWN.strip("\n").split("\n")],
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


def metadata_payload() -> dict:
    return {
        "id": KERNEL_ID,
        "title": TITLE,
        "code_file": "kaggle_baseline_v1.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "model_sources": [],
    }


def main() -> None:
    source_text = SOURCE.read_text()
    nb = build_notebook(source_text)
    NOTEBOOK.write_text(json.dumps(nb, indent=1))
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    (KERNEL_DIR / "kaggle_baseline_v1.ipynb").write_text(json.dumps(nb, indent=1))
    METADATA.write_text(json.dumps(metadata_payload(), indent=2) + "\n")
    print(f"wrote {NOTEBOOK}")
    print(f"wrote {KERNEL_DIR / 'kaggle_baseline_v1.ipynb'}")
    print(f"wrote {METADATA}")


if __name__ == "__main__":
    main()
