#!/usr/bin/env python3
"""Official-score learned_unet_orgdet_fnn_acf_scf_v1 only after a locked gate.

The 44b6 speed-change diagnostic must lock ``MAX_SPEED_CHANGE_UM`` before
this scorer runs. This script refuses until that gate exists. Do not
sweep leftovers. Do not invent scores. Never opens ``6bba`` during
tuning. A local official score is not a Kaggle run and not a
leaderboard number.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.learned_orgdet_fnn_acf_scf import (  # noqa: E402
    LOCKED_ORGDET_FNN_ACF_SCF_HASH,
    require_supported_gate,
)


def main() -> None:
    require_supported_gate()
    raise SystemExit(
        "learned_unet_orgdet_fnn_acf_scf_v1 gate is locked but this "
        f"scorer has no overlay path for hash {LOCKED_ORGDET_FNN_ACF_SCF_HASH}"
    )


if __name__ == "__main__":
    main()
