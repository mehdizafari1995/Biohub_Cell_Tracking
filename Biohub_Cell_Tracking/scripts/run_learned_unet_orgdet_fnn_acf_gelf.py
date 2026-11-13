#!/usr/bin/env python3
"""Official-score learned_unet_orgdet_fnn_acf_gelf_v1 only after a locked gate.

The 44b6 gap-length diagnostic did not support a TP-derived gate
(conflict FPs max 5.629 µm vs TP max 8.751 µm). This scorer refuses
until ``MAX_GAP_EDGE_UM`` is locked. Do not sweep leftovers. Do not
invent scores. Never opens ``6bba`` during tuning. A local official
score is not a Kaggle run and not a leaderboard number.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.learned_orgdet_fnn_acf_gelf import (  # noqa: E402
    LOCKED_ORGDET_FNN_ACF_GELF_HASH,
    require_supported_gate,
)


def main() -> None:
    require_supported_gate()
    raise SystemExit(
        "learned_unet_orgdet_fnn_acf_gelf_v1 gate is locked but this "
        f"scorer has no overlay path for hash {LOCKED_ORGDET_FNN_ACF_GELF_HASH}"
    )


if __name__ == "__main__":
    main()
