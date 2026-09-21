#!/usr/bin/env python3
"""Complete the unresolved 0.94-audit discussion IDs from the topics API.

Read-only. Does not launch ``learned_unet_orgop_v1``. Does not submit.
Does not download kernel outputs. HTML discussion pages are SPA shells;
only sanitized topics-API metadata is recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.experiments import git_version  # noqa: E402
from biohub_tracking.kaggle_api import list_competition_topics  # noqa: E402
from biohub_tracking.learned_orgop import LAUNCH_AUTHORIZED, LOCKED_ORGOP_HASH  # noqa: E402
from biohub_tracking.public_094_audit import (  # noqa: E402
    FOCUS3D_DISCUSSION_ID,
    FOCUS3D_TITLE,
    REQUESTED_DISCUSSION_IDS,
    classify_requested_discussions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "public_094_discussion_inventory.json",
    )
    args = parser.parse_args()
    if LAUNCH_AUTHORIZED:
        raise SystemExit("refusing to run inventory after an unauthorized orgop launch flag")
    listing = list_competition_topics()
    rows = classify_requested_discussions(listing["topics"], REQUESTED_DISCUSSION_IDS)
    focus = next((row for row in rows if row["id"] == FOCUS3D_DISCUSSION_ID), None)
    payload = {
        "audit_id": "public_094_discussion_inventory",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_version": git_version(ROOT),
        "orgop_hash": LOCKED_ORGOP_HASH,
        "orgop_launch_authorized": False,
        "kernel_launched": False,
        "competition_submitted": False,
        "topics_n_returned": listing["n_returned"],
        "topics_total_count": listing["total_count"],
        "topics_first_page_only": listing["first_page_only"],
        "requested_ids": list(REQUESTED_DISCUSSION_IDS),
        "discussions": rows,
        "focus3d_title_matches": bool(focus and focus.get("title") == FOCUS3D_TITLE),
        "notes": (
            "Read-only topics API. HTML bodies are SPA shells and were not "
            "used as sources. 740392 is absent from the first hot page. "
            "Did not launch orgop. Did not submit. Public 0.553 unused."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                "timestamp_utc": payload["timestamp_utc"],
                "n_returned": listing["n_returned"],
                "total_count": listing["total_count"],
                "retrieved_ids": [row["id"] for row in rows if row["retrieved"]],
                "missing_ids": [row["id"] for row in rows if not row["retrieved"]],
                "orgop_launch_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
