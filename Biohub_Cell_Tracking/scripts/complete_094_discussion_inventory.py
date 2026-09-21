#!/usr/bin/env python3
"""Complete the unresolved 0.94-audit discussion IDs from the topics API.

Read-only. Does not launch ``learned_unet_orgop_v1`` or
``learned_unet_retrain_v1``. Does not submit. Does not download kernel
outputs. HTML discussion pages are SPA shells; only sanitized topics-API
metadata and by-id HTTP status are recorded.
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
from biohub_tracking.kaggle_api import (  # noqa: E402
    get_competition_topic,
    list_competition_topics,
    search_kernels,
)
from biohub_tracking.learned_orgop import LOCKED_ORGOP_HASH  # noqa: E402
from biohub_tracking.public_094_audit import (  # noqa: E402
    FOCUS3D_DISCUSSION_ID,
    FOCUS3D_TITLE,
    ORGOP_LAUNCH_AUTHORIZED,
    REQUESTED_DISCUSSION_IDS,
    UNREAD_DISCUSSION_ID,
    apply_by_id_lookups,
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
    if ORGOP_LAUNCH_AUTHORIZED:
        raise SystemExit("refusing to mix this pre-launch inventory with an audit launch flag")
    listing = list_competition_topics()
    rows = classify_requested_discussions(listing["topics"], REQUESTED_DISCUSSION_IDS)
    missing_ids = [int(row["id"]) for row in rows if not row.get("retrieved")]
    lookups = [get_competition_topic(did) for did in missing_ids]
    rows = apply_by_id_lookups(rows, lookups)
    kernel_hits = search_kernels(str(UNREAD_DISCUSSION_ID), page_size=5)
    focus = next((row for row in rows if row["id"] == FOCUS3D_DISCUSSION_ID), None)
    unread = next((row for row in rows if row["id"] == UNREAD_DISCUSSION_ID), None)
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
        "by_id_lookups": [
            {
                "id": item.get("id"),
                "retrieved": item.get("retrieved"),
                "http_status": item.get("http_status"),
                "kind": item.get("kind"),
                "used_as_source": False,
                "body_retrieved": False,
            }
            for item in lookups
        ],
        "kernel_search_740392_n": kernel_hits["n"],
        "discussions": rows,
        "focus3d_title_matches": bool(focus and focus.get("title") == FOCUS3D_TITLE),
        "unread_740392_kind": None if unread is None else unread.get("kind"),
        "notes": (
            "Read-only topics API plus by-id lookup. HTML bodies are SPA "
            "shells and were not used as sources. 740392 is absent from the "
            "first hot page and the by-id endpoint 404s; kernel search is "
            "empty. Did not launch retrain or orgop. Did not submit. "
            "Public 0.553 unused."
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
                "unread_740392_kind": payload["unread_740392_kind"],
                "kernel_search_740392_n": payload["kernel_search_740392_n"],
                "orgop_launch_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
