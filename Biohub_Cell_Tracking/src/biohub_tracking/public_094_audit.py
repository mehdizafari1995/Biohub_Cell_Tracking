"""Read-only helpers for the public 0.94 parity audit.

Does not launch kernels. Does not submit. Public leaderboard 0.553 is
association_v2 only and is not a tune target. Discussion bodies that
are not independently retrieved must stay unused as sources.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

REQUESTED_DISCUSSION_IDS = (716952, 730160, 734604, 735352, 738217, 740392)
FOCUS3D_DISCUSSION_ID = 738217
FOCUS3D_TITLE = "focus3d : one of the best 3d cell segmentation"
ORGOP_HASH = "250c5bfacd4a8198"
ORGOP_LAUNCH_AUTHORIZED = False


def classify_requested_discussions(
    topics: Sequence[Mapping[str, Any]],
    requested_ids: Iterable[int] = REQUESTED_DISCUSSION_IDS,
) -> list[dict[str, Any]]:
    """Match requested discussion IDs against a sanitized topics page."""
    by_id: dict[int, Mapping[str, Any]] = {}
    for item in topics:
        try:
            tid = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        by_id[tid] = item
    rows: list[dict[str, Any]] = []
    for did in requested_ids:
        rec = by_id.get(int(did))
        if rec is None:
            rows.append(
                {
                    "id": int(did),
                    "retrieved": False,
                    "title": None,
                    "votes": None,
                    "commentCount": None,
                    "postDate": None,
                    "lastCommentPostDate": None,
                    "kind": "not_in_topics_first_page",
                    "used_as_source": False,
                    "body_retrieved": False,
                }
            )
            continue
        title = rec.get("title")
        rows.append(
            {
                "id": int(did),
                "retrieved": True,
                "title": title,
                "votes": rec.get("votes"),
                "commentCount": rec.get("commentCount"),
                "postDate": rec.get("postDate"),
                "lastCommentPostDate": rec.get("lastCommentPostDate"),
                "kind": "topics_api_metadata",
                "used_as_source": False,
                "body_retrieved": False,
            }
        )
    return rows
