from __future__ import annotations

from pathlib import Path

from biohub_tracking.learned_orgop import LAUNCH_AUTHORIZED, LOCKED_ORGOP_HASH
from biohub_tracking.public_094_audit import (
    FOCUS3D_DISCUSSION_ID,
    FOCUS3D_TITLE,
    REQUESTED_DISCUSSION_IDS,
    classify_requested_discussions,
)

ROOT = Path(__file__).resolve().parents[1]


def test_requested_ids_include_unresolved_focus3d_and_740392():
    assert REQUESTED_DISCUSSION_IDS == (716952, 730160, 734604, 735352, 738217, 740392)
    assert FOCUS3D_DISCUSSION_ID == 738217
    assert FOCUS3D_TITLE == "focus3d : one of the best 3d cell segmentation"
    assert LAUNCH_AUTHORIZED is False
    assert LOCKED_ORGOP_HASH == "250c5bfacd4a8198"


def test_classify_requested_discussions_marks_focus3d_and_missing_740392():
    topics = [
        {
            "id": 738217,
            "title": FOCUS3D_TITLE,
            "votes": 36,
            "commentCount": 50,
            "postDate": "2026-08-30T15:24:33.9075585Z",
            "lastCommentPostDate": "2026-09-20T08:19:52.7733333Z",
        },
        {
            "id": 735352,
            "title": "Possible big leaderboard shakeup",
            "votes": 9,
            "commentCount": 3,
        },
    ]
    rows = classify_requested_discussions(topics)
    by_id = {row["id"]: row for row in rows}
    assert by_id[738217]["retrieved"] is True
    assert by_id[738217]["title"] == FOCUS3D_TITLE
    assert by_id[738217]["used_as_source"] is False
    assert by_id[738217]["body_retrieved"] is False
    assert by_id[740392]["retrieved"] is False
    assert by_id[740392]["kind"] == "not_in_topics_first_page"
    assert by_id[740392]["used_as_source"] is False
    assert by_id[716952]["retrieved"] is False
    src = (ROOT / "scripts" / "complete_094_discussion_inventory.py").read_text()
    assert "list_competition_topics" in src
    assert "competitions/submit" not in src
    assert "kernels/push" not in src
    assert "LAUNCH_AUTHORIZED" in src
