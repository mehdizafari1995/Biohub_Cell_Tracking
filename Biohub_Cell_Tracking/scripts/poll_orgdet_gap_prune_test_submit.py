#!/usr/bin/env python3
"""Read-only poll: is learned_unet_orgdet_gap_prune_test_v1 submitted?

Queries kernel status, output names, and competition submissions. Never
downloads ``submission.csv``. Never calls ``competitions submit``. Never
overwrites locked slugs. A local poll is not a Kaggle scoring rerun.
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
    kernel_output_list,
    kernel_output_names,
    kernel_status,
    list_competition_submissions,
)
from biohub_tracking.learned_collect import (  # noqa: E402
    GAP_TEST_SLUG,
    ORGDET_GAP_PRUNE_TEST_CONFIG_HASH,
    ORGDET_GAP_PRUNE_TEST_SLUG,
    ORGDET_GAP_TEST_SLUG,
    ORGDET_TEST_SLUG,
    PRUNE_TEST_SLUG,
    TEST_SLUG,
    TRAIN_SLUG,
    safe_kernel_status,
)
from biohub_tracking.learned_orgdet_gap_prune_test import (  # noqa: E402
    ASSOCIATION_V2_PUBLIC_SCORE,
    ASSOCIATION_V2_SUBMISSION_REF,
    classify_orgdet_gap_prune_test_submit_poll,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default=ORGDET_GAP_PRUNE_TEST_SLUG)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "learned_unet_orgdet_gap_prune_test_v1_submit_poll.json",
    )
    args = parser.parse_args()
    if args.slug != ORGDET_GAP_PRUNE_TEST_SLUG:
        raise SystemExit(f"refusing unknown learned orgdet-gap-prune-test slug {args.slug!r}")
    if args.slug in {
        TRAIN_SLUG,
        TEST_SLUG,
        GAP_TEST_SLUG,
        PRUNE_TEST_SLUG,
        ORGDET_TEST_SLUG,
        ORGDET_GAP_TEST_SLUG,
    }:
        raise SystemExit("refusing to poll a locked train/no-gap/gap-test/prune-test/orgdet slug")
    raw = kernel_status(args.slug)
    status = safe_kernel_status(raw, args.slug)
    names = kernel_output_names(kernel_output_list(args.slug))
    submissions = list_competition_submissions()
    classified = classify_orgdet_gap_prune_test_submit_poll(
        kernel_complete=bool(status.get("is_complete")),
        kernel_failed=bool(status.get("is_error")),
        output_names=names,
        submissions=submissions,
    )
    payload = {
        "polled_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_version": git_version(ROOT),
        "orgdet_gap_prune_test_config_hash": ORGDET_GAP_PRUNE_TEST_CONFIG_HASH,
        "kernel": status,
        "output_names": names,
        "submissions": submissions,
        "association_v2_submission_ref": ASSOCIATION_V2_SUBMISSION_REF,
        "association_v2_public_score": ASSOCIATION_V2_PUBLIC_SCORE,
        **classified,
        "notes": (
            "Read-only poll. Did not download submission.csv. Did not submit. "
            "Did not overwrite locked slugs. Association_v2 ref 56399003 public "
            "API string 0.553, orgdet-test 56466744 public 0.612, and "
            "orgdet-gap-test 56583783 public 0.602 are not an "
            "orgdet-gap-prune-test leaderboard score."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                k: payload.get(k)
                for k in (
                    "polled_utc",
                    "decision",
                    "kernel_complete",
                    "output_names",
                    "n_submissions",
                    "submission_refs",
                    "human_action",
                    "is_competition_submission",
                    "is_leaderboard_score",
                )
            },
            indent=2,
            default=str,
        )
    )
    print("NOTE: read-only poll, not a competition submission.")


if __name__ == "__main__":
    main()
