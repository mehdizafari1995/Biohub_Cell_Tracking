#!/usr/bin/env python3
"""Read-only poll: is learned_unet_orgdet_fnn_acf_test_v1 submitted?

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
    ORGDET_FNN_ACF_TEST_CONFIG_HASH,
    ORGDET_FNN_ACF_TEST_SLUG,
    ORGDET_FNN_FAST_TEST_SLUG,
    ORGDET_TEST_SLUG,
    TEST_SLUG,
    TRAIN_SLUG,
    safe_kernel_status,
)
from biohub_tracking.learned_orgdet_fnn_acf_test import (  # noqa: E402
    ASSOCIATION_V2_PUBLIC_SCORE,
    ASSOCIATION_V2_SUBMISSION_REF,
    COMPETITION_DEADLINE_UTC,
    ORGDET_FNN_ACF_TEST_PRIVATE_SCORE,
    ORGDET_FNN_ACF_TEST_PUBLIC_SCORE,
    ORGDET_FNN_ACF_TEST_SCRIPT_VERSION_ID,
    ORGDET_FNN_ACF_TEST_SUBMISSION_REF,
    ORGDET_FNN_ACF_TEST_SUBMITTED_UTC,
    ORGDET_GAP_TEST_PUBLIC_SCORE,
    ORGDET_GAP_TEST_SUBMISSION_REF,
    ORGDET_TEST_PUBLIC_SCORE,
    ORGDET_TEST_SUBMISSION_REF,
    classify_orgdet_fnn_acf_test_submit_poll,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default=ORGDET_FNN_ACF_TEST_SLUG)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "learned_unet_orgdet_fnn_acf_test_v1_submit_poll.json",
    )
    args = parser.parse_args()
    if args.slug != ORGDET_FNN_ACF_TEST_SLUG:
        raise SystemExit(f"refusing unknown learned orgdet-fnn-acf-test slug {args.slug!r}")
    if args.slug in {TRAIN_SLUG, TEST_SLUG, ORGDET_TEST_SLUG, ORGDET_FNN_FAST_TEST_SLUG}:
        raise SystemExit("refusing to poll a locked train/no-gap/orgdet-test/fnn-fast-test slug")
    raw = kernel_status(args.slug)
    status = safe_kernel_status(raw, args.slug)
    names = kernel_output_names(kernel_output_list(args.slug))
    submissions = list_competition_submissions()
    classified = classify_orgdet_fnn_acf_test_submit_poll(
        kernel_complete=bool(status.get("is_complete")),
        kernel_failed=bool(status.get("is_error")),
        output_names=names,
        submissions=submissions,
    )
    payload = {
        "polled_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_version": git_version(ROOT),
        "orgdet_fnn_acf_test_config_hash": ORGDET_FNN_ACF_TEST_CONFIG_HASH,
        "kernel": status,
        "output_names": names,
        "submissions": submissions,
        "association_v2_submission_ref": ASSOCIATION_V2_SUBMISSION_REF,
        "association_v2_public_score": ASSOCIATION_V2_PUBLIC_SCORE,
        "orgdet_test_submission_ref": ORGDET_TEST_SUBMISSION_REF,
        "orgdet_test_public_score": ORGDET_TEST_PUBLIC_SCORE,
        "orgdet_gap_test_submission_ref": ORGDET_GAP_TEST_SUBMISSION_REF,
        "orgdet_gap_test_public_score": ORGDET_GAP_TEST_PUBLIC_SCORE,
        "orgdet_fnn_acf_test_submission_ref": ORGDET_FNN_ACF_TEST_SUBMISSION_REF,
        "orgdet_fnn_acf_test_public_score": ORGDET_FNN_ACF_TEST_PUBLIC_SCORE,
        "orgdet_fnn_acf_test_private_score": ORGDET_FNN_ACF_TEST_PRIVATE_SCORE,
        "orgdet_fnn_acf_test_script_version_id": ORGDET_FNN_ACF_TEST_SCRIPT_VERSION_ID,
        "orgdet_fnn_acf_test_submitted_utc": ORGDET_FNN_ACF_TEST_SUBMITTED_UTC,
        "competition_deadline_utc": COMPETITION_DEADLINE_UTC,
        **classified,
        "notes": (
            "Read-only poll. Did not download submission.csv. Did not submit. "
            "Did not overwrite locked slugs. Identity of 56644919 is "
            "mahdimzafari/biohub-learned-unet-orgdet-fnn-acf-test-v1 "
            "scriptVersionId 353195733. First-recorded public strings "
            "0.553 / 0.612 / 0.602 remain historical. Do not retune from "
            "0.63186 / 0.66560 or from any other public or private score. "
            "Competition deadline 2026-09-29T23:59:00Z has passed."
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
                    "new_submission_public_score",
                    "new_submission_private_score",
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
