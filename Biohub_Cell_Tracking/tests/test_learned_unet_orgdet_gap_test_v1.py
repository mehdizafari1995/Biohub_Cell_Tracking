from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from biohub_tracking.learned_orgdet_gap import LOCKED_ORGDET_GAP_HASH  # noqa: E402
from biohub_tracking.learned_orgdet_gap_test import (  # noqa: E402
    ASSOCIATION_V2_PUBLIC_SCORE,
    ASSOCIATION_V2_SUBMISSION_REF,
    LOCKED_ORGDET_GAP_TEST_HASH,
    ORGDET_GAP_TEST_KERNEL_SLUG,
    ORGDET_GAP_TEST_PUBLIC_SCORE,
    ORGDET_GAP_TEST_SUBMISSION_REF,
    classify_orgdet_gap_test_submit_poll,
    orgdet_gap_test_config_hash,
    orgdet_gap_test_config_payload,
)
from biohub_tracking.learned_orgdet_test import (  # noqa: E402
    LOCKED_ORGDET_TEST_HASH,
    ORGDET_TEST_KERNEL_SLUG,
)
from kaggle_learned_unet_orgdet_gap_test_v1 import (  # noqa: E402
    KERNEL_SLUG as NB_SLUG,
    LOCKED_ORGDET_GAP_TEST_HASH as NB_HASH,
    ORGDET_TEST_KERNEL_SLUG as NB_ORGDET_TEST,
    apply_gap_overlay_arrays,
    infer_config_hash as nb_hash,
    infer_config_payload as nb_payload,
)


_MAKE = importlib.util.spec_from_file_location("make_kaggle_kernel", ROOT / "scripts" / "make_kaggle_kernel.py")
assert _MAKE and _MAKE.loader
_make_mod = importlib.util.module_from_spec(_MAKE)
_MAKE.loader.exec_module(_make_mod)

_PUSH = importlib.util.spec_from_file_location("push_kaggle_kernel", ROOT / "scripts" / "push_kaggle_kernel.py")
assert _PUSH and _PUSH.loader
_push_mod = importlib.util.module_from_spec(_PUSH)
_PUSH.loader.exec_module(_push_mod)

_FETCH = importlib.util.spec_from_file_location("fetch_kernel_output", ROOT / "scripts" / "fetch_kernel_output.py")
assert _FETCH and _FETCH.loader
_fetch_mod = importlib.util.module_from_spec(_FETCH)
_FETCH.loader.exec_module(_fetch_mod)


def test_payload_matches_notebook_yaml_and_locked_hash():
    payload = orgdet_gap_test_config_payload()
    assert payload == nb_payload()
    assert orgdet_gap_test_config_hash(payload) == nb_hash(payload)
    assert orgdet_gap_test_config_hash(payload) == LOCKED_ORGDET_GAP_TEST_HASH
    assert LOCKED_ORGDET_GAP_TEST_HASH == "1ce4837ae16bee52"
    assert LOCKED_ORGDET_GAP_TEST_HASH == NB_HASH
    assert payload["orgdet_gap_config_hash"] == LOCKED_ORGDET_GAP_HASH == "ee31f36151325f9a"
    assert payload["orgdet_test_parent_hash"] == LOCKED_ORGDET_TEST_HASH
    assert payload["max_gap_frames"] == 2
    assert payload["interpolate_gaps"] is True
    assert payload["writes_submission_csv"] is True
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_orgdet_gap_test_v1.yaml").read_text())
    for key in (
        "gate_um",
        "max_gap_frames",
        "gap_scale",
        "assignment",
        "interpolate_gaps",
        "linker",
        "orgdet_gap_config_hash",
        "kernel_slug",
        "det_threshold",
        "pool_kernel_um",
    ):
        assert yaml_cfg[key] == payload[key]


def test_notebook_gap_overlay_matches_library():
    from biohub_tracking.learned_gap import apply_locked_overlay_arrays

    coords = np.asarray([[0, 8, 40, 40], [2, 8, 42, 40]], dtype=np.int64)
    nb_coords, nb_edges, n_gap, n_interp = apply_gap_overlay_arrays(coords, [], gate_um=12.0, max_gap_frames=2)
    lib_coords, lib_edges, lib_meta = apply_locked_overlay_arrays(coords, [])
    assert n_gap == 1
    assert n_interp == 1
    assert n_gap == lib_meta["n_gap_links"]
    assert n_interp == lib_meta["n_interpolated"]
    assert [tuple(row) for row in nb_coords.tolist()] == [tuple(row) for row in lib_coords.tolist()]
    assert nb_edges == lib_edges


def test_slug_guards():
    from biohub_tracking.learned_collect import (
        ORGDET_GAP_TEST_CONFIG_HASH,
        ORGDET_GAP_TEST_SLUG,
        ORGDET_TEST_SLUG,
    )

    assert ORGDET_GAP_TEST_KERNEL_SLUG == NB_SLUG
    assert ORGDET_TEST_KERNEL_SLUG == NB_ORGDET_TEST
    assert ORGDET_GAP_TEST_SLUG == ORGDET_GAP_TEST_KERNEL_SLUG
    assert ORGDET_GAP_TEST_CONFIG_HASH == LOCKED_ORGDET_GAP_TEST_HASH
    assert ORGDET_TEST_KERNEL_SLUG in _push_mod.FORBIDDEN_SLUGS
    assert ORGDET_TEST_SLUG in _push_mod.FORBIDDEN_SLUGS
    assert ORGDET_GAP_TEST_KERNEL_SLUG in _push_mod.FORBIDDEN_SLUGS
    assert ORGDET_GAP_TEST_KERNEL_SLUG in _fetch_mod.ALLOWED_SLUGS
    assert "learned_unet_orgdet_gap_test_v1" in _make_mod.KERNELS
    assert _make_mod.KERNELS["learned_unet_orgdet_gap_test_v1"]["kernel_id"] == ORGDET_GAP_TEST_KERNEL_SLUG
    assert _make_mod.KERNELS["learned_unet_orgdet_gap_test_v1"]["enable_gpu"] is True
    collect = ROOT / "scripts" / "collect_learned_unet_orgdet_gap_test_v1.py"
    text = collect.read_text()
    assert ORGDET_GAP_TEST_KERNEL_SLUG in text
    assert "competitions submit" not in text
    assert "artifacts" in text and "kaggle" in text


def test_classify_orgdet_gap_test_submit_poll_requires_ui_when_only_association_exists():
    association = {
        "ref": ASSOCIATION_V2_SUBMISSION_REF,
        "status": "complete",
        "publicScore": ASSOCIATION_V2_PUBLIC_SCORE,
        "private_score_returned": False,
        "date": "2026-09-20T15:12:52.307Z",
    }
    poll = classify_orgdet_gap_test_submit_poll(
        kernel_complete=True,
        kernel_failed=False,
        output_names=["run_manifest.json", "submission.csv"],
        submissions=[association],
    )
    assert poll["decision"] == "human_action_required_ui_submit"
    assert poll["orgdet_gap_test_submission_present"] is False
    assert poll["association_v2_submission_present"] is True
    assert poll["is_competition_submission"] is False
    assert poll["is_leaderboard_score"] is False
    assert "Kaggle UI" in poll["human_action"]
    assert "CLI" in poll["human_action"]
    assert "orgdet-gap-test-v1" in poll["human_action"]
    assert "orgdet-test" in poll["human_action"]
    assert classify_orgdet_gap_test_submit_poll(
        kernel_complete=True,
        kernel_failed=False,
        output_names=["run_manifest.json"],
        submissions=[association],
    )["decision"] == "kernel_missing_csv"
    extra = classify_orgdet_gap_test_submit_poll(
        kernel_complete=True,
        kernel_failed=False,
        output_names=["submission.csv"],
        submissions=[association, {"ref": 999, "status": "pending", "publicScore": None}],
    )
    assert extra["decision"] == "new_submission_present"
    assert extra["orgdet_gap_test_submission_present"] is False
    scored = classify_orgdet_gap_test_submit_poll(
        kernel_complete=True,
        kernel_failed=False,
        output_names=["submission.csv"],
        submissions=[
            association,
            {
                "ref": ORGDET_GAP_TEST_SUBMISSION_REF,
                "status": "complete",
                "publicScore": ORGDET_GAP_TEST_PUBLIC_SCORE,
                "date": "2026-09-26T16:09:48.773Z",
                "kernelSlug": "mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1",
                "scriptVersionId": 351808956,
            },
        ],
    )
    assert scored["decision"] == "public_score_returned"
    assert scored["orgdet_gap_test_submission_present"] is True
    assert scored["new_submission_public_score"] == ORGDET_GAP_TEST_PUBLIC_SCORE
    assert scored["is_leaderboard_score"] is False
    assert "not a tune target" in scored["human_action"]
    poll_src = (ROOT / "scripts" / "poll_orgdet_gap_test_submit.py").read_text()
    assert "list_competition_submissions" in poll_src
    assert "competitions/submit" not in poll_src
    assert "kernels/push" not in poll_src
    assert "download_kernel_output" not in poll_src
    live = json.loads((ROOT / "reports" / "learned_unet_orgdet_gap_test_v1_submit_poll.json").read_text())
    assert live["orgdet_gap_test_config_hash"] == LOCKED_ORGDET_GAP_TEST_HASH
    assert live["is_leaderboard_score"] is False
    assert live["is_competition_submission"] is False
    assert live["decision"] in {
        "human_action_required_ui_submit",
        "new_submission_present",
        "public_score_returned",
        "wait_kernel",
        "kernel_missing_csv",
        "record_kernel_error",
    }
