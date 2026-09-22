from __future__ import annotations

from pathlib import Path

import yaml

from biohub_tracking.learned_retrain import LOCKED_RETRAIN_HASH
from biohub_tracking.learned_retrain_lanes import (
    LANE_A_COMPARATOR,
    LANE_A_HASH,
    LANE_B_COMPARATOR,
    LANE_B_HASH,
    LANE_C_HASH,
    lane_a_payload,
    lane_b_payload,
    lane_c_payload,
    lane_decision,
    select_winner,
)

ROOT = Path(__file__).resolve().parents[1]


def test_retrain_lanes_are_immutable_and_match_yaml():
    assert LOCKED_RETRAIN_HASH == "92a419926fa7e40b"
    assert LANE_A_HASH == "1db21f87d00ce083"
    assert LANE_B_HASH == "594f6fbc03c974cf"
    assert LANE_C_HASH == "f924802e5f716a61"
    for payload, name in (
        (lane_a_payload(), "learned_unet_retrain_infer_v1.yaml"),
        (lane_b_payload(), "learned_unet_retrain_orgdet_v1.yaml"),
        (lane_c_payload(), "learned_unet_retrain_orgop_v1.yaml"),
    ):
        cfg = yaml.safe_load((ROOT / "configs" / name).read_text())
        for key, value in payload.items():
            assert cfg[key] == value, key
        assert payload["opens_6bba"] is False
        assert payload["writes_submission_csv"] is False
        assert payload["parent_retrain_hash"] == LOCKED_RETRAIN_HASH


def test_retrain_lane_selection_rejects_no_strict_gain():
    assert LANE_A_COMPARATOR == 0.6557507105877639
    assert LANE_B_COMPARATOR == 0.6795779484295005
    assert lane_decision(0.6557507105877639, LANE_A_COMPARATOR) == "reject_lane"
    assert lane_decision(0.6557507105877640, LANE_A_COMPARATOR) == "keep_lane"
    assert select_winner(
        {"A": 0.65, "B": 0.67, "C": 0.67},
        {"A": LANE_A_COMPARATOR, "B": LANE_B_COMPARATOR, "C": 0.67},
    ) is None
    assert select_winner(
        {"A": 0.66, "B": 0.68, "C": 0.69},
        {"A": LANE_A_COMPARATOR, "B": LANE_B_COMPARATOR, "C": 0.68},
    ) == "C"
