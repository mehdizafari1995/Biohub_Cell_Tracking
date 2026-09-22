from __future__ import annotations

from pathlib import Path

import yaml

from biohub_tracking.learned_retrain import LOCKED_RETRAIN_HASH
import pytest

from biohub_tracking.learned_retrain_lanes import (
    LANE_A_COMPARATOR,
    LANE_A_HASH,
    LANE_A_SLUG,
    LANE_B_COMPARATOR,
    LANE_B_HASH,
    LANE_B_SLUG,
    LANE_C_HASH,
    LANE_C_SLUG,
    ORGDET_HOLDOUT_CHAMPION,
    assert_tuning_only,
    decide_all_lanes,
    lane_a_payload,
    lane_b_payload,
    lane_c_payload,
    lane_decision,
    refuse_incomplete_retrain,
    require_lane_graph_paths,
    select_winner,
)
from biohub_tracking.learned_split import HoldoutLeakError

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


def test_retrain_lane_slugs_are_new_and_do_not_open_6bba():
    assert LANE_A_SLUG == "mahdimzafari/biohub-learned-unet-retrain-infer-v1"
    assert LANE_B_SLUG == "mahdimzafari/biohub-learned-unet-retrain-orgdet-v1"
    assert LANE_C_SLUG == "mahdimzafari/biohub-learned-unet-retrain-orgop-v1"
    assert LANE_A_SLUG != "mahdimzafari/biohub-learned-unet-infer-v1"
    assert LANE_B_SLUG != "mahdimzafari/biohub-learned-unet-orgdet-v1"
    assert LANE_C_SLUG != "mahdimzafari/biohub-learned-unet-orgop-v1"
    for payload in (lane_a_payload(), lane_b_payload(), lane_c_payload()):
        assert payload["opens_6bba"] is False
        assert payload["writes_submission_csv"] is False


def test_retrain_lanes_refuse_incomplete_helper_and_holdout_role():
    refuse_incomplete_retrain("complete", ["retrain_manifest.json", "learned_unet_retrain_v1_last.pt"])
    with pytest.raises(SystemExit, match="do not invent"):
        refuse_incomplete_retrain("running", [])
    with pytest.raises(SystemExit, match="submission.csv"):
        refuse_incomplete_retrain("complete", ["submission.csv"])
    assert_tuning_only("tuning")
    with pytest.raises(SystemExit, match="holdout"):
        assert_tuning_only("holdout")


def test_retrain_lanes_require_both_44b6_graphs_and_refuse_6bba(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="do not invent"):
        require_lane_graph_paths(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="44b6_12dfb391"):
        require_lane_graph_paths(empty)
    (empty / "44b6_12dfb391.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="44b6_267148e4"):
        require_lane_graph_paths(empty)
    leak = tmp_path / "leak"
    leak.mkdir()
    (leak / "6bba_05db0fb1.json").write_text("{}")
    with pytest.raises(HoldoutLeakError, match="6bba"):
        require_lane_graph_paths(leak)
    ready = tmp_path / "ready"
    ready.mkdir()
    (ready / "44b6_12dfb391.json").write_text("{}")
    (ready / "44b6_267148e4.json").write_text("{}")
    paths = require_lane_graph_paths(ready)
    assert [path.name for path in paths] == ["44b6_12dfb391.json", "44b6_267148e4.json"]


def test_retrain_lanes_decide_all_without_opening_6bba():
    rejected = decide_all_lanes({"A": LANE_A_COMPARATOR, "B": LANE_B_COMPARATOR, "C": LANE_B_COMPARATOR})
    assert rejected["winner"] is None
    assert rejected["may_open_6bba"] is False
    assert rejected["opens_6bba"] is False
    assert rejected["decisions"] == {"A": "reject_lane", "B": "reject_lane", "C": "reject_lane"}
    kept = decide_all_lanes({"A": 0.66, "B": 0.68, "C": 0.69})
    assert kept["winner"] == "C"
    assert kept["may_open_6bba"] is True
    assert kept["opens_6bba"] is False
    assert kept["holdout_champion"] == ORGDET_HOLDOUT_CHAMPION
    with pytest.raises(ValueError, match="do not invent"):
        decide_all_lanes({"A": 0.66})


def test_retrain_lane_scorer_script_is_fail_closed():
    source = (ROOT / "scripts" / "run_learned_unet_retrain_lanes.py").read_text()
    assert "refuse_incomplete_retrain" in source
    assert "require_lane_graph_paths" in source
    assert "decide_all_lanes" in source
    assert "opens_6bba" in source
    assert "kaggle competitions submit" not in source
    assert "6bba" in source
