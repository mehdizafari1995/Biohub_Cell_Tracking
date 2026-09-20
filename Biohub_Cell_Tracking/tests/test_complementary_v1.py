from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

from biohub_tracking.config import load_config, pipeline_config_hash
from biohub_tracking.validation import LOCKED_V1_TUNING, LOCKED_V1_VALIDATION

ROOT = Path(__file__).resolve().parents[1]

_RUN = importlib.util.spec_from_file_location(
    "run_complementary_detection", ROOT / "scripts" / "run_complementary_detection.py"
)
assert _RUN and _RUN.loader
_run_mod = importlib.util.module_from_spec(_RUN)
_RUN.loader.exec_module(_run_mod)


def test_complementary_config_copies_locked_association_and_keeps_primary():
    cand = load_config(ROOT / "configs" / "complementary_v1.yaml")
    assoc = load_config(ROOT / "configs" / "association_v2.yaml")
    base = load_config(ROOT / "configs" / "baseline_v1.yaml")
    raw_assoc = yaml.safe_load((ROOT / "configs" / "association_v2.yaml").read_text())
    raw_base = yaml.safe_load((ROOT / "configs" / "baseline_v1.yaml").read_text())
    raw_cand = yaml.safe_load((ROOT / "configs" / "complementary_v1.yaml").read_text())
    assert raw_assoc["detection"]["method"] == "blob_dog"
    assert "complementary_method" not in raw_assoc["detection"]
    assert "complementary_method" not in raw_base["detection"]
    assert raw_cand["detection"]["method"] == raw_base["detection"]["method"]
    assert raw_cand["detection"]["threshold"] == raw_base["detection"]["threshold"]
    assert raw_cand["detection"]["min_distance_um"] == raw_base["detection"]["min_distance_um"]
    assert raw_cand["linking"]["gate_um"] == raw_assoc["linking"]["gate_um"]
    assert raw_cand["linking"]["max_gap_frames"] == raw_assoc["linking"]["max_gap_frames"]
    assert raw_cand["linking"]["intensity_weight"] == raw_assoc["linking"]["intensity_weight"]
    assert raw_cand["linking"]["allow_divisions"] is False
    assert cand.detection.complementary_method == "local_max"
    assert cand.detection.complementary_min_distance_um == 6.0
    assert cand.detection.complementary_intensity_ratio == 0.45
    assert assoc.detection.complementary_method == "none"
    assert base.detection.complementary_method == "none"
    assert pipeline_config_hash(cand) != pipeline_config_hash(assoc)
    assert pipeline_config_hash(cand) == "2c4866928d7262b6"
    priors = json.loads((ROOT / "docs" / "complementary_priors.json").read_text())
    locked = priors["locked_operating_point"]
    assert locked is not None
    assert priors["status"] == "rejected_on_holdout"
    assert priors["locked_config_hash"] == "2c4866928d7262b6"
    assert locked["complementary_method"] == "local_max"
    assert locked["complementary_min_distance_um"] == 6.0
    assert locked["complementary_intensity_ratio"] == 0.45
    assert locked["complementary_threshold"] == 0.04
    assert priors["do_not_retune_after_holdout"] is True
    assert tuple(priors["holdout_datasets"]) == LOCKED_V1_VALIDATION
    assert tuple(priors["tuning_datasets"]) == LOCKED_V1_TUNING


def test_runner_refuses_holdout_until_lock_file_records_operating_point(tmp_path: Path):
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"locked_operating_point": None}))
    try:
        _run_mod.assert_tuning_only("validation", empty)
    except SystemExit as exc:
        assert "refusing to score holdout" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
    _run_mod.assert_tuning_only("tuning", empty)
    locked = tmp_path / "locked.json"
    locked.write_text(json.dumps({"locked_operating_point": {"complementary_method": "local_max"}}))
    _run_mod.assert_tuning_only("validation", locked)
    scored = tmp_path / "scored.json"
    scored.write_text(
        json.dumps(
            {
                "locked_operating_point": {"complementary_method": "local_max"},
                "holdout_scored": True,
            }
        )
    )
    try:
        _run_mod.assert_tuning_only("validation", scored)
    except SystemExit as exc:
        assert "re-score" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
    assert LOCKED_V1_TUNING == ("44b6_12dfb391", "44b6_267148e4")
    assert LOCKED_V1_VALIDATION == ("6bba_05db0fb1", "6bba_c328f2fd")
    assert _run_mod.TUNING_DATASETS == LOCKED_V1_TUNING
    assert _run_mod.VALIDATION_DATASETS == LOCKED_V1_VALIDATION


def test_priors_holdout_is_one_shot_and_not_retuned():
    priors = json.loads((ROOT / "docs" / "complementary_priors.json").read_text())
    assert priors["status"] == "rejected_on_holdout"
    assert priors["holdout_scored"] is True
    assert priors["do_not_retune_after_holdout"] is True
    measured = priors.get("measured_holdout_official") or {}
    assert measured.get("not_a_leaderboard_score") is True
    assert measured.get("decision") == "reject_complementary_v1"
    assert measured.get("association_v2") == 0.5915697467955445
    assert measured.get("complementary_v1") == 0.5913844796149841
    assert measured["complementary_v1"] < measured["association_v2"]
    try:
        _run_mod.assert_tuning_only("validation", ROOT / "docs" / "complementary_priors.json")
    except SystemExit as exc:
        assert "re-score" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
