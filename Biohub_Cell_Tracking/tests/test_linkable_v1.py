from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

from biohub_tracking.config import DetectionConfig, load_config, pipeline_config_hash
from biohub_tracking.detect import (
    detect_cells,
    filter_temporally_linkable,
    merge_sequence_detections,
    select_linkable_coords,
)
from biohub_tracking.validation import LOCKED_V1_TUNING, LOCKED_V1_VALIDATION

ROOT = Path(__file__).resolve().parents[1]
SCALE = (1.625, 0.40625, 0.40625)

_RUN = importlib.util.spec_from_file_location(
    "run_linkable_detection", ROOT / "scripts" / "run_linkable_detection.py"
)
assert _RUN and _RUN.loader
_run_mod = importlib.util.module_from_spec(_RUN)
_RUN.loader.exec_module(_run_mod)


def test_complementary_v1_hash_unchanged_when_temporal_link_is_off():
    cand = load_config(ROOT / "configs" / "complementary_v1.yaml")
    assert cand.detection.complementary_require_temporal_link is False
    assert "complementary_require_temporal_link" not in cand.to_dict()["detection"]
    assert pipeline_config_hash(cand) == "2c4866928d7262b6"


def test_linkable_config_copies_locked_association_and_adds_temporal_gate():
    cand = load_config(ROOT / "configs" / "linkable_v1.yaml")
    assoc = load_config(ROOT / "configs" / "association_v2.yaml")
    base = load_config(ROOT / "configs" / "baseline_v1.yaml")
    comp = load_config(ROOT / "configs" / "complementary_v1.yaml")
    raw_cand = yaml.safe_load((ROOT / "configs" / "linkable_v1.yaml").read_text())
    raw_assoc = yaml.safe_load((ROOT / "configs" / "association_v2.yaml").read_text())
    raw_base = yaml.safe_load((ROOT / "configs" / "baseline_v1.yaml").read_text())
    raw_comp = yaml.safe_load((ROOT / "configs" / "complementary_v1.yaml").read_text())
    assert raw_cand["detection"]["method"] == raw_base["detection"]["method"]
    assert raw_cand["detection"]["min_distance_um"] == raw_base["detection"]["min_distance_um"]
    assert raw_cand["detection"]["complementary_min_distance_um"] == raw_comp["detection"]["complementary_min_distance_um"]
    assert raw_cand["detection"]["complementary_intensity_ratio"] == raw_comp["detection"]["complementary_intensity_ratio"]
    assert raw_cand["detection"]["complementary_require_temporal_link"] is True
    assert raw_cand["detection"]["complementary_link_gate_um"] == raw_assoc["linking"]["gate_um"]
    assert raw_cand["detection"]["complementary_link_max_dt"] == 1
    assert raw_cand["linking"]["gate_um"] == raw_assoc["linking"]["gate_um"]
    assert raw_cand["linking"]["max_gap_frames"] == raw_assoc["linking"]["max_gap_frames"]
    assert raw_cand["linking"]["allow_divisions"] is False
    assert cand.detection.complementary_require_temporal_link is True
    assert assoc.detection.complementary_require_temporal_link is False
    assert base.detection.complementary_method == "none"
    assert pipeline_config_hash(cand) != pipeline_config_hash(comp)
    assert pipeline_config_hash(cand) != pipeline_config_hash(assoc)
    assert pipeline_config_hash(cand) == "fa975421edb6136e"


def test_select_linkable_keeps_inside_association_gate_and_drops_outside():
    extra = np.array(
        [
            [10.0, 10.0, 30.0],  # 20 xy voxels = 8.125 µm
            [10.0, 10.0, 50.0],  # 40 xy voxels = 16.25 µm
        ],
        dtype=float,
    )
    primary = np.array([[10.0, 10.0, 10.0]], dtype=float)
    kept = select_linkable_coords(extra, primary, SCALE, gate_um=12.0)
    assert len(kept) == 1
    assert np.allclose(kept[0], extra[0])


def test_filter_drops_peaks_without_adjacent_primary_and_ignores_extra_chains():
    primary = {
        0: np.array([[10.0, 10.0, 10.0]], dtype=float),
        1: np.zeros((0, 3), dtype=float),
        2: np.zeros((0, 3), dtype=float),
    }
    extra = {
        1: np.array([[10.0, 10.0, 30.0]], dtype=float),  # 8.125 µm from t=0 primary
        2: np.array([[10.0, 10.0, 30.0]], dtype=float),  # only another extra at t=1
    }
    kept = filter_temporally_linkable(extra, primary, SCALE, gate_um=12.0, max_dt=1)
    assert len(kept[1]) == 1
    assert len(kept[2]) == 0


def test_merge_sequence_attaches_only_linkable_extras_when_required():
    primary = {
        0: np.array([[4.0, 8.0, 8.0]], dtype=float),
        1: np.array([[4.0, 8.0, 8.0]], dtype=float),
    }
    extra = {
        1: np.array([[4.0, 8.0, 28.0]], dtype=float),  # 8.125 µm
    }
    off = DetectionConfig(complementary_method="local_max")
    on = DetectionConfig(
        complementary_method="local_max",
        complementary_require_temporal_link=True,
        complementary_link_gate_um=12.0,
        complementary_link_max_dt=1,
    )
    merged_off = merge_sequence_detections(primary, extra, off, SCALE)
    merged_on = merge_sequence_detections(primary, extra, on, SCALE)
    assert len(merged_off[1]) == 2
    assert len(merged_on[1]) == 2
    far = {1: np.array([[4.0, 8.0, 80.0]], dtype=float)}
    merged_far = merge_sequence_detections(primary, far, on, SCALE)
    assert len(merged_far[1]) == 1


def test_single_frame_detect_cells_does_not_attach_linkable_extras():
    image = np.zeros((16, 32, 32), dtype=np.float32)
    image[8, 8, 8] = 1.0
    image[8, 8, 28] = 0.6
    cfg = DetectionConfig(
        method="blob_dog",
        min_sigma=1.5,
        max_sigma=4.0,
        threshold=0.05,
        normalize="minmax",
        min_distance_um=4.0,
        suppress_duplicates=True,
        anisotropic=True,
        complementary_method="local_max",
        complementary_require_temporal_link=True,
        complementary_link_gate_um=12.0,
        complementary_link_max_dt=1,
    )
    coords = detect_cells(image, cfg, SCALE)
    cfg_off = DetectionConfig(**{**cfg.__dict__, "complementary_require_temporal_link": False})
    merged = detect_cells(image, cfg_off, SCALE)
    assert len(coords) <= len(merged)


def test_runner_refuses_holdout_until_lock_and_refuses_second_score(tmp_path: Path):
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
    locked.write_text(json.dumps({"locked_operating_point": {"complementary_require_temporal_link": True}}))
    _run_mod.assert_tuning_only("validation", locked)
    scored = tmp_path / "scored.json"
    scored.write_text(
        json.dumps(
            {
                "locked_operating_point": {"complementary_require_temporal_link": True},
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
    priors = json.loads((ROOT / "docs" / "linkable_priors.json").read_text())
    assert priors["locked_operating_point"] is not None
    assert priors["locked_config_hash"] == "fa975421edb6136e"
    assert priors["candidate_config_hash"] == "fa975421edb6136e"
    assert priors["do_not_score_holdout_until_locked"] is True
    assert tuple(priors["holdout_datasets"]) == LOCKED_V1_VALIDATION
    assert tuple(priors["tuning_datasets"]) == LOCKED_V1_TUNING
    assert priors["do_not_retune_after_holdout"] is True
    assert priors["status"] == "rejected_on_holdout"
    assert priors["holdout_scored"] is True


def test_priors_holdout_is_one_shot_and_not_retuned():
    priors = json.loads((ROOT / "docs" / "linkable_priors.json").read_text())
    assert priors["status"] == "rejected_on_holdout"
    assert priors["holdout_scored"] is True
    assert priors["do_not_retune_after_holdout"] is True
    measured = priors.get("measured_holdout_official") or {}
    assert measured.get("not_a_leaderboard_score") is True
    assert measured.get("decision") == "reject_linkable_v1"
    assert measured.get("association_v2") == 0.5915697467955445
    assert measured.get("linkable_v1") == 0.5914576909101106
    assert measured["linkable_v1"] < measured["association_v2"]
    try:
        _run_mod.assert_tuning_only("validation", ROOT / "docs" / "linkable_priors.json")
    except SystemExit as exc:
        assert "re-score" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
