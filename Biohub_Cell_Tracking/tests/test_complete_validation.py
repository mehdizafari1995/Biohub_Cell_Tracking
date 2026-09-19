from __future__ import annotations

import json
from pathlib import Path

import pytest

from biohub_tracking.graphs import TrackGraph
from biohub_tracking.official_eval import official_scorer_available
from biohub_tracking.splits import load_split_manifest
from biohub_tracking.validation import (
    FrozenSplitError,
    assert_frozen_v1_identity,
    assert_specimen_separated,
    assemble_complete_validation,
    extract_annotated_divisions,
    inventory_validation,
    plan_complete_validation,
    score_graphs_official,
)

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "docs" / "splits" / "v1.json"

pytestmark = pytest.mark.skipif(not official_scorer_available(), reason="official scorer source missing")


def _graph(name, nodes, edges, estimated=10.0):
    node_id, t, z, y, x = zip(*nodes)
    return TrackGraph(
        dataset=name,
        node_id=node_id,
        t=t,
        z=z,
        y=y,
        x=x,
        edges=edges,
        estimated_n_nodes=estimated,
    )


def _division_pair(dataset: str, recovered: bool = True) -> tuple[TrackGraph, TrackGraph]:
    gt = _graph(
        dataset,
        [
            (1, 0, 10, 10, 10),
            (2, 1, 10, 8, 10),
            (3, 1, 10, 12, 10),
            (4, 2, 10, 8, 10),
            (5, 2, 10, 12, 10),
        ],
        [(1, 2), (1, 3), (2, 4), (3, 5)],
        estimated=5,
    )
    if recovered:
        pred = gt.copy()
        pred.dataset = dataset
        return pred, gt
    pred = _graph(
        dataset,
        [
            (1, 0, 10, 10, 10),
            (2, 1, 10, 8, 10),
            (3, 1, 10, 12, 10),
        ],
        [(1, 2)],
        estimated=5,
    )
    return pred, gt


def test_frozen_v1_is_specimen_separated():
    manifest = load_split_manifest(V1)
    assert_specimen_separated(manifest)
    assert_frozen_v1_identity(manifest)
    plan = plan_complete_validation(manifest)
    assert plan["validation_datasets"] == ["6bba_05db0fb1", "6bba_c328f2fd"]
    assert plan["validation_embryos"] == ["6bba"]
    assert plan["expected_annotated_divisions"] == 5
    assert plan["retune_after_holdout"] is False
    assert {row["dataset"]: row["n_divisions"] for row in plan["expected_events"]} == {
        "6bba_05db0fb1": 3,
        "6bba_c328f2fd": 2,
    }


def test_shared_specimen_is_rejected():
    manifest = {
        "version": "test",
        "tuning_embryos": ["6bba"],
        "validation_embryos": ["6bba"],
        "tuning": ["6bba_aaaa"],
        "validation": ["6bba_bbbb"],
    }
    with pytest.raises(FrozenSplitError, match="specimen"):
        assert_specimen_separated(manifest)


def test_rewritten_v1_is_rejected():
    manifest = json.loads(V1.read_text())
    manifest["validation"] = ["6bba_05db0fb1"]
    with pytest.raises(FrozenSplitError, match="validation"):
        assert_frozen_v1_identity(manifest)


def test_extract_annotated_divisions_lists_parent_and_children():
    _, gt = _division_pair("toy")
    events = extract_annotated_divisions(gt)
    assert len(events) == 1
    assert events[0]["parent_id"] == 1
    assert events[0]["t"] == 0
    assert {c["node_id"] for c in events[0]["children"]} == {2, 3}


def test_official_scorer_recovers_annotated_division_event():
    pred, gt = _division_pair("toy_div", recovered=True)
    result = score_graphs_official(pred, gt)
    assert result["official"]["division_tp"] == 1
    assert result["official"]["division_fn"] == 0
    assert len(result["division_events"]) == 1
    assert result["division_events"][0]["official_recovered"] is True
    assert result["official_division_summary"]["division_tp"] == 1


def test_official_scorer_misses_linearized_division():
    pred, gt = _division_pair("toy_miss", recovered=False)
    result = score_graphs_official(pred, gt)
    assert result["official"]["division_tp"] == 0
    assert result["official"]["division_fn"] == 1
    assert result["division_events"][0]["official_recovered"] is False


def test_complete_report_requires_both_validation_sequences(tmp_path):
    manifest = load_split_manifest(V1)
    plan = plan_complete_validation(manifest)
    inventory = inventory_validation(tmp_path, plan["validation_datasets"])
    assert [item.dataset for item in inventory] == plan["validation_datasets"]
    assert all(item.status == "missing" for item in inventory)
    report = assemble_complete_validation(plan, inventory, [], "baseline.yaml")
    assert report["complete"] is False
    assert report["official_summary"] is None
    assert report["missing_datasets"] == plan["validation_datasets"]
    assert report["retune_after_holdout"] is False
    assert "missing" in (report["blocked_reason"] or "")


def test_two_sequence_official_summary_and_event_micro_average():
    pred_a, gt_a = _division_pair("6bba_aaa", recovered=True)
    pred_b, gt_b = _division_pair("6bba_bbb", recovered=False)
    scored = [score_graphs_official(pred_a, gt_a), score_graphs_official(pred_b, gt_b)]
    inventory = inventory_validation(Path("/no/such/train"), ["6bba_aaa", "6bba_bbb"])
    plan = {
        "split_version": "synthetic",
        "validation_datasets": ["6bba_aaa", "6bba_bbb"],
        "expected_annotated_divisions": 2,
        "retune_after_holdout": False,
    }
    report = assemble_complete_validation(plan, inventory, scored, "baseline.yaml")
    assert report["n_scored_samples"] == 2
    assert report["n_annotated_division_events"] == 2
    assert report["n_official_recovered_divisions"] == 1
    assert report["n_official_missed_divisions"] == 1
    assert report["official_summary"] is not None
    assert report["official_summary"]["division_tp"] == 1
    assert report["official_summary"]["division_fn"] == 1
    assert report["complete"] is False
