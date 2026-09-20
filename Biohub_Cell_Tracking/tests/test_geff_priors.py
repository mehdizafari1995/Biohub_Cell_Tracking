from __future__ import annotations

from pathlib import Path

from biohub_tracking.geff_priors import (
    TUNING_SPECIMEN,
    VALIDATION_SPECIMEN,
    collect_geff_division_priors,
    division_events,
    iter_specimen_geffs,
    lock_operating_point,
    specimen_of,
    summarise_priors,
)
from biohub_tracking.graphs import TrackGraph


def test_specimen_key_and_iterator(tmp_path: Path):
    assert specimen_of("44b6_12dfb391") == TUNING_SPECIMEN
    assert specimen_of("6bba_05db0fb1") == VALIDATION_SPECIMEN
    from biohub_tracking.geff_priors import REQUIRED_GEFF_ARRAYS

    def _touch_complete(root: Path) -> None:
        root.mkdir()
        for rel in REQUIRED_GEFF_ARRAYS:
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("x")

    _touch_complete(tmp_path / "44b6_aaaa.geff")
    _touch_complete(tmp_path / "6bba_bbbb.geff")
    (tmp_path / "44b6_partial.geff").mkdir()
    (tmp_path / "readme.txt").write_text("no")
    found = [p.name for p in iter_specimen_geffs(tmp_path, TUNING_SPECIMEN)]
    assert found == ["44b6_aaaa.geff"]
    val = [p.name for p in iter_specimen_geffs(tmp_path, VALIDATION_SPECIMEN)]
    assert val == ["6bba_bbbb.geff"]


def test_lock_operating_point_uses_tuning_only_and_stays_conservative():
    events = [
        {
            "dataset": "44b6_aaa",
            "specimen": "44b6",
            "parent_child_um": [4.8, 5.5],
            "sibling_um": [9.5],
            "dt": [1, 1],
            "sibling_alignment": [-0.7],
        },
        {
            "dataset": "44b6_bbb",
            "specimen": "44b6",
            "parent_child_um": [0.6, 8.3],
            "sibling_um": [8.8],
            "dt": [1, 1],
            "sibling_alignment": [-0.2],
        },
    ]
    summary = summarise_priors(events, "tuning_44b6_specimen_geffs")
    locked = lock_operating_point(summary)
    assert locked["n_events"] == 2
    assert locked["require_parent_predecessor"] is True
    assert locked["require_no_alternative_parent"] is True
    assert locked["max_parent_child_um"] <= 10.0
    assert locked["max_parent_child_um"] >= 8.3
    assert locked["min_child_separation_um"] >= 8.0
    assert locked["min_child_separation_um"] <= 8.8
    assert locked["max_sibling_alignment"] <= 0.25
    assert "Validation 6bba events were not used" in locked["note"]


def test_collect_excludes_validation_from_lock(tmp_path: Path):
    payload = collect_geff_division_priors(tmp_path)
    assert payload["tuning_specimen"] == "44b6"
    assert payload["validation_specimen_excluded"] == "6bba"
    assert payload["n_tuning_geffs_scanned"] == 0
    assert payload["locked_operating_point"]["require_parent_predecessor"] is True


def test_division_events_records_alignment():
    graph = TrackGraph(
        dataset="44b6_toy",
        node_id=[1, 2, 3],
        t=[0, 1, 1],
        z=[8, 8, 8],
        y=[40, 20, 60],
        x=[40, 40, 40],
        edges=[(1, 2), (1, 3)],
    )
    events = division_events(graph)
    assert len(events) == 1
    assert events[0]["specimen"] == "44b6"
    assert events[0]["dt"] == [1, 1]
    assert events[0]["sibling_alignment"]
    assert events[0]["sibling_alignment"][0] < 0
