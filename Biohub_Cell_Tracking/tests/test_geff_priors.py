from __future__ import annotations

from pathlib import Path

from biohub_tracking.geff_priors import (
    CURRENT_LOCKED_OPERATING_POINT,
    REQUIRED_GEFF_FILES,
    TUNING_SPECIMEN,
    VALIDATION_SPECIMEN,
    all_geff_files,
    apply_operating_point_to_yaml,
    atypical_reasons,
    collect_geff_division_priors,
    division_events,
    inventory_geff_names,
    is_typical_fork,
    iter_specimen_geffs,
    lock_operating_point,
    retain_or_update_lock,
    specimen_of,
    summarise_priors,
)
from biohub_tracking.graphs import TrackGraph


def test_specimen_key_and_iterator(tmp_path: Path):
    assert specimen_of("44b6_12dfb391") == TUNING_SPECIMEN
    assert specimen_of("6bba_05db0fb1") == VALIDATION_SPECIMEN
    def _touch_complete(root: Path) -> None:
        root.mkdir()
        for rel in REQUIRED_GEFF_FILES:
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


def test_typical_fork_excludes_close_twin_and_near_coincident():
    typical = {
        "dataset": "44b6_aaa",
        "n_children": 2,
        "dt": [1, 1],
        "parent_child_um": [4.8, 5.5],
        "sibling_um": [9.5],
        "sibling_alignment": [-0.7],
    }
    near = {
        "dataset": "44b6_2a2eff9f",
        "n_children": 2,
        "dt": [1, 1],
        "parent_child_um": [0.57, 8.38],
        "sibling_um": [8.75],
        "sibling_alignment": [],
    }
    close = {
        "dataset": "44b6_close",
        "n_children": 2,
        "dt": [1, 1],
        "parent_child_um": [3.0, 3.2],
        "sibling_um": [4.67],
        "sibling_alignment": [-0.9],
    }
    long_range = {
        "dataset": "44b6_long",
        "n_children": 2,
        "dt": [1, 1],
        "parent_child_um": [12.34, 4.0],
        "sibling_um": [9.2],
        "sibling_alignment": [-0.8],
    }
    assert is_typical_fork(typical)
    assert atypical_reasons(near) == ["near_coincident_daughter", "alignment_undefined"]
    assert "close_twin" in atypical_reasons(close)
    assert "parent_child_above_cap" in atypical_reasons(long_range)


def test_retain_or_update_lock_widens_parent_child_only():
    previous = {
        **CURRENT_LOCKED_OPERATING_POINT,
        "max_parent_child_um": 8.5,
        "min_child_separation_um": 8.7,
    }
    typical_wide = {
        "dataset": "44b6_587a1e22",
        "n_children": 2,
        "dt": [1, 1],
        "parent_child_um": [4.8, 9.92],
        "sibling_um": [9.1],
        "sibling_alignment": [-0.8],
    }
    close = {
        "dataset": "44b6_close",
        "n_children": 2,
        "dt": [1, 1],
        "parent_child_um": [3.0, 3.2],
        "sibling_um": [4.67],
        "sibling_alignment": [-0.9],
    }
    updated = retain_or_update_lock(previous, [typical_wide])
    assert updated["action"] == "update_typical"
    assert updated["max_parent_child_um"] == 10.0
    assert updated["min_child_separation_um"] == 8.7
    assert updated["max_sibling_alignment"] == 0.05
    retained = retain_or_update_lock(previous, [])
    assert retained["action"] == "retain"
    assert retained["max_parent_child_um"] == 8.5
    unfiltered = lock_operating_point(
        summarise_priors([typical_wide, close], "unfiltered")
    )
    assert unfiltered["min_child_separation_um"] == 8.0
    typical_tight_sib = {
        "dataset": "44b6_sib",
        "n_children": 2,
        "dt": [1, 1],
        "parent_child_um": [4.8, 5.1],
        "sibling_um": [8.375],
        "sibling_alignment": [-0.8],
    }
    sib_update = retain_or_update_lock(previous, [typical_tight_sib])
    assert sib_update["min_child_separation_um"] == 8.3
    assert sib_update["min_child_separation_um"] >= 8.2
    assert sib_update["max_sibling_alignment"] == 0.05


def test_apply_operating_point_keeps_yaml_comments(tmp_path: Path):
    dest = tmp_path / "division_v2.yaml"
    dest.write_text(
        "linking:\n"
        "  # comment\n"
        "  max_parent_child_um: 8.5\n"
        "  min_child_separation_um: 8.7\n"
        "  max_sibling_alignment: 0.05\n"
    )
    apply_operating_point_to_yaml(
        dest,
        {"max_parent_child_um": 10.0, "min_child_separation_um": 8.7, "max_sibling_alignment": 0.05},
    )
    text = dest.read_text()
    assert "max_parent_child_um: 10.0" in text
    assert "# comment" in text


def test_inventory_geff_names_and_required_files(tmp_path: Path):
    dest = tmp_path / "inventory.json"
    dest.write_text(
        '{"datasets": [{"kind": "geff", "split": "train", "dataset": "44b6_aaa", "embryo": "44b6"},'
        '{"kind": "geff", "split": "train", "dataset": "6bba_bbb", "embryo": "6bba"},'
        '{"kind": "zarr", "split": "train", "dataset": "44b6_aaa", "embryo": "44b6"}]}'
    )
    assert inventory_geff_names(dest, specimen="44b6") == ["44b6_aaa"]
    required = REQUIRED_GEFF_FILES
    full = all_geff_files()
    assert required[0] == "zarr.json"
    assert "nodes/ids/c/0" in required
    assert len(required) < len(full)
    assert set(required).issubset(set(full))


def test_collect_uses_typical_lock_not_unfiltered(tmp_path: Path):
    payload = collect_geff_division_priors(tmp_path)
    lock = payload["locked_operating_point"]
    assert lock["max_parent_child_um"] == CURRENT_LOCKED_OPERATING_POINT["max_parent_child_um"]
    assert lock["min_child_separation_um"] == CURRENT_LOCKED_OPERATING_POINT["min_child_separation_um"]
    assert lock["action"] == "retain"
    assert payload["n_typical_events"] == 0
    assert payload["unfiltered_lock_rejected"]["do_not_use"] is True


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
