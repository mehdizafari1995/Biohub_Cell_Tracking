from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

from kaggle_baseline_v1 import COLUMNS, audit_submission, config_hash, config_payload  # noqa: E402
from biohub_tracking.graphs import TrackGraph
from biohub_tracking.submit import graphs_to_submission, validate_submission


def _valid_df() -> pd.DataFrame:
    graph = TrackGraph(
        dataset="44b6_0113de3b",
        node_id=[1, 2, 3],
        t=[0, 1, 2],
        z=[10, 11, 12],
        y=[20, 21, 22],
        x=[30, 31, 32],
        edges=[(1, 2), (2, 3)],
    )
    return graphs_to_submission([graph])


def test_baseline_v1_config_is_frozen():
    payload = config_payload()
    assert payload["label"] == "baseline_v1"
    assert payload["anisotropic"] is True
    assert payload["suppress_duplicates"] is True
    assert payload["min_distance_um"] == 4.0
    assert payload["linking"] == "greedy"
    assert payload["allow_divisions"] is False
    assert config_hash(payload) == config_hash(config_payload())


def test_audit_accepts_valid_submission():
    df = _valid_df()
    errors = audit_submission(
        df,
        discovered=["44b6_0113de3b"],
        shapes={"44b6_0113de3b": (100, 64, 256, 256)},
        sample_names=["44b6_0113de3b", "44b6_0b24845f"],
    )
    assert errors == []
    assert list(df.columns) == COLUMNS


def test_audit_rejects_backward_edge_and_duplicate_node():
    df = _valid_df()
    df.loc[df["row_type"] == "edge", "source_id"] = [3, 2]
    df.loc[df["row_type"] == "edge", "target_id"] = [1, 3]
    errors = audit_submission(
        df,
        discovered=["44b6_0113de3b"],
        shapes={"44b6_0113de3b": (100, 64, 256, 256)},
        sample_names=["44b6_0113de3b"],
    )
    assert any("precede" in e for e in errors)

    df = _valid_df()
    df.loc[0, "node_id"] = 2
    errors = audit_submission(
        df,
        discovered=["44b6_0113de3b"],
        shapes={"44b6_0113de3b": (100, 64, 256, 256)},
        sample_names=[],
    )
    assert any("duplicate node_id" in e for e in errors)


def test_audit_requires_sample_datasets_that_exist_in_test():
    df = _valid_df()
    errors = audit_submission(
        df,
        discovered=["44b6_0113de3b", "6bba_05db0fb1"],
        shapes={"44b6_0113de3b": (100, 64, 256, 256), "6bba_05db0fb1": (100, 64, 256, 256)},
        sample_names=["44b6_0113de3b", "6bba_05db0fb1"],
    )
    assert any("missing discovered datasets" in e for e in errors)


def test_validate_submission_empty_and_out_of_bounds():
    empty = pd.DataFrame(columns=COLUMNS)
    assert validate_submission(empty) == ["submission.csv is empty"]
    df = _valid_df()
    df.loc[0, "z"] = 99
    errors = validate_submission(df, volume_shapes={"44b6_0113de3b": (3, 64, 256, 256)})
    assert any("z outside bounds" in e for e in errors)
