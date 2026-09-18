from biohub_tracking.graphs import TrackGraph
from biohub_tracking.submit import graphs_to_submission, read_submission, validate_submission, write_submission


def _toy_graph() -> TrackGraph:
    return TrackGraph(
        dataset="44b6_demo",
        node_id=[1, 2, 3],
        t=[0, 1, 2],
        z=[32, 33, 34],
        y=[128, 130, 131],
        x=[128, 125, 124],
        edges=[(1, 2), (2, 3)],
    )


def test_sample_submission_schema(tmp_path):
    df = graphs_to_submission([_toy_graph()])
    errors = validate_submission(df, required_datasets=["44b6_demo"])
    assert errors == []
    path = write_submission(df, tmp_path / "submission.csv")
    loaded = read_submission(path)
    assert list(loaded.columns) == [
        "id",
        "dataset",
        "row_type",
        "node_id",
        "t",
        "z",
        "y",
        "x",
        "source_id",
        "target_id",
    ]
    assert loaded.loc[0, "row_type"] == "node"
    assert loaded.loc[loaded["row_type"] == "edge", "node_id"].eq(-1).all()


def test_official_sample_submission_roundtrip():
    from pathlib import Path

    sample = Path(__file__).resolve().parents[1] / "data" / "sample_submission.csv"
    if not sample.exists():
        return
    df = read_submission(sample)
    errors = validate_submission(
        df,
        required_datasets=["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"],
    )
    assert errors == []
