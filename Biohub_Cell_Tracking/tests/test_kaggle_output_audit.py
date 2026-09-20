from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from biohub_tracking.graphs import TrackGraph
from biohub_tracking.kaggle_output_audit import (
    FROZEN_VALIDATION,
    PUBLIC_TEST_DATASETS,
    assemble_kaggle_output_report,
    classify_dataset,
    extract_kernel_output,
    independent_audit,
    score_graphs_against_geffs,
    summarize_submission,
)
from biohub_tracking.official_eval import official_scorer_available
from biohub_tracking.submit import graphs_to_submission


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


def _linear_public_test() -> list[TrackGraph]:
    graphs = []
    for i, name in enumerate(PUBLIC_TEST_DATASETS):
        base = 10 + i
        graphs.append(
            _graph(
                name,
                [
                    (1, 0, base, 20, 30),
                    (2, 1, base, 21, 30),
                    (3, 2, base, 22, 30),
                ],
                [(1, 2), (2, 3)],
                estimated=3,
            )
        )
    return graphs


def test_public_test_roles_match_frozen_split():
    assert classify_dataset("6bba_05db0fb1") == "frozen_validation_overlap"
    assert classify_dataset("6bba_c328f2fd") == "frozen_validation_overlap"
    assert classify_dataset("6bba_05b6850b") == "validation_specimen_not_in_locked_split"
    assert classify_dataset("44b6_0113de3b") == "tuning_specimen_public_test_copy"
    assert "6bba_c328f2fd" in FROZEN_VALIDATION
    assert "6bba_c328f2fd" not in PUBLIC_TEST_DATASETS


def test_independent_audit_accepts_valid_public_test_submission():
    df = graphs_to_submission(_linear_public_test())
    result = independent_audit(df)
    assert result["audit_ok"] is True
    assert result["errors"] == []
    assert result["leaderboard_score"] is None
    assert result["summary"]["n_predicted_divisions"] == 0
    assert result["summary"]["n_gap_edges"] == 0
    assert result["summary"]["datasets"] == list(PUBLIC_TEST_DATASETS)
    roles = {row["dataset"]: row["role"] for row in result["summary"]["per_dataset"]}
    assert roles["6bba_05db0fb1"] == "frozen_validation_overlap"


def test_independent_audit_rejects_gap_edge_and_division():
    graphs = _linear_public_test()
    graphs[0] = _graph(
        PUBLIC_TEST_DATASETS[0],
        [(1, 0, 10, 20, 30), (2, 2, 10, 22, 30)],
        [(1, 2)],
        estimated=2,
    )
    result = independent_audit(graphs_to_submission(graphs))
    assert result["audit_ok"] is False
    assert any("dt=1" in err for err in result["errors"])

    divided = _linear_public_test()
    divided[1] = _graph(
        PUBLIC_TEST_DATASETS[1],
        [
            (1, 0, 10, 20, 30),
            (2, 1, 10, 18, 30),
            (3, 1, 10, 22, 30),
        ],
        [(1, 2), (1, 3)],
        estimated=3,
    )
    result = independent_audit(graphs_to_submission(divided))
    assert result["audit_ok"] is False
    assert any("must not predict divisions" in err for err in result["errors"])


def test_independent_audit_rejects_non_integer_and_missing_dataset():
    df = graphs_to_submission(_linear_public_test())
    df["z"] = df["z"].astype("float64")
    df.loc[df.index[0], "z"] = 10.4
    result = independent_audit(df)
    assert result["audit_ok"] is False
    assert any("integer" in err for err in result["errors"])

    df = graphs_to_submission(_linear_public_test()[:3])
    result = independent_audit(df)
    assert result["audit_ok"] is False
    assert any("missing datasets" in err for err in result["errors"])


def test_worker_manifest_mismatch_is_a_warning():
    df = graphs_to_submission(_linear_public_test())
    result = independent_audit(df, worker_manifest={"n_nodes": 999, "n_edges": 1, "rows": 1})
    assert result["audit_ok"] is True
    assert result["warnings"]
    assert any("n_nodes" in warning for warning in result["warnings"])


def test_extract_kernel_output_keeps_only_expected_files(tmp_path):
    archive = tmp_path / "out.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("submission.csv", "id,dataset\n")
        zf.writestr("nested/run_manifest.json", json.dumps({"audit_ok": True}))
    written = extract_kernel_output(archive, tmp_path / "extracted")
    assert set(written) == {"submission.csv", "run_manifest.json"}
    assert written["submission.csv"].read_text().startswith("id,dataset")

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("weights.bin", "nope")
    with pytest.raises(ValueError, match="unexpected"):
        extract_kernel_output(bad, tmp_path / "bad_out")


def test_missing_geff_does_not_invent_an_official_score():
    graphs = _linear_public_test()
    rows = score_graphs_against_geffs(graphs, {})
    assert all(row["status"] == "missing_geff" for row in rows)
    assert all(row["official"] is None for row in rows)
    report = assemble_kaggle_output_report(
        {"audit_ok": True, "errors": [], "warnings": [], "summary": summarize_submission(graphs_to_submission(graphs))},
        rows,
        kernel_slug="mahdimzafari/biohub-association-v2",
        kernel_version=3,
        config_hash="2c6966b5204ba542",
        code_version="test",
    )
    assert report["official_summary_all_scored"] is None
    assert report["n_holdout_scored"] == 0
    assert report["complete_frozen_split"] is False
    assert report["public_lb_score"] is None
    assert report["private_lb_score"] is None
    assert report["retune_after_holdout"] is False


@pytest.mark.skipif(not official_scorer_available(), reason="official scorer source missing")
def test_official_score_of_submission_graph_is_not_a_leaderboard_number():
    pred = _graph(
        "6bba_05db0fb1",
        [(1, 0, 10, 10, 10), (2, 1, 10, 10, 10)],
        [(1, 2)],
        estimated=2,
    )
    gt = pred.copy()
    from biohub_tracking.official_eval import evaluate_official

    result = evaluate_official(pred, gt)
    row = {
        "dataset": "6bba_05db0fb1",
        "role": "frozen_validation_overlap",
        "status": "scored",
        "official": result.as_dict(),
        "local_reference": {"official_combined": 0.564},
        "delta_vs_local_reference": result.score - 0.564,
    }
    report = assemble_kaggle_output_report(
        {"audit_ok": True, "errors": [], "warnings": [], "summary": {"n_nodes": 2}},
        [row],
        kernel_slug="mahdimzafari/biohub-association-v2",
        kernel_version=3,
        config_hash="2c6966b5204ba542",
        code_version="test",
    )
    assert report["n_holdout_scored"] == 1
    assert report["public_test_overlap_scored"] is True
    assert report["complete_frozen_split"] is False
    assert report["public_lb_score"] is None
    assert report["official_summary_all_scored"]["score"] == pytest.approx(result.score)
    assert "not a Kaggle leaderboard score" in report["note"]
