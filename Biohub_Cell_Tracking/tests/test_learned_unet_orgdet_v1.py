from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from biohub_tracking.graphs import TrackGraph  # noqa: E402
from biohub_tracking.learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH  # noqa: E402
from biohub_tracking.learned_orgop import LOCKED_ORGOP_HASH as REJECTED_ORGOP_HASH  # noqa: E402
from biohub_tracking.learned_orgdet import (  # noqa: E402
    apply_orgdet_relink_arrays,
    INFER_KERNEL_SLUG,
    INFER_LOCKED_TUNE,
    INFER_STEMS,
    LOCKED_ORGDET_HASH,
    ORGDET_KERNEL_SLUG,
    ORGOP_KERNEL_SLUG,
    TRAIN_CONFIG_HASH as ORGDET_TRAIN_HASH,
    TRAIN_KERNEL_SLUG,
    apply_orgdet_relink,
    assert_tuning_only,
    edge_geometry,
    find_orgop_graphs,
    graph_from_payload,
    graph_payload,
    orgdet_config_hash,
    orgdet_config_payload,
    refuse_non_infer_stems,
    relink_summary,
    runtime_estimate,
    tuning_decision,
    write_graph_json,
)
from biohub_tracking.learned_split import HoldoutLeakError  # noqa: E402
from kaggle_learned_unet_orgdet_v1 import (  # noqa: E402
    CONFIG_LABEL,
    LOCKED_ORGDET_HASH as NB_HASH,
    assert_no_submission_csv,
    apply_orgdet_relink_arrays as nb_apply,
    main as orgdet_main,
    orgdet_config_hash as nb_hash,
    orgdet_config_payload as nb_payload,
)

_MAKE = importlib.util.spec_from_file_location("make_kaggle_kernel", ROOT / "scripts" / "make_kaggle_kernel.py")
assert _MAKE and _MAKE.loader
_make_mod = importlib.util.module_from_spec(_MAKE)
_MAKE.loader.exec_module(_make_mod)


def _graph(node_id, t, z, y, x, edges, dataset="44b6_12dfb391") -> TrackGraph:
    return TrackGraph(
        dataset=dataset,
        node_id=node_id,
        t=t,
        z=z,
        y=y,
        x=x,
        edges=edges,
    )


def test_orgdet_payload_matches_notebook_yaml_and_locked_hash():
    payload = orgdet_config_payload()
    assert payload == nb_payload()
    assert orgdet_config_hash(payload) == nb_hash(payload)
    assert orgdet_config_hash(payload) == LOCKED_ORGDET_HASH
    assert LOCKED_ORGDET_HASH == "80b20589198148b7"
    assert LOCKED_ORGDET_HASH == NB_HASH
    assert LOCKED_ORGDET_HASH != REJECTED_ORGOP_HASH
    assert REJECTED_ORGOP_HASH == "250c5bfacd4a8198"
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_orgdet_v1.yaml").read_text())
    assert yaml_cfg["label"] == CONFIG_LABEL
    assert yaml_cfg["infer_stems"] == list(INFER_STEMS)
    assert yaml_cfg["writes_submission_csv"] is False
    assert yaml_cfg["kernel_slug"] == ORGDET_KERNEL_SLUG
    assert yaml_cfg["train_kernel_slug"] == TRAIN_KERNEL_SLUG
    assert yaml_cfg["infer_kernel_slug"] == INFER_KERNEL_SLUG
    assert yaml_cfg["orgop_kernel_slug"] == ORGOP_KERNEL_SLUG
    assert yaml_cfg["train_config_hash"] == TRAIN_CONFIG_HASH == ORGDET_TRAIN_HASH
    assert yaml_cfg["infer_config_hash"] == LOCKED_INFER_HASH
    assert yaml_cfg["rejected_orgop_hash"] == REJECTED_ORGOP_HASH
    assert yaml_cfg["linker"] == "greedy_gate_12um"
    assert yaml_cfg["motion_model"] == "last_position"
    assert yaml_cfg["assignment"] == "greedy"
    assert yaml_cfg["max_link_um"] == 12.0
    assert yaml_cfg["max_gap_frames"] == 0
    assert yaml_cfg["uses_transformer"] is False
    assert yaml_cfg["uses_ilp"] is False
    assert yaml_cfg["retunes_detection"] is False
    assert yaml_cfg["source_graphs"] == "locked_learned_unet_orgop_graphs"
    assert yaml_cfg["det_threshold"] == 0.99
    assert yaml_cfg["pool_kernel_um"] == 3.0
    assert INFER_LOCKED_TUNE == 0.6557507105877639
    assert ORGDET_KERNEL_SLUG != ORGOP_KERNEL_SLUG
    assert ORGDET_KERNEL_SLUG != INFER_KERNEL_SLUG


def test_orgdet_refuses_holdout_and_non_tuning_stems():
    refuse_non_infer_stems(INFER_STEMS)
    try:
        refuse_non_infer_stems(["6bba_05db0fb1"])
        raise AssertionError("expected HoldoutLeakError")
    except HoldoutLeakError:
        pass
    try:
        refuse_non_infer_stems(["44b6_aaa"])
        raise AssertionError("expected HoldoutLeakError")
    except HoldoutLeakError:
        pass
    try:
        assert_tuning_only("validation")
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert "holdout" in str(exc)
    assert_tuning_only("tuning")
    try:
        apply_orgdet_relink(_graph([1], [0], [8], [10], [10], [], dataset="6bba_05db0fb1"))
        raise AssertionError("expected HoldoutLeakError")
    except HoldoutLeakError:
        pass


def test_greedy_is_one_to_one_and_gate_is_12um():
    near = _graph(
        [1, 2],
        [0, 1],
        [8, 8],
        [10, 12],
        [20, 20],
        [],
    )
    linked = apply_orgdet_relink(near)
    assert list(map(tuple, linked.edges.tolist())) == [(1, 2)]
    far = _graph(
        [1, 2],
        [0, 1],
        [8, 8],
        [10, 200],
        [20, 200],
        [(1, 2)],
    )
    unlinked = apply_orgdet_relink(far)
    assert unlinked.n_edges == 0
    assert unlinked.n_nodes == 2
    geom = edge_geometry(linked)
    assert geom["dt_unique"] == [1]
    assert geom["n_over_12um"] == 0
    assert geom["max_out_degree"] == 1
    runaway = _graph(
        [1, 2, 3],
        [0, 1, 2],
        [8, 8, 8],
        [10, 14, 80],
        [20, 20, 20],
        [],
    )
    gated = apply_orgdet_relink(runaway)
    assert list(map(tuple, gated.edges.tolist())) == [(1, 2)]
    assert 3 not in gated.out_edges().get(2, [])
    skipped = _graph(
        [1, 2],
        [0, 2],
        [8, 8],
        [10, 12],
        [20, 20],
        [],
    )
    assert apply_orgdet_relink(skipped).n_edges == 0


def test_relink_preserves_detections_and_is_deterministic():
    graph = _graph(
        [10, 11, 12, 13],
        [0, 0, 1, 1],
        [8, 8, 8, 8],
        [10, 40, 11, 41],
        [20, 20, 20, 20],
        [],
    )
    out = apply_orgdet_relink(graph)
    assert set(int(x) for x in out.node_id) == {10, 11, 12, 13}
    children = out.out_edges()
    assert children.get(10) == [12]
    assert children.get(11) == [13]
    assert all(len(v) <= 1 for v in children.values())
    assert all(len(v) <= 1 for v in out.in_edges().values())
    again = apply_orgdet_relink(graph)
    assert list(map(tuple, again.edges.tolist())) == list(map(tuple, out.edges.tolist()))
    summary = relink_summary(graph, out)
    assert summary["detection_counts_match"] is True
    assert summary["n_edges_before"] == 0
    assert summary["n_new_edges"] == 2
    nb_edges = nb_apply(graph.node_id, graph.t, graph.z, graph.y, graph.x)
    assert set(nb_edges) == set(tuple(row) for row in out.edges.tolist())
    assert apply_orgdet_relink_arrays is not None
    assert tuning_decision(0.6) == "reject_learned_unet_orgdet_v1"
    assert tuning_decision(INFER_LOCKED_TUNE) == "reject_learned_unet_orgdet_v1"
    assert tuning_decision(INFER_LOCKED_TUNE + 1e-9) == "keep_candidate_learned_unet_orgdet_v1"


def test_runtime_estimate_tuning_and_hidden_under_12h():
    tune = runtime_estimate(2)
    assert tune["under_limit"] is True
    assert tune["estimate_hours"] < 0.2
    hidden = runtime_estimate(199)
    assert hidden["under_limit"] is True
    assert hidden["estimate_hours"] < 12.0


def test_kernel_is_helper_not_submission_and_packages_cpu(tmp_path: Path, monkeypatch):
    source = (ROOT / "notebooks" / "kaggle_learned_unet_orgdet_v1.py").read_text()
    assert "to_csv(" not in source
    assert "graphs_to_submission" not in source
    assert "writes_submission_csv" in source
    assert "greedy_gate_12um" in source
    assert "uses_transformer" in source
    assert "uses_ilp" in source
    spec = _make_mod.KERNELS["learned_unet_orgdet_v1"]
    assert spec["kernel_id"] == ORGDET_KERNEL_SLUG
    assert spec.get("enable_gpu") is not True
    assert ORGOP_KERNEL_SLUG in spec.get("kernel_sources", [])
    meta = _make_mod.metadata_payload(spec)
    assert meta["enable_gpu"] is False
    assert meta["enable_internet"] is False
    assert meta["is_private"] is True
    assert meta["id"] == ORGDET_KERNEL_SLUG
    assert ORGOP_KERNEL_SLUG in meta["kernel_sources"]
    nb = _make_mod.build_notebook(source, spec["markdown"], enable_gpu=False)
    assert nb["metadata"]["kaggle"]["accelerator"] == "none"
    assert nb["metadata"]["kaggle"]["isInternetEnabled"] is False

    monkeypatch.setenv("BIOHUB_LEARNED_DRY_RUN", "1")
    monkeypatch.setenv("BIOHUB_WORKING", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    result = orgdet_main()
    assert result["status"] == "dry_run"
    assert result["config_hash"] == LOCKED_ORGDET_HASH
    assert result["is_competition_submission"] is False
    assert not (tmp_path / "submission.csv").exists()
    assert_no_submission_csv(tmp_path)
    payload = graph_payload("44b6_12dfb391", np.array([[0, 1, 2, 3]]), [(1, 1)])
    path = write_graph_json(tmp_path / "44b6_12dfb391.json", payload)
    graph = graph_from_payload(json.loads(path.read_text()))
    assert graph.dataset == "44b6_12dfb391"
    try:
        write_graph_json(tmp_path / "bad.json", graph_payload("6bba_05db0fb1", np.array([[0, 1, 2, 3]]), []))
        raise AssertionError("expected HoldoutLeakError")
    except HoldoutLeakError:
        pass


def test_blocked_no_graphs_writes_no_submission(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("BIOHUB_LEARNED_DRY_RUN", raising=False)
    monkeypatch.setenv("BIOHUB_WORKING", str(tmp_path))
    monkeypatch.setenv("BIOHUB_ORGOP_GRAPH_DIR", str(tmp_path / "missing"))
    monkeypatch.chdir(tmp_path)
    result = orgdet_main()
    assert result["status"] == "blocked_no_graphs"
    assert not (tmp_path / "submission.csv").exists()
    assert (tmp_path / "infer_manifest.json").exists()


def test_find_orgop_graphs_skips_zarr_and_holdout(tmp_path: Path):
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    (graphs / "44b6_12dfb391.json").write_text(json.dumps({"dataset": "44b6_12dfb391"}))
    (graphs / "44b6_267148e4.json").write_text(json.dumps({"dataset": "44b6_267148e4"}))
    zarr = tmp_path / "44b6_12dfb391.zarr"
    zarr.mkdir()
    (zarr / "44b6_12dfb391.json").write_text("{}")
    found = find_orgop_graphs([tmp_path])
    assert set(found) == set(INFER_STEMS)
    assert "zarr" not in found["44b6_12dfb391"].as_posix()


def test_push_forbids_locked_slugs_and_fetch_allows_orgdet():
    spec = importlib.util.spec_from_file_location(
        "push_kaggle_kernel",
        ROOT / "scripts" / "push_kaggle_kernel.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert TRAIN_KERNEL_SLUG in mod.FORBIDDEN_SLUGS
    assert INFER_KERNEL_SLUG in mod.FORBIDDEN_SLUGS
    assert ORGOP_KERNEL_SLUG in mod.FORBIDDEN_SLUGS
    assert ORGDET_KERNEL_SLUG not in mod.FORBIDDEN_SLUGS
    fetch = importlib.util.spec_from_file_location(
        "fetch_kernel_output",
        ROOT / "scripts" / "fetch_kernel_output.py",
    )
    assert fetch and fetch.loader
    fetch_mod = importlib.util.module_from_spec(fetch)
    fetch.loader.exec_module(fetch_mod)
    assert ORGDET_KERNEL_SLUG in fetch_mod.ALLOWED_SLUGS
    assert ORGOP_KERNEL_SLUG in fetch_mod.ALLOWED_SLUGS
    assert apply_orgdet_relink_arrays is not None
    assert nb_apply is not None


def test_keep_orgdet_tune_report_stays_44b6_and_beats_infer():
    report = json.loads((ROOT / "reports" / "learned_unet_orgdet_v1_tune.json").read_text())
    assert report["decision"] == "keep_candidate_learned_unet_orgdet_v1"
    assert report["config_hash"] == LOCKED_ORGDET_HASH
    assert report["learned_combined"] > INFER_LOCKED_TUNE
    assert report["is_leaderboard_score"] is False
    stems = [row["dataset"] for row in report["per_sequence"]]
    assert stems == list(INFER_STEMS)
    assert all(row["num_pred_nodes"] > 0 for row in report["per_sequence"])
    assert not (ROOT / "reports" / "learned_unet_orgdet_v1_holdout.json").exists()
    error = json.loads((ROOT / "reports" / "learned_unet_orgdet_v1_error.json").read_text())
    assert error["n_edges"] == [58617, 25440]
    assert error["detection_counts_match_orgop"] is True
    assert error["opened_holdout"] is False
    assert error["n_over_12um"] == [0, 0]
    assert error["official_tp_fp_fn"] == [877, 220, 173]
