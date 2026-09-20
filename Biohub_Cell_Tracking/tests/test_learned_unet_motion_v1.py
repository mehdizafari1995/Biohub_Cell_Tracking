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
from biohub_tracking.learned_link import LOCKED_LINK_HASH as REJECTED_V1_HASH  # noqa: E402
from biohub_tracking.learned_link_v2 import LOCKED_LINK_V2_HASH as REJECTED_V2_HASH  # noqa: E402
from biohub_tracking.learned_motion import (  # noqa: E402
    apply_motion_relink_arrays,
    INFER_KERNEL_SLUG,
    INFER_LOCKED_TUNE,
    INFER_STEMS,
    LOCKED_MOTION_HASH,
    MOTION_KERNEL_SLUG,
    REJECTED_LINK_V1_HASH,
    REJECTED_LINK_V1_SLUG,
    REJECTED_LINK_V2_HASH,
    REJECTED_LINK_V2_SLUG,
    TRAIN_CONFIG_HASH as MOTION_TRAIN_HASH,
    TRAIN_KERNEL_SLUG,
    apply_motion_relink,
    assert_tuning_only,
    edge_geometry,
    find_infer_graphs,
    graph_from_payload,
    graph_payload,
    motion_config_hash,
    motion_config_payload,
    refuse_non_infer_stems,
    relink_summary,
    runtime_estimate,
    tuning_decision,
    write_graph_json,
)
from biohub_tracking.learned_split import HoldoutLeakError  # noqa: E402
from kaggle_learned_unet_motion_v1 import (  # noqa: E402
    CONFIG_LABEL,
    LOCKED_MOTION_HASH as NB_HASH,
    assert_no_submission_csv,
    apply_motion_relink_arrays as nb_apply,
    main as motion_main,
    motion_config_hash as nb_hash,
    motion_config_payload as nb_payload,
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


def test_motion_payload_matches_notebook_yaml_and_locked_hash():
    payload = motion_config_payload()
    assert payload == nb_payload()
    assert motion_config_hash(payload) == nb_hash(payload)
    assert motion_config_hash(payload) == LOCKED_MOTION_HASH
    assert LOCKED_MOTION_HASH == "66853b6f1445c18b"
    assert LOCKED_MOTION_HASH == NB_HASH
    assert LOCKED_MOTION_HASH != REJECTED_V1_HASH
    assert LOCKED_MOTION_HASH != REJECTED_V2_HASH
    assert REJECTED_LINK_V1_HASH == "3a7423c6858ace41"
    assert REJECTED_LINK_V2_HASH == "1d4015dc499bd604"
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_motion_v1.yaml").read_text())
    assert yaml_cfg["label"] == CONFIG_LABEL
    assert yaml_cfg["infer_stems"] == list(INFER_STEMS)
    assert yaml_cfg["writes_submission_csv"] is False
    assert yaml_cfg["kernel_slug"] == MOTION_KERNEL_SLUG
    assert yaml_cfg["train_kernel_slug"] == TRAIN_KERNEL_SLUG
    assert yaml_cfg["infer_kernel_slug"] == INFER_KERNEL_SLUG
    assert yaml_cfg["train_config_hash"] == TRAIN_CONFIG_HASH == MOTION_TRAIN_HASH
    assert yaml_cfg["infer_config_hash"] == LOCKED_INFER_HASH
    assert yaml_cfg["linker"] == "constant_velocity_greedy_1to1"
    assert yaml_cfg["motion_model"] == "constant_velocity"
    assert yaml_cfg["max_link_um"] == 12.0
    assert yaml_cfg["max_gap_frames"] == 0
    assert yaml_cfg["uses_transformer"] is False
    assert yaml_cfg["rejected_link_v1_slug"] == REJECTED_LINK_V1_SLUG
    assert yaml_cfg["rejected_link_v2_slug"] == REJECTED_LINK_V2_SLUG
    assert INFER_LOCKED_TUNE == 0.6557507105877639
    assert MOTION_KERNEL_SLUG != REJECTED_LINK_V1_SLUG
    assert MOTION_KERNEL_SLUG != REJECTED_LINK_V2_SLUG


def test_motion_refuses_holdout_and_non_tuning_stems():
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
        apply_motion_relink(_graph([1], [0], [8], [10], [10], [], dataset="6bba_05db0fb1"))
        raise AssertionError("expected HoldoutLeakError")
    except HoldoutLeakError:
        pass


def test_constant_velocity_prefers_predicted_cell_over_nearest():
    # Track A moves +y by 4 voxels/frame. A distractor sits still, then
    # jumps near the last A position so last-position greedy would steal it.
    graph = _graph(
        [1, 2, 3, 4, 5, 6],
        [0, 0, 1, 1, 2, 2],
        [8, 8, 8, 8, 8, 8],
        [10, 40, 14, 40, 18, 16],
        [20, 20, 20, 20, 20, 20],
        [(1, 3), (2, 4)],
    )
    out = apply_motion_relink(graph)
    children = out.out_edges()
    assert 3 in children.get(1, [])
    assert 5 in children.get(3, [])
    assert 4 in children.get(2, [])
    assert graph.n_nodes == out.n_nodes
    summary = relink_summary(graph, out)
    assert summary["detection_counts_match"] is True
    assert out.metadata["n_velocity_links"] >= 1
    nb_edges = nb_apply(
        graph.node_id, graph.t, graph.z, graph.y, graph.x
    )
    assert (3, 5) in nb_edges


def test_first_link_is_last_position_and_gate_is_12um():
    near = _graph(
        [1, 2],
        [0, 1],
        [8, 8],
        [10, 12],
        [20, 20],
        [],
    )
    linked = apply_motion_relink(near)
    assert list(map(tuple, linked.edges.tolist())) == [(1, 2)]
    far = _graph(
        [1, 2],
        [0, 1],
        [8, 8],
        [10, 200],
        [20, 200],
        [(1, 2)],
    )
    unlinked = apply_motion_relink(far)
    assert unlinked.n_edges == 0
    assert unlinked.n_nodes == 2
    geom = edge_geometry(linked)
    assert geom["dt_unique"] == [1]
    assert geom["n_over_12um"] == 0
    assert geom["max_out_degree"] == 1
    # High velocity must not punch through the locked last-position 12 µm gate.
    # Last A at y=10 then y=14; prediction y=18. Far cell at y=80 is near a
    # runaway forecast if the last-position gate is omitted.
    runaway = _graph(
        [1, 2, 3],
        [0, 1, 2],
        [8, 8, 8],
        [10, 14, 80],
        [20, 20, 20],
        [],
    )
    gated = apply_motion_relink(runaway)
    assert list(map(tuple, gated.edges.tolist())) == [(1, 2)]
    assert 3 not in gated.out_edges().get(2, [])


def test_relink_preserves_detections_and_is_one_to_one():
    graph = _graph(
        [10, 11, 12, 13],
        [0, 0, 1, 1],
        [8, 8, 8, 8],
        [10, 40, 11, 41],
        [20, 20, 20, 20],
        [(10, 13), (11, 12)],
    )
    out = apply_motion_relink(graph)
    assert set(int(x) for x in out.node_id) == {10, 11, 12, 13}
    children = out.out_edges()
    assert children.get(10) == [12]
    assert children.get(11) == [13]
    assert all(len(v) <= 1 for v in children.values())
    assert all(len(v) <= 1 for v in out.in_edges().values())
    assert tuning_decision(0.6) == "reject_learned_unet_motion_v1"
    assert tuning_decision(INFER_LOCKED_TUNE) == "keep_candidate_learned_unet_motion_v1"


def test_runtime_estimate_tuning_and_hidden_under_12h():
    tune = runtime_estimate(2)
    assert tune["under_limit"] is True
    assert tune["estimate_hours"] < 0.2
    hidden = runtime_estimate(199)
    assert hidden["under_limit"] is True
    assert hidden["estimate_hours"] < 12.0


def test_kernel_is_helper_not_submission_and_packages_cpu(tmp_path: Path, monkeypatch):
    source = (ROOT / "notebooks" / "kaggle_learned_unet_motion_v1.py").read_text()
    assert "to_csv(" not in source
    assert "graphs_to_submission" not in source
    assert "writes_submission_csv" in source
    assert "constant_velocity" in source
    assert "uses_transformer" in source
    spec = _make_mod.KERNELS["learned_unet_motion_v1"]
    assert spec["kernel_id"] == MOTION_KERNEL_SLUG
    assert spec.get("enable_gpu") is not True
    assert INFER_KERNEL_SLUG in spec.get("kernel_sources", [])
    meta = _make_mod.metadata_payload(spec)
    assert meta["enable_gpu"] is False
    assert meta["enable_internet"] is False
    assert meta["is_private"] is True
    assert meta["id"] == MOTION_KERNEL_SLUG
    assert INFER_KERNEL_SLUG in meta["kernel_sources"]
    nb = _make_mod.build_notebook(source, spec["markdown"], enable_gpu=False)
    assert nb["metadata"]["kaggle"]["accelerator"] == "none"
    assert nb["metadata"]["kaggle"]["isInternetEnabled"] is False

    monkeypatch.setenv("BIOHUB_LEARNED_DRY_RUN", "1")
    monkeypatch.setenv("BIOHUB_WORKING", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    result = motion_main()
    assert result["status"] == "dry_run"
    assert result["config_hash"] == LOCKED_MOTION_HASH
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
    monkeypatch.delenv("BIOHUB_INFER_GRAPH_DIR", raising=False)
    monkeypatch.setenv("BIOHUB_WORKING", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    result = motion_main()
    assert result["status"] == "blocked_no_graphs"
    assert result["config_hash"] == LOCKED_MOTION_HASH
    assert result["is_competition_submission"] is False
    assert not (tmp_path / "submission.csv").exists()
    assert_no_submission_csv(tmp_path)
    empty = find_infer_graphs([tmp_path])
    assert empty == {}


def test_rejected_motion_tune_report_stays_44b6_and_below_infer():
    report = json.loads((ROOT / "reports" / "learned_unet_motion_v1_tune.json").read_text())
    assert report["decision"] == "reject_learned_unet_motion_v1"
    assert report["config_hash"] == LOCKED_MOTION_HASH
    assert report["learned_combined"] < INFER_LOCKED_TUNE
    assert report["is_leaderboard_score"] is False
    stems = [row["dataset"] for row in report["per_sequence"]]
    assert stems == list(INFER_STEMS)
    assert all(row["num_pred_nodes"] > 0 for row in report["per_sequence"])
    error = json.loads((ROOT / "reports" / "learned_unet_motion_v1_error.json").read_text())
    assert error["n_edges"] == [74025, 32123]
    assert error["detection_counts_match_infer"] is True
    assert error["opened_holdout"] is False
    assert error["n_over_12um"] == [0, 0]
    assert error["failure_mode"] == "constant_velocity_ranking_worse_than_last_position_greedy"


def test_push_forbids_locked_slugs_and_fetch_allows_motion():
    spec = importlib.util.spec_from_file_location(
        "push_kaggle_kernel",
        ROOT / "scripts" / "push_kaggle_kernel.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert TRAIN_KERNEL_SLUG in mod.FORBIDDEN_SLUGS
    assert INFER_KERNEL_SLUG in mod.FORBIDDEN_SLUGS
    assert "mahdimzafari/biohub-learned-unet-gap-test-v1" in mod.FORBIDDEN_SLUGS
    assert REJECTED_LINK_V1_SLUG in mod.FORBIDDEN_SLUGS
    assert REJECTED_LINK_V2_SLUG in mod.FORBIDDEN_SLUGS
    assert MOTION_KERNEL_SLUG not in mod.FORBIDDEN_SLUGS
    fetch = importlib.util.spec_from_file_location(
        "fetch_kernel_output",
        ROOT / "scripts" / "fetch_kernel_output.py",
    )
    assert fetch and fetch.loader
    fetch_mod = importlib.util.module_from_spec(fetch)
    fetch.loader.exec_module(fetch_mod)
    assert MOTION_KERNEL_SLUG in fetch_mod.ALLOWED_SLUGS
    assert apply_motion_relink_arrays is not None
    assert nb_apply is not None
