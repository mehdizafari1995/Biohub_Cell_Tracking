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

from biohub_tracking.learned_link import LOCKED_LINK_HASH as REJECTED_V1_HASH  # noqa: E402
from biohub_tracking.learned_link import assign_transformer_links as assign_v1  # noqa: E402
from biohub_tracking.learned_link_v2 import (  # noqa: E402
    INFER_KERNEL_SLUG,
    INFER_LOCKED_TUNE,
    INFER_STEMS,
    LINK_KERNEL_SLUG,
    LOCKED_INFER_HASH,
    LOCKED_LINK_V2_HASH,
    REJECTED_LINK_V1_HASH,
    REJECTED_LINK_V1_SLUG,
    TRAIN_CONFIG_HASH,
    TRAIN_KERNEL_SLUG,
    assign_transformer_links_v2,
    find_weights,
    graph_from_payload,
    graph_payload,
    link_config_hash,
    link_config_payload,
    pairwise_distances_um,
    refuse_non_infer_stems,
    runtime_estimate,
    verify_train_manifest,
    write_graph_json,
    assert_tuning_only,
)
from biohub_tracking.learned_split import HoldoutLeakError  # noqa: E402
from kaggle_learned_unet_link_v2 import (  # noqa: E402
    CONFIG_LABEL,
    assert_no_submission_csv,
    assign_transformer_links as nb_assign,
    link_config_hash as nb_hash,
    link_config_payload as nb_payload,
    main as link_main,
    pairwise_distances_um as nb_distances,
)

_MAKE = importlib.util.spec_from_file_location("make_kaggle_kernel", ROOT / "scripts" / "make_kaggle_kernel.py")
assert _MAKE and _MAKE.loader
_make_mod = importlib.util.module_from_spec(_MAKE)
_MAKE.loader.exec_module(_make_mod)


def test_link_v2_payload_matches_notebook_yaml_and_locked_hash():
    payload = link_config_payload()
    assert payload == nb_payload()
    assert link_config_hash(payload) == nb_hash(payload)
    assert link_config_hash(payload) == LOCKED_LINK_V2_HASH
    assert LOCKED_LINK_V2_HASH == "1d4015dc499bd604"
    assert LOCKED_LINK_V2_HASH != REJECTED_V1_HASH
    assert REJECTED_LINK_V1_HASH == "3a7423c6858ace41"
    assert REJECTED_V1_HASH == "3a7423c6858ace41"
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_link_v2.yaml").read_text())
    assert yaml_cfg["label"] == CONFIG_LABEL
    assert yaml_cfg["infer_stems"] == list(INFER_STEMS)
    assert yaml_cfg["writes_submission_csv"] is False
    assert yaml_cfg["kernel_slug"] == LINK_KERNEL_SLUG
    assert yaml_cfg["train_kernel_slug"] == TRAIN_KERNEL_SLUG
    assert yaml_cfg["infer_kernel_slug"] == INFER_KERNEL_SLUG
    assert yaml_cfg["train_config_hash"] == TRAIN_CONFIG_HASH
    assert yaml_cfg["infer_config_hash"] == LOCKED_INFER_HASH
    assert yaml_cfg["linker"] == "transformer_softmax_greedy_1to1_spatial"
    assert yaml_cfg["edge_activation"] == "softmax"
    assert yaml_cfg["edge_threshold"] is None
    assert yaml_cfg["max_link_um"] == 12.0
    assert yaml_cfg["rejected_link_v1_slug"] == REJECTED_LINK_V1_SLUG
    assert yaml_cfg["rejected_link_v1_hash"] == REJECTED_LINK_V1_HASH
    assert yaml_cfg["normalize"] == "window_minmax"
    assert INFER_LOCKED_TUNE == 0.6557507105877639
    assert LINK_KERNEL_SLUG != REJECTED_LINK_V1_SLUG


def test_link_v2_refuses_holdout_and_non_tuning_stems():
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


def test_v2_assignment_has_no_probability_cutoff_and_is_one_to_one():
    probs = np.array(
        [
            [0.12, 0.04, 0.01],
            [0.03, 0.11, 0.02],
            [0.01, 0.01, 0.09],
        ]
    )
    assert assign_v1(probs, threshold=0.5) == []
    links = assign_transformer_links_v2(probs)
    assert links[0][:2] == (0, 0)
    assert links[1][:2] == (1, 1)
    assert links[2][:2] == (2, 2)
    assert all(p < 0.5 for _i, _j, p in links)
    sources = [i for i, _j, _p in links]
    targets = [j for _i, j, _p in links]
    assert len(sources) == len(set(sources))
    assert len(targets) == len(set(targets))
    assert nb_assign(probs) == links
    empty = assign_transformer_links_v2(np.zeros((0, 0)))
    assert empty == []


def test_v2_spatial_gate_blocks_long_range_softmax_winners():
    scores = np.array([[0.9, 0.1], [0.2, 0.8]])
    distances = np.array([[40.0, 5.0], [6.0, 50.0]])
    ungated = assign_transformer_links_v2(scores)
    assert ungated[0][:2] == (0, 0)
    gated = assign_transformer_links_v2(scores, distances_um=distances, max_link_um=12.0)
    assert [(i, j) for i, j, _p in gated] == [(1, 0), (0, 1)]
    none = assign_transformer_links_v2(scores, distances_um=np.full((2, 2), 20.0), max_link_um=12.0)
    assert none == []
    src = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    tgt = np.array([[0.0, 0.0, 0.0]])
    dist = pairwise_distances_um(src, tgt, (1.0, 1.0, 0.5))
    assert dist.shape == (2, 1)
    assert abs(float(dist[0, 0]) - 0.0) < 1e-9
    assert abs(float(dist[1, 0]) - 5.0) < 1e-9
    assert np.allclose(nb_distances(src, tgt, (1.0, 1.0, 0.5)), dist)


def test_dense_softmax_below_half_now_writes_edges_inside_gate():
    n_src, n_tgt = 400, 80
    logits = np.zeros((n_src, n_tgt), dtype=np.float64)
    logits[0] = 3.0
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(np.clip(shifted, -40.0, 40.0))
    probs = exp / np.clip(exp.sum(axis=0, keepdims=True), 1e-8, None)
    assert float(probs.max()) < 0.5
    assert assign_v1(probs, threshold=0.5) == []
    distances = np.full((n_src, n_tgt), 4.0)
    distances[0, :] = 4.0
    distances[1:, :] = 40.0
    winners = assign_transformer_links_v2(probs, distances_um=distances, max_link_um=12.0)
    assert winners
    assert all(i == 0 for i, _j, _p in winners)
    assert len({j for _i, j, _p in winners}) == 1


def test_runtime_estimate_tuning_and_hidden_under_12h():
    tune = runtime_estimate(2)
    assert tune["under_limit"] is True
    assert tune["estimate_hours"] < 0.2
    hidden = runtime_estimate(199)
    assert hidden["under_limit"] is True
    assert hidden["estimate_hours"] < 12.0


def test_kernel_is_helper_not_submission_and_packages_gpu(tmp_path: Path, monkeypatch):
    source = (ROOT / "notebooks" / "kaggle_learned_unet_link_v2.py").read_text()
    assert "to_csv(" not in source
    assert "graphs_to_submission" not in source
    assert "writes_submission_csv" in source
    assert "transformer_softmax_greedy_1to1_spatial" in source
    assert "edge_threshold\": None" in source
    assert "max_link_um" in source
    assert "index_features" in source
    spec = _make_mod.KERNELS["learned_unet_link_v2"]
    assert spec["kernel_id"] == LINK_KERNEL_SLUG
    assert spec.get("enable_gpu") is True
    assert TRAIN_KERNEL_SLUG in spec.get("kernel_sources", [])
    meta = _make_mod.metadata_payload(spec)
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is False
    assert meta["is_private"] is True
    assert meta["id"] == LINK_KERNEL_SLUG
    assert TRAIN_KERNEL_SLUG in meta["kernel_sources"]
    nb = _make_mod.build_notebook(source, spec["markdown"], enable_gpu=True)
    assert nb["metadata"]["kaggle"]["accelerator"] == "gpu"
    assert nb["metadata"]["kaggle"]["isInternetEnabled"] is False

    train = tmp_path / "data" / "train"
    train.mkdir(parents=True)
    for name in INFER_STEMS:
        (train / f"{name}.zarr").mkdir()
    monkeypatch.setenv("BIOHUB_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("BIOHUB_LEARNED_DRY_RUN", "1")
    monkeypatch.chdir(tmp_path)
    result = link_main()
    assert result["status"] == "dry_run"
    assert result["config_hash"] == LOCKED_LINK_V2_HASH
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


def test_blocked_no_weights_records_search_and_writes_no_submission(tmp_path: Path, monkeypatch):
    train = tmp_path / "data" / "train"
    train.mkdir(parents=True)
    for name in INFER_STEMS:
        (train / f"{name}.zarr").mkdir()
    monkeypatch.setenv("BIOHUB_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("BIOHUB_LEARNED_DRY_RUN", raising=False)
    monkeypatch.delenv("BIOHUB_LEARNED_WEIGHTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    result = link_main()
    assert result["status"] == "blocked_no_weights"
    assert result["config_hash"] == LOCKED_LINK_V2_HASH
    assert "weight_search" in result
    assert result["is_competition_submission"] is False
    assert not (tmp_path / "submission.csv").exists()
    assert_no_submission_csv(tmp_path)


def test_push_forbids_locked_allows_link_v2_and_fetch_allows_link_v2():
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
    assert LINK_KERNEL_SLUG not in mod.FORBIDDEN_SLUGS
    fetch = importlib.util.spec_from_file_location(
        "fetch_kernel_output",
        ROOT / "scripts" / "fetch_kernel_output.py",
    )
    assert fetch and fetch.loader
    fetch_mod = importlib.util.module_from_spec(fetch)
    fetch.loader.exec_module(fetch_mod)
    assert REJECTED_LINK_V1_SLUG in fetch_mod.ALLOWED_SLUGS
    assert LINK_KERNEL_SLUG in fetch_mod.ALLOWED_SLUGS
    assert find_weights is not None
    assert verify_train_manifest is not None
