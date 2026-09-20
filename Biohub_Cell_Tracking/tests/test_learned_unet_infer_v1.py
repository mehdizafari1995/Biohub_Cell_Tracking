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

from biohub_tracking.learned_infer import (  # noqa: E402
    ASSOC_LOCKED_TUNE,
    INFER_KERNEL_SLUG,
    INFER_STEMS,
    LOCKED_INFER_HASH,
    TRAIN_CONFIG_HASH,
    TRAIN_KERNEL_SLUG,
    detect_cells_pooled,
    find_weights,
    graph_from_payload,
    graph_payload,
    greedy_gate_links,
    infer_config_hash,
    infer_config_payload,
    pool_kernel_from_um,
    refuse_non_infer_stems,
    scale_coords_to_original,
    verify_train_manifest,
    write_graph_json,
    assert_tuning_only,
)
from biohub_tracking.learned_split import HoldoutLeakError, config_hash  # noqa: E402
from kaggle_learned_unet_infer_v1 import (  # noqa: E402
    CONFIG_LABEL,
    assert_no_submission_csv,
    infer_config_hash as nb_hash,
    infer_config_payload as nb_payload,
    main as infer_main,
)

_MAKE = importlib.util.spec_from_file_location("make_kaggle_kernel", ROOT / "scripts" / "make_kaggle_kernel.py")
assert _MAKE and _MAKE.loader
_make_mod = importlib.util.module_from_spec(_MAKE)
_MAKE.loader.exec_module(_make_mod)

def test_infer_payload_matches_notebook_yaml_and_locked_hash():
    payload = infer_config_payload()
    assert payload == nb_payload()
    assert infer_config_hash(payload) == nb_hash(payload)
    assert infer_config_hash(payload) == LOCKED_INFER_HASH
    assert config_hash() == TRAIN_CONFIG_HASH
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_infer_v1.yaml").read_text())
    assert yaml_cfg["label"] == CONFIG_LABEL
    assert yaml_cfg["infer_stems"] == list(INFER_STEMS)
    assert yaml_cfg["writes_submission_csv"] is False
    assert yaml_cfg["kernel_slug"] == INFER_KERNEL_SLUG
    assert yaml_cfg["train_kernel_slug"] == TRAIN_KERNEL_SLUG
    assert yaml_cfg["train_config_hash"] == TRAIN_CONFIG_HASH
    assert yaml_cfg["linker"] == "greedy_gate_12um"
    assert yaml_cfg["gate_um"] == 12.0
    assert yaml_cfg["normalize"] == "window_minmax"
    assert ASSOC_LOCKED_TUNE == 0.6324600072329115


def test_infer_refuses_holdout_and_non_tuning_stems():
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


def test_peak_extraction_and_greedy_gate(tmp_path: Path):
    logits = np.full((5, 7, 7), -4.0, dtype=np.float32)
    logits[2, 3, 3] = 5.0
    logits[2, 3, 4] = 1.0
    peaks = detect_cells_pooled(logits, t=4, det_threshold=0.5, pool_kernel=(3, 3, 3))
    assert peaks.tolist() == [[4, 2, 3, 3]]
    scaled = scale_coords_to_original(peaks, [1, 4, 4])
    assert scaled.tolist() == [[4, 2, 12, 12]]
    kernel = pool_kernel_from_um(5.0, (1.625, 1.625, 1.625))
    assert kernel == (3, 3, 3)
    links = greedy_gate_links(
        np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0], [40.0, 0.0, 0.0]]),
        scale_zyx=(1.0, 1.0, 1.0),
        gate_um=12.0,
    )
    assert links == [(0, 0, 1.0)]
    payload = graph_payload("44b6_12dfb391", scaled, [(1, 1)])
    path = write_graph_json(tmp_path / "44b6_12dfb391.json", payload)
    graph = graph_from_payload(json.loads(path.read_text()))
    assert graph.dataset == "44b6_12dfb391"
    assert graph.n_nodes == 1
    try:
        write_graph_json(tmp_path / "bad.json", graph_payload("6bba_05db0fb1", scaled, []))
        raise AssertionError("expected HoldoutLeakError")
    except HoldoutLeakError:
        pass


def test_find_weights_and_train_manifest_guard(tmp_path: Path):
    dest = tmp_path / "weights"
    dest.mkdir()
    (dest / "learned_unet_v1_best.pt").write_bytes(b"not-a-real-checkpoint")
    found = find_weights([dest])
    assert found is not None and found.name == "learned_unet_v1_best.pt"
    (dest / "train_manifest.json").write_text(
        json.dumps(
            {
                "config_hash": TRAIN_CONFIG_HASH,
                "splits": {"train": ["44b6_aaa"], "test": list(INFER_STEMS)},
            }
        )
    )
    checked = verify_train_manifest(dest / "train_manifest.json")
    assert checked["splits"]["train"] == ["44b6_aaa"]
    (dest / "leaky.json").write_text(
        json.dumps({"splits": {"train": ["6bba_05db0fb1"], "test": []}})
    )
    try:
        verify_train_manifest(dest / "leaky.json")
        raise AssertionError("expected HoldoutLeakError")
    except HoldoutLeakError:
        pass


def test_kernel_is_helper_not_submission_and_packages_gpu(tmp_path: Path, monkeypatch):
    source = (ROOT / "notebooks" / "kaggle_learned_unet_infer_v1.py").read_text()
    assert "to_csv(" not in source
    assert "graphs_to_submission" not in source
    assert "writes_submission_csv" in source
    spec = _make_mod.KERNELS["learned_unet_infer_v1"]
    assert spec["kernel_id"] == INFER_KERNEL_SLUG
    assert spec.get("enable_gpu") is True
    assert TRAIN_KERNEL_SLUG in spec.get("kernel_sources", [])
    meta = _make_mod.metadata_payload(spec)
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is False
    assert meta["is_private"] is True
    assert meta["id"] == INFER_KERNEL_SLUG
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
    result = infer_main()
    assert result["status"] == "dry_run"
    assert result["config_hash"] == LOCKED_INFER_HASH
    assert result["is_competition_submission"] is False
    assert not (tmp_path / "submission.csv").exists()
    assert_no_submission_csv(tmp_path)


def test_push_and_fetch_lock_training_allow_infer():
    spec = importlib.util.spec_from_file_location(
        "push_kaggle_kernel",
        ROOT / "scripts" / "push_kaggle_kernel.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert TRAIN_KERNEL_SLUG in mod.FORBIDDEN_SLUGS
    assert INFER_KERNEL_SLUG not in mod.FORBIDDEN_SLUGS
    fetch = importlib.util.spec_from_file_location(
        "fetch_kernel_output",
        ROOT / "scripts" / "fetch_kernel_output.py",
    )
    assert fetch and fetch.loader
    fetch_mod = importlib.util.module_from_spec(fetch)
    fetch.loader.exec_module(fetch_mod)
    assert INFER_KERNEL_SLUG in fetch_mod.ALLOWED_SLUGS
    assert TRAIN_KERNEL_SLUG in fetch_mod.ALLOWED_SLUGS
