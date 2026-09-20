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

from biohub_tracking.learned_split import (  # noqa: E402
    HoldoutLeakError,
    INTERNAL_VAL_STEMS,
    KERNEL_SLUG,
    build_44b6_splits,
    config_hash,
    config_payload,
    list_tuning_pairs,
    refuse_holdout_stems,
)
from biohub_tracking.zarr_v3 import load_geff_arrays, write_array, write_geff_arrays  # noqa: E402
from kaggle_learned_unet_v1 import (  # noqa: E402
    CONFIG_LABEL,
    assert_no_submission_csv,
    config_hash as nb_hash,
    config_payload as nb_payload,
    main as kernel_main,
)

_MAKE = importlib.util.spec_from_file_location("make_kaggle_kernel", ROOT / "scripts" / "make_kaggle_kernel.py")
assert _MAKE and _MAKE.loader
_make_mod = importlib.util.module_from_spec(_MAKE)
_MAKE.loader.exec_module(_make_mod)


LOCKED_HASH = "4c38035d6657f2ad"


def test_learned_unet_v1_payload_is_stable_and_shared():
    payload = config_payload()
    assert payload == nb_payload()
    assert config_hash(payload) == nb_hash(payload)
    assert config_hash(payload) == LOCKED_HASH
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_v1.yaml").read_text())
    assert yaml_cfg["label"] == CONFIG_LABEL
    assert yaml_cfg["tuning_specimen"] == "44b6"
    assert yaml_cfg["holdout_specimens"] == ["6bba"]
    assert yaml_cfg["internal_val"] == list(INTERNAL_VAL_STEMS)
    assert yaml_cfg["writes_submission_csv"] is False
    assert yaml_cfg["kernel_slug"] == KERNEL_SLUG
    assert yaml_cfg["epochs"] == 3
    assert payload["edge_supervision"] == "gt_node_index"


def test_splits_refuse_holdout_and_keep_internal_val_on_44b6():
    stems = [
        "44b6_aaa",
        "44b6_12dfb391",
        "44b6_267148e4",
        "44b6_bbb",
    ]
    splits = build_44b6_splits(stems)
    assert splits["train"] == ["44b6_aaa", "44b6_bbb"]
    assert splits["test"] == ["44b6_12dfb391", "44b6_267148e4"]
    assert "6bba" not in json.dumps(splits)
    try:
        build_44b6_splits(["44b6_aaa", "6bba_05db0fb1"])
        raise AssertionError("expected HoldoutLeakError")
    except HoldoutLeakError:
        pass
    try:
        refuse_holdout_stems(["6bba_c328f2fd"])
        raise AssertionError("expected HoldoutLeakError")
    except HoldoutLeakError:
        pass
    try:
        build_44b6_splits(["44b6_12dfb391", "44b6_267148e4"])
        raise AssertionError("expected empty-train error")
    except ValueError:
        pass


def test_list_tuning_pairs_skips_holdout_volumes(tmp_path: Path):
    train = tmp_path / "train"
    train.mkdir()
    for name in ("44b6_aaa", "44b6_12dfb391", "6bba_05db0fb1", "6bba_other"):
        (train / f"{name}.zarr").mkdir()
        write_geff_arrays(
            train / f"{name}.geff",
            node_id=np.array([1, 2], dtype=np.int64),
            t=np.array([0, 1], dtype=np.int64),
            z=np.array([1, 1], dtype=np.int64),
            y=np.array([2, 3], dtype=np.int64),
            x=np.array([4, 5], dtype=np.int64),
            edges=np.array([[1, 2]], dtype=np.int64),
        )
    pairs = list_tuning_pairs(train)
    stems = [p["stem"] for p in pairs]
    assert stems == ["44b6_12dfb391", "44b6_aaa"]
    assert all(not s.startswith("6bba") for s in stems)


def test_geff_roundtrip_without_zarr_package(tmp_path: Path):
    path = tmp_path / "44b6_toy.geff"
    write_geff_arrays(
        path,
        node_id=np.array([10, 11, 12], dtype=np.int64),
        t=np.array([0, 0, 1], dtype=np.int64),
        z=np.array([1, 2, 1], dtype=np.int64),
        y=np.array([8, 9, 8], dtype=np.int64),
        x=np.array([3, 4, 5], dtype=np.int64),
        edges=np.array([[10, 12], [11, 12]], dtype=np.int64),
        estimated_number_of_nodes=99.0,
    )
    loaded = load_geff_arrays(path)
    assert list(loaded["node_id"]) == [10, 11, 12]
    assert loaded["edges"].shape == (2, 2)
    volume = tmp_path / "toy"
    write_array(volume, np.arange(8, dtype=np.uint16).reshape(2, 2, 2))
    from biohub_tracking.zarr_v3 import ZarrV2Array, ZarrV3Array, write_array_v2

    assert ZarrV3Array(volume).read().shape == (2, 2, 2)
    v2 = tmp_path / "v2arr"
    write_array_v2(v2, np.arange(6, dtype=np.int64).reshape(3, 2))
    assert ZarrV2Array(v2).read().tolist() == [[0, 1], [2, 3], [4, 5]]


def test_kernel_is_helper_not_submission_and_packages_gpu(tmp_path: Path, monkeypatch):
    source = (ROOT / "notebooks" / "kaggle_learned_unet_v1.py").read_text()
    assert "writes_submission_csv" in source
    assert "to_csv(" not in source
    assert "graphs_to_submission" not in source
    assert CONFIG_LABEL in source
    spec = _make_mod.KERNELS["learned_unet_v1"]
    assert spec["kernel_id"] == KERNEL_SLUG
    assert spec.get("enable_gpu") is True
    meta = _make_mod.metadata_payload(spec)
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is False
    assert meta["is_private"] is True
    assert meta["id"] == KERNEL_SLUG
    nb = _make_mod.build_notebook(source, spec["markdown"], enable_gpu=bool(spec.get("enable_gpu")))
    assert nb["metadata"]["kaggle"]["accelerator"] == "gpu"
    assert nb["metadata"]["kaggle"]["isInternetEnabled"] is False

    train = tmp_path / "data" / "train"
    train.mkdir(parents=True)
    for name in ("44b6_aaa", "44b6_12dfb391"):
        (train / f"{name}.zarr").mkdir()
        write_geff_arrays(
            train / f"{name}.geff",
            node_id=np.array([1, 2], dtype=np.int64),
            t=np.array([0, 1], dtype=np.int64),
            z=np.array([1, 1], dtype=np.int64),
            y=np.array([2, 3], dtype=np.int64),
            x=np.array([4, 5], dtype=np.int64),
            edges=np.array([[1, 2]], dtype=np.int64),
        )
    monkeypatch.setenv("BIOHUB_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("BIOHUB_LEARNED_DRY_RUN", "1")
    monkeypatch.chdir(tmp_path)
    result = kernel_main()
    assert result["status"] == "dry_run"
    assert result["config_hash"] == LOCKED_HASH
    assert result["is_competition_submission"] is False
    splits = json.loads((tmp_path / "dataset_splits.json").read_text())
    assert splits["train"] == ["44b6_aaa"]
    assert "6bba" not in json.dumps(splits)
    assert not (tmp_path / "submission.csv").exists()
    assert_no_submission_csv(tmp_path)


def test_push_script_allows_learned_kernel_and_locks_export_helpers():
    spec = importlib.util.spec_from_file_location(
        "push_kaggle_kernel",
        ROOT / "scripts" / "push_kaggle_kernel.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert KERNEL_SLUG not in mod.FORBIDDEN_SLUGS
    assert "mahdimzafari/biohub-tuning-zarr-export" in mod.FORBIDDEN_SLUGS
    assert "mahdimzafari/biohub-association-v2" in mod.FORBIDDEN_SLUGS
    fetch = importlib.util.spec_from_file_location(
        "fetch_kernel_output",
        ROOT / "scripts" / "fetch_kernel_output.py",
    )
    assert fetch and fetch.loader
    fetch_mod = importlib.util.module_from_spec(fetch)
    fetch.loader.exec_module(fetch_mod)
    assert KERNEL_SLUG in fetch_mod.ALLOWED_SLUGS
