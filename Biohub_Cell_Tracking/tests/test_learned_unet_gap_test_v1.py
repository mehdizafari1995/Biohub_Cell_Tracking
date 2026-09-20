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

from biohub_tracking.learned_collect import (  # noqa: E402
    GAP_TEST_CONFIG_HASH,
    GAP_TEST_SLUG,
    HOLDOUT_SLUG,
    INFER_SLUG,
    TEST_SLUG,
    TRAIN_SLUG,
)
from biohub_tracking.learned_gap import (  # noqa: E402
    LOCKED_GAP_HASH,
    apply_locked_overlay_arrays,
)
from biohub_tracking.learned_gap_test import (  # noqa: E402
    ESTIMATE_HIDDEN_VOLUMES,
    GAP_TEST_KERNEL_SLUG,
    LOCKED_GAP_TEST_HASH,
    gap_test_config_hash as locked_hash,
    gap_test_config_payload as locked_payload,
    refuse_overwrite_nogap_slug,
    runtime_estimate,
)
from biohub_tracking.learned_holdout import LOCKED_HOLDOUT_HASH  # noqa: E402
from biohub_tracking.learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH  # noqa: E402
from biohub_tracking.learned_test import TEST_KERNEL_SLUG  # noqa: E402
from kaggle_learned_unet_gap_test_v1 import (  # noqa: E402
    CONFIG_LABEL,
    KERNEL_SLUG,
    TEST_KERNEL_SLUG as NB_TEST_SLUG,
    apply_locked_gap_overlay,
    infer_config_hash as nb_hash,
    infer_config_payload as nb_payload,
    main as run_gap_test_kernel,
    runtime_estimate as nb_estimate,
)

_MAKE = importlib.util.spec_from_file_location("make_kaggle_kernel", ROOT / "scripts" / "make_kaggle_kernel.py")
assert _MAKE and _MAKE.loader
_make_mod = importlib.util.module_from_spec(_MAKE)
_MAKE.loader.exec_module(_make_mod)


def test_gap_test_payload_matches_notebook_yaml_and_locked_hash():
    payload = locked_payload()
    assert payload == nb_payload()
    assert locked_hash(payload) == nb_hash(payload)
    assert locked_hash(payload) == LOCKED_GAP_TEST_HASH
    assert locked_hash(payload) == GAP_TEST_CONFIG_HASH
    assert locked_hash(payload) == "85379c198ebc7241"
    assert payload["infer_config_hash"] == LOCKED_INFER_HASH
    assert payload["holdout_config_hash"] == LOCKED_HOLDOUT_HASH
    assert payload["train_config_hash"] == TRAIN_CONFIG_HASH
    assert payload["gap_config_hash"] == LOCKED_GAP_HASH
    assert payload["test_parent_hash"] == "7a8725b818fc609b"
    assert payload["writes_submission_csv"] is True
    assert payload["discovers_test_zarrs"] is True
    assert payload["hardcoded_test_stems"] == []
    assert payload["opens_train_volumes"] is False
    assert payload["max_gap_frames"] == 2
    assert payload["linker"] == "association_v2_gap_overlay"
    assert payload["kernel_slug"] == GAP_TEST_KERNEL_SLUG
    assert payload["test_kernel_slug"] == TEST_KERNEL_SLUG
    assert payload["kernel_slug"] != payload["test_kernel_slug"]
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_gap_test_v1.yaml").read_text())
    assert yaml_cfg["label"] == CONFIG_LABEL
    assert yaml_cfg["kernel_slug"] == KERNEL_SLUG == GAP_TEST_SLUG
    assert yaml_cfg["test_kernel_slug"] == TEST_SLUG
    assert yaml_cfg["max_gap_frames"] == 2
    assert yaml_cfg["writes_submission_csv"] is True


def test_runtime_estimate_hidden_test_under_12h():
    hidden = runtime_estimate(ESTIMATE_HIDDEN_VOLUMES)
    public = runtime_estimate(4)
    assert hidden == nb_estimate(ESTIMATE_HIDDEN_VOLUMES)
    assert hidden["under_limit"] is True
    assert hidden["estimate_hours"] < 12.0
    assert 5.0 <= hidden["estimate_hours"] <= 6.0
    assert public["under_limit"] is True
    assert public["estimate_sec"] < 8 * 60
    assert public["overlay_sec_per_volume"] > 0


def test_notebook_overlay_matches_locked_gap_helper():
    coords = np.asarray([[0, 8, 40, 40], [2, 8, 42, 40]], dtype=np.int64)
    edges: list[tuple[int, int]] = []
    nb_coords, nb_edges, n_gap, n_interp = apply_locked_gap_overlay(coords, edges)
    lib_coords, lib_edges, meta = apply_locked_overlay_arrays(coords, edges)
    assert n_gap == meta["n_gap_links"] == 1
    assert n_interp == meta["n_interpolated"] == 1
    assert np.array_equal(nb_coords, lib_coords)
    assert nb_edges == lib_edges
    keep_coords = np.asarray([[0, 8, 40, 40], [1, 8, 41, 40], [2, 8, 80, 40]], dtype=np.int64)
    keep_edges = [(1, 2)]
    after_coords, after_edges, n_gap2, n_interp2 = apply_locked_gap_overlay(keep_coords, keep_edges)
    assert n_gap2 == 0
    assert n_interp2 == 0
    assert after_edges == keep_edges
    assert np.array_equal(after_coords, keep_coords)


def test_kernel_discovers_test_and_packages_new_gpu_slug(tmp_path: Path, monkeypatch):
    source = (ROOT / "notebooks" / "kaggle_learned_unet_gap_test_v1.py").read_text()
    assert "apply_locked_gap_overlay" in source
    assert "submission.csv" in source
    assert "run_manifest.json" in source
    assert "list_test_zarrs" in source
    assert "INFER_STEMS" not in source
    assert "6bba_05db0fb1" not in source
    assert "44b6_0113de3b" not in source
    assert KERNEL_SLUG != TEST_KERNEL_SLUG
    assert NB_TEST_SLUG == TEST_KERNEL_SLUG
    spec = _make_mod.KERNELS["learned_unet_gap_test_v1"]
    assert spec["kernel_id"] == GAP_TEST_KERNEL_SLUG
    assert spec["kernel_id"] != TEST_SLUG
    assert spec.get("enable_gpu") is True
    assert TRAIN_SLUG in spec.get("kernel_sources", [])
    meta = _make_mod.metadata_payload(spec)
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is False
    assert meta["is_private"] is True
    assert meta["id"] == GAP_TEST_KERNEL_SLUG
    nb = _make_mod.build_notebook(source, spec["markdown"], enable_gpu=True)
    assert nb["metadata"]["kaggle"]["accelerator"] == "gpu"
    assert nb["metadata"]["kaggle"]["isInternetEnabled"] is False

    data = tmp_path / "data"
    (data / "test" / "pub_aaa.zarr").mkdir(parents=True)
    (data / "test" / "pub_bbb.zarr").mkdir()
    monkeypatch.setenv("BIOHUB_DATA_ROOT", str(data))
    monkeypatch.setenv("BIOHUB_LEARNED_DRY_RUN", "1")
    monkeypatch.chdir(tmp_path)
    result = run_gap_test_kernel()
    assert result["status"] == "dry_run"
    assert result["config_hash"] == GAP_TEST_CONFIG_HASH
    assert result["is_competition_submission"] is False
    assert result["datasets"] == ["pub_aaa", "pub_bbb"]
    assert result["estimate"]["under_limit"] is True
    assert (tmp_path / "run_manifest.json").exists()
    assert not (tmp_path / "submission.csv").exists()


def test_blocked_no_weights_records_search_and_writes_no_submission(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    (data / "test" / "pub_aaa.zarr").mkdir(parents=True)
    monkeypatch.setenv("BIOHUB_DATA_ROOT", str(data))
    monkeypatch.delenv("BIOHUB_LEARNED_DRY_RUN", raising=False)
    monkeypatch.delenv("BIOHUB_LEARNED_WEIGHTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    result = run_gap_test_kernel()
    assert result["status"] == "blocked_no_weights"
    assert result["config_hash"] == GAP_TEST_CONFIG_HASH
    assert "weight_search" in result
    assert result["is_competition_submission"] is False
    assert not (tmp_path / "submission.csv").exists()


def test_push_keeps_nogap_test_forbidden_and_allows_new_gap_slug():
    try:
        refuse_overwrite_nogap_slug(TEST_SLUG)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    refuse_overwrite_nogap_slug(GAP_TEST_SLUG)
    spec = importlib.util.spec_from_file_location(
        "push_kaggle_kernel",
        ROOT / "scripts" / "push_kaggle_kernel.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert TRAIN_SLUG in mod.FORBIDDEN_SLUGS
    assert INFER_SLUG in mod.FORBIDDEN_SLUGS
    assert HOLDOUT_SLUG in mod.FORBIDDEN_SLUGS
    assert TEST_SLUG in mod.FORBIDDEN_SLUGS
    assert GAP_TEST_SLUG in mod.FORBIDDEN_SLUGS
    fetch = importlib.util.spec_from_file_location(
        "fetch_kernel_output",
        ROOT / "scripts" / "fetch_kernel_output.py",
    )
    assert fetch and fetch.loader
    fetch_mod = importlib.util.module_from_spec(fetch)
    fetch.loader.exec_module(fetch_mod)
    assert GAP_TEST_SLUG in fetch_mod.ALLOWED_SLUGS
    assert TEST_SLUG in fetch_mod.ALLOWED_SLUGS
