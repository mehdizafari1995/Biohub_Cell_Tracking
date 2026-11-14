from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from biohub_tracking.learned_collect import (  # noqa: E402
    HOLDOUT_SLUG,
    INFER_SLUG,
    TEST_CONFIG_HASH,
    TEST_SLUG,
    TRAIN_SLUG,
)
from biohub_tracking.learned_holdout import LOCKED_HOLDOUT_HASH  # noqa: E402
from biohub_tracking.learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH  # noqa: E402
from biohub_tracking.learned_test import (  # noqa: E402
    COLUMNS,
    ESTIMATE_HIDDEN_VOLUMES,
    TEST_KERNEL_SLUG,
    append_submission_rows,
    audit_discovered_coverage,
    audit_graph,
    discovered_stems,
    graph_rows,
    list_test_zarrs,
    refuse_hardcoded_public_stems,
    refuse_train_volume,
    runtime_estimate,
    test_config_hash as locked_test_hash,
    test_config_payload as locked_test_payload,
    write_submission_header,
)
from kaggle_learned_unet_test_v1 import (  # noqa: E402
    CONFIG_LABEL,
    infer_config_hash as nb_hash,
    infer_config_payload as nb_payload,
    main as run_test_kernel,
    runtime_estimate as nb_estimate,
)

_MAKE = importlib.util.spec_from_file_location("make_kaggle_kernel", ROOT / "scripts" / "make_kaggle_kernel.py")
assert _MAKE and _MAKE.loader
_make_mod = importlib.util.module_from_spec(_MAKE)
_MAKE.loader.exec_module(_make_mod)


def test_test_payload_matches_notebook_yaml_and_locked_hash():
    payload = locked_test_payload()
    assert payload == nb_payload()
    assert locked_test_hash(payload) == nb_hash(payload)
    assert locked_test_hash(payload) == TEST_CONFIG_HASH
    assert locked_test_hash(payload) == "7a8725b818fc609b"
    assert payload["infer_config_hash"] == LOCKED_INFER_HASH
    assert payload["holdout_config_hash"] == LOCKED_HOLDOUT_HASH
    assert payload["train_config_hash"] == TRAIN_CONFIG_HASH
    assert payload["writes_submission_csv"] is True
    assert payload["discovers_test_zarrs"] is True
    assert payload["hardcoded_test_stems"] == []
    assert payload["opens_train_volumes"] is False
    assert payload["one_shot_holdout"] is False
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_test_v1.yaml").read_text())
    assert yaml_cfg["label"] == CONFIG_LABEL
    assert yaml_cfg["kernel_slug"] == TEST_KERNEL_SLUG
    assert yaml_cfg["train_kernel_slug"] == TRAIN_SLUG
    assert yaml_cfg["infer_kernel_slug"] == INFER_SLUG
    assert yaml_cfg["holdout_kernel_slug"] == HOLDOUT_SLUG
    assert yaml_cfg["linker"] == "greedy_gate_12um"
    assert yaml_cfg["gate_um"] == 12.0
    assert yaml_cfg["writes_submission_csv"] is True


def test_runtime_estimate_hidden_test_under_12h():
    hidden = runtime_estimate(ESTIMATE_HIDDEN_VOLUMES)
    public = runtime_estimate(4)
    assert hidden == nb_estimate(ESTIMATE_HIDDEN_VOLUMES)
    assert hidden["under_limit"] is True
    assert hidden["estimate_hours"] < 12.0
    assert 5.0 <= hidden["estimate_hours"] <= 6.0
    assert public["under_limit"] is True
    assert public["estimate_sec"] < 10 * 60


def test_discovers_test_zarrs_and_refuses_train_volumes(tmp_path: Path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "aa_one.zarr").mkdir()
    (test_dir / "bb_two.zarr").mkdir()
    (test_dir / "notes.txt").write_text("no")
    zarrs = list_test_zarrs(test_dir)
    assert discovered_stems(zarrs) == ["aa_one", "bb_two"]
    refuse_hardcoded_public_stems([])
    try:
        refuse_hardcoded_public_stems(["44b6_0113de3b"])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        refuse_train_volume(tmp_path / "train" / "44b6_aaa.zarr")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_submission_rows_and_graph_audit(tmp_path: Path):
    coords = [[0, 1, 2, 3], [1, 2, 3, 4]]
    edges = [[1, 2]]
    rows, next_id = graph_rows("demo", coords, edges, 0)
    assert next_id == 3
    assert rows[0][:3] == [0, "demo", "node"]
    assert rows[-1][:3] == [2, "demo", "edge"]
    dest = tmp_path / "submission.csv"
    write_submission_header(dest)
    append_submission_rows(dest, rows)
    with dest.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        body = list(reader)
    assert header == COLUMNS
    assert len(body) == 3
    assert audit_graph("demo", coords, edges, (100, 64, 256, 256)) == []
    assert audit_graph("demo", coords, [[1, 1]], None)
    assert audit_discovered_coverage(["a", "b"], ["a"], ["a", "c"]) == [
        "missing discovered datasets: ['b']"
    ]


def test_kernel_discovers_test_and_packages_gpu(tmp_path: Path, monkeypatch):
    source = (ROOT / "notebooks" / "kaggle_learned_unet_test_v1.py").read_text()
    assert "list_test_zarrs" in source
    assert "submission.csv" in source
    assert "run_manifest.json" in source
    assert "writes_submission_csv" in source
    assert "INFER_STEMS" not in source
    assert "6bba_05db0fb1" not in source
    assert "44b6_0113de3b" not in source
    spec = _make_mod.KERNELS["learned_unet_test_v1"]
    assert spec["kernel_id"] == TEST_KERNEL_SLUG
    assert spec["kernel_id"] != HOLDOUT_SLUG
    assert spec.get("enable_gpu") is True
    assert TRAIN_SLUG in spec.get("kernel_sources", [])
    meta = _make_mod.metadata_payload(spec)
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is False
    assert meta["is_private"] is True
    assert meta["id"] == TEST_KERNEL_SLUG
    nb = _make_mod.build_notebook(source, spec["markdown"], enable_gpu=True)
    assert nb["metadata"]["kaggle"]["accelerator"] == "gpu"
    assert nb["metadata"]["kaggle"]["isInternetEnabled"] is False

    data = tmp_path / "data"
    (data / "test" / "pub_aaa.zarr").mkdir(parents=True)
    (data / "test" / "pub_bbb.zarr").mkdir()
    monkeypatch.setenv("BIOHUB_DATA_ROOT", str(data))
    monkeypatch.setenv("BIOHUB_LEARNED_DRY_RUN", "1")
    monkeypatch.chdir(tmp_path)
    result = run_test_kernel()
    assert result["status"] == "dry_run"
    assert result["config_hash"] == TEST_CONFIG_HASH
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
    result = run_test_kernel()
    assert result["status"] == "blocked_no_weights"
    assert result["config_hash"] == TEST_CONFIG_HASH
    assert "weight_search" in result
    assert result["is_competition_submission"] is False
    assert not (tmp_path / "submission.csv").exists()


def test_push_locks_completed_test_slug_and_keeps_helpers_forbidden():
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
    fetch = importlib.util.spec_from_file_location(
        "fetch_kernel_output",
        ROOT / "scripts" / "fetch_kernel_output.py",
    )
    assert fetch and fetch.loader
    fetch_mod = importlib.util.module_from_spec(fetch)
    fetch.loader.exec_module(fetch_mod)
    assert TEST_SLUG in fetch_mod.ALLOWED_SLUGS
    assert HOLDOUT_SLUG in fetch_mod.ALLOWED_SLUGS
