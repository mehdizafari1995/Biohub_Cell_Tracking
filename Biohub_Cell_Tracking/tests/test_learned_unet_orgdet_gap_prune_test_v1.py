from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from biohub_tracking.learned_collect import (  # noqa: E402
    GAP_TEST_SLUG,
    HOLDOUT_SLUG,
    INFER_SLUG,
    ORGDET_GAP_PRUNE_TEST_CONFIG_HASH,
    ORGDET_GAP_PRUNE_TEST_SLUG,
    ORGDET_GAP_TEST_SLUG,
    ORGDET_TEST_SLUG,
    PRUNE_TEST_SLUG,
    TEST_SLUG,
    TRAIN_SLUG,
)
from biohub_tracking.learned_gap import LOCKED_GAP_HASH  # noqa: E402
from biohub_tracking.learned_gap_test import (  # noqa: E402
    ASSOCIATION_V2_PUBLIC_SCORE,
    ASSOCIATION_V2_SUBMISSION_REF,
)
from biohub_tracking.learned_test import ESTIMATE_HIDDEN_VOLUMES, TEST_KERNEL_SLUG  # noqa: E402
from biohub_tracking.learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH  # noqa: E402
from biohub_tracking.learned_orgdet_gap import LOCKED_ORGDET_GAP_HASH  # noqa: E402
from biohub_tracking.learned_orgdet_gap_prune import LOCKED_ORGDET_GAP_PRUNE_HASH  # noqa: E402
from biohub_tracking.learned_orgdet_gap_prune_test import (  # noqa: E402
    LOCKED_ORGDET_GAP_PRUNE_TEST_HASH,
    ORGDET_GAP_PRUNE_TEST_KERNEL_SLUG,
    classify_orgdet_gap_prune_test_submit_poll,
    compare_prune_vs_gap_counts,
    hidden_projection_from_rehearsal,
    orgdet_gap_prune_test_config_hash as locked_hash,
    orgdet_gap_prune_test_config_payload as locked_payload,
    refuse_overwrite_locked_slugs,
    runtime_estimate,
)
from biohub_tracking.learned_orgdet_gap_test import (  # noqa: E402
    LOCKED_ORGDET_GAP_TEST_HASH,
    ORGDET_GAP_TEST_KERNEL_SLUG,
)
from biohub_tracking.learned_orgdet_test import (  # noqa: E402
    LOCKED_ORGDET_TEST_HASH,
    ORGDET_TEST_KERNEL_SLUG,
)
from kaggle_learned_unet_orgdet_gap_prune_test_v1 import (  # noqa: E402
    CONFIG_LABEL,
    KERNEL_SLUG,
    ORGDET_GAP_TEST_KERNEL_SLUG as NB_ORGDET_GAP_TEST,
    apply_gap_overlay_arrays,
    drop_degree0_arrays,
    infer_config_hash as nb_hash,
    infer_config_payload as nb_payload,
    main as run_kernel,
    runtime_estimate as nb_estimate,
)

_MAKE = importlib.util.spec_from_file_location("make_kaggle_kernel", ROOT / "scripts" / "make_kaggle_kernel.py")
assert _MAKE and _MAKE.loader
_make_mod = importlib.util.module_from_spec(_MAKE)
_MAKE.loader.exec_module(_make_mod)

_PUSH = importlib.util.spec_from_file_location("push_kaggle_kernel", ROOT / "scripts" / "push_kaggle_kernel.py")
assert _PUSH and _PUSH.loader
_push_mod = importlib.util.module_from_spec(_PUSH)
_PUSH.loader.exec_module(_push_mod)

_FETCH = importlib.util.spec_from_file_location("fetch_kernel_output", ROOT / "scripts" / "fetch_kernel_output.py")
assert _FETCH and _FETCH.loader
_fetch_mod = importlib.util.module_from_spec(_FETCH)
_FETCH.loader.exec_module(_fetch_mod)


def test_payload_matches_notebook_yaml_and_locked_hash():
    payload = locked_payload()
    assert payload == nb_payload()
    assert locked_hash(payload) == nb_hash(payload)
    assert locked_hash(payload) == LOCKED_ORGDET_GAP_PRUNE_TEST_HASH
    assert locked_hash(payload) == ORGDET_GAP_PRUNE_TEST_CONFIG_HASH
    assert locked_hash(payload) == "52a30a5843e0a880"
    assert payload["train_config_hash"] == TRAIN_CONFIG_HASH
    assert payload["infer_config_hash"] == LOCKED_INFER_HASH
    assert payload["orgdet_gap_config_hash"] == LOCKED_ORGDET_GAP_HASH == "ee31f36151325f9a"
    assert payload["orgdet_gap_prune_config_hash"] == LOCKED_ORGDET_GAP_PRUNE_HASH == "862e3e7f7ac0330e"
    assert payload["orgdet_gap_test_parent_hash"] == LOCKED_ORGDET_GAP_TEST_HASH == "1ce4837ae16bee52"
    assert payload["orgdet_test_parent_hash"] == LOCKED_ORGDET_TEST_HASH
    assert LOCKED_GAP_HASH == "f4d888d190c45a4a"
    assert payload["writes_submission_csv"] is True
    assert payload["discovers_test_zarrs"] is True
    assert payload["hardcoded_test_stems"] == []
    assert payload["opens_train_volumes"] is False
    assert payload["max_gap_frames"] == 2
    assert payload["drop_degree0"] is True
    assert payload["linker"] == "orgdet_gap_then_drop_degree0"
    assert payload["kernel_slug"] == ORGDET_GAP_PRUNE_TEST_KERNEL_SLUG
    assert payload["orgdet_gap_test_kernel_slug"] == ORGDET_GAP_TEST_KERNEL_SLUG
    assert payload["kernel_slug"] != payload["orgdet_gap_test_kernel_slug"]
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "learned_unet_orgdet_gap_prune_test_v1.yaml").read_text())
    assert yaml_cfg["label"] == CONFIG_LABEL
    assert yaml_cfg["kernel_slug"] == KERNEL_SLUG == ORGDET_GAP_PRUNE_TEST_SLUG
    assert yaml_cfg["orgdet_gap_test_kernel_slug"] == ORGDET_GAP_TEST_SLUG
    assert yaml_cfg["drop_degree0"] is True
    assert yaml_cfg["writes_submission_csv"] is True
    for key in (
        "gate_um",
        "max_gap_frames",
        "gap_scale",
        "assignment",
        "interpolate_gaps",
        "drop_degree0",
        "linker",
        "orgdet_gap_prune_config_hash",
        "orgdet_gap_test_parent_hash",
        "det_threshold",
        "pool_kernel_um",
        "kernel_slug",
    ):
        assert yaml_cfg[key] == payload[key]


def test_runtime_estimate_hidden_test_under_12h():
    hidden = runtime_estimate(ESTIMATE_HIDDEN_VOLUMES)
    public = runtime_estimate(4)
    assert hidden == nb_estimate(ESTIMATE_HIDDEN_VOLUMES)
    assert hidden["under_limit"] is True
    assert hidden["estimate_hours"] < 12.0
    assert 4.0 <= hidden["estimate_hours"] <= 5.0
    assert public["under_limit"] is True
    assert public["estimate_sec"] < 8 * 60
    assert public["prune_overhead_sec_per_volume"] == 0.5
    projected = hidden_projection_from_rehearsal(312.69)
    assert projected["hidden_test"]["under_limit"] is True
    assert projected["fail_closed_approaches_12h"] is False


def test_notebook_gap_then_prune_drops_only_isolates():
    coords = np.asarray([[0, 8, 40, 40], [2, 8, 42, 40], [1, 8, 200, 200]], dtype=np.int64)
    edges: list[tuple[int, int]] = []
    gapped_coords, gapped_edges, n_gap, n_interp = apply_gap_overlay_arrays(coords, edges)
    assert n_gap == 1
    assert n_interp == 1
    pruned_coords, pruned_edges, n_dropped = drop_degree0_arrays(gapped_coords, gapped_edges)
    assert n_dropped == 1
    assert len(pruned_coords) == len(gapped_coords) - 1
    assert len(pruned_edges) == len(gapped_edges)
    linked = {int(s) for s, _d in pruned_edges} | {int(d) for _s, d in pruned_edges}
    assert linked == set(range(1, len(pruned_coords) + 1))
    keep_coords = np.asarray([[0, 8, 40, 40], [1, 8, 41, 40]], dtype=np.int64)
    keep_edges = [(1, 2)]
    after_coords, after_edges, n_dropped2 = drop_degree0_arrays(keep_coords, keep_edges)
    assert n_dropped2 == 0
    assert after_edges == keep_edges
    assert np.array_equal(after_coords, keep_coords)
    cmp = compare_prune_vs_gap_counts(
        {
            "n_nodes": len(pruned_coords),
            "n_edges": len(pruned_edges),
            "n_gap_links": n_gap,
            "n_interpolated": n_interp,
            "n_predicted_divisions": 0,
            "n_dropped_isolates": n_dropped,
        },
        {
            "n_nodes": len(gapped_coords),
            "n_edges": len(gapped_edges),
            "n_gap_links": n_gap,
            "n_interpolated": n_interp,
            "n_predicted_divisions": 0,
        },
    )
    assert cmp["ok"] is True


def test_kernel_discovers_test_and_packages_new_gpu_slug(tmp_path: Path, monkeypatch):
    source = (ROOT / "notebooks" / "kaggle_learned_unet_orgdet_gap_prune_test_v1.py").read_text()
    assert "apply_gap_overlay_arrays" in source
    assert "drop_degree0_arrays" in source
    assert "submission.csv" in source
    assert "run_manifest.json" in source
    assert "list_test_zarrs" in source
    assert "INFER_STEMS" not in source
    assert "6bba_05db0fb1" not in source
    assert "44b6_0113de3b" not in source
    assert KERNEL_SLUG != TEST_KERNEL_SLUG
    assert KERNEL_SLUG != ORGDET_GAP_TEST_KERNEL_SLUG
    assert KERNEL_SLUG != ORGDET_TEST_KERNEL_SLUG
    assert NB_ORGDET_GAP_TEST == ORGDET_GAP_TEST_KERNEL_SLUG
    spec = _make_mod.KERNELS["learned_unet_orgdet_gap_prune_test_v1"]
    assert spec["kernel_id"] == ORGDET_GAP_PRUNE_TEST_KERNEL_SLUG
    assert spec["kernel_id"] != ORGDET_GAP_TEST_SLUG
    assert spec["kernel_id"] != ORGDET_TEST_SLUG
    assert spec.get("enable_gpu") is True
    assert TRAIN_SLUG in spec.get("kernel_sources", [])
    meta = _make_mod.metadata_payload(spec)
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is False
    assert meta["is_private"] is True
    assert meta["id"] == ORGDET_GAP_PRUNE_TEST_KERNEL_SLUG
    nb = _make_mod.build_notebook(source, spec["markdown"], enable_gpu=True)
    assert nb["metadata"]["kaggle"]["accelerator"] == "gpu"
    assert nb["metadata"]["kaggle"]["isInternetEnabled"] is False

    data = tmp_path / "data"
    (data / "test" / "pub_aaa.zarr").mkdir(parents=True)
    (data / "test" / "pub_bbb.zarr").mkdir()
    monkeypatch.setenv("BIOHUB_DATA_ROOT", str(data))
    monkeypatch.setenv("BIOHUB_LEARNED_DRY_RUN", "1")
    monkeypatch.chdir(tmp_path)
    result = run_kernel()
    assert result["status"] == "dry_run"
    assert result["config_hash"] == ORGDET_GAP_PRUNE_TEST_CONFIG_HASH
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
    result = run_kernel()
    assert result["status"] == "blocked_no_weights"
    assert result["config_hash"] == ORGDET_GAP_PRUNE_TEST_CONFIG_HASH
    assert "weight_search" in result
    assert result["is_competition_submission"] is False
    assert not (tmp_path / "submission.csv").exists()


def test_push_keeps_parent_slugs_forbidden_and_allows_new_slug():
    try:
        refuse_overwrite_locked_slugs(ORGDET_GAP_TEST_SLUG)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        refuse_overwrite_locked_slugs(ORGDET_TEST_SLUG)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    refuse_overwrite_locked_slugs(ORGDET_GAP_PRUNE_TEST_SLUG)
    assert TRAIN_SLUG in _push_mod.FORBIDDEN_SLUGS
    assert INFER_SLUG in _push_mod.FORBIDDEN_SLUGS
    assert HOLDOUT_SLUG in _push_mod.FORBIDDEN_SLUGS
    assert TEST_SLUG in _push_mod.FORBIDDEN_SLUGS
    assert GAP_TEST_SLUG in _push_mod.FORBIDDEN_SLUGS
    assert PRUNE_TEST_SLUG in _push_mod.FORBIDDEN_SLUGS
    assert ORGDET_TEST_SLUG in _push_mod.FORBIDDEN_SLUGS
    assert ORGDET_GAP_TEST_SLUG in _push_mod.FORBIDDEN_SLUGS
    assert ORGDET_GAP_PRUNE_TEST_SLUG not in _push_mod.FORBIDDEN_SLUGS
    assert ORGDET_GAP_PRUNE_TEST_SLUG in _fetch_mod.ALLOWED_SLUGS
    assert ORGDET_GAP_TEST_SLUG in _fetch_mod.ALLOWED_SLUGS
    collect = ROOT / "scripts" / "collect_learned_unet_orgdet_gap_prune_test_v1.py"
    text = collect.read_text()
    assert ORGDET_GAP_PRUNE_TEST_KERNEL_SLUG in text
    assert "competitions submit" not in text
    assert "artifacts" in text and "kaggle" in text


def test_classify_submit_poll_requires_ui_when_only_known_rows_exist():
    association = {
        "ref": ASSOCIATION_V2_SUBMISSION_REF,
        "status": "complete",
        "publicScore": ASSOCIATION_V2_PUBLIC_SCORE,
        "private_score_returned": False,
        "date": "2026-09-20T15:12:52.307Z",
    }
    orgdet_test = {
        "ref": 56466744,
        "status": "complete",
        "publicScore": "0.612",
        "private_score_returned": False,
        "date": "2026-09-22T14:13:11.68Z",
    }
    poll = classify_orgdet_gap_prune_test_submit_poll(
        kernel_complete=True,
        kernel_failed=False,
        output_names=["run_manifest.json", "submission.csv"],
        submissions=[association, orgdet_test],
    )
    assert poll["decision"] == "human_action_required_ui_submit"
    assert poll["orgdet_gap_prune_test_submission_present"] is False
    assert poll["association_v2_submission_present"] is True
    assert poll["orgdet_test_submission_present"] is True
    assert poll["is_competition_submission"] is False
    assert poll["is_leaderboard_score"] is False
    assert "Kaggle UI" in poll["human_action"]
    assert "CLI" in poll["human_action"]
    assert "orgdet-gap-prune-test-v1" in poll["human_action"]
    assert classify_orgdet_gap_prune_test_submit_poll(
        kernel_complete=True,
        kernel_failed=False,
        output_names=["run_manifest.json"],
        submissions=[association, orgdet_test],
    )["decision"] == "kernel_missing_csv"
    extra = classify_orgdet_gap_prune_test_submit_poll(
        kernel_complete=True,
        kernel_failed=False,
        output_names=["submission.csv"],
        submissions=[association, orgdet_test, {"ref": 999, "status": "pending", "publicScore": None}],
    )
    assert extra["decision"] == "new_submission_present"
    assert extra["orgdet_gap_prune_test_submission_present"] is True
    poll_src = (ROOT / "scripts" / "poll_orgdet_gap_prune_test_submit.py").read_text()
    assert "list_competition_submissions" in poll_src
    assert "competitions/submit" not in poll_src
    assert "kernels/push" not in poll_src
    assert "download_kernel_output" not in poll_src


def test_compare_prune_vs_gap_rejects_edge_changes():
    bad = compare_prune_vs_gap_counts(
        {
            "n_nodes": 10,
            "n_edges": 9,
            "n_gap_links": 1,
            "n_interpolated": 1,
            "n_predicted_divisions": 0,
            "n_dropped_isolates": 1,
        },
        {"n_nodes": 11, "n_edges": 8, "n_gap_links": 1, "n_interpolated": 1, "n_predicted_divisions": 0},
    )
    assert bad["ok"] is False
    assert any("edges changed" in err for err in bad["errors"])
