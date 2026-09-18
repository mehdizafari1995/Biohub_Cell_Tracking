from biohub_tracking.splits import build_fixed_split, embryo_holdout_splits, save_split_manifest, load_split_manifest


def test_holdout_keeps_embryos_together(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    for name in ["44b6_aaa.zarr", "44b6_bbb.zarr", "6bba_ccc.zarr"]:
        (train / name).mkdir()
    splits = embryo_holdout_splits(train)
    assert set(splits["holdout_44b6"]["val"]) == {"44b6_aaa", "44b6_bbb"}
    assert splits["holdout_44b6"]["train"] == ["6bba_ccc"]
    assert "6bba_ccc" not in splits["holdout_6bba"]["train"]
    assert set(splits["holdout_6bba"]["val"]) == {"6bba_ccc"}


def test_fixed_split_is_specimen_disjoint(tmp_path):
    summaries = [
        {"dataset": "44b6_aaa", "embryo": "44b6", "n_nodes": 10, "n_edges": 8, "n_divisions": 0},
        {"dataset": "44b6_bbb", "embryo": "44b6", "n_nodes": 12, "n_edges": 9, "n_divisions": 2},
        {"dataset": "6bba_ccc", "embryo": "6bba", "n_nodes": 20, "n_edges": 18, "n_divisions": 1},
        {"dataset": "6bba_ddd", "embryo": "6bba", "n_nodes": 5, "n_edges": 4, "n_divisions": 0},
    ]
    manifest = build_fixed_split(summaries, max_per_embryo=1, tuning_embryos=["44b6"])
    assert manifest["tuning"] == ["44b6_bbb"]
    assert manifest["validation"] == ["6bba_ccc"]
    assert "44b6" not in manifest["validation_embryos"]
    path = save_split_manifest(manifest, tmp_path / "v1.json")
    loaded = load_split_manifest(path)
    assert loaded["tuning"] == manifest["tuning"]

