from biohub_tracking.splits import embryo_holdout_splits


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
