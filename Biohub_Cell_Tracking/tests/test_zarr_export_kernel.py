from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

from kaggle_zarr_export import (  # noqa: E402
    HOLDOUT_ZARRS,
    ZIP_NAME,
    iter_holdout_zarrs,
    wanted_zarr,
    zip_zarr_stores,
)

_MAKE = importlib.util.spec_from_file_location("make_kaggle_kernel", ROOT / "scripts" / "make_kaggle_kernel.py")
assert _MAKE and _MAKE.loader
_make_mod = importlib.util.module_from_spec(_MAKE)
_MAKE.loader.exec_module(_make_mod)
KERNELS = _make_mod.KERNELS
build_kernel = _make_mod.build_kernel
metadata_payload = _make_mod.metadata_payload

_DL = importlib.util.spec_from_file_location("download_geffs", ROOT / "scripts" / "download_geffs.py")
assert _DL and _DL.loader
_dl_mod = importlib.util.module_from_spec(_DL)
_DL.loader.exec_module(_dl_mod)


def _touch_zarr(root: Path, name: str, files: list[str]) -> Path:
    store = root / f"{name}.zarr"
    for rel in files:
        dest = store / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"chunk")
    return store


def test_wanted_zarr_is_only_the_two_holdouts():
    assert wanted_zarr(Path("train/6bba_05db0fb1.zarr")) is True
    assert wanted_zarr(Path("train/6bba_c328f2fd.zarr")) is True
    assert wanted_zarr(Path("train/6bba_ffffff.zarr")) is False
    assert wanted_zarr(Path("train/44b6_abc.zarr")) is False
    assert wanted_zarr(Path("train/6bba_05db0fb1.geff")) is False
    assert HOLDOUT_ZARRS == ("6bba_05db0fb1", "6bba_c328f2fd")


def test_iter_and_zip_skips_other_volumes_and_geffs(tmp_path: Path):
    train = tmp_path / "train"
    _touch_zarr(train, "6bba_05db0fb1", ["zarr.json", "0/c/0/0/0/0"])
    _touch_zarr(train, "6bba_c328f2fd", ["zarr.json"])
    _touch_zarr(train, "44b6_aaa", ["zarr.json", "0/c/0/0/0/0"])
    geff = train / "6bba_05db0fb1.geff"
    (geff / "nodes").mkdir(parents=True)
    (geff / "zarr.json").write_text("g")
    stores = iter_holdout_zarrs(train)
    assert [p.name for p in stores] == ["6bba_05db0fb1.zarr", "6bba_c328f2fd.zarr"]
    dest = tmp_path / ZIP_NAME
    stats = zip_zarr_stores(stores, dest)
    assert stats["n_stores"] == 2
    with zipfile.ZipFile(dest) as zf:
        names = zf.namelist()
    assert "train/6bba_05db0fb1.zarr/zarr.json" in names
    assert "train/6bba_05db0fb1.zarr/0/c/0/0/0/0" in names
    assert "train/6bba_c328f2fd.zarr/zarr.json" in names
    assert all("44b6_" not in n for n in names)
    assert all(".geff/" not in n for n in names)


def test_zip_refuses_non_holdout_store(tmp_path: Path):
    extra = _touch_zarr(tmp_path, "44b6_aaa", ["zarr.json"])
    try:
        zip_zarr_stores([extra], tmp_path / "bad.zip")
    except ValueError as exc:
        assert "non-holdout" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_extract_holdout_zarr_zip_skips_other_members(tmp_path: Path):
    zpath = tmp_path / "export.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("train/6bba_05db0fb1.zarr/zarr.json", b"a")
        zf.writestr("train/6bba_c328f2fd.zarr/0/c/0/0/0/0", b"b")
        zf.writestr("train/44b6_aaa.zarr/0/c/0/0/0/0", b"nope")
        zf.writestr("train/6bba_05db0fb1.geff/zarr.json", b"g")
        zf.writestr("other.txt", b"x")
    stats = _dl_mod.extract_holdout_zarr_zip(zpath, tmp_path / "data")
    assert stats["n_stores"] == 2
    assert (tmp_path / "data/train/6bba_05db0fb1.zarr/zarr.json").read_bytes() == b"a"
    assert (tmp_path / "data/train/6bba_c328f2fd.zarr/0/c/0/0/0/0").read_bytes() == b"b"
    assert not (tmp_path / "data/train/44b6_aaa.zarr").exists()
    assert not (tmp_path / "data/train/6bba_05db0fb1.geff").exists()


def test_extract_unwraps_worker_archive_inner_zip(tmp_path: Path):
    inner = tmp_path / "holdout_zarrs_6bba.zip"
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("train/6bba_05db0fb1.zarr/zarr.json", b"inner")
    outer = tmp_path / "worker.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.write(inner, arcname="holdout_zarrs_6bba.zip")
        zf.writestr("zarr_export_manifest.json", b"{}")
        zf.writestr("train/44b6_aaa.zarr/zarr.json", b"nope")
    stats = _dl_mod.extract_holdout_zarr_zip(outer, tmp_path / "data")
    assert stats["n_stores"] == 1
    assert (tmp_path / "data/train/6bba_05db0fb1.zarr/zarr.json").read_bytes() == b"inner"
    assert not (tmp_path / "data/train/44b6_aaa.zarr").exists()


def test_packaged_kernel_is_private_internet_off():
    spec = KERNELS["zarr_export"]
    meta = metadata_payload(spec)
    assert meta["id"] == "mahdimzafari/biohub-holdout-zarr-export"
    assert meta["id"] not in {
        "mahdimzafari/biohub-baseline-v1",
        "mahdimzafari/biohub-association-v2",
        "mahdimzafari/biohub-geff-export",
    }
    assert meta["is_private"] is True
    assert meta["enable_internet"] is False
    assert meta["enable_gpu"] is False
    assert meta["competition_sources"] == ["biohub-cell-tracking-during-development"]

    written = build_kernel("zarr_export")
    kernel_meta = json.loads((ROOT / spec["kernel_dir"] / "kernel-metadata.json").read_text())
    assert kernel_meta["is_private"] is True
    assert kernel_meta["enable_internet"] is False
    notebook = json.loads((ROOT / spec["notebook"]).read_text())
    assert notebook["metadata"]["kaggle"]["isInternetEnabled"] is False
    source = (ROOT / spec["source"]).read_text()
    assert "biohub_tracking" not in source
    assert "competitions submit" not in source.lower()
    assert "6bba_05db0fb1" in source
    assert "6bba_c328f2fd" in source
    assert "44b6_" not in source or "Do **not**" in source
    assert written
