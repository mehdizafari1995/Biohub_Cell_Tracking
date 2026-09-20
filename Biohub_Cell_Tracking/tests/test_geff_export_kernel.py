from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

from kaggle_geff_export import iter_geff_stores, wanted_geff, zip_geff_stores


def _touch_geff(root: Path, name: str, files: list[str]) -> Path:
    store = root / f"{name}.geff"
    for rel in files:
        dest = store / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("x")
    return store


def test_wanted_geff_is_tuning_or_holdout():
    assert wanted_geff(Path("train/44b6_abc.geff")) is True
    assert wanted_geff(Path("train/6bba_05db0fb1.geff")) is True
    assert wanted_geff(Path("train/6bba_c328f2fd.geff")) is True
    assert wanted_geff(Path("train/6bba_ffffff.geff")) is False
    assert wanted_geff(Path("train/44b6_abc.zarr")) is False


def test_iter_and_zip_skips_zarr_and_other_embryos(tmp_path: Path):
    train = tmp_path / "train"
    _touch_geff(train, "44b6_aaa", ["zarr.json", "nodes/ids/c/0"])
    _touch_geff(train, "6bba_05db0fb1", ["zarr.json"])
    _touch_geff(train, "6bba_other", ["zarr.json"])
    (train / "44b6_aaa.zarr" / "0").mkdir(parents=True)
    (train / "44b6_aaa.zarr" / "0" / "c").write_text("nope")
    stores = iter_geff_stores(train)
    assert [p.name for p in stores] == ["44b6_aaa.geff", "6bba_05db0fb1.geff"]
    dest = tmp_path / "out.zip"
    stats = zip_geff_stores(stores, dest)
    assert stats["n_stores"] == 2
    with zipfile.ZipFile(dest) as zf:
        names = zf.namelist()
    assert "train/44b6_aaa.geff/zarr.json" in names
    assert "train/6bba_05db0fb1.geff/zarr.json" in names
    assert all(".zarr/" not in n for n in names)
    assert all("6bba_other" not in n for n in names)


def test_extract_geff_export_zip_skips_zarr(tmp_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "download_geffs",
        ROOT / "scripts" / "download_geffs.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    zpath = tmp_path / "export.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("train/44b6_aaa.geff/zarr.json", b"g")
        zf.writestr("train/44b6_aaa.zarr/0/c/0", b"nope")
        zf.writestr("other.txt", b"x")
    stats = mod.extract_geff_export_zip(zpath, tmp_path / "data")
    assert stats["n_stores"] == 1
    assert (tmp_path / "data/train/44b6_aaa.geff/zarr.json").read_text() == "g"
    assert not (tmp_path / "data/train/44b6_aaa.zarr").exists()
