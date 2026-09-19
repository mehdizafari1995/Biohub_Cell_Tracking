# Kaggle kernel v1 failure

Private kernel `mahdimzafari/biohub-baseline-v1` version **1** failed after ~13 s.

```
ModuleNotFoundError: No module named 'zarr'
```

Internet was off, as required. The Kaggle Python 3.12 image used for this competition does not ship `zarr`.

## Tested fix (version 2)

`notebooks/kaggle_baseline_v1.py` now reads OME-Zarr v3 chunks without the `zarr` package:

1. Prefer `zarr` if present.
2. Else decode `c/t/0/0/0` blosc+zstd+bitshuffle chunks with `imagecodecs`, `numcodecs`, or `blosc2`.
3. Else `pip install --no-index --no-deps` a vendored `numcodecs` wheel if the kernel upload includes `wheels/numcodecs-*.whl`.

Local test: fallback frame 0 of `6bba_05db0fb1` matches `zarr.open_group(...)["0"][0]` exactly.

This is a Kaggle execution failure with a tested fix, not a local simulation.
