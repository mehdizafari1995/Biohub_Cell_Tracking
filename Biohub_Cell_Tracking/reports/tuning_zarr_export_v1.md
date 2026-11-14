# Private helper: biohub-tuning-zarr-export v1

Internet-off private kernel. **Not a competition submission.** No
`submission.csv`. Do not overwrite locked prediction or helper kernels.

## Purpose

Per-file Kaggle Zarr downloads 429 at ~90 s/file. This helper zips only the
two frozen-tuning volumes needed to official-score complementary_v1:

- `44b6_12dfb391.zarr`
- `44b6_267148e4.zarr`

Holdout `6bba_*` volumes and all GEFFs are refused.

## Worker

- Slug: `mahdimzafari/biohub-tuning-zarr-export`
- Version: **1 COMPLETE**
- Output: `tuning_zarrs_44b6.zip` plus `tuning_zarr_export_manifest.json`
- Worker archive: 963 516 043 bytes (ignored `artifacts/kaggle/`)
- Extracted locally: 2 stores, 204 files, no GEFFs

Kaggle public / private scores: not applicable; not submitted.
