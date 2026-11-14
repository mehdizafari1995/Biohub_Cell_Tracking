# Private kernel `mahdimzafari/biohub-holdout-zarr-export` v1

Helper kernel. **Not a competition submission.** No `submission.csv`. No leaderboard score.

## Why

Official holdout of `division_v2` needs the two frozen-validation **Zarr** volumes
(`6bba_05db0fb1`, `6bba_c328f2fd`; ≈1.0 GB). Per-file Kaggle downloads 429 at
~90 s/file. This private internet-off kernel attached the competition dataset
and zipped **only those two stores**.

## Estimate (before the run)

- Storage: two typical `(100, 64, 256, 256)` uint16 volumes, already blosc-compressed, ≈1.0 GB.
- CPU / GPU: CPU zip only. No GPU.
- Kaggle: 12 h CPU limit; expected under 5 minutes. Output well under the 20 GB worker cap.

## Result

- Status: **COMPLETE** (private, internet off, CPU).
- Output: `zarr_export_manifest.json` + `holdout_zarrs_6bba.zip` (993 198 228 bytes; 204 members; 13.43 s worker).
- Outer worker archive: 1 047 433 547 bytes, kept under ignored `artifacts/kaggle/`.
- Stores: **2 / 2** holdout Zarrs. No other volumes. No GEFFs.
- Local extract: both sequences complete (100 frames) under ignored `data/train/`.
- Do **not** commit the zip or extracted stores.

## Policy

Do not submit this kernel. Do not overwrite `biohub-baseline-v1`,
`biohub-association-v2`, or `biohub-geff-export`.
