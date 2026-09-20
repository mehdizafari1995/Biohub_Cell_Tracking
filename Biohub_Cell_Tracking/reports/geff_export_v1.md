# Private kernel `mahdimzafari/biohub-geff-export` v1

Helper kernel. **Not a competition submission.** No `submission.csv`. No leaderboard score.

## Why

Per-file Kaggle GEFF downloads hit HTTP 429 at ~one success per 90 s (13 required files × 73 stores is not runnable in a cron cycle). This private internet-off kernel attached the competition dataset and zipped only GEFF stores.

## Result

- Status: **COMPLETE** (private, internet off, CPU).
- Output: `geff_export_manifest.json` + `train_geffs_44b6.zip` (655 129 bytes uncompressed members; outer worker archive 321 999 bytes).
- Stores: **73** (71 `44b6` + holdout `6bba_05db0fb1` + `6bba_c328f2fd`).
- Files inside the GEFF zip: 1533 (full 21-file layout).
- Image volumes: **not copied**.
- Worker elapsed: 203.74 s.
- Local extract: 71/71 complete `44b6` GEFFs and 2/2 holdout GEFFs under ignored `data/train/`. Zip kept under ignored `artifacts/kaggle/`.

## Policy

Do not submit this kernel. Do not overwrite `biohub-baseline-v1` or `biohub-association-v2`. Do not commit the zip or extracted stores.
