# learned_unet_orgdet_fnn_acf_test_v1

Same family as `learned_unet_orgdet_fnn_acf_v1` (draft PR #7). Do **not**
merge. Do **not** overwrite
`mahdimzafari/biohub-learned-unet-orgdet-fnn-fast-test-v1` v1,
fnn-test v1, or elf-test v1. Do **not** retune hash `2d13d0e3c90df093`.
Do **not** score frozen `6bba` again. Do **not** submit.

## Operating point

Config: `configs/learned_unet_orgdet_fnn_acf_test_v1.yaml`.
Hash `2d13d0e3c90df093`. Parent overlay `a66075085ac64400` (official
44b6 **0.734620 vs FNN 0.724653**; official 6bba **0.755503 vs
0.752091**). Parent fnn-fast-test `9b9763e26bc0f037`. Parent FNN
`95e9300eb84ba71c`.

- Discover every `test/*.zarr`
- Locked orgdet detections: 0.99 / pool 3.0 / TTA
- Locked greedy 12 µm, then association_v2 gap-close
- Then drop leftover degree-0 nodes
- Then drop consecutive dt=1 edges longer than 8.0 µm
- Then keep only forward nearest-neighbor dt=1 edges (vectorized)
- Then sample 3×3×3-mean appearance from the sequence Zarr
- Then drop dt=1 edges with relative appearance > 0.38
- Then drop newly created degree-0 leftovers
- Writes `submission.csv` + `run_manifest.json`

Not another 12 µm assignment. Not a rank/div/reas/ELF/FNN/TCF retune.
Do **not** retune `max_rel_appearance`.

## Kernel

Private GPU slug `mahdimzafari/biohub-learned-unet-orgdet-fnn-acf-test-v1`
packaged. Internet off. Attaches competition +
`mahdimzafari/biohub-learned-unet-v1`.
**Not a competition submission. Not a leaderboard score.**

## Cost estimate (before rehearsal)

- Public: fnn-fast-test measured **320.23 s / 4 volumes** plus **15 s**
  3×3×3-mean appearance filter ≈ **380 s**
- Hidden: **95.06 s/volume × 199 ≈ 5.25 h**, peak ~1.2 GB, under the
  12 h GPU limit and the 90% fail-closed gate
- A local package is not a Kaggle run

## Status

**PACKAGED AND PUSHED.** Version **1 running**. New slug is **not**
in `FORBIDDEN_SLUGS` until v1 COMPLETE. Preferred packaged public-test
worker remains fnn-fast-test until this kernel is COMPLETE / `audit_ok`.
Do **not** overwrite fnn-fast-test v1. Do **not** push a second version.
Do **not** submit from the CLI.

## Next

Poll until COMPLETE. Collect into ignored `artifacts/kaggle/`. Do
**not** submit. Do **not** score 6bba again. Do **not** retune
`max_rel_appearance`. Do **not** use **0.602**, **0.612**, or **0.553**.
