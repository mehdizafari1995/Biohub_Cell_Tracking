# learned_unet_orgdet_fnn_acf_test_v1

Same family as `learned_unet_orgdet_fnn_acf_v1` (draft PR #7). Do **not**
merge. Do **not** overwrite this slug or
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
version **1 COMPLETE / inferred** at 2026-09-27. Internet off. Attaches
competition + `mahdimzafari/biohub-learned-unet-v1`.
**Not a competition submission. Not a leaderboard score.**

| metric | orgdet-fnn-fast-test v1 | orgdet-fnn-acf-test v1 |
|---|---|---|
| nodes | 197058 | **196290** |
| edges | 180125 | **178473** |
| gap links | 9499 | **9499** |
| interpolated | 13629 | **13629** |
| dropped isolates | 5080 | **5848** |
| dropped long dt=1 | 7441 | **7441** |
| dropped non-forward-NN | 7710 | **7710** |
| dropped inconsistent dt=1 | — | **1652** |
| divisions | 0 | 0 |
| runtime | 320.23 s | **323.98 s** |
| peak RAM | 1186 MB | **1187 MB** |
| worker audit | ok | ok |
| independent audit | ok | ok |

Per-sequence: `44b6_0113de3b` 26352/24745, `44b6_0b24845f` 74670/65853,
`6bba_05b6850b` 8159/7369, `6bba_05db0fb1` **87109/80506** matches
locked orgdet-fnn-acf holdout overlay (626 inconsistent dt=1).

Hidden-test projection from this rehearsal: **80.99 s/volume × 199 ≈
4.48 h**, peak ~1.2 GB, under the 12 h GPU limit and the 90%
fail-closed gate.

## Status

**READY_FOR_MANUAL_SUBMISSION_REVIEW.** Label
`KEEP_CANDIDATE_NOT_SUBMISSION_AUTHORIZED`. Preferred packaged public-test
worker for the ACF holdout champion. CSV stays under ignored
`artifacts/kaggle/`. Slug is in `FORBIDDEN_SLUGS`. Do **not** overwrite
v1.

## Next

Do **not** submit this slug from the CLI. Additional UI submit is
**not** authorized unless a later message separately authorizes that
extra click. Do **not** retune from **0.602**, **0.612**, or **0.553**.
A later 44b6-first hypothesis must still beat ACF **0.734620** by
**+0.002000**.
