# learned_unet_orgdet_fnn_test_v1

Same family as `learned_unet_orgdet_fnn_v1` (draft PR #7). Do **not**
merge. Do **not** overwrite this slug or
`mahdimzafari/biohub-learned-unet-orgdet-elf-test-v1` v1,
orgdet-gap-prune-test, orgdet-gap-test, orgdet-test, prune-test, or
gap-test. Do **not** retune hash `1af763e6ffc23188`. Do **not** score
frozen `6bba` again.

## Operating point

Config: `configs/learned_unet_orgdet_fnn_test_v1.yaml`.
Hash `1af763e6ffc23188`. Parent overlay `95e9300eb84ba71c` (official
44b6 **0.724653 vs ELF 0.709838**; official 6bba **0.752091 vs
0.737787**). Parent elf-test `ad375d39e13098d1`.

- Discover every `test/*.zarr`
- Locked orgdet detections: 0.99 / pool 3.0 / TTA
- Locked greedy 12 µm, then association_v2 gap-close
- Then drop leftover degree-0 nodes
- Then drop consecutive dt=1 edges longer than 8.0 µm
- Then keep only forward nearest-neighbor dt=1 edges
- Then drop newly created degree-0 leftovers
- Writes `submission.csv` + `run_manifest.json`

Not another 12 µm assignment. Not a short-fragment retune. Not another
8.0 µm dt=1 length retune.

## Kernel

Private GPU slug `mahdimzafari/biohub-learned-unet-orgdet-fnn-test-v1`
version **1 COMPLETE / inferred** at 2026-09-26. Internet off. Attaches
competition + `mahdimzafari/biohub-learned-unet-v1`.
**Not a competition submission.**

| metric | orgdet-elf-test v1 | orgdet-fnn-test v1 |
|---|---|---|
| nodes | 199977 | **197058** |
| edges | 187835 | **180125** |
| gap links | 9499 | **9499** |
| interpolated | 13629 | **13629** |
| dropped isolates | 2161 | **5080** |
| dropped long dt=1 | 7441 | **7441** |
| dropped non-forward-NN | 0 | **7710** |
| divisions | 0 | 0 |
| runtime | 352.87 s | **798.80 s** |
| peak RAM | 1161 MB | **1173 MB** |
| worker audit | ok | ok |
| independent audit | ok | ok |

Per-sequence: `44b6_0113de3b` 26695/25375, `44b6_0b24845f` 74703/65935,
`6bba_05b6850b` 8321/7683, `6bba_05db0fb1` **87339/81132** matches
locked orgdet-fnn holdout overlay.

Hidden-test projection from this rehearsal: **199.70 s/volume × 199 ≈
11.04 h**, peak ~1.2 GB, under the 12 h GPU limit but
`fail_closed_approaches_12h`. Naive per-edge nearest-neighbor scan was
much slower than the 1 s overhead estimate on dense public-test stems.

## Status

**Not** `READY_FOR_MANUAL_SUBMISSION_REVIEW` because the 90% hidden-runtime
gate failed (11.04 h ≥ 10.8 h). Label
`KEEP_CANDIDATE_NOT_SUBMISSION_AUTHORIZED`. Preferred packaged public-test
fallback remains `mahdimzafari/biohub-learned-unet-orgdet-elf-test-v1`
(4.88 h). CSV stays under ignored `artifacts/kaggle/`. Slug is in
`FORBIDDEN_SLUGS`. Do **not** overwrite v1.

## Next

Do **not** submit this slug from the CLI or UI unless a later message
separately authorizes that extra click. Do **not** retune from **0.602**,
**0.612**, or **0.553**. A later 44b6-first hypothesis must beat FNN
**0.724653** by **+0.002000**. A faster immutable FNN public-test slug
(same locked overlay, new hash/slug, no gate retune) may be needed before
this champion is hidden-safe.
