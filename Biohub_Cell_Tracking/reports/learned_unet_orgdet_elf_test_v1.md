# learned_unet_orgdet_elf_test_v1

Same family as `learned_unet_orgdet_elf_v1` (draft PR #7). Do **not**
merge. Do **not** overwrite this slug or
`mahdimzafari/biohub-learned-unet-orgdet-gap-prune-test-v1` v1,
orgdet-gap-test, orgdet-test, prune-test, or gap-test. Do **not** retune
hash `ad375d39e13098d1`. Do **not** score frozen `6bba` again.

## Operating point

Config: `configs/learned_unet_orgdet_elf_test_v1.yaml`.
Hash `ad375d39e13098d1`. Parent overlay `292c4f1f968f948d` (official
44b6 **0.709838 vs prune 0.693437**; official 6bba **0.737787 vs
0.723018**). Parent prune-test `52a30a5843e0a880`.

- Discover every `test/*.zarr`
- Locked orgdet detections: 0.99 / pool 3.0 / TTA
- Locked greedy 12 µm, then association_v2 gap-close
- Then drop leftover degree-0 nodes
- Then drop consecutive dt=1 edges longer than 8.0 µm
- Then drop newly created degree-0 leftovers
- Writes `submission.csv` + `run_manifest.json`

Not another 12 µm assignment. Not a short-fragment retune.

## Kernel

Private GPU slug `mahdimzafari/biohub-learned-unet-orgdet-elf-test-v1`
version **1 COMPLETE / inferred** at 2026-09-26. Internet off. Attaches
competition + `mahdimzafari/biohub-learned-unet-v1`.
**Not a competition submission.**

| metric | orgdet-gap-prune-test v1 | orgdet-elf-test v1 |
|---|---|---|
| nodes | 201671 | **199977** |
| edges | 195276 | **187835** |
| gap links | 9499 | **9499** |
| interpolated | 13629 | **13629** |
| dropped isolates | 467 | **2161** |
| dropped long dt=1 | 0 | **7441** |
| divisions | 0 | 0 |
| runtime | 331.08 s | **352.87 s** |
| peak RAM | 1164 MB | **1161 MB** |
| worker audit | ok | ok |
| independent audit | ok | ok |

Per-sequence: `44b6_0113de3b` 26771/25667, `44b6_0b24845f` 76487/70345,
`6bba_05b6850b` 8397/7864, `6bba_05db0fb1` **88322/83959** matches
locked orgdet-elf holdout overlay.

Hidden-test projection from this rehearsal: **88.22 s/volume × 199 ≈
4.88 h**, peak ~1.2 GB, under the 12 h GPU limit.

## Status

`READY_FOR_MANUAL_SUBMISSION_REVIEW`. Label
`KEEP_CANDIDATE_NOT_SUBMISSION_AUTHORIZED` until a later message
authorizes a UI click. CSV stays under ignored `artifacts/kaggle/`.
Slug is in `FORBIDDEN_SLUGS`. Do **not** overwrite v1.

## Next

Additional UI submit of this slug is **not** authorized unless a later
message separately authorizes that extra click. Do **not** submit from
the CLI. Do **not** retune from **0.602**, **0.612**, or **0.553**.
