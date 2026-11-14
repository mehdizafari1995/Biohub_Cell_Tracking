# learned_unet_orgdet_gap_prune_test_v1

Same family as `learned_unet_orgdet_gap_prune_v1` (draft PR #7). Do **not**
merge. Do **not** overwrite this slug or
`mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1` v1, orgdet-test,
prune-test, or gap-test. Do **not** retune hash `52a30a5843e0a880`.
Do **not** score frozen `6bba` again.

## Operating point

Config: `configs/learned_unet_orgdet_gap_prune_test_v1.yaml`.
Hash `52a30a5843e0a880`. Parent overlay `862e3e7f7ac0330e` (official
6bba **0.723018 vs orgdet-gap 0.722805**). Parent orgdet-gap-test
`1ce4837ae16bee52`.

- Discover every `test/*.zarr`
- Locked orgdet detections: 0.99 / pool 3.0 / TTA
- Locked greedy 12 µm, then association_v2 gap-close
- Then drop leftover degree-0 nodes
- Writes `submission.csv` + `run_manifest.json`

## Kernel

Private GPU slug `mahdimzafari/biohub-learned-unet-orgdet-gap-prune-test-v1`
version **1 COMPLETE / inferred** at 2026-09-26T19:54:59Z. Internet
off. Attaches competition + `mahdimzafari/biohub-learned-unet-v1`.
**Not a competition submission.**

| metric | orgdet-gap-test v1 | orgdet-gap-prune-test v1 |
|---|---|---|
| nodes | 202138 | **201671** |
| edges | 195276 | **195276** |
| gap links | 9499 | **9499** |
| interpolated | 13629 | **13629** |
| dropped isolates | 0 | **467** |
| divisions | 0 | 0 |
| runtime | 312.69 s | **331.08 s** |
| peak RAM | 1176 MB | **1164 MB** |
| worker audit | ok | ok |
| independent audit | ok | ok |

Per-sequence: `44b6_0113de3b` 26815/25892, `44b6_0b24845f` 77621/75022,
`6bba_05b6850b` 8458/8075, `6bba_05db0fb1` **88777/86287** matches
locked orgdet-gap-prune holdout overlay (88925−148 / 86287).

Hidden-test projection from this rehearsal: **82.77 s/volume × 199 ≈
4.58 h**, peak ~1.2 GB, under the 12 h GPU limit.

## Status

`READY_FOR_MANUAL_SUBMISSION_REVIEW`. Label
`KEEP_CANDIDATE_NOT_SUBMISSION_AUTHORIZED` until a later message
authorizes a UI click. CSV stays under ignored `artifacts/kaggle/`.
Slug is in `FORBIDDEN_SLUGS`. Do **not** overwrite v1.

## Submit poll (2026-09-26T19:55:35Z, classifier corrected)

Read-only. Kernel **complete**. Outputs `run_manifest.json` +
`submission.csv`. Competition list has **three** rows:

- association_v2 **56399003** public API **`0.553`**
- orgdet-test **56466744** public API **`0.612`**
- orgdet-gap-test **56583783** public API **`0.602`** (`scriptVersionId` **351808956**)

This slug is **not** on the competition list. Decision
`human_action_required_ui_submit`. Did **not** submit. Public **0.602**
is orgdet-gap-test only and was **not** used for tuning.

## Next

Additional UI submit of this slug is **not** authorized unless a later
message separately authorizes that extra click. Do **not** submit from
the CLI. Do **not** retune from **0.602**, **0.612**, or **0.553**.
