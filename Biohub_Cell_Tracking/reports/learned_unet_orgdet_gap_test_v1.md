# learned_unet_orgdet_gap_test_v1

Same family as `learned_unet_orgdet_gap_v1` (draft PR #7). Do **not**
merge. Do **not** overwrite this slug or orgdet-test v1. Do **not**
retune hash `1ce4837ae16bee52`. Do **not** score frozen `6bba` again.

## Operating point

Config: `configs/learned_unet_orgdet_gap_test_v1.yaml`.
Hash `1ce4837ae16bee52`. Parent overlay `ee31f36151325f9a` (official
6bba **0.722805 vs orgdet 0.705226**).

- Discover every `test/*.zarr`
- Locked orgdet detections: 0.99 / pool 3.0 / TTA
- Locked greedy 12 µm, then association_v2 gap-close
- Writes `submission.csv` + `run_manifest.json`

## Kernel

Private GPU slug `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1`
version **1 COMPLETE / inferred** at 2026-09-22T09:05:18Z. Internet
off. Attaches competition + `mahdimzafari/biohub-learned-unet-v1`.
**Not a competition submission.**

| metric | orgdet-test v1 | orgdet-gap-test v1 |
|---|---|---|
| nodes | 188509 | **202138** |
| edges | 172148 | **195276** |
| gap links | 0 | **9499** |
| interpolated | 0 | **13629** |
| dropped isolates | 0 | **0** |
| divisions | 0 | 0 |
| runtime | 334.23 s | **312.69 s** |
| peak RAM | 1183 MB | **1176 MB** |
| worker audit | ok | ok |
| independent audit | ok | ok |

Per-sequence: `44b6_0113de3b` 26897/25892, `44b6_0b24845f` 77793/75022,
`6bba_05b6850b` 8523/8075, `6bba_05db0fb1` **88925/86287** matches
locked orgdet-gap holdout overlay (84328+4597 / 78493+7794).

Hidden-test projection from this rehearsal: **78.17 s/volume × 199 ≈
4.32 h**, peak ~1.2 GB, under the 12 h GPU limit.

## Status

`READY_FOR_MANUAL_SUBMISSION_REVIEW`. Label
`KEEP_CANDIDATE_NOT_SUBMISSION_AUTHORIZED` until a human submits from
the Kaggle UI. CSV stays under ignored `artifacts/kaggle/`. Slug is in
`FORBIDDEN_SLUGS`. Do **not** overwrite v1.

## Submit poll (2026-09-22T13:46:09Z)

Read-only reconfirm. Kernel still **complete**. Outputs still
`run_manifest.json` + `submission.csv`. Competition list still exactly
one row (association_v2 **56399003**, public API string **`0.553`**).
Decision `human_action_required_ui_submit`. Did **not** download the
CSV. Did **not** submit. `0.553` is **not** this slug's leaderboard
score. Prior poll **2026-09-22T12:42:26Z** is unchanged.

## Submit poll (2026-09-26T19:55:35Z)

Read-only. Kernel still **complete**. Outputs still
`run_manifest.json` + `submission.csv`. Competition list now has three
rows. Identity-proven row **56583783** is this notebook (`scriptVersionId`
**351808956**), status **complete**, public API string **`0.602`**,
private **not returned**. Decision `public_score_returned`. That
**0.602** is **not** orgdet-test **0.612**, **not** association_v2
**0.553**, and **not** a local official holdout score. Do **not**
invent extra digits. Do **not** retune from **0.602**. Do **not**
resubmit. Do **not** re-poll **56583783** as pending.

## Next

Do **not** resubmit this slug. Preferred unsubmitted worker is now
`learned_unet_orgdet_gap_prune_test_v1` v1 (`READY_FOR_MANUAL_SUBMISSION_REVIEW`).
Additional UI submit of that prune-test slug is **not** authorized
unless a later message separately authorizes that extra click.
