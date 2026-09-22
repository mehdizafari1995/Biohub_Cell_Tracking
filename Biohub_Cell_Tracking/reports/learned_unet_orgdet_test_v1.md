# learned_unet_orgdet_test_v1

Same experiment family as `learned_unet_orgdet_holdout_v1` (draft PR #7).
Do **not** merge. Do not overwrite `baseline_v1`, `association_v2`,
`division_v2`, `complementary_v1`, `linkable_v1`, rejected linker
overlays, or locked learned train/infer/holdout/test/**gap-test**/
**prune-test** / **orgdet-holdout** slugs. Do **not** retune. Do
**not** score frozen `6bba` again.

## Hypothesis

Locked `learned_unet_orgdet_holdout_v1` kept on frozen `6bba`
(**0.705226 vs prune 0.701378**) at organizer detections
(`det_threshold=0.99`, `pool_kernel_um=3.0`, flip TTA) plus greedy
12 µm last-position linking. Applying that locked operating point to
every public or hidden `test/*.zarr` at runtime can write a
schema-valid `submission.csv`. This is a packaging / runtime
rehearsal, not a retune. Public-test stems are not used to change
thresholds. Frozen `6bba` is not official-scored again.

## Operating point (locked; do not retune)

Config: `configs/learned_unet_orgdet_test_v1.yaml`. Hash
`c45c859610d29e90`. Parent orgdet `80b20589198148b7`. Parent orgdet
holdout `912f0c19b572a054`. Training hash `4c38035d6657f2ad`. Rejected
orgop `250c5bfacd4a8198`. No-gap test parent `7a8725b818fc609b`.
Gap-test parent `85379c198ebc7241`. Prune-test parent `074a71dec68304e8`.

- Discovers every `test/*.zarr` at runtime. No hard-coded public stems
- Never opens `train/` volumes
- Detection: `det_threshold` 0.99, TTA flip-x/y/xy, `pool_kernel_um` 3.0
- Consecutive linker: greedy 12 µm, 1-1, last-position. No gap overlay.
  No isolate drop. No ILP. No transformer ranking.
- Writes `submission.csv` + `run_manifest.json`
- New slug `mahdimzafari/biohub-learned-unet-orgdet-test-v1`
- Does **not** overwrite `mahdimzafari/biohub-learned-unet-prune-test-v1`

## Cost estimate (before push)

Orgdet-holdout v1 already measured **173.8 s / 2 volumes** (1185 MB)
at this operating point. Conservative hidden-test budget is
**86.9 s/volume × 199 ≈ 4.8 h**, under the 12 h GPU limit.

| set | volumes | estimate | under 12 h GPU limit |
|---|---|---|---|
| public test | 4 | ~5.8 min | yes |
| hidden test (train-sized) | ~199 | **~4.8 h** | yes |

See `reports/learned_unet_orgdet_test_v1_estimate.json`. Estimate only.
Not a Kaggle run and not a leaderboard score.

## Kaggle test-time notebook

Private GPU kernel `mahdimzafari/biohub-learned-unet-orgdet-test-v1`.
Internet off. GPU on (same T4 class as orgdet-holdout). Attaches
competition data + locked `mahdimzafari/biohub-learned-unet-v1`
weights. Refuses public 199-video / DeepCenter checkpoints. **Not a
competition submission.** Do **not** submit from the CLI.

## Kaggle rehearsal (not a leaderboard score)

Private GPU kernel `mahdimzafari/biohub-learned-unet-orgdet-test-v1`
version **1 COMPLETE / inferred**. Internet off. Environment
`kaggle_input`. T4 x2, torch 2.10+cu128. Weights
`/kaggle/input/notebooks/mahdimzafari/biohub-learned-unet-v1/learned_unet_v1_last.pt`.
Worker `audit_ok`. Independent stream audit `audit_ok`. **Not submitted.**
Competition list still exactly association_v2 **56399003**. Frozen
`6bba` was not official-scored again.

| metric | prune-test v1 | orgdet-test v1 |
|---|---|---|
| nodes | 250864 | **188509** |
| edges | 242412 | **172148** |
| gap links | 13912 | **0** |
| interpolated | 19953 | **0** |
| dropped isolates | 796 | **0** |
| divisions | 0 | 0 |
| runtime | 348.18 s | **334.23 s** |
| peak RAM | 1202 MB | **1183 MB** |
| worker audit | ok | ok |
| independent audit | ok | ok |

Per-sequence: `44b6_0113de3b` 26355/24972, `44b6_0b24845f` 69893/61602,
`6bba_05b6850b` 7933/7081, `6bba_05db0fb1` **84328/78493** matches locked
orgdet-holdout graphs exactly.

Hidden-test projection from this rehearsal: **83.56 s/volume × 199 ≈
4.62 h**, peak ~1.2 GB, under the 12 h GPU limit. Fail-closed gate
did not trigger.

## Status

`READY_FOR_MANUAL_SUBMISSION_REVIEW`. Do **not** submit from the CLI.
Do **not** overwrite v1. CSV stays under ignored `artifacts/kaggle/`.

Read-only submit poll **2026-09-22T04:50:14Z**: kernel still `complete`,
outputs still `run_manifest.json` + `submission.csv`, competition list
still exactly one row (association_v2 **56399003** / **0.553**). Decision
`human_action_required_ui_submit`. Did **not** download CSV. Did **not**
submit. Public `0.553` is **not** an orgdet-test leaderboard score.

## Next

`HUMAN_ACTION_REQUIRED` — Mahdi must **Submit to Competition** from the
Kaggle UI on `mahdimzafari/biohub-learned-unet-orgdet-test-v1` version 1
**before the 2026-09-22 23:59 UTC entry deadline**. Do not submit
prune-test or gap-test now that orgdet-test v1 is COMPLETE / `audit_ok`.
Do not resubmit association_v2 **56399003**. Do not overwrite the
running retrain helper. Do not score frozen `6bba` again.
