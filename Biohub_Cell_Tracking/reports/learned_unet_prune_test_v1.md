# learned_unet_prune_test_v1

Same experiment family as `learned_unet_prune_v1` (draft PR #7). Do **not**
merge. Do not overwrite `baseline_v1`, `association_v2`, `division_v2`,
`complementary_v1`, `linkable_v1`, rejected linker overlays, or locked
learned train/infer/holdout/test/**gap-test** slugs. Do **not** retune.
Do **not** score frozen `6bba` again.

## Hypothesis

Locked `TemporalUNet3D` detections plus the locked `learned_unet_gap_v1`
overlay and the locked `learned_unet_prune_v1` degree-0 drop can be
applied to every public or hidden `test/*.zarr` at runtime and write a
schema-valid `submission.csv`. This is a packaging / runtime rehearsal,
not a retune. Public-test stems are not used to change thresholds.
Frozen `6bba` is not official-scored again.

## Operating point (locked; do not retune)

Config: `configs/learned_unet_prune_test_v1.yaml`. Hash `074a71dec68304e8`.
Parent infer `d7578b5f64ddac08`. Parent holdout `d374a21f03dddc04`.
Parent gap overlay `f4d888d190c45a4a`. Parent prune `115753ea16aaf021`.
No-gap test-time parent `7a8725b818fc609b`. Gap-test parent
`85379c198ebc7241`. Training hash `4c38035d6657f2ad`.

- Discovers every `test/*.zarr` at runtime. No hard-coded public stems
- Never opens `train/` volumes
- Detection: `det_threshold` 0.5, TTA flip-x/y/xy, `pool_kernel_um` 5.0
- Consecutive linker: greedy 12 µm, 1-1, then locked gap overlay, then drop isolates
- Writes `submission.csv` + `run_manifest.json`
- New slug `mahdimzafari/biohub-learned-unet-prune-test-v1`
- Does **not** overwrite `mahdimzafari/biohub-learned-unet-gap-test-v1`

## Cost estimate (before push)

Gap-test public rehearsal already measured **330.0 s / 4 volumes**
(1184 MB). Isolate drop is a cheap CPU scan (~0.5 s/volume; holdout
prune overlay was 9.83 s / 2 including gap). Conservative hidden-test
budget remains holdout-based **97.5 s/volume × 199 ≈ 5.4 h**, under
the 12 h GPU limit.

| set | volumes | estimate | under 12 h GPU limit |
|---|---|---|---|
| public test | 4 | ~6.5 min | yes |
| hidden test (train-sized) | ~199 | **~5.4 h** | yes |

## Kaggle test-time notebook

Private GPU kernel `mahdimzafari/biohub-learned-unet-prune-test-v1`.
Internet off. GPU on (same T4 class as gap-test). Attaches competition
data + locked `mahdimzafari/biohub-learned-unet-v1` weights. **Not a
competition submission.** Do **not** submit from the CLI.

## Kaggle rehearsal (not a leaderboard score)

Private GPU kernel `mahdimzafari/biohub-learned-unet-prune-test-v1`
version **1 COMPLETE / inferred**. Internet off. Environment
`kaggle_input`. T4 x2, torch 2.10+cu128. Weights
`/kaggle/input/notebooks/mahdimzafari/biohub-learned-unet-v1/learned_unet_v1_last.pt`.
**Not submitted.** Competition list still exactly association_v2
**56399003**.

| metric | gap-test v1 | prune-test v1 |
|---|---|---|
| nodes | 251660 | **250864** |
| edges | 242412 | **242412** |
| gap links | 13912 | 13912 |
| interpolated | 19953 | 19953 |
| dropped isolates | — | **796** |
| divisions | 0 | 0 |
| runtime | 330.02 s | **348.18 s** |
| peak RAM | 1184 MB | **1202 MB** |
| worker audit | ok | ok |
| independent audit | ok | ok |

Per-sequence isolate drops: `44b6_0113de3b` 220, `44b6_0b24845f` 233,
`6bba_05b6850b` 139, `6bba_05db0fb1` 204. Edges match locked gap-test
on every stem, including holdout-overlap `6bba_05db0fb1` 100197.

Hidden-test projection from this rehearsal: **87.05 s/volume × 199 ≈
4.81 h**, peak ~1.2 GB, under the 12 h GPU limit. Fail-closed gate
did not trigger.

## Status

`READY_FOR_MANUAL_SUBMISSION_REVIEW`. Do **not** submit from the CLI.
`HUMAN_ACTION_REQUIRED`: Mahdi must **Submit to Competition** from the
Kaggle UI on `mahdimzafari/biohub-learned-unet-prune-test-v1` version 1.
Do not overwrite this slug.
