# learned_unet_infer_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not** merge.
Do not overwrite `baseline_v1`, `association_v2`, `division_v2`,
`complementary_v1`, `linkable_v1`, or the completed training helper
`mahdimzafari/biohub-learned-unet-v1`.

## Hypothesis

Specimen-separated `TemporalUNet3D` detections, linked with the locked
association_v2 12 µm 1-1 gate (no gap-close), can raise node recall on
frozen-tuning `44b6` without the extra-`T_pred` failure of complementary
local-max overlays. Transformer edge scores are **not** used in this first
infer pass so the comparison isolates detection.

## Operating point (locked from 44b6; holdout not run)

Config: `configs/learned_unet_infer_v1.yaml`. Hash `d7578b5f64ddac08`.
Training hash remains `4c38035d6657f2ad`.

- Infer stems only: `44b6_12dfb391`, `44b6_267148e4`
- Holdout `6bba` is refused
- `det_threshold` 0.5, TTA flip-x/y/xy, `pool_kernel_um` 5.0
- Normalisation matches training: per-window min-max, not organizer quantiles
- Linker: greedy 12 µm, max_children=1, max_parents=1, no gap-close
- Writes JSON graphs + `infer_manifest.json`. Does **not** write `submission.csv`
- Worker used `/kaggle/input/notebooks/mahdimzafari/biohub-learned-unet-v1/learned_unet_v1_last.pt` (epoch 2 = best train loss)

Do **not** retune these gates after the first frozen-validation official score.

## Kaggle helper

Private GPU kernel `mahdimzafari/biohub-learned-unet-infer-v1`. Internet off.
**Not a submission.** Attaches `mahdimzafari/biohub-learned-unet-v1`.

| version | status | notes |
|---|---|---|
| 1 | COMPLETE / `blocked_no_weights` | 5.08 s. Hard-coded `/kaggle/input/biohub-learned-unet-v1` miss. |
| 2 | COMPLETE / `blocked_no_weights` | 3.74 s. `/kaggle/input` = `{competitions, notebooks}`. |
| 3 | **COMPLETE / inferred** | 187.0 s, peak 1136 MB, T4 x2. Graphs: `44b6_12dfb391` 80741/74147, `44b6_267148e4` 36951/32238. No `submission.csv`. |

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4` only.

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes |
|---|---|---|---|---|---|---|
| association_v2 | **0.632460** | 0.64514 | 0.63246 | 0.97758 | 0/0/2 | 93104 |
| learned_unet_infer_v1 | **0.655751** | 0.69001 | 0.65575 | 0.97004 | 0/0/2 | 117692 |

Per-sequence learned: `44b6_12dfb391` 0.747072 (recall 0.994), `44b6_267148e4` 0.446373 (recall 0.946). Extra T_pred vs estimated nodes, especially on `267148e4`. Decision **`keep_candidate_learned_unet_infer_v1`**. Do **not** look at `6bba` yet. See `reports/learned_unet_infer_v1_tune.json`.

## Next

One-shot official holdout is packaged as **new** helper
`mahdimzafari/biohub-learned-unet-holdout-v1` (hash `d374a21f03dddc04`).
Do not overwrite this 44b6 infer slug. Reject on combined-score
regression. Do not retune. See `reports/learned_unet_holdout_v1.md`.
