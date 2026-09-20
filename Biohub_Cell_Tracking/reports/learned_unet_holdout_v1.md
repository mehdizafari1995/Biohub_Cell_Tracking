# learned_unet_holdout_v1

Same experiment family as `learned_unet_v1` / `learned_unet_infer_v1` (draft PR #7).
Do **not** merge. Do not overwrite `baseline_v1`, `association_v2`,
`division_v2`, `complementary_v1`, `linkable_v1`,
`mahdimzafari/biohub-learned-unet-v1`, or
`mahdimzafari/biohub-learned-unet-infer-v1`.

## Hypothesis

Locked `TemporalUNet3D` detections (44b6-only training, locked infer
operating point `d7578b5f64ddac08`) generalize to frozen holdout `6bba`
and raise official combined score versus locked association_v2
**0.591570**. Transformer edge scores are still unused so the comparison
isolates detection. One-shot only. Do not retune.

## Operating point (locked; do not retune)

Config: `configs/learned_unet_holdout_v1.yaml`. Hash `d374a21f03dddc04`.
Parent infer hash `d7578b5f64ddac08`. Training hash `4c38035d6657f2ad`.

- Infer stems only: `6bba_05db0fb1`, `6bba_c328f2fd`
- Tuning `44b6` volumes are refused
- `det_threshold` 0.5, TTA flip-x/y/xy, `pool_kernel_um` 5.0
- Normalisation matches training: per-window min-max
- Linker: greedy 12 µm, max_children=1, max_parents=1, no gap-close
- Writes JSON graphs + `infer_manifest.json`. Does **not** write `submission.csv`
- Worker used `/kaggle/input/notebooks/mahdimzafari/biohub-learned-unet-v1/learned_unet_v1_last.pt`

## Cost estimate (before push)

- Storage: competition `train/` already on Kaggle; no extra download
- GPU: two 100-frame volumes. 44b6 infer v3 was 187 s / 1136 MB on T4 x2.
  Holdout v1 measured **184.4 s / 1177 MB**. Well under the 12 h GPU limit
- Output: two JSON graphs + manifest. No volumes, weights, or `submission.csv`

## Kaggle helper

Private GPU kernel `mahdimzafari/biohub-learned-unet-holdout-v1`. Internet off.
**Not a submission.** Attaches `mahdimzafari/biohub-learned-unet-v1`.

| version | status | notes |
|---|---|---|
| 1 | **COMPLETE / inferred** | 184.4 s, peak 1177 MB, T4 x2. Graphs: `6bba_05db0fb1` 97062/89394, `6bba_c328f2fd` 52161/47595. No `submission.csv`. |

Do **not** overwrite this slug. Do not score `6bba` again.

## Official frozen-holdout score (not leaderboard)

Local official scorer `075fc5f5` once on `6bba_05db0fb1` + `6bba_c328f2fd`.

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes |
|---|---|---|---|---|---|---|
| association_v2 | **0.591570** | 0.58881 | 0.59157 | 0.83824 | 0/0/5 | 96336 |
| learned_unet_holdout_v1 | **0.693605** | 0.72769 | 0.69360 | 0.95713 | 0/0/5 | 149223 |

Per-sequence learned: `6bba_05db0fb1` 0.719310 (recall 0.979) vs association_v2 0.564336; `6bba_c328f2fd` 0.633142 (recall 0.935) vs 0.660194. Combined wins; the second sequence is mixed. Extra T_pred vs estimated nodes, especially on `c328f2fd`. Decision **`keep_candidate_learned_unet_holdout_v1`**. Do **not** retune. Do **not** score holdout twice. See `reports/learned_unet_holdout_v1_holdout.json`.

This is a local official score, not a Kaggle leaderboard number.

## Next

Package a **new** private internet-off test-time notebook (new slug; do not
overwrite train/infer/holdout helpers). Discover every `test/*.zarr` at
runtime. Write `submission.csv`. Estimate ~5 h for a train-sized hidden
test at ~90 s/volume. Do **not** submit from the CLI. Do not retune.
