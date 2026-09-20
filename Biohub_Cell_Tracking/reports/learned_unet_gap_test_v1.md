# learned_unet_gap_test_v1

Same experiment family as `learned_unet_gap_v1` (draft PR #7). Do **not**
merge. Do not overwrite `baseline_v1`, `association_v2`, `division_v2`,
`complementary_v1`, `linkable_v1`, `learned_unet_division_v1`, or locked
learned train/infer/holdout/**no-gap test** slugs. Do **not** retune. Do
**not** score frozen `6bba` again.

## Hypothesis

Locked `TemporalUNet3D` detections plus the locked `learned_unet_gap_v1`
overlay (12 µm, `max_gap_frames=2`, interpolate leftover unmatched nodes)
can be applied to every public or hidden `test/*.zarr` at runtime and write
a schema-valid `submission.csv`. This is a generalization / runtime
hypothesis, not a retune. Public-test stems are not used to change
thresholds. Frozen `6bba` is not official-scored again.

## Operating point (locked; do not retune)

Config: `configs/learned_unet_gap_test_v1.yaml`. Hash `85379c198ebc7241`.
Parent infer `d7578b5f64ddac08`. Parent holdout `d374a21f03dddc04`.
Parent gap overlay `f4d888d190c45a4a`. No-gap test-time parent
`7a8725b818fc609b`. Training hash `4c38035d6657f2ad`.

- Discovers every `test/*.zarr` at runtime. No hard-coded public stems
- Never opens `train/` volumes
- Detection: `det_threshold` 0.5, TTA flip-x/y/xy, `pool_kernel_um` 5.0
- Consecutive linker: greedy 12 µm, 1-1, then locked gap overlay
- Writes `submission.csv` + `run_manifest.json`
- New slug `mahdimzafari/biohub-learned-unet-gap-test-v1`
- Does **not** overwrite `mahdimzafari/biohub-learned-unet-test-v1`

## Cost estimate (before push)

Holdout infer **184.4 s / 2 volumes** (92.2 s/volume) plus overlay
**9.687 s / 2 holdout graphs** (4.8435 s/volume). Combined
**97.0435 s/volume**. Peak RAM observed on holdout infer **1177 MB**.

| set | volumes | estimate | under 12 h GPU limit |
|---|---|---|---|
| public test | 4 | ~6.5 min | yes |
| hidden test (train-sized) | ~199 | **~5.4 h** | yes |

No-gap public test already measured 306.5 s / 1184 MB. Overlay on four
public stems is expected to add on the order of 15 s (scaled from 9.7 s
on two holdout graphs). Hidden-test still under 12 h.

## Kaggle test-time notebook

Private GPU kernel `mahdimzafari/biohub-learned-unet-gap-test-v1`.
Internet off. Attaches `mahdimzafari/biohub-learned-unet-v1`. Writes
`submission.csv`. **Not a competition submission** until Mahdi submits
from the Kaggle UI. This repo does not authorize `kaggle competitions
submit`.

| version | status | notes |
|---|---|---|
| 1 | **COMPLETE / inferred** | 330.0 s, peak 1184 MB. Public test four stems. 251660 nodes / 242412 edges / 13912 gap links / 19953 interpolated / 0 divisions. Worker `audit_ok`. Independent stream audit `audit_ok`. Weights `/kaggle/input/notebooks/mahdimzafari/biohub-learned-unet-v1/learned_unet_v1_last.pt`. |

`6bba_05db0fb1` counts (103432/100197) match the locked holdout gap
overlay (97062+6370 nodes / 89394+10803 edges). Frozen `6bba` was **not**
official-scored again. CSV stays under ignored `artifacts/kaggle/`. Do
**not** overwrite this slug. Do **not** overwrite the no-gap
`biohub-learned-unet-test-v1` v1.

Estimate vs measured: public 4 volumes estimated 388 s; measured 330.0 s.

## Next

If a public leaderboard number is wanted: `HUMAN_ACTION_REQUIRED` — Mahdi
must **Submit to Competition** from the Kaggle UI on this slug version 1.
Do **not** submit from the CLI. Do **not** retune. Do **not** score
`6bba` again. Prefer this gap slug over submitting the no-gap
`biohub-learned-unet-test-v1` v1.
