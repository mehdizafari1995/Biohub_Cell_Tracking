# learned_unet_test_v1

Same experiment family as `learned_unet_v1` / infer / holdout (draft PR #7).
Do **not** merge. Do not overwrite `baseline_v1`, `association_v2`,
`division_v2`, `complementary_v1`, `linkable_v1`,
`mahdimzafari/biohub-learned-unet-v1`,
`mahdimzafari/biohub-learned-unet-infer-v1`, or
`mahdimzafari/biohub-learned-unet-holdout-v1`.

## Hypothesis

Locked `TemporalUNet3D` detections (44b6-only training, locked infer
operating point `d7578b5f64ddac08`, holdout lock `d374a21f03dddc04`) can
be applied to every public or hidden `test/*.zarr` at runtime and write a
schema-valid `submission.csv`. This is a generalization / runtime
hypothesis, not a retune. Public-test stems are not used to change
thresholds. Frozen `6bba` is not official-scored again.

## Operating point (locked; do not retune)

Config: `configs/learned_unet_test_v1.yaml`. Hash `7a8725b818fc609b`.
Parent infer hash `d7578b5f64ddac08`. Parent holdout hash `d374a21f03dddc04`.
Training hash `4c38035d6657f2ad`.

- Discovers every `test/*.zarr` at runtime. No hard-coded public stems
- Never opens `train/` volumes
- `det_threshold` 0.5, TTA flip-x/y/xy, `pool_kernel_um` 5.0
- Normalisation matches training: per-window min-max
- Linker: greedy 12 µm, max_children=1, max_parents=1, no gap-close
- Writes `submission.csv` + `run_manifest.json`
- Weights from `/kaggle/input/notebooks/mahdimzafari/biohub-learned-unet-v1/`
  (`learned_unet_v1_best.pt`, fallback `learned_unet_v1_last.pt`)

## Cost estimate (before push)

Holdout v1 measured **184.4 s / 2 volumes** (92.2 s/volume), peak **1177 MB**
on T4 x2.

| set | volumes | estimate | under 12 h GPU limit |
|---|---|---|---|
| public test | 4 | ~6.1 min | yes |
| hidden test (train-sized) | ~199 | **~5.1 h** | yes |

Peak RAM ~1.2 GB. Submission CSV for a train-sized hidden test is expected
to be hundreds of MB and stays out of git.

## Kaggle test-time notebook

Private GPU kernel `mahdimzafari/biohub-learned-unet-test-v1`. Internet off.
Attaches `mahdimzafari/biohub-learned-unet-v1`. Writes `submission.csv`.
**Not a competition submission** until Mahdi submits from the Kaggle UI.
This repo does not authorize `kaggle competitions submit`.

| version | status | notes |
|---|---|---|
| 1 | **COMPLETE / inferred** | 306.5 s, peak 1184 MB. Public test four stems. 231707 nodes / 208547 edges / 0 divisions. Worker `audit_ok`. Independent stream audit `audit_ok`. Weights `/kaggle/input/notebooks/mahdimzafari/biohub-learned-unet-v1/learned_unet_v1_last.pt`. |

`6bba_05db0fb1` counts (97062/89394) match the locked holdout helper graphs.
Frozen `6bba` was **not** official-scored again. CSV stays under ignored
`artifacts/kaggle/`. Do **not** overwrite this slug.

## Next

If a public leaderboard number is wanted: Mahdi must **Submit to Competition**
from the Kaggle UI. Do **not** submit from the CLI. Do **not** retune. Do
**not** score `6bba` again.
