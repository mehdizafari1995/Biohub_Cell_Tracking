# learned_unet_link_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite `baseline_v1`, `association_v2`, `division_v2`,
`complementary_v1`, `linkable_v1`, `learned_unet_division_v1`, or locked
learned train/infer/holdout/test/gap-test slugs. Do **not** retune locked
gates. Do **not** score frozen `6bba` until this 44b6 lock is documented.

## Hypothesis

Locked `TemporalUNet3D` detections plus the trained `SimpleNodeTransformer`
pairwise scores (softmax over sources, threshold 0.5, greedy 1-1) can raise
the official frozen-tuning `44b6` score versus locked
`learned_unet_infer_v1` **0.655751**. The first infer pass isolated
detection by using a greedy 12 µm gate and explicitly left transformer
edge scores unused. This pass isolates the learned linker. Frozen `6bba`
is refused.

## Operating point (44b6 only; holdout not run)

Config: `configs/learned_unet_link_v1.yaml`. Hash `3a7423c6858ace41`.
Training hash `4c38035d6657f2ad`. Parent infer `d7578b5f64ddac08`.

- Infer stems only: `44b6_12dfb391`, `44b6_267148e4`
- Holdout `6bba` is refused
- Detection unchanged: `det_threshold` 0.5, TTA flip-x/y/xy, `pool_kernel_um` 5.0
- Normalisation matches training: per-window min-max
- Linker: transformer softmax, threshold 0.5, max_children=1, max_parents=1
- Pair MLP is row-chunked (score-preserving) to bound GPU memory
- Writes JSON graphs + `infer_manifest.json`. Does **not** write `submission.csv`
- New slug `mahdimzafari/biohub-learned-unet-link-v1`

Do **not** retune after the first frozen-validation official score.

## Cost estimate (before push)

Locked infer helper **187.0 s / 2 volumes** (93.5 s/volume). Transformer
pair scoring is estimated at **1.6×** that encode time because each
window already runs `encode` and then scores one node pair.

| set | volumes | estimate | under 12 h GPU limit |
|---|---|---|---|
| frozen tuning | 2 | **~5.0 min** | yes |
| hidden test if later promoted | ~199 | **~8.3 h** | yes |

Peak RAM budget **2500 MB** (infer 1136 MB plus chunked pair tensors).
T4 16 GB is enough. This helper does not write `submission.csv`.

## Kaggle helper

Private GPU kernel `mahdimzafari/biohub-learned-unet-link-v1`. Internet
off. Attaches `mahdimzafari/biohub-learned-unet-v1`. **Not a submission.**

| version | status | notes |
|---|---|---|
| 1 | **running** | Pushed this run. 44b6 transformer linker. No `submission.csv`. |

Official 44b6 tuning score and one-shot `6bba` holdout are **not** run
until this helper finishes and graphs are downloaded into ignored
`artifacts/kaggle/`. `learned_unet_gap_v1` remains the holdout champion.
`learned_unet_gap_test_v1` v1 remains the preferred public-test worker.

## Next

Poll kernel v1. If COMPLETE, download graphs (do not commit them),
official-score frozen tuning vs `learned_unet_infer_v1` 0.655751, then
one-shot `6bba` only after a documented 44b6 keep. Reject on combined
score regression. Do not submit from the CLI.
