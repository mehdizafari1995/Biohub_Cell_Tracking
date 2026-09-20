# learned_unet_link_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite `baseline_v1`, `association_v2`, `division_v2`,
`complementary_v1`, `linkable_v1`, `learned_unet_division_v1`, or locked
learned train/infer/holdout/test/gap-test slugs. Do **not** retune locked
gates. Frozen `6bba` was **not** scored.

## Hypothesis

Locked `TemporalUNet3D` detections plus the trained `SimpleNodeTransformer`
pairwise scores (softmax over sources, threshold 0.5, greedy 1-1) can raise
the official frozen-tuning `44b6` score versus locked
`learned_unet_infer_v1` **0.655751**. The first infer pass isolated
detection by using a greedy 12 µm gate and explicitly left transformer
edge scores unused. This pass isolates the learned linker. Frozen `6bba`
is refused.

## Operating point (44b6 only; rejected on tuning)

Config: `configs/learned_unet_link_v1.yaml`. Hash `3a7423c6858ace41`.
Training hash `4c38035d6657f2ad`. Parent infer `d7578b5f64ddac08`.

- Infer stems only: `44b6_12dfb391`, `44b6_267148e4`
- Holdout `6bba` is refused
- Detection unchanged: `det_threshold` 0.5, TTA flip-x/y/xy, `pool_kernel_um` 5.0
- Normalisation matches training: per-window min-max
- Linker: transformer softmax, threshold 0.5, max_children=1, max_parents=1
- Pair MLP is row-chunked (score-preserving) to bound GPU memory
- Writes JSON graphs + `infer_manifest.json`. Does **not** write `submission.csv`
- Slug `mahdimzafari/biohub-learned-unet-link-v1` v1 **COMPLETE / inferred**

Do **not** retune this hash. Do **not** overwrite the completed slug.

## Kaggle helper

Private GPU kernel `mahdimzafari/biohub-learned-unet-link-v1`. Internet
off. Attaches `mahdimzafari/biohub-learned-unet-v1`. **Not a submission.**

| version | status | notes |
|---|---|---|
| 1 | **COMPLETE / inferred** | 162.1 s, peak 1210 MB, T4 x2. Graphs: `44b6_12dfb391` 80741/0, `44b6_267148e4` 36951/0. No `submission.csv`. |

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4` only.
Graphs from the COMPLETE helper. GEFFs from `mahdimzafari/biohub-geff-export`
v1. **Not a Kaggle run. Not a leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes / edges |
|---|---|---|---|---|---|---|
| learned_unet_infer_v1 | **0.655751** | 0.69001 | 0.65575 | 0.97004 | 0/0/2 | 117692 / 106385 |
| learned_unet_link_v1 | **0.000000** | 0.0 | 0.0 | 0.0 | 0/0/2 | 117692 / **0** |

Per-sequence linker: both volumes official **0.0**. Detection counts match
locked infer exactly (80741 / 36951). The official scorer warns and returns
0.0 when the predicted graph has no edges, so the reported node recall is
also 0.0 by that short-circuit. Runtime 1.47 s, peak 415 MB.

Decision **`reject_learned_unet_link_v1`**. Do **not** retune. Frozen `6bba`
was refused because tuning did not keep the candidate.

## Error analysis

Softmax is taken over sources for each target. With hundreds of detections
per frame, even a strongly peaked source stays below the locked 0.5
probability cutoff, so `assign_transformer_links` writes no edges. This is
an assignment-gate failure, not a detection regression. See
`reports/learned_unet_link_v1_error.json`.

## Next

Keep `learned_unet_gap_v1` / `learned_unet_gap_test_v1` as the candidate.
A later run may package a **new** slug (`learned_unet_link_v2`) that keeps
these locked detections and replaces `prob > 0.5` with a documented
argmax / no-cutoff 1-1 assignment, optionally plus the locked 12 µm spatial
gate. Do **not** overwrite this rejected v1 slug. Do **not** score `6bba`
for this family until a 44b6 keep. If a public leaderboard number is
wanted, Mahdi submits `biohub-learned-unet-gap-test-v1` v1 from the Kaggle
UI. Do **not** submit from the CLI.
