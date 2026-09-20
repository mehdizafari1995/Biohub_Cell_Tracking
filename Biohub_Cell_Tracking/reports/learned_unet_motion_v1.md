# learned_unet_motion_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite `baseline_v1`, `association_v2`, `division_v2`,
`complementary_v1`, `linkable_v1`, `learned_unet_division_v1`, rejected
`learned_unet_link_v1` / `learned_unet_link_v2`, or locked learned
train/infer/holdout/test/gap-test slugs. Do **not** retune locked gates.
Frozen `6bba` was **not** scored.

## Hypothesis

Rejected transformer linkers ranked worse than last-position greedy on
the same locked detections. Constant-velocity / motion-aware 1-1
assignment, with the locked association_v2 **12 µm last-position**
spatial limit, should reduce identity switches versus last-position
greedy (`learned_unet_infer_v1` **0.655751**) on frozen-tuning `44b6`.
Frozen `6bba` is refused until that 44b6 comparison keeps the candidate.

## Operating point (44b6 only; rejected on tuning)

Config: `configs/learned_unet_motion_v1.yaml`. Hash `66853b6f1445c18b`.
Training hash `4c38035d6657f2ad`. Parent infer `d7578b5f64ddac08`.
Rejected transformer hashes `3a7423c6858ace41` and `1d4015dc499bd604`
stay locked.

- Infer stems only: `44b6_12dfb391`, `44b6_267148e4`
- Holdout `6bba` is refused
- Detection unchanged (locked infer nodes)
- Linker: constant-velocity ranking, greedy 1-1, hard last-position
  `max_link_um=12.0`, `max_gap_frames=0`, no interpolation, no intensity,
  no transformer
- Writes no `submission.csv`
- Helper slug `mahdimzafari/biohub-learned-unet-motion-v1` packaged, not
  pushed (local overlay is sufficient)

Do **not** retune this hash. Do **not** score frozen `6bba` for this overlay.

## Cost estimate (before score)

- Storage: reuse infer-helper graphs + two `44b6` GEFFs from
  `mahdimzafari/biohub-geff-export` v1. No volumes, no weights.
- CPU: re-link 117692 locked detections. Measured **9.8 s / 579 MB**.
- No GPU. Not a Kaggle run.

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4` only.
Graphs from private helper `mahdimzafari/biohub-learned-unet-infer-v1` v3.
GEFFs from private helper `mahdimzafari/biohub-geff-export` v1. **Not a
Kaggle run. Not a leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes / edges |
|---|---|---|---|---|---|---|
| learned_unet_infer_v1 | **0.655751** | 0.69001 | 0.65575 | 0.97004 | 0/0/2 | 117692 / 106385 |
| learned_unet_motion_v1 | **0.583722** | 0.61333 | 0.58372 | 0.97004 | 0/0/2 | 117692 / 106148 |

Per-sequence motion: `44b6_12dfb391` 0.701347 (infer 0.747072),
`44b6_267148e4` 0.333768 (infer 0.446373). Detection counts match
locked infer exactly (80741 / 36951). Official matched-edge
TP/FP/FN = 828/300/222 vs infer 877/221/173. Runtime 9.82 s, peak 579 MB.

Decision **`reject_learned_unet_motion_v1`**. Do **not** retune. Frozen
`6bba` was refused because tuning did not keep the candidate.

## Error analysis

The motion linker wrote 1-1 consecutive edges inside the locked 12 µm
last-position gate (max 11.94 µm; 0 edges over the gate; 0 predicted
divisions). Constant-velocity ranking among those gated candidates is
worse than last-position greedy: fewer official TPs and more conflicting
FPs on the same detections. See `reports/learned_unet_motion_v1_error.json`.

## Next

Keep `learned_unet_gap_v1` / `learned_unet_gap_test_v1` as the candidate.
Do **not** retune this motion hash. Do **not** package another
velocity-threshold variant without a new documented hypothesis. A later
run may start a **new** slug that keeps locked learned detections and
tests Hungarian 1-1 inside the same 12 µm last-position gate. Do **not**
score `6bba` for this family. If a public leaderboard number is wanted,
Mahdi submits `biohub-learned-unet-gap-test-v1` v1 from the Kaggle UI.
Do **not** submit from the CLI.
