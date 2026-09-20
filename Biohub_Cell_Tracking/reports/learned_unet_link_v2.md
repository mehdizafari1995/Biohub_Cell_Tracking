# learned_unet_link_v2

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite `baseline_v1`, `association_v2`, `division_v2`,
`complementary_v1`, `linkable_v1`, `learned_unet_division_v1`, rejected
`learned_unet_link_v1`, or locked learned train/infer/holdout/test/gap-test
slugs. Do **not** retune locked gates. Frozen `6bba` is **not** opened
until a documented 44b6 keep.

## Hypothesis

Rejected `learned_unet_link_v1` wrote **0 edges** because softmax over
hundreds of sources stays below the locked 0.5 probability cutoff.
Keeping the same locked detections and transformer scores, greedy 1-1
assignment on **descending scores with no 0.5 cutoff**, plus the locked
association_v2 **12 µm** spatial gate, can write plausible edges and
raise the official frozen-tuning `44b6` score versus locked
`learned_unet_infer_v1` **0.655751**. Frozen `6bba` is refused until
that 44b6 comparison keeps the candidate.

## Operating point (44b6 only; packaged, not yet scored)

Config: `configs/learned_unet_link_v2.yaml`. Hash `1d4015dc499bd604`.
Training hash `4c38035d6657f2ad`. Parent infer `d7578b5f64ddac08`.
Rejected parent `learned_unet_link_v1` hash `3a7423c6858ace41` stays
locked.

- Infer stems only: `44b6_12dfb391`, `44b6_267148e4`
- Holdout `6bba` is refused
- Detection unchanged: `det_threshold` 0.5, TTA flip-x/y/xy, `pool_kernel_um` 5.0
- Normalisation matches training: per-window min-max
- Linker: transformer softmax, **no probability cutoff**, greedy 1-1,
  `max_link_um=12.0`
- Pair MLP is row-chunked (score-preserving) to bound GPU memory
- Writes JSON graphs + `infer_manifest.json`. Does **not** write `submission.csv`
- New slug `mahdimzafari/biohub-learned-unet-link-v2` version **1 running**

Do **not** overwrite rejected v1. Do **not** score `6bba` until a 44b6 keep.

## Kaggle helper

Private GPU kernel `mahdimzafari/biohub-learned-unet-link-v2`. Internet
off. Attaches `mahdimzafari/biohub-learned-unet-v1`. **Not a submission.**

| version | status | notes |
|---|---|---|
| 1 | **running** | Pushed this run. Hash `1d4015dc499bd604`. No outputs yet. |

Estimate from locked infer_v1 (187.0 s / 2 volumes) × 1.65 transformer
+ spatial-gate overhead: ~5.1 min for the two 44b6 volumes; ~8.5 h if
later promoted to ~199 hidden volumes. Peak RAM budget 2.5 GB. Under
the 12 h GPU limit.

## Official frozen-tuning score (not leaderboard)

Not run this packaging step. Local official scorer `075fc5f5` on
`44b6_12dfb391` + `44b6_267148e4` only after the helper is COMPLETE.
Compare to locked `learned_unet_infer_v1` **0.655751**. **Not a Kaggle
run. Not a leaderboard score.**

## Next

Poll helper v1. If COMPLETE, download graphs into ignored
`artifacts/kaggle/` and official-score 44b6. Keep `learned_unet_gap_v1`
/ `learned_unet_gap_test_v1` as the current candidate until that score
is recorded. If a public leaderboard number is wanted, Mahdi submits
`biohub-learned-unet-gap-test-v1` v1 from the Kaggle UI. Do **not**
submit from the CLI.
