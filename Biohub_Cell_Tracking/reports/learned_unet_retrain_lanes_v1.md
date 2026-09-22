# learned_unet_retrain lanes A/B/C

Same experiment family as `learned_unet_retrain_v1` (draft PR #7).
Do **not** merge. Do **not** overwrite locked train/infer/holdout/test/
gap-test/prune-test/orgop/orgdet slugs. Do **not** push a second
retrain version. Frozen `6bba` stays closed until a lane strictly wins.

## Hypothesis

After the authorized +3-epoch 44b6 continuation completes, official
scores on the two frozen-tuning sequences can tell whether the new
weights beat locked comparators. Lanes were hashed **before** any 44b6
score. No sweep.

| lane | config | hash | comparator |
|---|---|---|---|
| A | locked infer 0.5 / pool 5.0 / greedy 12 µm | `1db21f87d00ce083` | infer **0.655751** |
| B | locked orgdet 0.99 / pool 3.0 / greedy 12 µm | `594f6fbc03c974cf` | orgdet **0.679578** |
| C | orgdet geometry + softmax 0.5 + fixed ILP | `f924802e5f716a61` | lane B official combined |

Reserved slugs (not launched this cycle):
`mahdimzafari/biohub-learned-unet-retrain-infer-v1`,
`mahdimzafari/biohub-learned-unet-retrain-orgdet-v1`,
`mahdimzafari/biohub-learned-unet-retrain-orgop-v1`.

## Fail-closed scorer (packaged, not run)

`scripts/run_learned_unet_retrain_lanes.py` official-scores existing
44b6 graphs only. It refuses:

- role != `tuning`
- incomplete retrain helper (`running` / 0 outputs)
- `submission.csv` on the helper
- missing `44b6_12dfb391.json` / `44b6_267148e4.json`
- any `6bba` stem

A local official score is not a Kaggle run and not a leaderboard
number. The script never opens `6bba` even if `may_open_6bba` is
true; a later locked cycle must do that once.

## This cycle

Retrain helper v1 is still **RUNNING** (poll **2026-09-22T05:53:06Z**,
0 outputs). The scorer therefore cannot run. Did **not** invent a
score. Did **not** launch lane infer kernels. Did **not** push a
second retrain version. Champion remains
`learned_unet_orgdet_holdout_v1` **0.705226**.
