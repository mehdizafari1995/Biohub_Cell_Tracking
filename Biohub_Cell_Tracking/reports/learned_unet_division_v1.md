# learned_unet_division_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not** merge.
Do not overwrite `baseline_v1`, `association_v2`, `division_v2`,
`complementary_v1`, `linkable_v1`, or locked learned train/infer/holdout/test
slugs. Do **not** retune. Frozen `6bba` was **not** scored.

## Hypothesis

Locked `division_v2` gates (44b6 GEFF geometry, hash `0fa756963e5d42d1`)
applied to locked `learned_unet_infer_v1` graphs may recover annotated
divisions that the 1-1 learned linker never proposes, without lowering the
official 44b6 combined score. Learned graphs have no intensity, so the
intensity-sum gate is unused. This is not a retune of detection or gates.

## Operating point (locked; rejected on 44b6)

Config: `configs/learned_unet_division_v1.yaml`. Hash `8761fb66cc2254f3`.
Parent infer hash `d7578b5f64ddac08`. Parent holdout hash `d374a21f03dddc04`.
Training hash `4c38035d6657f2ad`. Division gates hash `0fa756963e5d42d1`.

- Source graphs: locked learned infer JSON only
- Gates: max parent–child 10.0 µm, min sibling 8.3 µm, alignment cosine ≤ 0.05,
  predecessor required, no alternative 1-1 parent inside 12 µm
- Intensity unused
- Writes no `submission.csv`

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4` only.
Graphs from private helper `mahdimzafari/biohub-learned-unet-infer-v1` v3.
GEFFs from private helper `mahdimzafari/biohub-geff-export` v1. **Not a
Kaggle run. Not a leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | extra edges |
|---|---|---|---|---|---|---|
| learned_unet_infer_v1 | **0.655751** | 0.69001 | 0.65575 | 0.97004 | 0/0/2 | 0 |
| learned_unet_division_v1 | **0.655235** | 0.68947 | 0.65524 | 0.97004 | 0/1/2 | 282 |

Per-sequence overlay: `44b6_12dfb391` 117 forks, combined unchanged at
0.747072 (0/0/1). `44b6_267148e4` 165 forks, combined 0.445220 vs 0.446373
(+1 edge FP, +1 division FP, 0/1/1). Official event scorer recovered **0 / 2**
annotated forks. Runtime 8.6 s, peak 594 MB.

Decision **`reject_learned_unet_division_v1`**. Do **not** retune. Frozen
`6bba` holdout was refused because tuning did not keep the candidate.

## Next

Keep `learned_unet_holdout_v1` / `learned_unet_test_v1` as the candidate.
If a public leaderboard number is wanted, Mahdi submits
`biohub-learned-unet-test-v1` v1 from the Kaggle UI. Do **not** submit from
the CLI.
