# learned_unet_retrain lanes A/B/C

Same experiment family as `learned_unet_retrain_v1` (draft PR #7).
Do **not** merge. Do **not** overwrite locked slugs. Frozen `6bba`
stayed closed. Official 44b6 scores below are **not** leaderboard
numbers.

## Hypothesis

After the authorized +3-epoch 44b6 continuation, official scores on
the two frozen-tuning sequences would tell whether the new weights
beat locked comparators. Lanes were hashed **before** any 44b6 score.
No sweep.

| lane | config | hash | comparator | official combined | decision |
|---|---|---|---|---|---|
| A | locked infer 0.5 / pool 5.0 / greedy 12 µm | `1db21f87d00ce083` | infer **0.655751** | **0.626370** | `reject_lane` |
| B | locked orgdet 0.99 / pool 3.0 / greedy 12 µm | `594f6fbc03c974cf` | orgdet **0.679578** | **0.642049** | `reject_lane` |
| C | orgdet geometry + softmax 0.5 + fixed ILP | `f924802e5f716a61` | lane B **0.642049** | **0.0** | `reject_lane` |

Winner: **none**. `may_open_6bba=false`. `opens_6bba=false`.

## Helpers (not submissions)

- Retrain weights: `mahdimzafari/biohub-learned-unet-retrain-v1` v1 COMPLETE / trained. Hash `92a419926fa7e40b`. 10461 s, 1817 MB. 17989 train candidate edges. reload 0.0.
- Lane A v2 COMPLETE / inferred. 146.3 s, 1118 MB. Graphs 69997/65214 and 35471/31289. Checkpoint hash `92a419926fa7e40b`.
- Lane B v2 COMPLETE / inferred. 150.5 s, 1181 MB. Graphs 63356/59236 and 30896/27391. Checkpoint hash `92a419926fa7e40b`.
- Lane C v1 errored with 0 files: `main()` compared lane hash to parent orgop `250c5bfacd4a8198`. Proven implementation defect. v2 COMPLETE / inferred after the guard used `f924802e5f716a61`. 157.1 s, 1230 MB. Softmax>0.5 candidates **0 / 1**. Local ILP: `ilp_skipped_no_candidates` and `tracksdata_ilp` (1 edge). Kaggle had no `tracksdata`.

## Official 44b6 (not leaderboard)

Scorer `075fc5f5`. Two frozen-tuning GEFFs only.

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes / edges |
|---|---|---|---|---|---|---|
| learned_unet_infer_v1 | **0.655751** | 0.69001 | 0.65575 | 0.97004 | 0/0/2 | 117692 / 106385 |
| learned_unet_orgdet_v1 | **0.679578** | — | 0.67958 | — | 0/0/2 | — |
| retrain A | **0.626370** | 0.64818 | 0.62637 | 0.98254 | 0/0/2 | 105468 / 96503 |
| retrain B | **0.642049** | 0.65493 | 0.64205 | 0.98254 | 0/0/2 | 94252 / 86627 |
| retrain C | **0.0** | 0.0 | 0.0 | 0.48571* | 0/0/2 | 94252 / 1 |

\*Scorer short-circuits empty-edge graphs. Per-sequence C: 63356/0 and 30896/1.

Retrain raised recall vs infer (0.983 vs 0.970) but lost official TPs/FPs
enough to drop adjusted edge Jaccard. Softmax>0.5 on 0.99 detections
stayed near-empty (same failure mode as rejected orgop, despite 17.9k
train-window candidates on GT pairs).

## Decision

`reject_learned_unet_retrain_lanes_v1`. Do **not** retune. Do **not**
open `6bba`. Champion remains `learned_unet_orgdet_holdout_v1`
**0.705226**. Public API `0.553` is association_v2 **56399003** only.
