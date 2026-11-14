# learned_unet_prune_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not** merge.
Do not overwrite `baseline_v1`, `association_v2`, `division_v2`,
`complementary_v1`, `linkable_v1`, `learned_unet_division_v1`, rejected
linker overlays, or locked learned train/infer/holdout/test/gap-test slugs.
Do **not** retune. Do **not** score frozen `6bba` again for this overlay.
This is **not** another 12 µm assignment.

## Hypothesis

Locked `learned_unet_gap_v1` already recovers FN edges. Leftover degree-0
detections after that overlay inflate `T_pred` without contributing edges
(the extra-`T_pred` failure mode of complementary/linkable). Dropping those
isolates on frozen-tuning `44b6` should raise adjusted edge Jaccard versus
locked gap **0.658452** without rewriting consecutive or gap links. Isolated
nodes can still match sparse GT, so official 44b6 score decides. Frozen
`6bba` is refused until that comparison keeps the candidate.

## Operating point (locked; one-shot holdout done)

Config: `configs/learned_unet_prune_v1.yaml`. Hash `115753ea16aaf021`.
Parent infer `d7578b5f64ddac08`. Parent gap `f4d888d190c45a4a`. Training
hash `4c38035d6657f2ad`. Locked holdout combined
`LEARNED_PRUNE_LOCKED_HOLDOUT` = **0.7013778258203534**.

- Source graphs: locked learned infer / holdout JSON only
- Locked gap overlay first (12 µm, `max_gap_frames=2`, interpolate)
- Then drop in-degree=0 and out-degree=0 nodes
- Existing dt=1 and gap edges kept
- Intensity unused
- Writes no `submission.csv`

Do **not** retune these gates. Do **not** score frozen `6bba` again.

## Cost estimate (before score)

- Storage: reuse infer/holdout-helper graphs + two `44b6` / two `6bba`
  GEFFs from `mahdimzafari/biohub-geff-export` v1. No volumes, no weights.
- CPU: gap overlay + isolate scan. Measured **6.85 s / 606 MB** (tune) and
  **9.83 s / 836 MB** (holdout).
- No GPU. No new kernel. Not a Kaggle run.

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4` only.
Graphs from private helper `mahdimzafari/biohub-learned-unet-infer-v1` v3.
GEFFs from private helper `mahdimzafari/biohub-geff-export` v1. **Not a
Kaggle run. Not a leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes |
|---|---|---|---|---|---|---|
| learned_unet_gap_v1 | **0.658452** | 0.70241 | 0.65845 | 0.97667 | 0/0/2 | 127571 |
| learned_unet_prune_v1 | **0.662081** | 0.70584 | 0.66208 | 0.97603 | 0/0/2 | 127146 |

Dropped 425 isolates (269 + 156). Extra edges versus infer still 16852
(same as gap). Per-sequence overlay:

| sequence | gap combined | prune combined | recall gap → prune | dropped |
|---|---|---|---|---|
| `44b6_12dfb391` | 0.745296 | 0.750528 | 0.9962 → 0.9949 | 269 |
| `44b6_267148e4` | 0.460171 | 0.460595 | 0.9571 → 0.9571 | 156 |

Combined wins. Edge TP on the first sequence rose 700→703 after isolates
stopped stealing GT matches. Official event scorer recovered **0 / 2**
annotated forks. Runtime 6.85 s, peak 606 MB.

Decision **`keep_candidate_learned_unet_prune_v1`**. See
`reports/learned_unet_prune_v1_tune.json`.

## Official frozen-holdout score (not leaderboard)

Local official scorer `075fc5f5` once on `6bba_05db0fb1` + `6bba_c328f2fd`.
Graphs from private helper `mahdimzafari/biohub-learned-unet-holdout-v1` v1
(97062/89394 and 52161/47595). GEFFs from `mahdimzafari/biohub-geff-export`
v1. **Not a Kaggle run. Not a leaderboard score.** Do **not** score again.

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes |
|---|---|---|---|---|---|---|
| learned_unet_gap_v1 | **0.701026** | 0.74335 | 0.70103 | 0.97588 | 0/0/5 | 159131 |
| learned_unet_prune_v1 | **0.701378** | 0.74335 | 0.70138 | 0.97490 | 0/0/5 | 158640 |

Dropped 491 isolates (204 + 287). Edge Jaccard unchanged (same 16804 extra
edges as gap). Per-sequence overlay:

| sequence | gap combined | prune combined | recall gap → prune | dropped |
|---|---|---|---|---|
| `6bba_05db0fb1` | 0.718133 | 0.718354 | 0.9870 → 0.9870 | 204 |
| `6bba_c328f2fd` | 0.661175 | 0.661834 | 0.9648 → 0.9628 | 287 |

Combined wins on both sequences. Edge TP/FP/FN match locked gap
(1051/210/132 and 429/98/71). Official event scorer recovered **0 / 5**
annotated forks. Runtime 9.83 s, peak 836 MB. Official deps:
`tracksdata==0.1.0rc10`, `polars==1.44.2`, `geff==1.3.1.1.3`.

Decision **`keep_candidate_learned_unet_prune_v1`**. Do **not** retune. Do
**not** score this overlay on `6bba` again. See
`reports/learned_unet_prune_v1_holdout.json`.

This is a local official score, not a Kaggle leaderboard number.

## Next

Package a **new** public-test slug `learned_unet_prune_test_v1` (locked
infer + locked gap overlay + drop degree-0). Do **not** overwrite
`mahdimzafari/biohub-learned-unet-gap-test-v1` v1. Until that kernel is
COMPLETE, gap-test v1 remains the packaged public-test worker. Do **not**
submit from the CLI.
