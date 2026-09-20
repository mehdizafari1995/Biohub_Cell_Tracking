# learned_unet_gap_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not** merge.
Do not overwrite `baseline_v1`, `association_v2`, `division_v2`,
`complementary_v1`, `linkable_v1`, `learned_unet_division_v1`, or locked
learned train/infer/holdout/test slugs. Do **not** retune. Frozen `6bba`
has **not** been scored for this overlay.

## Hypothesis

The first learned infer pass isolated detection: UNet local-max + locked
association_v2 12 µm 1-1 gate **without** gap-close (`max_gap_frames=0`).
Applying the locked association_v2 gap-close (12 µm, `max_gap_frames=2`,
`sqrt(dt)` scaled gate, interpolate) only to leftover unmatched nodes
should recover FN edges on frozen-tuning `44b6` without rewriting existing
dt=1 links or retuning detection. Learned graphs have no intensity, so the
association intensity term is unused.

## Operating point (locked from 44b6; holdout not run)

Config: `configs/learned_unet_gap_v1.yaml`. Hash `f4d888d190c45a4a`.
Parent infer hash `d7578b5f64ddac08`. Parent holdout hash `d374a21f03dddc04`.
Training hash `4c38035d6657f2ad`.

- Source graphs: locked learned infer JSON only
- Existing dt=1 edges kept
- Gap-close: dt=2 or 3, gate `12 * sqrt(dt)` µm, greedy 1-1
- Interpolated integer voxels on skipped frames
- Intensity unused
- Writes no `submission.csv`

Do **not** retune these gates after the first frozen-validation official score.

## Cost estimate (before score)

- Storage: reuse infer-helper graphs + two 44b6 GEFFs from
  `mahdimzafari/biohub-geff-export` v1. No volumes, no weights.
- CPU: re-link ~117k nodes. Measured **7.2 s / 607 MB**.
- No GPU. No new kernel. Not a Kaggle run.

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4` only.
Graphs from private helper `mahdimzafari/biohub-learned-unet-infer-v1` v3.
GEFFs from private helper `mahdimzafari/biohub-geff-export` v1. **Not a
Kaggle run. Not a leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | extra edges | extra nodes |
|---|---|---|---|---|---|---|---|
| learned_unet_infer_v1 | **0.655751** | 0.69001 | 0.65575 | 0.97004 | 0/0/2 | 0 | 0 |
| learned_unet_gap_v1 | **0.658452** | 0.70241 | 0.65845 | 0.97667 | 0/0/2 | 16852 | 9879 |

Per-sequence overlay:

| sequence | infer combined | gap combined | recall infer → gap | extra nodes |
|---|---|---|---|---|
| `44b6_12dfb391` | 0.747072 | 0.745296 | 0.9937 → 0.9962 | 5557 |
| `44b6_267148e4` | 0.446373 | 0.460171 | 0.9464 → 0.9571 | 4322 |

Combined wins. The first sequence is mixed (extra `T_pred` lowered adj-edge
Jaccard). Edge TP rose on both sequences (687→700 and 190→204). Official
event scorer recovered **0 / 2** annotated forks. Runtime 7.2 s, peak 607 MB.

Decision **`keep_candidate_learned_unet_gap_v1`**. Do **not** look at `6bba`
yet. Do **not** retune. See `reports/learned_unet_gap_v1_tune.json`.

## Next

One-shot official holdout on locked `learned_unet_holdout_v1` graphs
(`6bba_05db0fb1` + `6bba_c328f2fd`) with the same overlay. Reject on
combined regression versus locked holdout **0.693605**. Do not overwrite
locked slugs. Do not submit from the CLI.
