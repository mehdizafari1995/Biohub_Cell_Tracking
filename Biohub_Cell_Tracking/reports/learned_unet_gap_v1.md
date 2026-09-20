# learned_unet_gap_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not** merge.
Do not overwrite `baseline_v1`, `association_v2`, `division_v2`,
`complementary_v1`, `linkable_v1`, `learned_unet_division_v1`, or locked
learned train/infer/holdout/test slugs. Do **not** retune. Do **not** score
frozen `6bba` again for this overlay.

## Hypothesis

The first learned infer pass isolated detection: UNet local-max + locked
association_v2 12 µm 1-1 gate **without** gap-close (`max_gap_frames=0`).
Applying the locked association_v2 gap-close (12 µm, `max_gap_frames=2`,
`sqrt(dt)` scaled gate, interpolate) only to leftover unmatched nodes
should recover FN edges on frozen-tuning `44b6` without rewriting existing
dt=1 links or retuning detection. Learned graphs have no intensity, so the
association intensity term is unused.

## Operating point (locked; one-shot holdout done)

Config: `configs/learned_unet_gap_v1.yaml`. Hash `f4d888d190c45a4a`.
Parent infer hash `d7578b5f64ddac08`. Parent holdout hash `d374a21f03dddc04`.
Training hash `4c38035d6657f2ad`. Locked holdout combined
`LEARNED_GAP_LOCKED_HOLDOUT` = **0.7010255203349978**.

- Source graphs: locked learned infer / holdout JSON only
- Existing dt=1 edges kept
- Gap-close: dt=2 or 3, gate `12 * sqrt(dt)` µm, greedy 1-1
- Interpolated integer voxels on skipped frames
- Intensity unused
- Writes no `submission.csv`

Do **not** retune these gates. Do **not** score frozen `6bba` again.

## Cost estimate (before score)

- Storage: reuse holdout-helper graphs + two `6bba` GEFFs from
  `mahdimzafari/biohub-geff-export` v1. No volumes, no weights.
- CPU: re-link ~149k holdout nodes. Measured **9.7 s / 845 MB**.
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

Decision **`keep_candidate_learned_unet_gap_v1`**. See
`reports/learned_unet_gap_v1_tune.json`.

## Official frozen-holdout score (not leaderboard)

Local official scorer `075fc5f5` once on `6bba_05db0fb1` + `6bba_c328f2fd`.
Graphs from private helper `mahdimzafari/biohub-learned-unet-holdout-v1` v1
(97062/89394 and 52161/47595). GEFFs from `mahdimzafari/biohub-geff-export`
v1. **Not a Kaggle run. Not a leaderboard score.** Do **not** score again.

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes |
|---|---|---|---|---|---|---|
| learned_unet_holdout_v1 | **0.693605** | 0.72769 | 0.69360 | 0.95713 | 0/0/5 | 149223 |
| learned_unet_gap_v1 | **0.701026** | 0.74335 | 0.70103 | 0.97588 | 0/0/5 | 159131 |

Per-sequence overlay:

| sequence | no-gap combined | gap combined | recall no-gap → gap | extra nodes | extra edges |
|---|---|---|---|---|---|
| `6bba_05db0fb1` | 0.719310 | 0.718133 | 0.9788 → 0.9870 | 6370 | 10803 |
| `6bba_c328f2fd` | 0.633142 | 0.661175 | 0.9354 → 0.9648 | 3538 | 6001 |

Combined wins. The first sequence is mixed (edge TP 1030→1051, but extra
`T_pred` lowered adj-edge Jaccard). The second sequence improved (edge TP
397→429). Official event scorer recovered **0 / 5** annotated forks. Extra
edges 16804. Interpolated nodes 9908. Runtime 9.7 s, peak 845 MB. Official
deps: `tracksdata==0.1.0rc10`, `polars==1.44.2`, `geff==1.3.1.1.3`.

Decision **`keep_candidate_learned_unet_gap_v1`**. Do **not** retune. Do
**not** score this overlay on `6bba` again. See
`reports/learned_unet_gap_v1_holdout.json`.

This is a local official score, not a Kaggle leaderboard number.

## Next

Gap test-time slug `mahdimzafari/biohub-learned-unet-gap-test-v1` v1 is
**COMPLETE / inferred** (330.0 s, `audit_ok`, hash `85379c198ebc7241`).
Do **not** overwrite it or `mahdimzafari/biohub-learned-unet-test-v1` v1.
Do **not** retune. Do **not** score frozen `6bba` again. Do not submit
from the CLI.
