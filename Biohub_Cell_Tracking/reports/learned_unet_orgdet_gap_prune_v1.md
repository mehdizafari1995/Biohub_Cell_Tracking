# learned_unet_orgdet_gap_prune_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or orgdet/orgdet-gap slugs. Do **not** retune. One-shot holdout is done.
Do **not** score frozen `6bba` again for this overlay. This is **not**
another 12 µm assignment.

## Hypothesis

Locked `learned_unet_orgdet_gap_v1` already recovers leftover unmatched
orgdet nodes. Leftover degree-0 detections after that overlay inflate
`T_pred` without contributing edges (the extra-`T_pred` failure mode of
complementary/linkable, and the keep-mode of `learned_unet_prune_v1`).
Dropping those isolates on frozen-tuning `44b6` should raise adjusted
edge Jaccard versus locked orgdet-gap **0.687221** without rewriting
consecutive or gap links. Isolated nodes can still match sparse GT, so
official 44b6 score decides. Frozen `6bba` is refused until that
comparison keeps the candidate.

## Operating point (locked; one-shot holdout done)

Config: `configs/learned_unet_orgdet_gap_prune_v1.yaml`. Hash
`862e3e7f7ac0330e`. Parent orgdet `80b20589198148b7`. Parent orgdet-gap
`ee31f36151325f9a`. Parent gap gates `f4d888d190c45a4a`. Training hash
`4c38035d6657f2ad`. Locked holdout combined
`LEARNED_ORGDET_GAP_PRUNE_LOCKED_HOLDOUT` = **0.7230176371507055**.

- Source: locked orgdet (or empty-edge orgop) 44b6 / 6bba graphs
- Locked orgdet relink when the source graph has 0 edges
- Locked orgdet-gap overlay first (12 µm, `max_gap_frames=2`, interpolate)
- Then drop in-degree=0 and out-degree=0 nodes
- Existing dt=1 and gap edges kept
- Intensity unused. No transformer. No ILP.
- Writes no `submission.csv` from this overlay scorer

Do **not** retune these gates. Do **not** score frozen `6bba` again.

## Cost estimate (before score)

- Storage: reuse orgop / orgdet-holdout helper graphs + two `44b6` / two
  `6bba` GEFFs from `mahdimzafari/biohub-geff-export` v1. No volumes, no
  weights.
- CPU: orgdet relink + gap overlay + isolate scan. Measured **10.40 s /
  528 MB** (tune) and **7.64 s / 776 MB** (holdout).
- No GPU. No new kernel. Not a Kaggle run.

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4` only.
Orgop detections 62480/0 and 28672/0, then locked orgdet relink, then gap,
then isolate drop. GEFFs from `mahdimzafari/biohub-geff-export` v1.
**Not a Kaggle run. Not a leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes |
|---|---|---|---|---|---|---|
| learned_unet_orgdet_gap_v1 | **0.687221** | 0.70382 | 0.68722 | 0.97667 | 0/0/2 | 96977 |
| learned_unet_orgdet_gap_prune_v1 | **0.693437** | 0.70993 | 0.69344 | 0.97603 | 0/0/2 | 96750 |

Dropped 227 isolates (127 + 100). Extra edges versus orgdet still 9951
(same as orgdet-gap). Per-sequence overlay:

| sequence | gap combined | prune combined | recall gap → prune | dropped |
|---|---|---|---|---|
| `44b6_12dfb391` | 0.774579 | 0.783806 | 0.9962 → 0.9949 | 127 |
| `44b6_267148e4` | 0.487928 | 0.488201 | 0.9571 → 0.9571 | 100 |

Combined wins. Edge TP on the first sequence rose 699→704 after isolates
stopped stealing GT matches. Official event scorer recovered **0 / 2**
annotated forks. Runtime 10.40 s, peak 528 MB.

Decision **`keep_candidate_learned_unet_orgdet_gap_prune_v1`**. See
`reports/learned_unet_orgdet_gap_prune_v1_tune.json`.

## Official frozen-holdout score (not leaderboard)

Local official scorer `075fc5f5` once on `6bba_05db0fb1` + `6bba_c328f2fd`.
Graphs from private helper `mahdimzafari/biohub-learned-unet-orgdet-holdout-v1`
v1 (84328/78493 and 37482/35147). GEFFs from `mahdimzafari/biohub-geff-export`
v1. **Not a Kaggle run. Not a leaderboard score.** Do **not** score again.

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes |
|---|---|---|---|---|---|---|
| learned_unet_orgdet_gap_v1 | **0.722805** | 0.74265 | 0.72281 | 0.95762 | 0/0/5 | 127966 |
| learned_unet_orgdet_gap_prune_v1 | **0.723018** | 0.74265 | 0.72302 | 0.95762 | 0/0/5 | 127670 |

Dropped 296 isolates (148 + 148). Edge Jaccard unchanged (same 10427 extra
edges as orgdet-gap). Per-sequence overlay:

| sequence | gap combined | prune combined | recall gap → prune | dropped |
|---|---|---|---|---|
| `6bba_05db0fb1` | 0.739289 | 0.739450 | 0.9837 → 0.9837 | 148 |
| `6bba_c328f2fd` | 0.684139 | 0.684472 | 0.9315 → 0.9315 | 148 |

Combined wins on both sequences. Edge TP/FP/FN match locked orgdet-gap
(1052/201/131 and 414/90/86). Official event scorer recovered **0 / 5**
annotated forks. Runtime 7.64 s, peak 776 MB. Official deps:
`tracksdata==0.1.0rc10`, `polars==1.44.2`, `geff==1.3.1.1.3`.

Decision **`keep_candidate_learned_unet_orgdet_gap_prune_v1`**. This is
the new holdout champion. Do **not** retune. Do **not** score this
overlay on `6bba` again. See
`reports/learned_unet_orgdet_gap_prune_v1_holdout.json`.

This is a local official score, not a Kaggle leaderboard number.

## Next

`learned_unet_orgdet_gap_prune_test_v1` v1 is **COMPLETE / inferred /
`audit_ok`** (hash `52a30a5843e0a880`). Independent audit `audit_ok`.
`READY_FOR_MANUAL_SUBMISSION_REVIEW`. Do **not** overwrite v1. Do
**not** submit from the CLI. Additional UI submit is **not**
authorized unless a later message separately authorizes that extra
click. Public **0.602** is orgdet-gap-test **56583783** only.
