# learned_unet_orgdet_fnn_acf_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or orgdet/elf/fnn slugs. Do **not** retune. This is **not** another 12 µm
assignment, **not** a rank/div/reas retune, and **not** another 8.0 µm
dt=1 length retune.

## Hypothesis

Official 44b6 FNN residue was still association-dominated (174 conflict
FPs / 144 FNs; 0 predicted divisions). Rank, unused-child division, and
claimed-child reassignment stay closed.

On both frozen-tuning FNN graphs, every official-matched TP dt=1 edge
had 3×3×3-mean relative appearance change ≤ 0.3756, while 13 conflict
FPs and 1644 unmatched leftover dt=1 edges exceeded 0.38. Appearance is
sampled from the two exported tuning Zarrs (`44b6_12dfb391`,
`44b6_267148e4`). Dropping those already-chosen dt=1 edges, then new
degree-0 leftovers, should raise adjusted edge Jaccard versus FNN
**0.724653** without rewriting the 12 µm linker, gap gates, ELF, or
FNN keep rule. Official 44b6 had to exceed FNN by **≥ 0.002000**
(threshold **0.726653**). Frozen `6bba` stayed closed unless that gate
kept. Hash `a66075085ac64400` was locked before the 44b6 score.

## Operating point (kept)

Config: `configs/learned_unet_orgdet_fnn_acf_v1.yaml`. Hash `a66075085ac64400`.
Parent FNN `95e9300eb84ba71c`. Parent ELF `292c4f1f968f948d`.
Parent prune `862e3e7f7ac0330e`. Parent orgdet-gap `ee31f36151325f9a`.
Parent orgdet `80b20589198148b7`. Train hash `4c38035d6657f2ad`.

- Source: locked orgop / orgdet graphs
- Locked orgdet+gap+prune+ELF+FNN first
- Then drop a dt=1 edge when
  `|I_dst − I_src| / max(|I_src|, |I_dst|, 1e-6) > 0.38`
  where `I` is the 3×3×3-mean sampled from the sequence Zarr
- Then drop newly created degree-0 leftovers
- Non-dt=1 edges stay. No transformer. No ILP. No 12 µm retune
- Writes no `submission.csv`

Do **not** retune `max_rel_appearance`. Do **not** retune `nn_tie_eps`.
Do **not** retune `max_dt1_edge_um`. Do **not** score 6bba again.

## Cost estimate (before score)

- Storage: reuse orgop / orgdet-holdout helper graphs + GEFFs from
  `mahdimzafari/biohub-geff-export` v1 + the two exported tuning Zarrs
  from `mahdimzafari/biohub-tuning-zarr-export` v1. Holdout Zarrs from
  `mahdimzafari/biohub-holdout-zarr-export` v1 only after the 44b6 keep.
  No weights committed.
- CPU: locked FNN overlay + per-timepoint 3×3×3 sample.
  Measured **15.11 s / 562 MB** on 44b6; **13.63 s / 767 MB** on 6bba.
- No new GPU training. Not a Kaggle run for the overlay scores.

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4`
only. GEFFs from `mahdimzafari/biohub-geff-export` v1. Zarrs from
`mahdimzafari/biohub-tuning-zarr-export` v1. **Not a Kaggle run. Not a
leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes |
|---|---|---|---|---|---|---|
| learned_unet_orgdet_fnn_v1 | **0.724653** | 0.74020 | 0.72465 | 0.97425 | 0/0/2 | 95019 |
| learned_unet_orgdet_fnn_acf_v1 | **0.734620** | 0.74959 | 0.73462 | 0.97425 | 0/0/2 | 94142 |

Dropped 1657 inconsistent dt=1 edges (648 + 1009) and 877 new isolates.
Official TP/FP/FN **907 / 160 / 143** vs FNN **906 / 174 / 144**.
Per-sequence:

| sequence | fnn combined | acf combined | dropped edges |
|---|---|---|---|
| `44b6_12dfb391` | 0.811357 | 0.814642 | 648 |
| `44b6_267148e4` | 0.520601 | 0.541120 | 1009 |

Delta versus FNN **+0.009967**, above the preregistered **+0.002000**
gate (**0.726653**). Official event scorer recovered **0 / 2** annotated
forks. Runtime 15.11 s, peak 562 MB.

Decision **`keep_candidate_learned_unet_orgdet_fnn_acf_v1`**. Hash was
locked before the 6bba look. See
`reports/learned_unet_orgdet_fnn_acf_v1_tune.json`.

## Official frozen-holdout score (not leaderboard)

One-shot local official scorer on `6bba_05db0fb1` + `6bba_c328f2fd`.
Source graphs from `mahdimzafari/biohub-learned-unet-orgdet-holdout-v1`.
Zarrs from `mahdimzafari/biohub-holdout-zarr-export` v1. **Not a Kaggle
run. Not a leaderboard score.** Do **not** score again.

| candidate | combined | edge J | adj edge J | recall | divisions |
|---|---|---|---|---|---|
| learned_unet_orgdet_fnn_v1 | **0.752091** | 0.77110 | 0.75209 | 0.95501 | 0/0/5 |
| learned_unet_orgdet_fnn_acf_v1 | **0.755503** | 0.77436 | 0.75550 | 0.95501 | 0/0/5 |

Dropped 879 inconsistent dt=1 edges (626 + 253). Official TP/FP/FN
**1462 / 205 / 221** vs FNN **1462 / 213 / 221**. Per-sequence:
`6bba_05db0fb1` 0.776181; `6bba_c328f2fd` 0.707083. Runtime 13.63 s,
peak 767 MB.

Decision **`keep_candidate_learned_unet_orgdet_fnn_acf_v1`**. New
holdout champion. Public **0.602 / 0.612 / 0.553** were not used.

## Next

Package `learned_unet_orgdet_fnn_acf_test_v1` only. Do **not** overwrite
fnn-fast-test v1, fnn-test v1, or elf-test v1. Do **not** submit from
the CLI. Do **not** score 6bba again. Do **not** retune
`max_rel_appearance`.
