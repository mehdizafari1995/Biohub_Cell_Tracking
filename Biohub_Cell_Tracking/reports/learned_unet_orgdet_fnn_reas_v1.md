# learned_unet_orgdet_fnn_reas_v1 (rejected on holdout)

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or orgdet/elf/fnn slugs. Do **not** retune locked FNN or `division_v2`.
This is **not** another 12 µm assignment, **not** a rank_v1 retune, and
**not** a relaunch of rejected `learned_unet_orgdet_fnn_div_v1` (that
overlay only attached unused children).

## Hypothesis

Official 44b6 FNN oracle recovered both annotated forks on the same
detections (2/2) while locked FNN predicts **0** divisions.
`learned_unet_orgdet_fnn_div_v1` was an official no-op because those
daughters were already claimed 1-1. Conservatively stealing an
already-claimed FNN child when locked `division_v2` geometry holds and
the new parent is strictly closer was supposed to recover those forks
without rewriting the 12 µm linker, gap, ELF, or FNN keep rule.

Hash `39fb20d675e21d0e` was locked before any overlay score. Official
44b6 had to exceed FNN by **≥ 0.002000** (threshold **0.726653**) before
any one-shot 6bba versus **0.752091**.

## Operating point (rejected on holdout)

Config: `configs/learned_unet_orgdet_fnn_reas_v1.yaml`.
Hash `39fb20d675e21d0e`. Parent FNN `95e9300eb84ba71c`.
Rejected unused-child overlay `e184e6ce5bd90e46`.
Division numeric gates `0fa756963e5d42d1`.

- Source: locked FNN detections
- Geometry: max parent–child 10.0 µm, min sibling 8.3 µm, alignment
  cosine ≤ 0.05, predecessor required, min displacement 1.0 µm
- Reassign only an already-claimed dt=1 child whose current parent has
  out-degree 1 and is farther than the new parent
- Does **not** attach unused children
- Intensity unused. No ILP. No transformer. No 12 µm retune
- Writes no `submission.csv`
- No kernel packaged

Do **not** retune. Do **not** score 6bba again.

## Cost estimate (before score)

- Storage: reuse orgop / orgdet-holdout graphs + GEFFs from
  `mahdimzafari/biohub-geff-export` v1. Appearance Zarrs not required.
- CPU: locked FNN rebuild plus claimed-child scan. Budget 76 s / 900 MB.
  Measured **40.4 s / 533 MB** on 44b6; **58.0 s / 741 MB** on 6bba.
- No GPU kernel. Not a Kaggle run.

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4`
only. GEFFs from `mahdimzafari/biohub-geff-export` v1. Orgop graphs
from `mahdimzafari/biohub-learned-unet-orgop-v1` v1. **Not a Kaggle
run. Not a leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | reassigned |
|---|---|---|---|---|---|---|
| learned_unet_orgdet_fnn_v1 | **0.724653** | 0.74020 | 0.72465 | 0.97425 | 0/0/2 | 0 |
| learned_unet_orgdet_fnn_reas_v1 | **0.727378** | 0.74304 | 0.72738 | 0.97425 | 0/2/2 | 102 |

Official matched-edge TP/FP/FN **908 / 172 / 142** vs FNN **906 / 174 /
144**. Official event scorer recovered **0 / 2** annotated forks and
added **2** division FPs. Per-sequence:

| sequence | FNN combined | reas combined | reassigned |
|---|---|---|---|
| 44b6_12dfb391 | 0.811357 | 0.811357 | 8 |
| 44b6_267148e4 | 0.520601 | 0.528652 | 94 |

Delta versus FNN **+0.002726**, above the preregistered **+0.002000**
gate (**0.726653**). The gain is association on `44b6_267148e4`
(204/86/73 vs 202/88/75), not recovered GT forks. Runtime 40.37 s,
peak 533 MB.

Tuning decision **`keep_candidate_learned_unet_orgdet_fnn_reas_v1`**.
Hash was locked before the 6bba look. See
`reports/learned_unet_orgdet_fnn_reas_v1_tune.json`.

## Official frozen-holdout score (not leaderboard)

One-shot local official scorer on `6bba_05db0fb1` + `6bba_c328f2fd`.
Source graphs from `mahdimzafari/biohub-learned-unet-orgdet-holdout-v1`.
**Not a Kaggle run. Not a leaderboard score.** Do **not** score again.

| candidate | combined | edge J | adj edge J | recall | divisions | reassigned |
|---|---|---|---|---|---|---|
| learned_unet_orgdet_fnn_v1 | **0.752091** | 0.77110 | 0.75209 | 0.95501 | 0/0/5 | 0 |
| learned_unet_orgdet_fnn_reas_v1 | **0.750390** | 0.76935 | 0.75039 | 0.95501 | 0/2/5 | 191 |

Official TP/FP/FN **1461 / 216 / 222** vs FNN **1462 / 213 / 221**.
Per-sequence:

| sequence | FNN combined | reas combined | reassigned |
|---|---|---|---|
| 6bba_05db0fb1 | 0.774163 | 0.771682 | 134 |
| 6bba_c328f2fd | 0.700745 | 0.700745 | 57 |

Holdout loss is on `6bba_05db0fb1` (1052/146/131 vs 1053/143/130).
`6bba_c328f2fd` is identical to FNN. Runtime 57.96 s, peak 741 MB.

Decision **`reject_learned_unet_orgdet_fnn_reas_v1`**. Status
`rejected_on_holdout`. Champion remains `learned_unet_orgdet_fnn_v1`.
Frozen `6bba` must not be scored again for this hash. Do **not** retune.

Public **0.602 / 0.612 / 0.553** were not used.

## Next experiment (not launched)

Do not retune `learned_unet_orgdet_fnn_reas_v1`, `fnn_div`,
`division_v2`, FNN, ELF, TCF, or rank_v1. Champion remains
`learned_unet_orgdet_fnn_v1`. Preferred packaged public-test worker
remains fnn-fast-test `9b9763e26bc0f037`. If continuing, preregister
one new 44b6-first hash that is not a reas/rank/div/12 µm retune.
Prefer a two-Zarr appearance-consistency filter on remaining
forward-NN dt=1 edges. Official-score 44b6 vs FNN **0.724653** with
gate **0.726653** before any one-shot 6bba versus **0.752091**. Do not
overwrite fnn-fast-test v1.
