# learned_unet_orgdet_fnn_div_v1 (rejected on tuning)

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or orgdet/elf/fnn slugs. Do **not** retune locked FNN or `division_v2`.
Frozen `6bba` was **not** opened. This is **not** another 12 µm
assignment, **not** a rank_v1 retune, and **not** a relaunch of
`learned_unet_division_v1` (that overlay sat on infer graphs).

## Hypothesis

Official 44b6 FNN oracle recovered both annotated forks on the same
detections (2/2) while locked FNN predicts **0** divisions. Applying
already-locked `division_v2` gates as a second-child overlay on locked
FNN graphs was supposed to recover those forks without rewriting the
12 µm linker, gap, ELF, or FNN keep rule.

Hash `e184e6ce5bd90e46` was locked before any overlay score. Official
44b6 had to exceed FNN by **≥ 0.002000** (threshold **0.726653**).

## Operating point (rejected)

Config: `configs/learned_unet_orgdet_fnn_div_v1.yaml`.
Hash `e184e6ce5bd90e46`. Parent FNN `95e9300eb84ba71c`.
Division gates `0fa756963e5d42d1`.

- Source: locked FNN detections
- Gates: max parent–child 10.0 µm, min sibling 8.3 µm, alignment
  cosine ≤ 0.05, predecessor required, no alternative 1-1 parent
  inside 12 µm
- Intensity unused. No ILP. No transformer. No 12 µm retune
- Writes no `submission.csv`
- No kernel packaged

Do **not** retune. Do **not** open `6bba`.

## Cost estimate (before score)

- Storage: reuse orgop graphs + two 44b6 GEFFs. Appearance Zarrs not
  required.
- CPU: locked FNN rebuild plus `propose_divisions`. Budget 76 s /
  900 MB. Measured **16.1 s / 514 MB**.
- No GPU kernel. Not a Kaggle run.

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4`
only. GEFFs from `mahdimzafari/biohub-geff-export` v1. Orgop graphs
from `mahdimzafari/biohub-learned-unet-orgop-v1` v1. **Not a Kaggle
run. Not a leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | extra edges |
|---|---|---|---|---|---|---|
| learned_unet_orgdet_fnn_v1 | **0.724653** | 0.74020 | 0.72465 | 0.97425 | 0/0/2 | 0 |
| learned_unet_orgdet_fnn_div_v1 | **0.724653** | 0.74020 | 0.72465 | 0.97425 | 0/0/2 | 51 |

Official matched-edge TP/FP/FN **906 / 174 / 144** identical to FNN.
Official event scorer recovered **0 / 2** annotated forks. Predicted
forks 16 + 35. Those extra edges were unmatched under sparse GT (0
official edge FPs added; 0 official division FPs). Per-sequence:

| sequence | FNN combined | div combined | extra edges |
|---|---|---|---|
| 44b6_12dfb391 | 0.811357 | 0.811357 | 16 |
| 44b6_267148e4 | 0.520601 | 0.520601 | 35 |

Decision **`reject_learned_unet_orgdet_fnn_div_v1`**. Locked
`division_v2` only attaches unused children. The two recoverable GT
daughters are already claimed by 1-1 FNN links, so the overlay is an
official no-op and misses the **+0.002000** gate. Frozen `6bba` not
opened. Do **not** retune this hash.

Public **0.602 / 0.612 / 0.553** were not used.

## Next experiment (not launched)

Do not retune `learned_unet_orgdet_fnn_div_v1`, `division_v2`, FNN,
ELF, TCF, or rank_v1. Champion remains `learned_unet_orgdet_fnn_v1`.
If continuing, preregister one new 44b6-first hash that is not another
12 µm assignment and not a rejected overlay. Prefer a conservative
reassignment of an already-claimed FNN child when locked
`division_v2` geometry holds, or a two-Zarr appearance-consistency
filter on remaining forward-NN dt=1 edges. Official-score 44b6 vs FNN
**0.724653** with gate **0.726653** before any one-shot 6bba versus
**0.752091**. Do not overwrite fnn-fast-test v1.
