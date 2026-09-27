# learned_unet_orgdet_fnn_rank_v1 (rejected on tuning)

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or orgdet/elf/fnn slugs. Do **not** retune locked FNN. Frozen `6bba` was
**not** opened. This is **not** another 12 µm assignment, **not** a
short-fragment retune, and **not** another 8.0 µm dt=1 length retune.

## Hypothesis

Official 44b6 FNN oracle (same detections, official scorer `075fc5f5`)
showed association-dominated residue with candidate-edge recall
**0.996** and candidate-oracle **1.048979** vs FNN **0.724653**. A
learned pairwise ranker that only sees candidates already inside the
locked envelope, trained with hard negatives on 44b6 only, then
constrained 1-in / 2-out selection, was supposed to recover those
edges without rewriting the 12 µm linker, gap gates, ELF, or FNN keep
rule. Link-head canary `f7dc3d3292ce813d` was **`canary_ok`**.

Hash `2748c4a2b2823c0a` was locked before any ranker score. Official
44b6 had to exceed FNN by **≥ 0.002000** (threshold **0.726653**).

## Operating point (rejected)

Config: `configs/learned_unet_orgdet_fnn_rank_v1.yaml`.
Hash `2748c4a2b2823c0a`. Parent FNN `95e9300eb84ba71c`.

- Source: locked FNN detections
- Candidate set: dt ∈ {1, 2}, 12 µm · √dt, locked
- Coordinate embeddings (no Zarr appearance; no 69-volume train)
- Transformer pair head; hard negatives; 44b6 only
- Constrained selection: max 1 incoming, max 2 outgoing
- Intensity unused. No ILP. No velocity. No 12 µm retune
- Writes no `submission.csv`
- Helper packaged as `mahdimzafari/biohub-learned-unet-orgdet-fnn-rank-v1`.
  **Not pushed.**

Do **not** retune. Do **not** open `6bba`.

## Cost estimate (before score)

- Storage: reuse orgop graphs + two 44b6 GEFFs. Appearance Zarrs not
  required for this helper.
- CPU: one epoch over 1751832 envelope pairs. Measured **405.9 s /
  3814 MB**.
- No GPU kernel. Not a Kaggle run.

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4`
only. GEFFs from `mahdimzafari/biohub-geff-export` v1. Orgop graphs
from `mahdimzafari/biohub-learned-unet-orgop-v1` v1. **Not a Kaggle
run. Not a leaderboard score.**

A first pass with an extra logit>0 cutoff wrote 0 edges (official
**0.0**). That cutoff is not in the locked payload; it was an
implementation defect of the same class as `link_v1` softmax+0.5.
Selection was corrected to rank-order + constraints only and scored
once.

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes / edges |
|---|---|---|---|---|---|---|
| learned_unet_orgdet_fnn_v1 | **0.724653** | 0.74020 | 0.72465 | 0.97425 | 0/0/2 | 95019 / 88812 |
| learned_unet_orgdet_fnn_rank_v1 | **0.006663** | 0.00681 | 0.00666 | 0.97361 | 0/309/2 | 94952 / 93464 |

Official matched-edge TP/FP/FN **8 / 124 / 1042** vs FNN **906 / 174 /
144**. Train pairs: 1019 true / 1750813 hard. Predicted forks 23014 +
11993. Per-sequence:

| sequence | FNN combined | rank combined |
|---|---|---|
| 44b6_12dfb391 | 0.811357 | 0.006868 |
| 44b6_267148e4 | 0.520601 | 0.006089 |

Decision **`reject_learned_unet_orgdet_fnn_rank_v1`**. Class-imbalanced
BCE plus unconstrained two-child permission lost almost all official
TPs and added 309 division FPs. Frozen `6bba` not opened. Do **not**
retune this hash.

Public **0.602 / 0.612 / 0.553** were not used.

## Next experiment (not launched)

Do not retune `learned_unet_orgdet_fnn_rank_v1`. Champion remains
`learned_unet_orgdet_fnn_v1`. The preferred next family
`learned_unet_orgdet_fnn_div_v1` hash `e184e6ce5bd90e46` was later
official-scored on 44b6 at **0.724653** and rejected. Do not relaunch
rank or div. Do not overwrite fnn-fast-test v1.
