# learned_unet_orgdet_fnn_rank_canary_v1

CPU link-head canary for preregistered `learned_unet_orgdet_fnn_rank_v1`.
Do **not** merge. Do **not** push a kernel. Frozen `6bba` was not
opened. No `submission.csv`. No 199-video weights. The ranker itself
was **not** official-scored.

## Hypothesis

Previous transformer linkers (`link_v1`, `link_v2`, `orgop`) failed
because the pair head stayed near init. The FNN oracle showed
association-fixable residue inside the locked 12 µm / `sqrt(dt)`
envelope (candidate-edge recall **0.996**, candidate-oracle
**1.048979** vs FNN **0.724653**). A ranker is fail-closed until this
canary shows nonzero link-head learning on hard negatives that already
sit inside that envelope.

## Operating point

Config: `configs/learned_unet_orgdet_fnn_rank_canary_v1.yaml`. Hash
`f7dc3d3292ce813d`. Parent rank `2748c4a2b2823c0a`. Parent FNN
`95e9300eb84ba71c`. Train hash `4c38035d6657f2ad`.

- Synthetic 44b6-refusing graph pairs, seed 0
- 6 true / 5 hard-negative / 5 out-of-envelope excluded
- Parent 0 is a two-child fork; one dt=2 pair is included
- Loss: pairwise logits + softplus ranking on in-envelope pairs only
- Freeze UNet / detect head; 80 AdamW steps; lr `1e-3`
- Architecture imported from `notebooks/kaggle_learned_unet_v1.py`

## Result (not leaderboard)

| gate | value |
|---|---|
| detection_loss finite | 0.525 |
| first link_loss / grad | 1.361 / 0.547 |
| last link_loss | 8.50e-5 |
| ranking acc | **6 / 6** |
| min true−hard logit | **20.677** |
| mean true / hard logit | 10.828 / −10.197 |
| checkpoint reload max abs | **0.0** |
| runtime | 1.12 s CPU |
| peak RAM | 375.4 MB |
| 6bba opened | no |
| official 44b6 scored | no |
| kernel pushed | no |

Decision **`canary_ok`**. Not a scoring candidate. Rank hash
`2748c4a2b2823c0a` remains `preregistered_not_scored`.

## Next experiment (not launched)

Package one 44b6-only ranker helper for `learned_unet_orgdet_fnn_rank_v1`
hash `2748c4a2b2823c0a`. Prefer the already exported tuning Zarrs
(`44b6_12dfb391`, `44b6_267148e4`) before any 69-volume train. Official
score 44b6 vs FNN **0.724653** with gate **0.726653** before any `6bba`
look. Do not overwrite fnn-fast-test v1. Do not retune 12 µm, ELF, FNN,
or TCF.
