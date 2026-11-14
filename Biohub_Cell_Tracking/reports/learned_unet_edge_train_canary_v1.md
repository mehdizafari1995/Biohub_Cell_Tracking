# learned_unet_edge_train_canary_v1

CPU evidence canary after rejected `learned_unet_orgop_v1`. Do **not**
merge. Do **not** push a kernel. Frozen `6bba` was not opened. No
`submission.csv`. No 199-video weights.

## Hypothesis

The locked 3-epoch transformer is near init, so organizer softmax>0.5
writes zero candidates. The same architecture can still learn peaked
edges if edge loss actually moves the pair head.

## Operating point

Config: `configs/learned_unet_edge_train_canary_v1.yaml`. Hash
`1a3d524e4d922701`. Parent orgop `250c5bfacd4a8198`. Train hash
`4c38035d6657f2ad`. Synthetic 12×12 1-1 targets, freeze UNet, 60 AdamW
steps, lr `1e-3`, seed 0. Architecture imported from
`notebooks/kaggle_learned_unet_v1.py`.

## Result (not leaderboard)

| gate | value |
|---|---|
| detection_loss finite | 1.400 |
| first edge_loss / grad | 0.172 / 0.018 |
| last edge_loss | 5.26e-4 |
| eval softmax `prob_max` | 0.996 |
| pairs > 0.5 | **12 / 12** |
| checkpoint reload max abs | **0.0** |
| runtime | 0.60 s CPU |
| 6bba opened | no |
| kernel pushed | no |

Decision **`canary_ok`**. Not a scoring candidate.

## Compute projection (locked train v3 rate)

3-epoch 69-video helper was 11732 s. Scaled:

- 3 epochs: 3.26 h (already done)
- **10 epochs: 10.86 h, under 12 h**
- 20 epochs: 21.73 h (over)
- 50-epoch organizer scale: 54.31 h (over)

## Next experiment (not launched)

`learned_unet_edge_train_v1`: new slug, 44b6-only, log separate
detection/edge losses, budget **10 epochs** or equivalent window cap,
no 199-video weights. Official-score 44b6 vs infer **0.655751** before
any `6bba` look. Do not overwrite orgop v1 or prune-test v1.
