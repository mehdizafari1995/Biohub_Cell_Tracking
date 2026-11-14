# 44b6 official-scorer ELF residue

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Frozen `6bba` was **not** opened for this decomposition. This is
**not** a Kaggle run and **not** a leaderboard score. Do **not** retune
locked ELF `292c4f1f968f948d`. Do **not** use public **0.602** /
**0.612** / **0.553**.

## Scope

Official scorer `075fc5f5` on frozen-tuning `44b6_12dfb391` +
`44b6_267148e4` only. Graphs rebuilt from locked
`mahdimzafari/biohub-learned-unet-orgop-v1` detections, then locked
`learned_unet_orgdet_elf_v1`. GEFFs from `mahdimzafari/biohub-geff-export`
v1. Combined score matched locked ELF **0.709838**.

Official TP/FP/FN **910 / 203 / 140**. Node recall **0.976**. Divisions
**0 / 0 / 2**. Predicted out-degree ≥2: **0**.

## Residue

| bucket | count |
|---|---|
| conflict FP edges | 203 |
| official FN edges | 140 |
| TP edge length max | 7.6219 µm |
| FP edge length max | 7.9608 µm |
| interpolated nodes after ELF | 5767 (25 matched GT) |
| TP edges touching interpolations | 33 |
| FP edges touching interpolations | 34 |
| TP two-hop cosine min | -1.0 |
| predicted divisions | 0 |

Turning-angle, unique-degree, all-interpolation drop, and another 8.0 µm
dt=1 length filter are no-ops or would retune a rejected/locked lever.

Forward nearest-neighbor status of existing consecutive edges:

| class | n | not source-NN at t+1 |
|---|---|---|
| official-style TP | 910 | 10 |
| conflict FP | 203 | 30 |

Dominant remaining mode is **association**. Assignment overlays stay
closed. The unused lever is an edge *filter* on already-chosen ELF
links: keep a dt=1 edge only when the destination is a nearest neighbor
of the source at t+1 (ties kept).

## Preregistered hypothesis

`learned_unet_orgdet_fnn_v1`, hash `95e9300eb84ba71c`. Parent ELF
`292c4f1f968f948d`. Not another 12 µm assignment. Not a short-fragment
retune. Not another 8.0 µm dt=1 length retune. Strict tune gate:
official 44b6 must exceed ELF **0.709838** by **≥ 0.002000** (threshold
**0.711838**). Frozen `6bba` refused until that keep.

See `reports/learned_unet_orgdet_fnn_v1.md`.
