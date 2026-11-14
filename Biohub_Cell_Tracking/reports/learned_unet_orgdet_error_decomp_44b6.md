# 44b6 official-scorer error decomposition

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Frozen `6bba` was **not** opened. This is **not** a Kaggle run
and **not** a leaderboard score. Do **not** retune locked hashes. Do
**not** use public **0.602** / **0.612** / **0.553**.

## Scope

Official scorer `075fc5f5` on frozen-tuning `44b6_12dfb391` +
`44b6_267148e4` only. Graphs rebuilt from locked
`mahdimzafari/biohub-learned-unet-orgop-v1` detections. GEFFs from
`mahdimzafari/biohub-geff-export` v1. Compared locked overlays:

| overlay | hash | combined | matches locked |
|---|---|---|---|
| `learned_unet_orgdet_v1` | `80b20589198148b7` | **0.679578** | yes |
| `learned_unet_orgdet_gap_v1` | `ee31f36151325f9a` | **0.687221** | yes |
| `learned_unet_orgdet_gap_prune_v1` | `862e3e7f7ac0330e` | **0.693437** | yes |

Runtime 25.19 s, peak 562 MB. Official deps `tracksdata==0.1.0rc10`,
`polars==1.44.2`, `geff==1.3.1.1.3`.

## Prune residue (champion)

Official TP/FP/FN **908 / 229 / 142**. Node recall **0.976**. Divisions
**0 / 0 / 2**.

| bucket | count |
|---|---|
| detection FN edges (unmatched GT endpoint) | 27 |
| association FN edges (both GT endpoints matched) | 115 |
| missed GT nodes | 16 |
| unmatched pred nodes (sparse GT; not node FPs) | 95698 |
| unsupported t±1 leftovers after prune | 0 |
| short fragments ≤3 nodes | 462 (0 GT matches) |
| short fragments ≤7 nodes | 1954 (0 GT matches on both stems) |

Detection FN share **0.190**. Dominant mode **association**. Assignment
overlays stay closed. t±1 unsupported-node filtering is a no-op after
isolate prune.

## Preregistered hypothesis

From 44b6 evidence only: drop connected components with fewer than 8
nodes after locked orgdet+gap+prune
(`learned_unet_orgdet_tcf_v1`, hash `a6b46dfcf410478f`). Not another
12 µm assignment. Strict tune gate: official 44b6 must exceed
**0.693437** by **≥ 0.002000** (threshold **0.695437**). Frozen `6bba`
refused until that keep.

See `reports/learned_unet_orgdet_error_decomp_44b6.json` and
`reports/learned_unet_orgdet_tcf_v1.md`.
