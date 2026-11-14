# learned_unet_orgdet_tcf_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or orgdet/orgdet-gap slugs. Do **not** retune. Frozen `6bba` was **not**
opened. This is **not** another 12 µm assignment.

## Hypothesis

Official 44b6 error decomposition of locked
`learned_unet_orgdet_gap_prune_v1` was association-dominated (115 vs 27
detection FN edges). Assignment overlays stay closed. Unsupported t±1
leftovers were already 0 after prune. Fragments of size ≤7 had **0**
sparse-GT matches on both tuning sequences and still inflate `T_pred`.
Dropping those short components after locked prune should raise
adjusted edge Jaccard versus **0.693437** without rewriting
consecutive or gap links. Official 44b6 must exceed prune by
**≥ 0.002000** (threshold **0.695437**). Frozen `6bba` stayed closed
unless that gate kept.

## Operating point (rejected on tuning)

Config: `configs/learned_unet_orgdet_tcf_v1.yaml`. Hash `a6b46dfcf410478f`.
Parent prune `862e3e7f7ac0330e`. Parent orgdet-gap `ee31f36151325f9a`.
Parent orgdet `80b20589198148b7`. Train hash `4c38035d6657f2ad`.

- Source: locked orgop / orgdet 44b6 graphs
- Locked orgdet+gap+prune first
- Then drop connected components with fewer than 8 nodes
- Existing dt=1 and gap edges inside kept components stay
- Intensity unused. No transformer. No ILP.
- Writes no `submission.csv`

Do **not** retune `min_component_nodes`. Do **not** open `6bba`.

## Cost estimate (before score)

- Storage: reuse orgop helper graphs + two `44b6` GEFFs from
  `mahdimzafari/biohub-geff-export` v1. No volumes, no weights.
- CPU: orgdet relink + gap + isolate prune + component scan. Measured
  **10.63 s / 512 MB**.
- No GPU. No new kernel. Not a Kaggle run.

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4`
only. GEFFs from `mahdimzafari/biohub-geff-export` v1. **Not a Kaggle
run. Not a leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes |
|---|---|---|---|---|---|---|
| learned_unet_orgdet_gap_prune_v1 | **0.693437** | 0.70993 | 0.69344 | 0.97603 | 0/0/2 | 96750 |
| learned_unet_orgdet_tcf_v1 | **0.695181** | 0.70993 | 0.69518 | 0.97603 | 0/0/2 | 94796 |

Dropped 1954 short-fragment nodes (1094 + 860) and 1486 short-fragment
edges. Official TP/FP/FN stayed **908 / 229 / 142** (edge Jaccard
unchanged). Gain was T_pred only: **+0.001743**, below the
preregistered **+0.002000** gate (**0.695437**). Per-sequence:

| sequence | prune combined | tcf combined | dropped |
|---|---|---|---|
| `44b6_12dfb391` | 0.783806 | 0.785284 | 1094 |
| `44b6_267148e4` | 0.488201 | 0.490547 | 860 |

Combined improved but missed the strict gate by **0.000257**. Official
event scorer recovered **0 / 2** annotated forks. Runtime 10.63 s,
peak 512 MB.

Decision **`reject_learned_unet_orgdet_tcf_v1`**. Frozen `6bba` was
not opened. Do **not** retune fragment length. Do **not** treat this
as a holdout candidate. See
`reports/learned_unet_orgdet_tcf_v1_tune.json`.

This is a local official score, not a Kaggle leaderboard number.

## Next

Holdout champion remains `learned_unet_orgdet_gap_prune_v1`
(**0.723018**). Preferred public-test worker remains
`learned_unet_orgdet_gap_prune_test_v1` v1 **COMPLETE / `audit_ok`**.
Do **not** overwrite v1. Do **not** submit from the CLI. Do **not**
relaunch TCF with a different `min_component_nodes`.
