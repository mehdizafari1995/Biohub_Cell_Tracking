# learned_unet_orgdet_fnn_acf_ipc_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or orgdet/elf/fnn/acf slugs. Do **not** retune. This is **not** another
12 µm assignment, **not** a rank/div/reas/ACF/BNN retune, and **not**
another 8.0 µm dt=1 length retune.

## Hypothesis

Official 44b6 ACF residue was still association-dominated (160 conflict
FPs / 143 FNs; 0 predicted divisions). Global backward-NN among all
nodes at t−1 is closed: `learned_unet_orgdet_fnn_acf_bnn_v1` dropped
true parents as well as conflict FPs. Locked FNN already keeps only
source→forward-NN destinations. Incoming conflicts could remain:
several already-chosen ACF parents can still claim one child. When a
child has two or more incoming dt=1 edges, keep only the nearest of
those already-chosen parents (ties kept). Unique incoming dt=1 edges
stay. That is not global t−1 nearest-neighbor. Official 44b6 had to
exceed ACF by **≥ 0.002000** (threshold **0.736620**). Frozen `6bba`
stayed closed unless that gate kept. Hash `7302409595bee523` was
locked before the 44b6 score.

## Operating point (rejected)

Config: `configs/learned_unet_orgdet_fnn_acf_ipc_v1.yaml`. Hash
`7302409595bee523`. Parent ACF `a66075085ac64400`. Rejected BNN
`bdb38cf13e226ced`.

- Source: locked orgop / orgdet graphs
- Locked orgdet+gap+prune+ELF+FNN+ACF first
- Then, only on children with ≥2 incoming dt=1 edges, keep the
  nearest already-chosen parent (`nn_tie_eps=1e-9`)
- Unique incoming dt=1 edges stay
- Then drop newly created degree-0 leftovers
- Non-dt=1 edges stay. No transformer. No ILP. No 12 µm retune
- Writes no `submission.csv`

Do **not** retune `nn_tie_eps`. Do **not** retune `max_rel_appearance`.
Do **not** open `6bba` for this hash.

## Cost estimate (before score)

- Storage: reuse orgop graphs + GEFFs from `mahdimzafari/biohub-geff-export`
  v1 + the two exported tuning Zarrs from
  `mahdimzafari/biohub-tuning-zarr-export` v1. Holdout Zarrs were **not**
  downloaded. No weights committed.
- CPU: locked ACF overlay + incoming-conflict keep. Measured
  **19.62 s / 552 MB** on 44b6.
- No new GPU training. Not a Kaggle run for the overlay score.

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4`
only. GEFFs from `mahdimzafari/biohub-geff-export` v1. Zarrs from
`mahdimzafari/biohub-tuning-zarr-export` v1. **Not a Kaggle run. Not a
leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes |
|---|---|---|---|---|---|---|
| learned_unet_orgdet_fnn_acf_v1 | **0.734620** | 0.74959 | 0.73462 | 0.97425 | 0/0/2 | 94142 |
| learned_unet_orgdet_fnn_acf_ipc_v1 | **0.734620** | 0.74959 | 0.73462 | 0.97425 | 0/0/2 | 94142 |

Dropped **0** conflict incoming edges. Conflict children **0 / 0**.
Official TP/FP/FN **907 / 160 / 143**, identical to locked ACF.
Per-sequence:

| sequence | acf combined | ipc combined | dropped edges | conflict children |
|---|---|---|---|---|
| `44b6_12dfb391` | 0.814642 | 0.814642 | 0 | 0 |
| `44b6_267148e4` | 0.541120 | 0.541120 | 0 | 0 |

Delta versus ACF **0.000000**, below the preregistered **+0.002000**
gate (**0.736620**). After locked ACF, no child has two incoming dt=1
edges, so the incoming-parent filter is a no-op. Official event scorer
recovered **0 / 2** annotated forks. Runtime 19.62 s, peak 552 MB.

Decision **`reject_learned_unet_orgdet_fnn_acf_ipc_v1`**. Frozen `6bba`
was not opened. No kernel. Do **not** retune. See
`reports/learned_unet_orgdet_fnn_acf_ipc_v1_tune.json`.

## Next

A later unused lever, if one is launched, must not be this incoming
already-chosen-parent rule and must not be rejected global backward-NN.
Prefer a **gap-edge appearance** filter: ACF only touches dt=1, so
dt>1 gap-close edges are still unfiltered. Official 44b6 must still
exceed ACF **0.734620 + 0.002000**. Do **not** overwrite acf-test v1.
Do **not** submit. Do **not** score 6bba for this hash. Do **not**
use public **0.602 / 0.612 / 0.553**.
