# learned_unet_orgdet_fnn_acf_bnn_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or orgdet/elf/fnn/acf slugs. Do **not** retune. This is **not** another
12 µm assignment, **not** a rank/div/reas/ACF retune, and **not**
another 8.0 µm dt=1 length retune.

## Hypothesis

Official 44b6 ACF residue was still association-dominated (160 conflict
FPs / 143 FNs; 0 predicted divisions). Locked FNN already keeps only
source→forward-NN destinations. Incoming conflicts remain: several
parents can still claim one child. On already-chosen ACF dt=1 edges,
keep an edge only when the source is a nearest neighbor of the
destination among **all** nodes at t−1 (ties kept), then drop new
degree-0 leftovers. That is the unused symmetric filter of FNN. It
does not reassign, does not shrink the 12 µm linker/gap gates, and is
not an ELF / FNN / ACF / TCF / rank / division retune. Official 44b6
had to exceed ACF by **≥ 0.002000** (threshold **0.736620**). Frozen
`6bba` stayed closed unless that gate kept. Hash `bdb38cf13e226ced`
was locked before the 44b6 score.

## Operating point (rejected)

Config: `configs/learned_unet_orgdet_fnn_acf_bnn_v1.yaml`. Hash
`bdb38cf13e226ced`. Parent ACF `a66075085ac64400`. Parent FNN
`95e9300eb84ba71c`. Parent ELF `292c4f1f968f948d`.

- Source: locked orgop / orgdet graphs
- Locked orgdet+gap+prune+ELF+FNN+ACF first
- Then keep a dt=1 edge only when the source is a nearest neighbor of
  the destination among all nodes at t−1 (`nn_tie_eps=1e-9`)
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
- CPU: locked ACF overlay + vectorized backward-NN. Measured
  **16.83 s / 557 MB** on 44b6.
- No new GPU training. Not a Kaggle run for the overlay score.

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4`
only. GEFFs from `mahdimzafari/biohub-geff-export` v1. Zarrs from
`mahdimzafari/biohub-tuning-zarr-export` v1. **Not a Kaggle run. Not a
leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes |
|---|---|---|---|---|---|---|
| learned_unet_orgdet_fnn_acf_v1 | **0.734620** | 0.74959 | 0.73462 | 0.97425 | 0/0/2 | 94142 |
| learned_unet_orgdet_fnn_acf_bnn_v1 | **0.734152** | 0.74834 | 0.73415 | 0.97361 | 0/0/2 | 93387 |

Dropped 1148 non-backward-NN dt=1 edges (410 + 738) and 755 new isolates.
Official TP/FP/FN **901 / 154 / 149** vs ACF **907 / 160 / 143**.
Per-sequence:

| sequence | acf combined | bnn combined | dropped edges |
|---|---|---|---|
| `44b6_12dfb391` | 0.814642 | 0.813465 | 410 |
| `44b6_267148e4` | 0.541120 | 0.540629 | 738 |

Delta versus ACF **−0.000468**, below the preregistered **+0.002000**
gate (**0.736620**). Official event scorer recovered **0 / 2** annotated
forks. Runtime 16.83 s, peak 557 MB.

Decision **`reject_learned_unet_orgdet_fnn_acf_bnn_v1`**. Global
backward-NN among all t−1 nodes dropped true parents (6 fewer TPs) as
well as conflict FPs. Frozen `6bba` was not opened. No kernel. Do
**not** retune. See `reports/learned_unet_orgdet_fnn_acf_bnn_v1_tune.json`.

## Next

A later unused lever, if one is launched, must not be this global
backward-NN rule. Prefer a filter that only resolves incoming conflicts
among **already-chosen** ACF parents, or a gap-edge appearance filter
(ACF only touches dt=1). Official 44b6 must still exceed ACF
**0.734620 + 0.002000**. Do **not** overwrite acf-test v1. Do **not**
submit. Do **not** score 6bba for this hash. Do **not** use public
**0.602 / 0.612 / 0.553**.
