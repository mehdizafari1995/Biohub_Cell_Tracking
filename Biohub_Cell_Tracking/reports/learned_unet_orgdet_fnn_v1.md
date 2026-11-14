# learned_unet_orgdet_fnn_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or orgdet/orgdet-gap/orgdet-gap-prune/elf slugs. Do **not** retune. This is
**not** another 12 µm assignment, **not** a short-fragment retune, and
**not** another 8.0 µm dt=1 length retune.

## Hypothesis

Official 44b6 ELF residue was association-dominated (203 conflict FPs /
140 FNs; 0 predicted divisions). Turning-angle, unique-degree, all-
interpolation drop, and another 8.0 µm dt=1 length filter stay closed.

On both frozen-tuning ELF graphs, 30 of 203 conflict FPs were not the
source's nearest neighbor at t+1, versus 10 of 910 TPs. Dropping those
already-chosen dt=1 edges, then new degree-0 leftovers, should raise
adjusted edge Jaccard versus ELF **0.709838** without rewriting the 12 µm
linker, gap gates, or ELF length gate. Official 44b6 had to exceed ELF by
**≥ 0.002000** (threshold **0.711838**). Frozen `6bba` stayed closed
unless that gate kept. Hash `95e9300eb84ba71c` was locked before the
44b6 score.

## Operating point (kept)

Config: `configs/learned_unet_orgdet_fnn_v1.yaml`. Hash `95e9300eb84ba71c`.
Parent ELF `292c4f1f968f948d`. Parent prune `862e3e7f7ac0330e`.
Parent orgdet-gap `ee31f36151325f9a`. Parent orgdet `80b20589198148b7`.
Train hash `4c38035d6657f2ad`.

- Source: locked orgop / orgdet graphs
- Locked orgdet+gap+prune+ELF first
- Then keep a dt=1 edge only when the destination is a nearest neighbor
  of the source at t+1 (ties kept, `nn_tie_eps=1e-9`)
- Then drop newly created degree-0 leftovers
- Non-dt=1 edges stay. Intensity unused. No transformer. No ILP.
- Writes no `submission.csv`

Do **not** retune `nn_tie_eps`. Do **not** retune `max_dt1_edge_um`.
Do **not** score 6bba again.

## Cost estimate (before score)

- Storage: reuse orgop / orgdet-holdout helper graphs + GEFFs from
  `mahdimzafari/biohub-geff-export` v1. No volumes, no weights.
- CPU: locked ELF overlay + per-node nearest-neighbor scan.
  Measured **72.14 s / 530 MB** on 44b6; **114.96 s / 734 MB** on 6bba.
- No new GPU training. Not a Kaggle run for the overlay scores.

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4`
only. GEFFs from `mahdimzafari/biohub-geff-export` v1. **Not a Kaggle
run. Not a leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes |
|---|---|---|---|---|---|---|
| learned_unet_orgdet_elf_v1 | **0.709838** | 0.72626 | 0.70984 | 0.97603 | 0/0/2 | 96173 |
| learned_unet_orgdet_fnn_v1 | **0.724653** | 0.74020 | 0.72465 | 0.97425 | 0/0/2 | 95019 |

Dropped 2949 non-forward-NN dt=1 edges (1307 + 1642) and 1154 new
isolates. Official TP/FP/FN **906 / 174 / 144** vs ELF **910 / 203 /
140**. Per-sequence:

| sequence | elf combined | fnn combined | dropped edges |
|---|---|---|---|
| `44b6_12dfb391` | 0.792144 | 0.811357 | 1307 |
| `44b6_267148e4` | 0.516398 | 0.520601 | 1642 |

Delta versus ELF **+0.014815**, above the preregistered **+0.002000**
gate (**0.711838**). Official event scorer recovered **0 / 2** annotated
forks. Runtime 72.14 s, peak 530 MB.

Decision **`keep_candidate_learned_unet_orgdet_fnn_v1`**. Hash was locked
before the 6bba look. See `reports/learned_unet_orgdet_fnn_v1_tune.json`.

## Official frozen-holdout score (not leaderboard)

One-shot local official scorer on `6bba_05db0fb1` + `6bba_c328f2fd`.
Source graphs from `mahdimzafari/biohub-learned-unet-orgdet-holdout-v1`.
**Not a Kaggle run. Not a leaderboard score.** Do **not** score again.

| candidate | combined | edge J | adj edge J | recall | divisions |
|---|---|---|---|---|---|
| learned_unet_orgdet_elf_v1 | **0.737787** | 0.75737 | 0.73779 | 0.95582 | 0/0/5 |
| learned_unet_orgdet_fnn_v1 | **0.752091** | 0.77110 | 0.75209 | 0.95501 | 0/0/5 |

Dropped 3566 non-forward-NN dt=1 edges (2827 + 739). Official TP/FP/FN
**1462 / 213 / 221**. Per-sequence: `6bba_05db0fb1` 0.774163;
`6bba_c328f2fd` 0.700745. Runtime 114.96 s, peak 734 MB.

Decision **`keep_candidate_learned_unet_orgdet_fnn_v1`**. New holdout
champion. Public **0.602 / 0.612 / 0.553** were not used.

## Next

Public-test slug `mahdimzafari/biohub-learned-unet-orgdet-fnn-test-v1`
v1 is **COMPLETE / inferred / `audit_ok`**. Independent `audit_ok`.
`6bba_05db0fb1` **87339/81132** matches this overlay. Hidden **11.04 h**
trips `fail_closed_approaches_12h`, so it is **not**
`READY_FOR_MANUAL_SUBMISSION_REVIEW`. Preferred packaged fallback remains
elf-test. Do **not** overwrite either slug. Do **not** submit from the
CLI. Additional UI submit is **not** authorized unless a later message
separately authorizes that extra click.
