# learned_unet_orgdet_elf_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or orgdet/orgdet-gap/orgdet-gap-prune slugs. Do **not** retune. This is
**not** another 12 µm assignment and **not** a short-fragment retune.

## Hypothesis

Official 44b6 error decomposition of locked
`learned_unet_orgdet_gap_prune_v1` was association-dominated (115 vs 27
detection FN edges). Assignment overlays stay closed. Short-fragment TCF
missed the +0.002000 gate on `T_pred` only.

On both frozen-tuning prune graphs, every official-matched TP consecutive
edge was shorter than 8.0 µm (max 7.6219 µm on `44b6_267148e4`, max
5.8590 µm on `44b6_12dfb391`), while 26 conflict FPs exceeded 8.0 µm.
Dropping dt=1 edges longer than 8.0 µm after locked prune, then dropping
newly created degree-0 leftovers, should raise adjusted edge Jaccard
versus **0.693437** without rewriting the 12 µm linker or gap gates.
Official 44b6 had to exceed prune by **≥ 0.002000** (threshold
**0.695437**). Frozen `6bba` stayed closed unless that gate kept.

## Operating point (kept)

Config: `configs/learned_unet_orgdet_elf_v1.yaml`. Hash `292c4f1f968f948d`.
Parent prune `862e3e7f7ac0330e`. Parent orgdet-gap `ee31f36151325f9a`.
Parent orgdet `80b20589198148b7`. Train hash `4c38035d6657f2ad`.

- Source: locked orgop / orgdet graphs
- Locked orgdet+gap+prune first
- Then drop consecutive dt=1 edges longer than 8.0 µm
- Then drop newly created degree-0 leftovers
- Existing shorter dt=1 and non-dt=1 edges stay
- Intensity unused. No transformer. No ILP.
- Writes no `submission.csv`

Do **not** retune `max_dt1_edge_um`. Do **not** score 6bba again.

## Cost estimate (before score)

- Storage: reuse orgop / orgdet-holdout helper graphs + GEFFs from
  `mahdimzafari/biohub-geff-export` v1. No volumes, no weights.
- CPU: orgdet relink + gap + isolate prune + edge-length filter.
  Measured **11.26 s / 529 MB** on 44b6; **7.77 s** on 6bba.
- No new GPU training. Not a Kaggle run for the overlay scores.

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4`
only. GEFFs from `mahdimzafari/biohub-geff-export` v1. **Not a Kaggle
run. Not a leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes |
|---|---|---|---|---|---|---|
| learned_unet_orgdet_gap_prune_v1 | **0.693437** | 0.70993 | 0.69344 | 0.97603 | 0/0/2 | 96750 |
| learned_unet_orgdet_elf_v1 | **0.709838** | 0.72626 | 0.70984 | 0.97603 | 0/0/2 | 96173 |

Dropped 2247 long dt=1 edges (840 + 1407) and 577 new isolates.
Official TP/FP/FN **910 / 203 / 140** vs prune **908 / 229 / 142**.
Per-sequence:

| sequence | prune combined | elf combined | dropped edges |
|---|---|---|---|
| `44b6_12dfb391` | 0.783806 | 0.792144 | 840 |
| `44b6_267148e4` | 0.488201 | 0.516398 | 1407 |

Delta versus prune **+0.016401**, above the preregistered **+0.002000**
gate (**0.695437**). Official event scorer recovered **0 / 2** annotated
forks. Runtime 11.26 s, peak 529 MB.

Decision **`keep_candidate_learned_unet_orgdet_elf_v1`**. Hash was locked
before the 6bba look. See `reports/learned_unet_orgdet_elf_v1_tune.json`.

## Official frozen-holdout score (not leaderboard)

One-shot local official scorer on `6bba_05db0fb1` + `6bba_c328f2fd`.
Source graphs from `mahdimzafari/biohub-learned-unet-orgdet-holdout-v1`.
**Not a Kaggle run. Not a leaderboard score.** Do **not** score again.

| candidate | combined | edge J | adj edge J | recall | divisions |
|---|---|---|---|---|---|
| learned_unet_orgdet_gap_prune_v1 | **0.723018** | 0.74265 | 0.72302 | 0.95762 | 0/0/5 |
| learned_unet_orgdet_elf_v1 | **0.737787** | 0.75737 | 0.73779 | 0.95582 | 0/0/5 |

Dropped 2888 long dt=1 edges (2328 + 560). Official TP/FP/FN
**1464 / 250 / 219**. Per-sequence: `6bba_05db0fb1` 0.756179;
`6bba_c328f2fd` 0.694883. Runtime 7.67 s.

Decision **`keep_candidate_learned_unet_orgdet_elf_v1`**. New holdout
champion. Public **0.602 / 0.612 / 0.553** were not used.

## Next

Public-test slug `learned_unet_orgdet_elf_test_v1` hash `ad375d39e13098d1`
v1 is **COMPLETE / inferred / `audit_ok`**. Independent `audit_ok`.
`READY_FOR_MANUAL_SUBMISSION_REVIEW`. Do **not** overwrite
`mahdimzafari/biohub-learned-unet-orgdet-gap-prune-test-v1` v1. Do **not**
overwrite elf-test v1. Do **not** submit from the CLI.
