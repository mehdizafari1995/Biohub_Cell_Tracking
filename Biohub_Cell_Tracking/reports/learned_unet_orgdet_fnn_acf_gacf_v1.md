# learned_unet_orgdet_fnn_acf_gacf_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or orgdet/elf/fnn/acf slugs. Do **not** retune. This is **not** another
12 µm assignment, **not** a rank/div/reas/ACF/BNN/IPC retune, and **not**
another 8.0 µm dt=1 length retune.

## Hypothesis

Official 44b6 ACF residue was still association-dominated (160 conflict
FPs / 143 FNs; 0 predicted divisions). Locked ACF only drops dt=1 edges
whose 3×3×3-mean relative appearance exceeds 0.38. Incoming-parent IPC
was a no-op after ACF. Global backward-NN is closed.

Gap-close still links leftover detections across dt∈{2,3} and
interpolates those spans to dt=1 hops. The two real endpoints stay
unfiltered by ACF. Dropping a gap-close chain when those real endpoints
have relative appearance > 0.38 (same locked gate, not a retune), then
new degree-0 leftovers, should raise adjusted edge Jaccard versus ACF
**0.734620** without rewriting the 12 µm linker, gap gates, ELF, FNN,
or ACF. Official 44b6 had to exceed ACF by **≥ 0.002000** (threshold
**0.736620**). Frozen `6bba` stayed closed unless that gate kept. Hash
`5ce49a6d175c8032` was locked before the 44b6 score.

## Operating point (rejected)

Config: `configs/learned_unet_orgdet_fnn_acf_gacf_v1.yaml`. Hash
`5ce49a6d175c8032`. Parent ACF `a66075085ac64400`. Rejected BNN
`bdb38cf13e226ced`. Rejected IPC `7302409595bee523`.

- Source: locked orgop / orgdet graphs
- Locked orgdet+gap+prune+ELF+FNN+ACF first
- Then drop a gap-close chain when the two real endpoints satisfy
  `|I_dst − I_src| / max(|I_src|, |I_dst|, 1e-6) > 0.38`
  Remaining dt≥2 edges use the same locked gate
- Then drop newly created degree-0 leftovers
- Consecutive dt=1 edges stay (ACF already filtered them)
- No transformer. No ILP. No 12 µm retune
- Writes no `submission.csv`

Do **not** retune `max_rel_appearance`. Do **not** retune `max_gap_dt`.
Do **not** open `6bba` for this hash.

## Cost estimate (before score)

- Storage: reuse orgop graphs + GEFFs from `mahdimzafari/biohub-geff-export`
  v1 + the two exported tuning Zarrs from
  `mahdimzafari/biohub-tuning-zarr-export` v1. Holdout Zarrs were **not**
  downloaded. No weights committed.
- CPU: locked ACF overlay + gap-chain appearance walk. Measured
  **19.01 s / 581 MB** on 44b6.
- No new GPU training. Not a Kaggle run for the overlay score.

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4`
only. GEFFs from `mahdimzafari/biohub-geff-export` v1. Zarrs from
`mahdimzafari/biohub-tuning-zarr-export` v1. **Not a Kaggle run. Not a
leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes |
|---|---|---|---|---|---|---|
| learned_unet_orgdet_fnn_acf_v1 | **0.734620** | 0.74959 | 0.73462 | 0.97425 | 0/0/2 | 94142 |
| learned_unet_orgdet_fnn_acf_gacf_v1 | **0.734745** | 0.74959 | 0.73475 | 0.97425 | 0/0/2 | 94004 |

Dropped **160** inconsistent gap-close edges (49 + 111) across **61**
chains (19 + 42) after checking 2253 gap chains. Official TP/FP/FN
**907 / 160 / 143**, identical to locked ACF. Per-sequence:

| sequence | acf combined | gacf combined | dropped edges | chains |
|---|---|---|---|---|
| `44b6_12dfb391` | 0.814642 | 0.814696 | 49 | 19 |
| `44b6_267148e4` | 0.541120 | 0.541417 | 111 | 42 |

Delta versus ACF **+0.000125**, below the preregistered **+0.002000**
gate (**0.736620**). Official matched-edge counts did not change. The
dropped chains were leftover unmatched gap-close links, so only `T_pred`
moved. Official event scorer recovered **0 / 2** annotated forks.
Runtime 19.01 s, peak 581 MB.

Decision **`reject_learned_unet_orgdet_fnn_acf_gacf_v1`**. Frozen `6bba`
was not opened. No kernel. Do **not** retune. See
`reports/learned_unet_orgdet_fnn_acf_gacf_v1_tune.json`.

## Next

A later unused lever, if one is launched, must not be this gap-edge
appearance rule and must not be rejected BNN / IPC / rank / div / reas.
Preferred unused lever: **gap-edge length** (ELF is dt=1 only). Official
44b6 must still exceed ACF **0.734620 + 0.002000**. Do **not** overwrite
acf-test v1. Do **not** submit. Do **not** score 6bba for this hash. Do
**not** use public **0.602 / 0.612 / 0.553**.
