# learned_unet_orgdet_fnn_acf_snm_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or orgdet/elf/fnn/acf slugs. Do **not** retune. This is **not** another
12 µm assignment, **not** a rank/div/reas/ACF/BNN/IPC/GACF/GELF/TAC/SCF
retune, and **not** another 8.0 µm dt=1 length, 0.38 appearance, or
`nn_tie_eps` retune.

## Hypothesis

Official 44b6 ACF residue was still association-dominated (160 conflict
FPs / 143 FNs; 0 predicted divisions). Locked ACF only drops dt=1
edges whose 3×3×3-mean relative appearance exceeds 0.38. ELF only
drops dt=1 edges longer than 8.0 µm. FNN already keeps the source's
nearest neighbor at t+1. Gap-edge appearance GACF, gap-edge length
GELF, dt=1 turning-angle TAC, and dt=1 unique-pred speed-change SCF
are closed.

Remaining conflict FPs are almost all dt=1. If official-matched TP
dt=1 forward-NN hops had a strictly positive second-nearest margin
(d2 − d1) while conflict-FP hops sat below a 0.1 µm round-down of
that TP min, a single gate could drop those FPs without retuning
ELF/FNN/ACF/TAC/SCF. Leftover unmatched edges are **not** negatives
and must not drive a gate. Relative margin is recorded, not gated.
Official 44b6 would have had to exceed ACF by **≥ 0.002000**
(threshold **0.736620**) before opening frozen `6bba`.

## Diagnostic (rejected; overlay not scored)

Config: `configs/learned_unet_orgdet_fnn_acf_snm_v1.yaml`. Hash
`87f46b2bfee49fe8`. Parent ACF `a66075085ac64400`. Rejected SCF
`a259d184c9f99046`. `min_second_nn_margin_um` stays **null**.

Local rebuild of locked ACF on `44b6_12dfb391` + `44b6_267148e4` using
orgop graphs plus the two exported tuning Zarrs. dt=1 hops with a
second t+1 neighbor vs official 7 µm sparse-GT hop class. **Not a
Kaggle run. Not a leaderboard score. Not an official overlay score.**

dt=1 second-nearest margins:

| class | n | min µm | max µm |
|---|---|---|---|
| TP | 907 | **0.0** | 12.160 |
| conflict FP | 160 | 0.081 | 11.830 |
| leftover | 86088 | 0.0 | — |

Relative margin min is **0.0** for TP and leftover, **0.041** for
conflict FP, so it is not a gate. TP min **0.0** means at least one
official-matched TP has a tied second neighbor. Conflict-FP min
**0.081 µm** sits **above** every zero-margin TP. A TP-min round-down
cannot drop conflict FPs without also dropping TPs (`tp_covers_full_margin`).
No threshold sweep. Per-sequence:
`44b6_12dfb391` 704/83/60119 TP/conflict/leftover;
`44b6_267148e4` 203/77/25969. ACF rebuild counts match TAC/SCF
(64228/60906 and 29914/26249 nodes/edges). Official TP/conflict counts
match locked ACF **907/160**.

Decision **`reject_learned_unet_orgdet_fnn_acf_snm_v1`**. Reason
`tp_covers_full_margin`. Overlay apply stays refuse-closed while the
µm gate is unlocked. Frozen `6bba` not opened. No kernel. Parent ACF
champion and acf-test v1 unchanged. No threshold sweep. Leftovers
were not used as negatives.

A dt=1 forward-NN second-nearest-margin filter cannot move the +0.002
gate on these ACF graphs. Unique-pred kinematics (angle and vector
|Δv|) and second-NN spatial margin are now closed.

Runtime 14.85 s, peak 231 MB. Machine-readable copy:
`reports/learned_unet_orgdet_fnn_acf_snm_v1_diag.json`.
