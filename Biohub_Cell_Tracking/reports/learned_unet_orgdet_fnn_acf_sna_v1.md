# learned_unet_orgdet_fnn_acf_sna_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or orgdet/elf/fnn/acf slugs. Do **not** retune. This is **not** another
12 µm assignment, **not** a rank/div/reas/ACF/BNN/IPC/GACF/GELF/TAC/SCF/SNM
retune, and **not** another 8.0 µm dt=1 length, 0.38 appearance, or
`nn_tie_eps` retune.

## Hypothesis

Official 44b6 ACF residue was still association-dominated (160 conflict
FPs / 143 FNs; 0 predicted divisions). Locked ACF only drops dt=1
edges whose dest-vs-source 3×3×3-mean relative appearance exceeds 0.38.
ELF only drops dt=1 edges longer than 8.0 µm. FNN already keeps the
source's nearest neighbor at t+1. Gap-edge appearance GACF, gap-edge
length GELF, dt=1 turning-angle TAC, dt=1 unique-pred speed-change SCF,
and dt=1 forward-NN second-nearest spatial margin SNM are closed.

Remaining conflict FPs are almost all dt=1. If official-matched TP dt=1
forward-NN hops had dest vs spatial-second-NN 3×3×3-mean relative
appearance strictly above zero while conflict-FP hops sat below a 0.01
round-down of that TP min, a single gate could drop those FPs without
retuning ELF/FNN/ACF/SNM. Leftover unmatched edges are **not**
negatives and must not drive a gate. Dest-vs-source and dest
disadvantage (dest-vs-src minus second-vs-src) are recorded, not gated.
Official 44b6 would have had to exceed ACF by **≥ 0.002000**
(threshold **0.736620**) before opening frozen `6bba`.

## Diagnostic (rejected; overlay not scored)

Config: `configs/learned_unet_orgdet_fnn_acf_sna_v1.yaml`. Hash
`9b8811d3fae3a953`. Parent ACF `a66075085ac64400`. Rejected SNM
`87f46b2bfee49fe8`. `min_dest_second_nn_rel_appearance` stays **null**.

Local rebuild of locked ACF on `44b6_12dfb391` + `44b6_267148e4` using
orgop graphs plus the two exported tuning Zarrs. dt=1 hops with a
spatial second t+1 neighbor vs official 7 µm sparse-GT hop class. **Not a
Kaggle run. Not a leaderboard score. Not an official overlay score.**

dt=1 dest vs spatial-second-NN relative appearance:

| class | n | min | max |
|---|---|---|---|
| TP | 907 | **1.11e-5** | 0.651 |
| conflict FP | 160 | 1.38e-4 | 0.478 |
| leftover | 86088 | 0.0 | — |

TP min **1.11e-5** rounds down at 0.01 to **0.00**, so the proposed
gate covers the full delta (`tp_covers_full_delta`). Conflict-FP min
**1.38e-4** sits **above** every near-zero TP. A TP-min round-down
cannot drop conflict FPs without also dropping TPs. Dest-vs-source max
is **0.376** for TP (matches locked ACF 0.38) and **0.357** for conflict
FP. Dest disadvantage max is **0.256** for TP and **0.288** for conflict
FP; those recorded stats were not used as a gate. No threshold sweep.
Per-sequence: `44b6_12dfb391` 704/83/60119 TP/conflict/leftover;
`44b6_267148e4` 203/77/25969. ACF rebuild counts match SNM
(64228/60906 and 29914/26249 nodes/edges). Official TP/conflict counts
match locked ACF **907/160**.

Decision **`reject_learned_unet_orgdet_fnn_acf_sna_v1`**. Reason
`tp_covers_full_delta`. Overlay apply stays refuse-closed while the
appearance gate is unlocked. Frozen `6bba` not opened. No kernel. Parent
ACF champion and acf-test v1 unchanged. No threshold sweep. Leftovers
were not used as negatives.

A dt=1 dest vs spatial-second-NN appearance filter cannot move the
+0.002 gate on these ACF graphs. Unique-pred kinematics, second-NN
spatial margin, and dest vs second-NN appearance are now closed.

Runtime 17.89 s, peak 303 MB. Machine-readable copy:
`reports/learned_unet_orgdet_fnn_acf_sna_v1_diag.json`.
