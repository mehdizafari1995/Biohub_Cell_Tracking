# learned_unet_orgdet_fnn_acf_scf_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or orgdet/elf/fnn/acf slugs. Do **not** retune. This is **not** another
12 µm assignment, **not** a rank/div/reas/ACF/BNN/IPC/GACF/GELF/TAC
retune, and **not** another 8.0 µm dt=1 length or 0.38 appearance retune.

## Hypothesis

Official 44b6 ACF residue was still association-dominated (160 conflict
FPs / 143 FNs; 0 predicted divisions). Locked ACF only drops dt=1
edges whose 3×3×3-mean relative appearance exceeds 0.38. ELF only
drops dt=1 edges longer than 8.0 µm. Gap-edge appearance GACF,
gap-edge length GELF, and dt=1 turning-angle TAC are closed.

Remaining conflict FPs are almost all dt=1. If official-matched TP
unique-predecessor dt=1 two-hops had smaller vector speed-change
|Δv| than conflict-FP two-hops, a single 0.1 µm/frame round-up of
TP max could drop those FPs without retuning ELF/FNN/ACF/TAC.
Leftover unmatched edges are **not** negatives and must not drive a
gate. Relative |Δv| is recorded, not gated. Official 44b6 would have
had to exceed ACF by **≥ 0.002000** (threshold **0.736620**) before
opening frozen `6bba`.

## Diagnostic (rejected; overlay not scored)

Config: `configs/learned_unet_orgdet_fnn_acf_scf_v1.yaml`. Hash
`a259d184c9f99046`. Parent ACF `a66075085ac64400`. Rejected TAC
`4dfc5514c65496e5`. `max_speed_change_um` stays **null**.

Local rebuild of locked ACF on `44b6_12dfb391` + `44b6_267148e4` using
orgop graphs plus the two exported tuning Zarrs. Unique-predecessor
dt=1 |Δv| vs official 7 µm sparse-GT hop class. Zero-length hops are
included (unlike TAC). **Not a Kaggle run. Not a leaderboard score.
Not an official overlay score.**

Unique-pred dt=1 speed-changes:

| class | n | min µm/frame | max µm/frame |
|---|---|---|---|
| TP | 855 | 0.0 | **5.997** |
| conflict FP | 145 | 0.0 | **4.596** |
| leftover | 79168 | — | 7.961 |

Relative |Δv| maxed at **1.0** for TP, conflict FP, and leftover, so
it is not a gate. Proposed one-shot gate = round-up of TP max =
**6.0 µm/frame**. Conflict FPs above that gate: **0**. TPs above that
gate: **0**. Leftovers above that gate: 126 (ignored as negatives).
No threshold sweep. Per-sequence:
`44b6_12dfb391` 672/74/56838 TP/conflict/leftover;
`44b6_267148e4` 183/71/22330. ACF rebuild counts match TAC
(64228/60906 and 29914/26249 nodes/edges).

Decision **`reject_learned_unet_orgdet_fnn_acf_scf_v1`**. Reason
`no_conflict_fp_above_gate`. Conflict-FP |Δv| sits **below** the TP
envelope, so a TP-max round-up cannot drop conflict FPs without also
dropping TPs. Overlay apply stays refuse-closed while the µm/frame
gate is unlocked. Frozen `6bba` not opened. No kernel. Parent ACF
champion and acf-test v1 unchanged. No threshold sweep. Leftovers
were not used as negatives.

A dt=1 unique-predecessor speed-change / acceleration filter cannot
move the +0.002 gate on these ACF graphs. Unique-pred kinematics
(turning angle and vector |Δv|) are now closed.

Runtime 13.14 s, peak 241 MB. Machine-readable copy:
`reports/learned_unet_orgdet_fnn_acf_scf_v1_diag.json`.
