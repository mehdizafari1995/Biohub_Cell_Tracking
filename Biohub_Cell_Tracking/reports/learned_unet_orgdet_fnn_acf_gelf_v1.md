# learned_unet_orgdet_fnn_acf_gelf_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or orgdet/elf/fnn/acf slugs. Do **not** retune. This is **not** another
12 µm assignment, **not** a rank/div/reas/ACF/BNN/IPC/GACF retune, and
**not** another 8.0 µm dt=1 length retune.

## Hypothesis

Official 44b6 ACF residue was still association-dominated (160 conflict
FPs / 143 FNs; 0 predicted divisions). Locked ACF only drops dt=1
edges whose 3×3×3-mean relative appearance exceeds 0.38. ELF only
drops dt=1 edges longer than 8.0 µm. Gap-edge appearance GACF is
closed (leftover unmatched only).

Gap-close still links leftover detections across dt∈{2,3}. If official
matched TP gap chains were shorter than conflict-FP gap chains, a
single 0.1 µm round-up of TP max could drop those FPs without touching
dt=1 ACF/FNN/ELF. Leftover unmatched chains are **not** negatives and
must not drive a gate. Official 44b6 would have had to exceed ACF by
**≥ 0.002000** (threshold **0.736620**) before opening frozen `6bba`.

## Diagnostic (rejected; overlay not scored)

Config: `configs/learned_unet_orgdet_fnn_acf_gelf_v1.yaml`. Hash
`59812f90385ed4d2`. Parent ACF `a66075085ac64400`. Rejected GACF
`5ce49a6d175c8032`. `max_gap_edge_um` stays **null**.

Local rebuild of locked ACF on `44b6_12dfb391` + `44b6_267148e4` using
orgop graphs plus the two exported tuning Zarrs. Official 7 µm
sparse-GT hop classes on gap chains (pre-gap sources through
interpolants, dt 2..3). **Not a Kaggle run. Not a leaderboard score.
Not an official overlay score.**

| class | n | min µm | max µm |
|---|---|---|---|
| TP | 14 | 0.0 | **8.751** |
| conflict FP | 4 | 0.0 | **5.629** |
| leftover | 2235 | — | 20.619 |

Per-dt: dt=2 has 10/3/1471 TP/conflict/leftover; dt=3 has 4/1/764.
Per-sequence: `44b6_12dfb391` 7/1/1243; `44b6_267148e4` 7/3/992.
Proposed one-shot gate = round-up of TP max = **8.8 µm**. Conflict FPs
above that gate: **0**. Leftovers above that gate: 262 (ignored as
negatives). No threshold sweep.

Decision **`reject_learned_unet_orgdet_fnn_acf_gelf_v1`**. Reason
`no_conflict_fp_above_gate`. Overlay apply stays refuse-closed while
the µm gate is unlocked. Frozen `6bba` not opened. No kernel. Parent
ACF champion and acf-test v1 unchanged.

Remaining ACF conflict FPs are almost all dt=1: gap-close contributed
only 4 of 160, and those 4 are shorter than the TP envelope. Another
gap-edge filter cannot move the +0.002 gate.

Machine-readable copy:
`reports/learned_unet_orgdet_fnn_acf_gelf_v1_diag.json`.
