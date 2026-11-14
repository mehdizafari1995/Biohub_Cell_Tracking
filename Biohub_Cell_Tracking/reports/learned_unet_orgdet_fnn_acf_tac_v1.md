# learned_unet_orgdet_fnn_acf_tac_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or orgdet/elf/fnn/acf slugs. Do **not** retune. This is **not** another
12 µm assignment, **not** a rank/div/reas/ACF/BNN/IPC/GACF/GELF retune,
and **not** another 8.0 µm dt=1 length or 0.38 appearance retune.

## Hypothesis

Official 44b6 ACF residue was still association-dominated (160 conflict
FPs / 143 FNs; 0 predicted divisions). Locked ACF only drops dt=1
edges whose 3×3×3-mean relative appearance exceeds 0.38. ELF only
drops dt=1 edges longer than 8.0 µm. Gap-edge appearance GACF and
gap-edge length GELF are closed.

Remaining conflict FPs are almost all dt=1. If official-matched TP
unique-predecessor dt=1 two-hops were more collinear than conflict-FP
two-hops, a single 1-degree round-up of TP max turning angle could
drop those FPs without retuning ELF/FNN/ACF. Leftover unmatched edges
are **not** negatives and must not drive a gate. Official 44b6 would
have had to exceed ACF by **≥ 0.002000** (threshold **0.736620**)
before opening frozen `6bba`.

## Diagnostic (rejected; overlay not scored)

Config: `configs/learned_unet_orgdet_fnn_acf_tac_v1.yaml`. Hash
`4dfc5514c65496e5`. Parent ACF `a66075085ac64400`. Rejected GELF
`59812f90385ed4d2`. `max_turn_deg` stays **null**.

Local rebuild of locked ACF on `44b6_12dfb391` + `44b6_267148e4` using
orgop graphs plus the two exported tuning Zarrs. Unique-predecessor
dt=1 two-hop turning angle vs official 7 µm sparse-GT hop class.
Degenerate zero-length hops are not gated. **Not a Kaggle run. Not a
leaderboard score. Not an official overlay score.**

Measurable (non-degenerate) unique-pred dt=1 turns:

| class | n | min deg | max deg | min cosine |
|---|---|---|---|---|
| TP | 563 | 0.0 | **180.0** | **-1.0** |
| conflict FP | 112 | 0.0 | 180.0 | — |
| leftover | 53564 | — | 180.0 | — |

Also 25929 degenerate (zero-length) hops, not used as a gate. Total
classified two-hops 80168. Per-sequence including degenerates:
`44b6_12dfb391` 672/74/56838 TP/conflict/leftover;
`44b6_267148e4` 183/71/22330. ACF rebuild counts match GELF
(64228/60906 and 29914/26249 nodes/edges).

Decision **`reject_learned_unet_orgdet_fnn_acf_tac_v1`**. Reason
`tp_covers_full_angle`. Official-matched TPs already include 180°
U-turns, so no 1-degree round-up of TP max can sit below any conflict
FP without also dropping TPs. Overlay apply stays refuse-closed while
the degree gate is unlocked. Frozen `6bba` not opened. No kernel.
Parent ACF champion and acf-test v1 unchanged. No threshold sweep.
Leftovers were not used as negatives.

A dt=1 turning-angle / collinearity filter cannot move the +0.002
gate on these ACF graphs.

Runtime 15.50 s, peak 230 MB. Machine-readable copy:
`reports/learned_unet_orgdet_fnn_acf_tac_v1_diag.json`.
