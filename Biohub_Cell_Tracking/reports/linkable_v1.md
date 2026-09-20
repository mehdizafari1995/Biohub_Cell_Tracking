# linkable_v1 — temporally linkable complementary peaks

New experiment family after rejected `complementary_v1`. Does **not** overwrite
`baseline_v1`, `association_v2`, `division_v2`, or `complementary_v1`. Not a
Kaggle run. No leaderboard score.

Learned 3D UNet + transformer was estimated first and is **blocked** on this
worker (no GPU; organizer default split is not specimen-separated). See
`docs/learned_detection_estimate.md`. This family is the documented CPU fallback.

## Hypothesis

`complementary_v1` holdout recall was identical to association_v2; extra T_pred
(+312 nodes) lowered adjusted edge Jaccard. `linkable_v1` keeps the same
complementary spatial gates (copied, not retuned) and attaches a complementary
peak only when a **primary** detection in an adjacent frame sits inside the
locked 12 µm association gate. Complementary-to-complementary chains are not
used. Tune on `44b6` only.

## Operating point

`configs/linkable_v1.yaml`, pipeline hash `fa975421edb6136e`.
Locked from frozen **tuning** (`44b6` only), then scored **once** on frozen
`6bba`. Status: **rejected_on_holdout**. Do not retune.

## Official tuning score

Local official scorer `075fc5f5` on frozen tuning `44b6_12dfb391` +
`44b6_267148e4`. Shared association_v2 linker. **Not a Kaggle run. Not a
leaderboard score.** Runtime 279.8 s, peak RAM 1422 MB.

| candidate | official combined | edge J | adj edge J | recall | TP/FP/FN div | pred nodes |
|---|---|---|---|---|---|---|
| association_v2 | **0.632460** | 0.64514 | 0.63246 | 0.97758 | 0/0/2 | 93104 |
| linkable_v1 | **0.633168** | 0.64588 | 0.63317 | 0.97822 | 0/0/2 | 93133 |

Per-sequence official combined:

| dataset | association_v2 | linkable_v1 | pred nodes (B → F) |
|---|---|---|---|
| `44b6_12dfb391` | 0.667400 | **0.668404** | 66980 → 66992 |
| `44b6_267148e4` | **0.542891** | 0.542841 | 26124 → 26141 |

Tuning decision was `keep_candidate_linkable_v1` (+0.000708 aggregate).
Fewer extra nodes than rejected complementary_v1. That operating point was
locked. Holdout was not used to choose thresholds.

## Official holdout score (once)

Local official scorer `075fc5f5` on frozen validation `6bba_05db0fb1` +
`6bba_c328f2fd` after private helper `mahdimzafari/biohub-holdout-zarr-export`
v1. **Not a Kaggle run. Not a leaderboard score.** Runtime 285.6 s, peak RAM
1501 MB. Parent commit `2b2d2ff` plus this working tree. Config hashes:
association `0a97ad15532c26dc`, linkable `fa975421edb6136e`.

| candidate | official combined | edge J | adj edge J | recall | pred nodes | TP/FP/FN div |
|---|---|---|---|---|---|---|
| association_v2 | **0.591570** | 0.58881 | 0.59157 | 0.83824 | 96336 | 0/0/5 |
| linkable_v1 | 0.591458 | 0.58881 | 0.59146 | 0.83824 | 96525 | 0/0/5 |

Per-sequence official combined:

| dataset | association_v2 | linkable_v1 | det nodes (B → F) |
|---|---|---|---|
| `6bba_05db0fb1` | **0.564336** | 0.564314 | 61809 → 61837 |
| `6bba_c328f2fd` | **0.660194** | 0.659855 | 26591 → 26740 |

Decision: **`reject_linkable_v1`**. Combined score regressed (−0.000112).
Node recall and raw edge Jaccard were identical; extra T_pred (+189 nodes)
lowered adj-edge Jaccard. The temporal filter reduced extras versus
complementary_v1 (+312 nodes, 0.591384) but still recovered no annotated
cells. Do **not** retune the 12 µm / max_dt=1 gate after this result.
association_v2 remains the scoring candidate.

## Tests

`PYTHONPATH=src python -m pytest Biohub_Cell_Tracking/tests` — 109 passed
on this branch after the holdout lock and one-shot refusal.
