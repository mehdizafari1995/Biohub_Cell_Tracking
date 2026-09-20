# complementary_v1 — complementary 3D detection on locked association_v2

New experiment family. Does **not** overwrite `baseline_v1`, `association_v2`,
`division_v2`, or rejected `division_v1`. Not a Kaggle run. No leaderboard score.

## Hypothesis

Official holdout of locked association_v2 has node recall 0.838 versus 0.978 on
tuning. Three of five annotated holdout forks sit inside locked division_v2
gates and were still missed because the predicted graph never produced a
parent + orphan pair. Widening those gates after seeing holdout would be
leakage and would not fix topology that lacks the second cell.

`complementary_v1` keeps the locked anisotropic DoG + 4 µm NMS primary detector
and the locked association_v2 linker (`max_children=1`). A second local-max
pass may add a peak only when:

1. It is at least **6.0 µm** from every primary detection (larger than the 4 µm
   NMS, so not a duplicate of a crowded neighbor).
2. Its intensity is at least **0.45 ×** the primary median intensity.
3. Remaining complementary peaks are self-NMS'd at 4 µm.

## Operating point

`configs/complementary_v1.yaml`, pipeline hash `2c4866928d7262b6`.
Locked from frozen **tuning** (`44b6` only), then scored **once** on frozen
`6bba`. Status: **rejected_on_holdout**. Do not retune.

## Official tuning score

Local official scorer `075fc5f5` on frozen tuning `44b6_12dfb391` +
`44b6_267148e4`. Shared association_v2 linker. **Not a Kaggle run. Not a
leaderboard score.** Runtime 267.7 s, peak RAM 656 MB.

| candidate | official combined | edge J | adj edge J | recall | TP/FP/FN div |
|---|---|---|---|---|---|
| association_v2 | **0.632460** | 0.64514 | 0.63246 | 0.97758 | 0/0/2 |
| complementary_v1 | **0.633135** | 0.64588 | 0.63313 | 0.97822 | 0/0/2 |

Per-sequence official combined:

| dataset | association_v2 | complementary_v1 | pred nodes (A → E) |
|---|---|---|---|
| `44b6_12dfb391` | 0.667400 | **0.668400** | 66980 → 66995 |
| `44b6_267148e4` | **0.542891** | 0.542732 | 26124 → 26178 |

Tuning decision was `keep_candidate_complementary_v1` (+0.000675 aggregate).
That operating point was locked. Holdout was not used to choose thresholds.

## Official holdout score (once)

Local official scorer `075fc5f5` on frozen validation `6bba_05db0fb1` +
`6bba_c328f2fd` after private helper `mahdimzafari/biohub-holdout-zarr-export`
v1. **Not a Kaggle run. Not a leaderboard score.** Runtime 275.1 s, peak RAM
758 MB. Code `d735680`. Config hashes: association `0a97ad15532c26dc`,
complementary `2c4866928d7262b6`.

| candidate | official combined | edge J | adj edge J | recall | pred nodes | TP/FP/FN div |
|---|---|---|---|---|---|---|
| association_v2 | **0.591570** | 0.58881 | 0.59157 | 0.83824 | 96336 | 0/0/5 |
| complementary_v1 | 0.591384 | 0.58881 | 0.59138 | 0.83824 | 96648 | 0/0/5 |

Per-sequence official combined:

| dataset | association_v2 | complementary_v1 | det nodes (B → E) |
|---|---|---|---|
| `6bba_05db0fb1` | **0.564336** | 0.564313 | 61809 → 61839 |
| `6bba_c328f2fd` | **0.660194** | 0.659601 | 26591 → 26845 |

Decision: **`reject_complementary_v1`**. Combined score regressed (−0.000185).
Node recall and raw edge Jaccard were identical; extra T_pred (+312 nodes)
lowered adj-edge Jaccard. Identity switches stayed 135 / 483. No extra
annotated cells were recovered. Do **not** retune isolation or intensity
gates after this result. association_v2 remains the scoring candidate.

## Helper kernels

Private `mahdimzafari/biohub-tuning-zarr-export` version 1 **COMPLETE**.
Zipped only the two frozen-tuning Zarrs (919 MB worker archive). Not a
submission.

Private `mahdimzafari/biohub-holdout-zarr-export` version 1 **COMPLETE**.
Zipped only the two frozen-holdout Zarrs. Reused for this holdout score.
Not a submission. Do not overwrite baseline_v1 / association_v2 /
geff-export / holdout-zarr-export / tuning-zarr-export.

## Tests

`PYTHONPATH=src python -m pytest Biohub_Cell_Tracking/tests` — 101 passed
on this branch after the holdout lock and one-shot refusal.
