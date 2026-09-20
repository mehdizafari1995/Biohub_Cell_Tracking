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

## Candidate (not locked)

`configs/complementary_v1.yaml`, pipeline hash `2c4866928d7262b6`.
`docs/complementary_priors.json` still has `locked_operating_point: null`.

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

Decision: `keep_candidate_complementary_v1`. Aggregate did not regress
(+0.000675). The first sequence recovered one extra annotated node and one
edge; the second added 54 nodes with unchanged recall and a small adj-edge
penalty. Do **not** lock yet. Do **not** score `6bba` until a lock is recorded.

## Helper kernel

Private `mahdimzafari/biohub-tuning-zarr-export` version 1 **COMPLETE**.
Zipped only the two frozen-tuning Zarrs (919 MB worker archive). Not a
submission. Do not overwrite baseline_v1 / association_v2 / geff-export /
holdout-zarr-export.

## Tests

`PYTHONPATH=src python -m pytest Biohub_Cell_Tracking/tests` — 99+ passed
on this branch after the complementary family was added.
