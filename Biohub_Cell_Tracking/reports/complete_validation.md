# Complete specimen-separated validation

Harness: `scripts/run_complete_validation.py`, `scripts/run_holdout_pair.py`, and `biohub_tracking.validation`.

## What it scores

- Both frozen validation sequences from `docs/splits/v1.json`: `6bba_05db0fb1` (3 annotated divisions) and `6bba_c328f2fd` (2 annotated divisions).
- Official combined score via `tracking_cellmot.metrics` when volumes exist.
- Each annotated-division event via `tracking_cellmot.division_metrics.score_divisions` (local grandparent → parent → children → grandchildren window).

It refuses a rewritten v1 split and does not retune after seeing holdout.

## This environment (2026-09-20)

Both holdout GEFFs and Zarrs are present locally (helper kernels; ignored `data/train/`). Official pair:

| config | complete | official combined | divisions TP/FP/FN |
|---|---|---|---|
| association_v2 | true | 0.591570 | 0/0/5 |
| division_v2 | true | 0.591570 | 0/0/5 |

These are local official scores, not leaderboard scores. See `reports/holdout_pair_division_v2.md`.
