# Complete specimen-separated validation

Harness: `scripts/run_complete_validation.py` and `biohub_tracking.validation`.

## What it scores

- Both frozen validation sequences from `docs/splits/v1.json`: `6bba_05db0fb1` (3 annotated divisions) and `6bba_c328f2fd` (2 annotated divisions).
- Official combined score via `tracking_cellmot.metrics` when volumes exist.
- Each annotated-division event via `tracking_cellmot.division_metrics.score_divisions` (local grandparent → parent → children → grandchildren window).

It refuses a rewritten v1 split and does not retune after seeing holdout.

## This environment (2026-09-19)

Train volumes are not present. The harness therefore reports `complete=false` and does not invent an official score. Re-run after placing:

```
data/train/6bba_05db0fb1.zarr
data/train/6bba_05db0fb1.geff
data/train/6bba_c328f2fd.zarr
data/train/6bba_c328f2fd.geff
```

Earlier local official numbers (experiment branch, not this machine) remain in that branch’s `reports/abc_comparison.md`. They are local official scores, not leaderboard scores.
