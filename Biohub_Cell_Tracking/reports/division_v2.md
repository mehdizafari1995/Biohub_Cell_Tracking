# division_v2 — conservative forks on locked association_v2

New experiment family. Does **not** overwrite `association_v2` or `division_v1`.
Not a Kaggle run. No leaderboard score.

## Hypothesis

`division_v1` attached any orphan within 9 µm of a one-child parent and used
`min_child_separation_um=6` (below every tuning GT sibling ≥ 8.75 µm). Frozen
validation: **0 TP / 3 FP / 5 FN**, official combined 0.590 vs association_v2
0.592. False forks hurt the edge term.

`division_v2` keeps locked association_v2 detections and linking
(`max_children=1` during association) and adds a second child only when every
conservative check passes:

1. Parent already has a predecessor (not a birth).
2. Second child is an orphan (`in_degree=0`).
3. No other node at the parent time sits within the 12 µm association gate of
   that orphan (do not steal a plausible 1-1 continuation).
4. Both parent–child distances ≤ 8.5 µm (covers every observed 44b6 event).
5. Sibling separation ≥ 8.7 µm (just below the 44b6 minimum 8.75 µm).
6. Parent→child displacements are not same-direction (`cosine ≤ 0.05`).
   Observed 44b6 alignments are all ≤ −0.66 when both daughters move ≥ 1 µm.
7. Optional intensity sum tolerance 0.45 if intensities are present.

Success when first official holdout is run: no combined-score regression vs
association_v2 (0.592) and `division_fp == 0` preferred. `TP > 0` is bonus.

## Priors (tuning specimen only)

GEFF coordinates only. No validation images were opened. `6bba` events are
inventoried as historical and were **not** used to lock gates.

| set | n GEFF | n events | parent–child max / p90 | sibling min / p10 | alignment max |
|---|---|---|---|---|---|
| original three `44b6` | 3 | 3 | 8.38 / 7.74 | 8.75 / 8.90 | −0.66 |
| this run `44b6` complete stores | 17 | 4 | 8.38 / 7.51 | 8.75 / 8.97 | −0.66 |

Fourth event: `44b6_341df25f`. All annotated forks have `dt=1`. 54 remaining
train `44b6` GEFFs were not complete when the lock was written. Expand those
**before** the first frozen-validation official score; do not retune after
holdout.

Locked operating point (`configs/division_v2.yaml`,
`docs/division_priors.json` → `locked_operating_point`):

- `max_parent_child_um=8.5`
- `min_child_separation_um=8.7`
- `max_sibling_alignment=0.05`
- `require_parent_predecessor=true`
- `require_no_alternative_parent=true`
- `alternative_parent_gate_um=12.0`
- `intensity_sum_tolerance=0.45`

## Official score

Not run this cycle. Train volumes for `6bba_05db0fb1` and `6bba_c328f2fd` are
not on this machine (~1.0 GB compressed). A local simulation is not a Kaggle
run. Do not invent a leaderboard number.

## Tests

`PYTHONPATH=src python -m pytest Biohub_Cell_Tracking/tests` after this change.
See `docs/PROJECT_STATE.md` for the recorded count.
