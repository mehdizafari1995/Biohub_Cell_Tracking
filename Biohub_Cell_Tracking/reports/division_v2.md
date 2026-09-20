# division_v2 — conservative forks on locked association_v2

New experiment family. Does **not** overwrite `association_v2` or `division_v1`.
Not a Kaggle run. No leaderboard score.

## Hypothesis

`division_v1` attached any orphan within 9 µm of a one-child parent and used
`min_child_separation_um=6` (below every typical 44b6 sibling). Frozen
validation: **0 TP / 3 FP / 5 FN**, official combined 0.590 vs association_v2
0.592. False forks hurt the edge term.

`division_v2` keeps locked association_v2 detections and linking
(`max_children=1` during association) and adds a second child only when every
conservative check passes:

1. Parent already has a predecessor (not a birth).
2. Second child is an orphan (`in_degree=0`).
3. No other node at the parent time sits within the 12 µm association gate of
   that orphan (do not steal a plausible 1-1 continuation).
4. Both parent–child distances ≤ **10.0 µm**.
5. Sibling separation ≥ **8.3 µm**.
6. Parent→child displacements are not same-direction (`cosine ≤ 0.05`).
7. Optional intensity sum tolerance 0.45 if intensities are present.

Success when first official holdout is run: no combined-score regression vs
association_v2 (0.592) and `division_fp == 0` preferred. `TP > 0` is bonus.

## Priors (tuning specimen only, complete)

GEFF coordinates only. No validation images were opened. `6bba` GEFFs are
present for inventory and were **not** used to lock gates.

| set | n GEFF | n events | parent–child max / p90 | sibling min / p10 | alignment max |
|---|---|---|---|---|---|
| original three `44b6` | 3 | 3 | 8.38 / 7.74 | 8.75 / 8.90 | −0.66 |
| first lock (incomplete) | 17 | 4 | 8.38 / 7.51 | 8.75 / 8.97 | −0.66 |
| **all train `44b6`** | **71** | **26** (14 typical / 12 atypical) | typical 9.92 / 7.11 | typical 8.38 / 8.62 | typical −0.33 |

Typical = bipolar, `dt=1`, both daughters ≥ 1 µm, sibling ≥ 8.0 µm,
parent–child ≤ 10.0 µm, anti-aligned. Atypical leftovers: 6 close twins,
near-coincident daughters, one parent–child above 10 µm. An unfiltered
`lock_operating_point` on all 26 events would drop `min_child_separation_um`
to **8.0**; that lock is recorded as `unfiltered_lock_rejected` and is not used.

Locked operating point (`configs/division_v2.yaml`, config hash
`0fa756963e5d42d1`, `docs/division_priors.json` → `locked_operating_point`):

- `max_parent_child_um=10.0` (covers typical max 9.92 µm)
- `min_child_separation_um=8.3` (covers typical min 8.38 µm; floor 8.2)
- `max_sibling_alignment=0.05`
- `require_parent_predecessor=true`
- `require_no_alternative_parent=true`
- `alternative_parent_gate_um=12.0`
- `intensity_sum_tolerance=0.45`

Do not retune these gates after the first frozen-validation official score.

## Official score

Run once on the frozen split after a private helper kernel zipped only the two
holdout Zarrs (`mahdimzafari/biohub-holdout-zarr-export` v1). Shared
`baseline_v1` detections. Official scorer `075fc5f5`.

| candidate | official combined | TP / FP / FN |
|---|---|---|
| association_v2 | **0.591570** | 0 / 0 / 5 |
| division_v2 | **0.591570** | 0 / 0 / 5 |

Per-sequence combined: `6bba_05db0fb1` 0.564336, `6bba_c328f2fd` 0.660194
(identical on both candidates). Runtime 139.4 s, peak RAM 761 MB.

**Decision:** keep division_v2 as a no-op overlay. No combined-score regression
and no extra false forks. `TP > 0` did not happen. Do not retune gates.
This is not a Kaggle run and not a leaderboard number.

Three of five holdout GT events sit inside the locked gates and were still
missed (predicted topology / detections). The other two exceed 10 µm
parent–child. See `reports/holdout_pair_division_v2.md`.

## Tests

`PYTHONPATH=src python -m pytest Biohub_Cell_Tracking/tests` — 85 passed
(2026-09-20, this branch).
