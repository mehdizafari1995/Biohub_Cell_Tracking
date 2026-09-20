# Official holdout pair: association_v2 vs division_v2

Local official scorer `075fc5f5` on frozen `docs/splits/v1.json`.
Shared `baseline_v1` detections. **Not a Kaggle run. Not a leaderboard score.**
Do not retune locked gates after this result.

## Decision

`keep_division_v2` as a no-op conservative overlay:

- Combined score **identical** to locked association_v2: **0.5915697467955445**
- `division_fp == 0` (no extra false forks)
- `division_tp == 0` (none of 5 annotated events recovered)

association_v2 remains the scoring candidate. division_v2 did not regress
and did not help. Do not overwrite `association_v2.yaml` or `division_v2.yaml`.

## Per-sequence official combined

| dataset | GT divisions | association_v2 | division_v2 |
|---|---|---|---|
| `6bba_05db0fb1` | 3 | 0.564336 | 0.564336 |
| `6bba_c328f2fd` | 2 | 0.660194 | 0.660194 |
| **aggregate** | **5** | **0.591570** | **0.591570** |

Node recall 0.838 on both. Edge Jaccard 0.5888 / adj 0.5916 on both.
Runtime 139.4 s, peak RAM 761 MB, code `751587f` plus this runner.
Config hashes: association `2c6966b5204ba542`, division `0fa756963e5d42d1`.

## Why 0 TP (GEFF geometry already inventoried, unused for locking)

Holdout GT forks were inventoried before this score and were **not** used to
choose thresholds. After the score they explain the misses:

| event | parent–child µm | sibling µm | alignment | inside 10.0 / 8.3 / 0.05? |
|---|---|---|---|---|
| `6bba_05db0fb1` t=24 | 6.95 / 5.75 | 12.49 | −0.93 | yes |
| `6bba_05db0fb1` t=52 | 2.60 / 6.36 | 8.52 | −0.77 | yes |
| `6bba_05db0fb1` t=62 | 7.37 / 2.03 | 9.35 | −0.97 | yes |
| `6bba_c328f2fd` t=66 | 13.38 / 4.18 | 17.13 | −0.86 | no (parent–child) |
| `6bba_c328f2fd` t=28 | 11.21 / 12.32 | 14.92 | +0.20 | no (parent–child and alignment) |

Three of five annotated events sit inside the locked gates. They were still
missed because the predicted graph never produced a parent + orphan pair that
passed predecessor / no-alternative-parent checks. Widening gates after seeing
holdout would be leakage and would not fix those three.

## Tests

`PYTHONPATH=src python -m pytest Biohub_Cell_Tracking/tests` — 85 passed.
