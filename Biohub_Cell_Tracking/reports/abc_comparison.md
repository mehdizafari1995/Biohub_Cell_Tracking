# A/B/C candidate comparison

Locked detection for all three: anisotropic DoG + 4 µm NMS (`baseline_v1`). Official scorer `075fc5f5`. Frozen split `docs/splits/v1.json`. Previous registry rows were not overwritten.

Scores are **local official combined scores**, not Kaggle leaderboard numbers.

| | A baseline_v1 | B association_v2 | C division_v1 | D division_v2 |
|---|---|---|---|---|
| Linking | greedy 15 µm | greedy 12 µm gate + 2-frame gap close + intensity 0.5 | B + conservative two-child proposals | B + predecessor / anti-align / no-steal forks |
| Tuning official score | 0.585 | **0.632** | 0.630 | priors only (71 GEFFs) |
| Tuning raw / adj edge J | 0.590 / 0.585 | 0.645 / 0.632 | 0.643 / 0.630 | not scored |
| Tuning ID switches / fragments | 169 / 274 | 150 / 181 | (same detections) | not scored |
| Frozen val official score | 0.537 | **0.592** | 0.590 | not scored |
| Val raw / adj edge J | 0.530 / 0.537 | 0.589 / 0.592 | 0.587 / 0.590 | not scored |
| Val node recall | 0.804 | 0.838 | 0.838 | not scored |
| Val ID switches / fragments | 150 / 603 | 135 / 483 | — | not scored |
| Val divisions TP/FP/FN | 0/0/5 | 0/0/5 | 0/3/5 | not scored |
| Val division Jaccard | 0.0 | 0.0 | 0.0 | not scored |

Per-sequence frozen validation (official combined):

| dataset | A | B |
|---|---|---|
| `6bba_05db0fb1` (3 GT divisions) | 0.498 | 0.564 |
| `6bba_c328f2fd` (2 GT divisions) | 0.634 | 0.660 |

`6bba_05db0fb1` A matches the earlier complete-sequence aniso run `20260918T053718Z_val_aniso` (0.498). That previous number was one sequence, not the full split.

## Association ablations (tuning only, then one locked val pass)

Independent components vs greedy baseline (tuning official score 0.585, 169 switches):

| variant | official | raw J | adj J | switches | fragments | recall | keep? |
|---|---|---|---|---|---|---|---|
| constant-velocity | 0.533 | 0.538 | 0.533 | 209 | — | 0.948 | no |
| gate 12 µm | 0.609 | 0.615 | 0.609 | 164 | — | 0.948 | yes |
| gate 20 µm | 0.582 | 0.587 | 0.582 | 164 | — | 0.948 | no |
| gap close 1 | 0.600 | 0.609 | 0.600 | 162 | — | 0.964 | yes |
| gap close 2 | 0.622 | 0.634 | 0.622 | 151 | — | 0.975 | yes |
| gated Hungarian | 0.458 | 0.462 | 0.458 | 242 | — | 0.948 | no |
| intensity cost | 0.603 | 0.608 | 0.603 | 159 | — | 0.948 | yes |
| start/end penalties | 0.529 | 0.534 | 0.529 | 194 | — | 0.948 | no |
| boundary-aware LAP | 0.547 | 0.552 | 0.547 | 193 | — | 0.948 | no |
| **association_v2** (12 µm + gap2 + intensity, greedy) | **0.632** | 0.645 | 0.632 | 150 | 181 | 0.978 | locked |

Hungarian was never applied to all detections; ungated pairs were excluded. Even that gated Hungarian still lost to greedy on this dense field.

## Division

GEFF-only priors (no validation images during threshold choice): every annotated division is `dt=1`. Tuning three events: parent–child median 5.2 µm, p90 7.7 µm, sibling min 8.75 µm. Locked `max_parent_child_um=9`, `min_child_separation_um=6`.

On frozen validation, division_v1 scored **0 TP / 3 FP / 5 FN**. Combined score dropped vs B (0.590 vs 0.592) because false forks hurt the edge term. n=5 annotated events (3+2). Do not claim generalization.

Visual inspection of the three `6bba_05db0fb1` events (t=24, 52, 62): dense MIP, sparse GT circles, predictions nearby. Two `6bba_c328f2fd` daughters sit 12–13 µm from the parent, outside the 9 µm tuning prior. Raising that gate after seeing validation would be leakage.

## D `division_v2` (this family)

Locked from 17 complete `44b6` GEFFs (4 events) only. See `reports/division_v2.md`. Official holdout has not been run because the two validation Zarr volumes are not on this machine. Do not invent a score.

## Sanity sequence

`docs/splits/v2.json` adds `44b6_2a2eff9f` (100 frames, 1 GT division). Same specimen prefix as tuning. Locked association_v2 official score 0.801, recall 0.995, 16 switches. This is **not** a third-embryo holdout; the downloaded GEFF inventory only has prefixes `44b6` and `6bba`.

## Kaggle

A has a real Kaggle kernel execution (`mahdimzafari/biohub-baseline-v1` v2). B now also has a real private internet-off kernel execution (`mahdimzafari/biohub-association-v2` v3, 593.53 s, `audit_ok`) on public `test/`. The downloaded worker CSV was independently audited (`reports/kaggle_association_v2_v3_independent_audit.md`): schema `audit_ok`, and official score **0.564336** on public-test overlap `6bba_05db0fb1` (0/0/3 divisions), matching the local 0.564 figure. That is not a leaderboard score. `6bba_c328f2fd` is not in public test. Extra public-test stems (`44b6_*`, `6bba_05b6850b`) are not holdout and were not used to retune. C did not improve locally and has no kernel. D is local-only so far. No competition submit was performed. No public or private leaderboard score was returned.
