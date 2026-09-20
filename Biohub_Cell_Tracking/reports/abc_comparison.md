# A/B/C candidate comparison

Locked detection for all three: anisotropic DoG + 4 µm NMS (`baseline_v1`). Official scorer `075fc5f5`. Frozen split `docs/splits/v1.json`. Previous registry rows were not overwritten.

Scores are **local official combined scores**, not Kaggle leaderboard numbers.

| | A baseline_v1 | B association_v2 | C division_v1 | D division_v2 |
|---|---|---|---|---|
| Linking | greedy 15 µm | greedy 12 µm gate + 2-frame gap close + intensity 0.5 | B + conservative two-child proposals | B + predecessor / anti-align / no-steal forks |
| Tuning official score | 0.585 | **0.632** | 0.630 | priors only (71 GEFFs) |
| Tuning raw / adj edge J | 0.590 / 0.585 | 0.645 / 0.632 | 0.643 / 0.630 | not scored |
| Tuning ID switches / fragments | 169 / 274 | 150 / 181 | (same detections) | not scored |
| Frozen val official score | 0.537 | **0.592** | 0.590 | **0.592** (identical to B) |
| Val raw / adj edge J | 0.530 / 0.537 | 0.589 / 0.592 | 0.587 / 0.590 | 0.589 / 0.592 |
| Val node recall | 0.804 | 0.838 | 0.838 | 0.838 |
| Val ID switches / fragments | 150 / 603 | 135 / 483 | — | same detections as B |
| Val divisions TP/FP/FN | 0/0/5 | 0/0/5 | 0/3/5 | 0/0/5 |
| Val division Jaccard | 0.0 | 0.0 | 0.0 | 0.0 |

Per-sequence frozen validation (official combined):

| dataset | A | B | D |
|---|---|---|---|
| `6bba_05db0fb1` (3 GT divisions) | 0.498 | 0.564 | 0.564 |
| `6bba_c328f2fd` (2 GT divisions) | 0.634 | 0.660 | 0.660 |

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

Locked from all 71 train `44b6` GEFFs (14 typical / 12 atypical). Official
holdout vs association_v2 on identical detections: **0.591570** both,
**0 TP / 0 FP / 5 FN**. Keep as a no-op overlay. Do not retune. See
`reports/division_v2.md` and `reports/holdout_pair_division_v2.md`.

## E `complementary_v1` (rejected on holdout)

Locked complementary local-max on locked baseline_v1 detections + locked
association_v2 linker. Official tuning combined **0.633135** vs association_v2
**0.632460**. Official holdout **0.591384 vs 0.591570**. Recall unchanged;
extra T_pred (+312 nodes). Rejected. Do not retune. See
`reports/complementary_v1.md` and `reports/complementary_v1_holdout.json`.

## G `learned_unet_holdout_v1` (locked; one-shot holdout done)

Specimen-separated TemporalUNet3D + 12 µm 1-1 linker. Official 44b6 tune
**0.655751** vs association_v2 **0.632460**. Official frozen holdout
**0.693605 vs association_v2 0.591570**. Recall 0.957, 0/0/5 divisions,
149223 pred nodes. Per-sequence mixed (`6bba_05db0fb1` 0.719 vs 0.564;
`6bba_c328f2fd` 0.633 vs 0.660). Combined keep. Decision
`keep_candidate_learned_unet_holdout_v1`. Do not retune. Do not score the
no-gap holdout candidate again. See `reports/learned_unet_holdout_v1.md`.

## H `learned_unet_gap_v1` (kept on one-shot holdout)

Locked association_v2 gap-close overlay (12 µm, `max_gap_frames=2`,
interpolate) on leftover unmatched nodes of locked learned graphs. Official
44b6 tune **0.658452** vs learned infer **0.655751**. Official frozen
holdout **0.701026 vs no-gap 0.693605**. Recall 0.976, 0/0/5 divisions,
159131 pred nodes, 16804 extra edges, 9908 interpolated nodes. Per-sequence
mixed (`6bba_05db0fb1` 0.718 vs 0.719; `6bba_c328f2fd` 0.661 vs 0.633).
Combined keep. Decision `keep_candidate_learned_unet_gap_v1`. Do not
retune. Do not score this overlay on `6bba` again. See
`reports/learned_unet_gap_v1.md`.

## Sanity sequence

`docs/splits/v2.json` adds `44b6_2a2eff9f` (100 frames, 1 GT division). Same specimen prefix as tuning. Locked association_v2 official score 0.801, recall 0.995, 16 switches. This is **not** a third-embryo holdout; the downloaded GEFF inventory only has prefixes `44b6` and `6bba`.

## Kaggle

A has a real Kaggle kernel execution (`mahdimzafari/biohub-baseline-v1` v2). B now also has a real private internet-off kernel execution (`mahdimzafari/biohub-association-v2` v3, 593.53 s, `audit_ok`) on public `test/`. The downloaded worker CSV was independently audited (`reports/kaggle_association_v2_v3_independent_audit.md`): schema `audit_ok`, and official score **0.564336** on public-test overlap `6bba_05db0fb1` (0/0/3 divisions), matching the local 0.564 figure. That is not a leaderboard score. `6bba_c328f2fd` is not in public test. Extra public-test stems (`44b6_*`, `6bba_05b6850b`) are not holdout and were not used to retune. C did not improve locally and has no kernel. D is local-only and identical to B on holdout. E is local-only and worse than B on holdout. G has a private GPU holdout helper (`mahdimzafari/biohub-learned-unet-holdout-v1` v1, 184.4 s) that wrote graphs only, not `submission.csv`. A new private GPU test-time notebook (`mahdimzafari/biohub-learned-unet-test-v1` v1, 306.5 s, peak 1184 MB, hash `7a8725b818fc609b`) discovered all four public `test/*.zarr` stems and wrote `submission.csv` (`audit_ok`, 231707 nodes / 208547 edges / 0 divisions). That is the no-gap public-test worker, not a leaderboard score. H (`learned_unet_gap_v1`) is local-official on locked holdout graphs (9.7 s / 845 MB). A new private GPU test-time notebook (`mahdimzafari/biohub-learned-unet-gap-test-v1` v1, 330.0 s, peak 1184 MB, hash `85379c198ebc7241`) applied the same locked overlay on public `test/` and wrote `submission.csv` (`audit_ok`, 251660 nodes / 242412 edges / 0 divisions). `6bba_05db0fb1` counts match the locked holdout gap overlay. That is not a leaderboard score. No competition submit was performed by this worker.
