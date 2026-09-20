# Local runs (not a Kaggle submission)

No notebook has been submitted to Kaggle. No leaderboard score is claimed.

## Official scorer

**Available.** Vendored from https://github.com/royerlab/kaggle-cell-tracking-competition commit `075fc5f5a52d11077f9dc2b074644618f26939e2` (BSD-3-Clause). Matching dependency: `tracksdata==0.1.0rc10` @ `7523bb39ed9ab1bf88512911fd6c69e46fcad188`, `polars==1.44.2`, `geff==1.3.1.1.3`. Torch is not imported.

Reported competition-style scores use `tracking_cellmot.metrics.evaluate` + `per_sample_metrics` + `summarise` (the same aggregation as the organizer `scripts/evaluate.py`). `biohub_tracking.evaluate` is a labelled diagnostic. On every compared prediction in this repo the two agreed on edge TP/FP/FN and Jaccards (gap 0).

### Combined score vs Edge Jaccard

```
score = adj_edge_jaccard + 0.1 × division_jaccard   # division term dropped if TP+FP+FN = 0
adj_edge_jaccard = max(0, edge_jaccard · (1 − 0.1 · (N_pred − N_true) / N_true))
```

The Edge Jaccard column is **raw**. The combined score uses **adjusted** edge Jaccard. That is why a run can show raw `0.395` and combined `0.387`. `evaluate_datasets` in the official repo adds raw Jaccard into `score` and is not used here.

## Coordinate audit

See `docs/coordinate_audit.md`. Matching is per integer `t` on physical `(z,y,x) × (1.625, 0.40625, 0.40625)` µm, 7 µm gate. OME-Zarr T scale `1.0` unit `second` is metadata, not a confirmed 1 Hz interval. Unmatched predictions are not node FPs.

## GEFF inventory and frozen split

16 train GEFFs downloaded before selecting images. Five contain annotated divisions. Kaggle file listing was interrupted by HTTP 429; unique-store count is therefore **incomplete**.

Frozen before tuning in `docs/splits/v1.json` (specimen prefix `44b6` vs `6bba`; finer embryo identity is not in the file names):

| role | dataset | annotated nodes | edges | divisions | estimated_n_nodes | complete zarr locally |
|---|---|---|---|---|---|---|
| tuning | `44b6_12dfb391` | 788 | 773 | 1 | 58672 | 42 / 100 frames (429) |
| tuning | `44b6_267148e4` | 280 | 277 | 1 | 19130 | 0 / 100 |
| validation | `6bba_05db0fb1` | 1229 | 1183 | 3 | 69800 | **100 / 100** |
| validation | `6bba_c328f2fd` | 511 | 500 | 2 | 31228 | **100 / 100** |

Complete-sequence official numbers below are **one validation video** (`6bba_05db0fb1`), not the full split and not the hidden test.

## Partial-volume diagnostic (earlier crops; not complete-sequence)

| experiment | config | official score | raw edge J | adj edge J | notes |
|---|---|---|---|---|---|
| `20260918T051429Z_baseline_official` | greedy baseline | 0.382 | 0.390 | 0.382 | 12+3 frames, 0 GT divisions |
| `20260918T051450Z_hungarian_official` | Hungarian | 0.387 | 0.395 | 0.387 | same crops |

## Tuning (`44b6_12dfb391`, 42 available frames, official scorer)

Chosen **before** looking at validation. Hungarian linking was worse (more conflicting edges). Physical anisotropic DoG + 4 µm NMS won.

| experiment | config | official score | raw edge J | adj | node recall | ID switches | broken | runtime s | RAM MB |
|---|---|---|---|---|---|---|---|---|---|
| `tune_baseline` | `baseline.yaml` | 0.556 | 0.549 | 0.556 | 0.941 | 70 | 101 | 33.3 | 661 |
| `tune_hungarian` | `hungarian.yaml` | 0.415 | 0.410 | 0.415 | 0.941 | 102 | 133 | 33.2 | 665 |
| `tune_nms` | `physical_nms.yaml` | 0.665 | 0.652 | 0.665 | 0.941 | 45 | 76 | 32.6 | 637 |
| **selected** `tune_aniso` | `physical_aniso.yaml` | **0.719** | 0.712 | 0.719 | 0.951 | 35 | 61 | 33.2 | 665 |

The GT division on this sequence is not in the downloaded 42 frames, so the division term was dropped here.

## Complete-sequence validation (`6bba_05db0fb1`, T=100, 3 GT divisions)

Official aggregation on this one sample. **Not a leaderboard score.**

| experiment | config | official score | raw edge J | adj edge J | div J (TP/FP/FN) | node recall | n_pred | missed GT | ID switches | runtime s | peak RAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `20260918T053433Z_val_baseline` | greedy DoG | 0.428 | 0.419 | 0.428 | 0.0 (0/0/3) | 0.710 | 54472 | 356 | 119 | 78.8 | 1062 |
| `20260918T053556Z_val_hungarian` | Hungarian | 0.364 | 0.356 | 0.364 | 0.0 (0/0/3) | 0.710 | 54472 | 356 | 158 | 77.6 | 1079 |
| **`20260918T053718Z_val_aniso`** | anisotropic DoG + NMS, greedy | **0.498** | 0.492 | 0.498 | 0.0 (0/0/3) | 0.770 | 61809 | 283 | 102 | 79.3 | 1141 |

Oracle (GT centroids, greedy 15 µm, no division edges): official edge Jaccard **0.592**, not 1.0, with 319 identity switches. Perfect detections do not yield reliable linking in this dense field. Baseline `allow_divisions: false` recovers **zero** of the three annotated divisions even with GT centroids.

Unmatched predictions (~60k vs 1229 annotated nodes) are not node false positives. `estimated_number_of_nodes=69800`.

Error overlays: `docs/figures/errors/` including `6bba_05db0fb1_val_aniso_overlay.png` and `6bba_05db0fb1_errors_t50.png`.

## Kaggle path

`notebooks/kaggle_inference.py` uses the selected `physical_aniso` detection settings and greeds links. It globbs `test/*.zarr` (no hard-coded public names).

**Local simulation** (`scripts/simulate_kaggle.py`, outbound TCP blocked): 1 complete sequence, `submission.csv` rows=119080, schema checks passed, 0.583 s/frame. Estimate if hidden test has 200 sequences of T=100: 3.24 h, **6.48 h with 2× margin**, under the 12 h limit. The 200-sequence count is **unverified** (inventory 429). Using the slower eval-path 0.79 s/frame × 200 × 100 × 2 = 8.8 h, still under 12 h. This simulation is **not** a Kaggle execution.

Exact submit steps: `docs/kaggle_submit.md`.

## Remaining blockers

- Full train GEFF/zarr inventory incomplete (Kaggle 429).
- Two frozen-split volumes not fully downloaded (`44b6_267148e4`, rest of `44b6_12dfb391` / `6bba_c328f2fd`).
- Official score on the **full** validation split (two sequences) not yet available.
- Greedy one-child linking never predicts divisions; division Jaccard stays 0.
- Dense-field identity switches remain after the detection improvement.
- No actual Kaggle scoring rerun.

## Later update (2026-09-18, after the notes above)

Split volumes listed as incomplete above were finished (all four v1 zarrs are 100/100). `docs/splits/v1.json` was not changed. `docs/splits/v2.json` adds sanity sequence `44b6_2a2eff9f` only.

### Actual Kaggle kernel (not a competition submit)

Private kernel `mahdimzafari/biohub-baseline-v1` **version 1** failed: no `zarr` package. **Version 2 COMPLETE** on public test with internet off. Worker numbers from `run_manifest.json`: 595.24 s, 1.49 s/frame, 435 MB, 224788 rows, audit_ok. Decoder `blosc2 4.1.2`. No public or private leaderboard score was returned. Manual submit click: `docs/kaggle_submit.md`.

Runtime projection from **this Kaggle run** (not the local 0.583 s/frame simulation): 1.49 s/frame. If hidden test ≈ train (16 sequences × 100 frames) ≈ 40 minutes. If someone still assumes 200 × 100 frames: 8.3 h, and 2× margin exceeds 12 h. The 200-sequence figure remains unverified.

### Full frozen validation (two sequences, official scorer)

New runs; previous tables above are unchanged.

| experiment | config | official | raw J | adj | div J (TP/FP/FN) | recall | switches | fragments |
|---|---|---|---|---|---|---|---|---|
| `val_baseline_v1` | A greedy 15 µm | 0.537 | 0.530 | 0.537 | 0.0 (0/0/5) | 0.804 | 150 | 603 |
| `val_association_v2` | B 12 µm + gap2 + intensity | **0.592** | 0.589 | 0.592 | 0.0 (0/0/5) | 0.838 | 135 | 483 |
| `division_v1_val` | C + two-child proposals | 0.590 | 0.587 | 0.590 | 0.0 (0/3/5) | 0.838 | — | — |

`6bba_05db0fb1` A is 0.498, matching `20260918T053718Z_val_aniso`. Comparison write-up: `reports/abc_comparison.md`.

