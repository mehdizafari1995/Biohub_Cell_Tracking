# Project state

Last updated: 2026-09-20 (linkable_v1 rejected on holdout). Authoritative implementation: `Biohub_Cell_Tracking/`.

## Repository

- Private GitHub repo: `mehdizafari1995/Biohub_Cell_Tracking`
- Base branch: `main` (baseline + official scorer + privacy audit + coordination docs + complete-validation harness)
- Association experiment (keep open, do not merge): `cursor/kaggle-baseline-association-0b83` (PR #2)
- Division experiment (keep open, do not merge): `cursor/competition-operational-strategy-3381` (PR #4). Official holdout done; do not retune.
- Complementary experiment (keep open, do not merge): `cursor/complementary-detection-v1` (PR #5). Locked from 44b6, scored once on 6bba, **rejected**.
- Linkable experiment (keep open, do not merge): `cursor/linkable-complementary-v1`. Locked from 44b6, scored once on 6bba, **rejected**.
- Command channel: issue #1 (this worker still gets 403 on issues)

## What is locked

- Frozen embryo-level split `docs/splits/v1.json`: tune on specimen `44b6`, validate on `6bba`. Do not retune after seeing validation.
- `docs/splits/v2.json` only adds sanity sequence `44b6_2a2eff9f` (same specimen prefix as tuning, not a third embryo).
- Official scorer: vendored `royerlab/kaggle-cell-tracking-competition` @ `075fc5f5a52d11077f9dc2b074644618f26939e2`. Dependencies: `tracksdata==0.1.0rc10`, `polars==1.44.2`, `geff==1.3.1.1.3`.
- Frozen greedy baseline: `configs/baseline.yaml`
- Kaggle kernel config (anisotropic DoG + 4 µm NMS + greedy 15 µm): `configs/baseline_v1.yaml`
- Locked association: `configs/association_v2.yaml` (12 µm gate + 2-frame gap close + intensity 0.5, greedy). Kernel config hash `2c6966b5204ba542`. Do not overwrite after holdout.
- Locked conservative division: `configs/division_v2.yaml` (association_v2 + predecessor / no-alternative-parent / anti-alignment). Gates from **all 71 train `44b6` GEFFs**: `max_parent_child_um=10.0`, `min_child_separation_um=8.3`, `max_sibling_alignment=0.05`. Config hash `0fa756963e5d42d1`. Do not retune after holdout.

## Rejected (do not retune)

- `configs/complementary_v1.yaml`: locked baseline_v1 primary DoG + complementary local-max (6.0 µm isolation, 0.45 intensity ratio) + locked association_v2 linker. Hash `2c4866928d7262b6`. Tuning 0.633135 vs association_v2 0.632460; holdout **0.591384 vs 0.591570**. Status `rejected_on_holdout`. See `docs/complementary_priors.json`.
- `configs/linkable_v1.yaml`: complementary_v1 spatial gates plus temporal link filter (primary neighbor within 12 µm at t±1). Hash `fa975421edb6136e`. Tuning 0.633168 vs association_v2 0.632460; holdout **0.591458 vs 0.591570**. Status `rejected_on_holdout`. See `docs/linkable_priors.json`.

## Measured results (local official scorer, not leaderboard)

On frozen validation (`6bba_05db0fb1` + `6bba_c328f2fd`):

| candidate | official combined | notes |
|---|---|---|
| A `baseline_v1` | 0.537 | Kaggle kernel v2 ran on public test; not submitted |
| B `association_v2` | **0.591570** | local official on both frozen sequences; Kaggle kernel v3 independently audited |
| C `division_v1` | 0.590 | 0 TP / 3 FP / 5 FN; rejected |
| D `division_v2` | **0.591570** | identical to B; 0 TP / 0 FP / 5 FN; keep as no-op overlay; do not retune |
| E `complementary_v1` | 0.591384 | rejected; recall unchanged; extra T_pred +312; do not retune |
| F `linkable_v1` | 0.591458 | rejected; recall unchanged; extra T_pred +189; do not retune |

Independent official score of the **downloaded** association_v2 kernel v3 CSV vs the train GEFF for public-test overlap `6bba_05db0fb1`: **0.564336**. That is not a leaderboard score. `6bba_c328f2fd` is not in public test.

See `Biohub_Cell_Tracking/reports/abc_comparison.md`, `reports/division_v2.md`, `reports/holdout_pair_division_v2.md`, `reports/complementary_v1.md`, `reports/linkable_v1.md`, `docs/learned_detection_estimate.md`, and `experiments/registry.csv`.

## Kaggle

- Private kernel `mahdimzafari/biohub-baseline-v1` version 2 **COMPLETE** on public `test/` (595 s, 1.49 s/frame, peak RAM 435 MB, `audit_ok`).
- Private kernel `mahdimzafari/biohub-association-v2` version 3 **COMPLETE** on public `test/` (593.53 s, 1.48 s/frame, peak RAM 421 MB, `audit_ok`).
- Private helper kernel `mahdimzafari/biohub-geff-export` version 1 **COMPLETE** (203.74 s). Zipped 73 train GEFFs. **Not a submission.**
- Private helper kernel `mahdimzafari/biohub-holdout-zarr-export` version 1 **COMPLETE** (13.43 s). Zipped only the two holdout Zarrs (993 MB). **Not a submission.**
- Private helper kernel `mahdimzafari/biohub-tuning-zarr-export` version 1 **COMPLETE**. Zipped only the two frozen-tuning Zarrs (919 MB). **Not a submission.** Do not overwrite the five kernels above.
- Competition submit: **not performed**. No public or private leaderboard score.
- `Kaggle` is used only through configured secrets / `KAGGLE_API_TOKEN`. Never print tokens.

## Complete validation harness

`scripts/run_complete_validation.py` and `scripts/run_holdout_pair.py` inventory both frozen validation sequences and score them with the official scorer plus per-event `score_divisions`. Complementary scoring is `scripts/run_complementary_detection.py` (one-shot holdout after lock; a second holdout score is refused). Linkable scoring is `scripts/run_linkable_detection.py` (same one-shot protocol).

## Tests

`PYTHONPATH=src python -m pytest Biohub_Cell_Tracking/tests` — 109 passed (2026-09-20, this branch).

## Privacy

Tracked tree has no competition volumes, credentials, weights, submissions, or run artifacts. Ignore rules cover `data/`, `submission.csv`, `artifacts/{runs,tmp,kaggle,cache}`, `.venv/`, `kaggle.json`, `.env*`.
