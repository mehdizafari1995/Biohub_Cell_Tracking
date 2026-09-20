# Project state

Last updated: 2026-09-20 (division_v2 conservative family). Authoritative implementation: `Biohub_Cell_Tracking/`.

## Repository

- Private GitHub repo: `mehdizafari1995/Biohub_Cell_Tracking`
- Base branch: `main` (baseline + official scorer + privacy audit + coordination docs + complete-validation harness)
- Association experiment (keep open, do not merge): `cursor/kaggle-baseline-association-0b83` (PR #2)
- Active experiment branch: `cursor/competition-operational-strategy-3381` (conservative `division_v2` on locked `association_v2`)
- Command channel: issue #1 (this worker still gets 403 on issues)

## What is locked

- Frozen embryo-level split `docs/splits/v1.json`: tune on specimen `44b6`, validate on `6bba`. Do not retune after seeing validation.
- `docs/splits/v2.json` only adds sanity sequence `44b6_2a2eff9f` (same specimen prefix as tuning, not a third embryo).
- Official scorer: vendored `royerlab/kaggle-cell-tracking-competition` @ `075fc5f5a52d11077f9dc2b074644618f26939e2`. Dependencies: `tracksdata==0.1.0rc10`, `polars==1.44.2`, `geff==1.3.1.1.3`.
- Frozen greedy baseline: `configs/baseline.yaml`
- Kaggle kernel config (anisotropic DoG + 4 µm NMS + greedy 15 µm): `configs/baseline_v1.yaml`
- Locked association: `configs/association_v2.yaml` (12 µm gate + 2-frame gap close + intensity 0.5, greedy). Kernel config hash `2c6966b5204ba542`. Do not overwrite after holdout.
- New conservative division candidate: `configs/division_v2.yaml` (association_v2 + predecessor / no-alternative-parent / anti-alignment filters). Gates locked from 17 complete `44b6` GEFFs (4 events). Do not retune after holdout.

## Measured results (local official scorer, not leaderboard)

On frozen validation (`6bba_05db0fb1` + `6bba_c328f2fd`):

| candidate | official combined | notes |
|---|---|---|
| A `baseline_v1` | 0.537 | Kaggle kernel v2 ran on public test; not submitted |
| B `association_v2` | **0.592** | local official on both frozen sequences; Kaggle kernel v3 independently audited |
| C `division_v1` | 0.590 | 0 TP / 3 FP / 5 FN; rejected |
| D `division_v2` | not scored | priors locked; official holdout needs train volumes |

Independent official score of the **downloaded** association_v2 kernel v3 CSV vs the train GEFF for public-test overlap `6bba_05db0fb1`: **0.564336**. That is not a leaderboard score. `6bba_c328f2fd` is not in public test.

See `Biohub_Cell_Tracking/reports/abc_comparison.md`, `reports/division_v2.md`, `reports/kaggle_association_v2_v3_independent_audit.md`, and `experiments/registry.csv`.

## Kaggle

- Private kernel `mahdimzafari/biohub-baseline-v1` version 2 **COMPLETE** on public `test/` (595 s, 1.49 s/frame, peak RAM 435 MB, `audit_ok`).
- Private kernel `mahdimzafari/biohub-association-v2` version 3 **COMPLETE** on public `test/` (593.53 s, 1.48 s/frame, peak RAM 421 MB, `audit_ok`).
- Competition submit: **not performed**. No public or private leaderboard score.
- `Kaggle` is used only through configured secrets / `KAGGLE_API_TOKEN`. Never print tokens.

## Complete validation harness

`scripts/run_complete_validation.py` inventories both frozen validation sequences and scores available graphs with the official scorer plus per-event `score_divisions`. This environment still lacks the two holdout Zarr volumes.

## Tests

`PYTHONPATH=src python -m pytest Biohub_Cell_Tracking/tests` — 69 passed (2026-09-20, this branch).

## Privacy

Tracked tree has no competition volumes, credentials, weights, submissions, or run artifacts. Ignore rules cover `data/`, `submission.csv`, `artifacts/{runs,tmp,kaggle,cache}`, `.venv/`, `kaggle.json`, `.env*`.
