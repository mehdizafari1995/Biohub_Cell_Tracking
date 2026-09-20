# Project state

Last updated: 2026-09-20 (packaged learned_unet_link_v1 44b6 transformer linker; hash 3a7423c6858ace41). Authoritative implementation: `Biohub_Cell_Tracking/`.

## Repository

- Private GitHub repo: `mehdizafari1995/Biohub_Cell_Tracking`
- Base branch: `main` (baseline + official scorer + privacy audit + coordination docs + complete-validation harness)
- Association experiment (keep open, do not merge): `cursor/kaggle-baseline-association-0b83` (PR #2)
- Division experiment (keep open, do not merge): `cursor/competition-operational-strategy-3381` (PR #4). Official holdout done; do not retune.
- Complementary experiment (keep open, do not merge): `cursor/complementary-detection-v1` (PR #5). Locked from 44b6, scored once on 6bba, **rejected**.
- Linkable experiment (keep open, do not merge): `cursor/linkable-complementary-v1` (PR #6). Locked from 44b6, scored once on 6bba, **rejected**.
- Learned UNet experiment (keep open, do not merge): `cursor/learned-unet-v1` (PR #7). Training v3 COMPLETE / `trained`. Infer v3 COMPLETE / `inferred` (44b6 tune 0.655751). Holdout v1 COMPLETE / `inferred`. Official frozen `6bba` no-gap **0.693605 vs association_v2 0.591570** (`keep_candidate_learned_unet_holdout_v1`). Test-time slug `mahdimzafari/biohub-learned-unet-test-v1` v1 **COMPLETE / inferred** on public `test/` (306.5 s, 1184 MB, `audit_ok`, hash `7a8725b818fc609b`). `learned_unet_division_v1` overlay **rejected** on 44b6 (**0.655235 vs 0.655751**, 0/1/2, 282 extra edges). `learned_unet_gap_v1` overlay **kept** after one-shot official holdout (**0.701026 vs no-gap 0.693605**, hash `f4d888d190c45a4a`, 16804 extra edges, 0/0/5). Gap test-time slug `mahdimzafari/biohub-learned-unet-gap-test-v1` v1 **COMPLETE / inferred** on public `test/` (330.0 s, 1184 MB, `audit_ok`, hash `85379c198ebc7241`, 251660 nodes / 242412 edges / 13912 gap links / 0 divisions). Transformer-linker helper `mahdimzafari/biohub-learned-unet-link-v1` is packaged (hash `3a7423c6858ace41`, 44b6 only, holdout refused). Official 44b6 score is **not** run until that helper finishes. Do not retune locked gates. Do not score the no-gap holdout twice. Do not score the gap holdout twice. Do not overwrite train/infer/holdout/test/gap-test slugs.
- Command channel: issue #1 (this worker still gets 403 on issues)

## What is locked

- Frozen embryo-level split `docs/splits/v1.json`: tune on specimen `44b6`, validate on `6bba`. Do not retune after seeing validation.
- `docs/splits/v2.json` only adds sanity sequence `44b6_2a2eff9f` (same specimen prefix as tuning, not a third embryo).
- Official scorer: vendored `royerlab/kaggle-cell-tracking-competition` @ `075fc5f5a52d11077f9dc2b074644618f26939e2`. Dependencies: `tracksdata==0.1.0rc10`, `polars==1.44.2`, `geff==1.3.1.1.3`.
- Frozen greedy baseline: `configs/baseline.yaml`
- Kaggle kernel config (anisotropic DoG + 4 µm NMS + greedy 15 µm): `configs/baseline_v1.yaml`
- Locked association: `configs/association_v2.yaml` (12 µm gate + 2-frame gap close + intensity 0.5, greedy). Kernel config hash `2c6966b5204ba542`. Do not overwrite after holdout.
- Locked conservative division: `configs/division_v2.yaml` (association_v2 + predecessor / no-alternative-parent / anti-alignment). Gates from **all 71 train `44b6` GEFFs**: `max_parent_child_um=10.0`, `min_child_separation_um=8.3`, `max_sibling_alignment=0.05`. Config hash `0fa756963e5d42d1`. Do not retune after holdout.
- Locked learned infer (44b6 only): `configs/learned_unet_infer_v1.yaml`. Hash `d7578b5f64ddac08`. Official tune **0.655751**. Do not retune.
- Locked learned holdout (one-shot 6bba): `configs/learned_unet_holdout_v1.yaml`. Hash `d374a21f03dddc04`. Official holdout **0.693605**. Do not retune. Do not score again.
- Locked learned test-time (discover `test/*.zarr`): `configs/learned_unet_test_v1.yaml`. Hash `7a8725b818fc609b`. Same operating point as infer/holdout (no gap-close). Kernel v1 COMPLETE. Do not retune. Do not overwrite.
- Locked learned gap overlay: `configs/learned_unet_gap_v1.yaml`. Hash `f4d888d190c45a4a`. Official tune **0.658452** vs learned infer **0.655751**. Official holdout **0.701026** vs no-gap **0.693605**. Do not retune. Do not score holdout again.
- Locked learned gap test-time (discover `test/*.zarr` + locked overlay): `configs/learned_unet_gap_test_v1.yaml`. Hash `85379c198ebc7241`. Kernel v1 COMPLETE. Do not retune. Do not overwrite.
- Packaged learned transformer linker (44b6 only): `configs/learned_unet_link_v1.yaml`. Hash `3a7423c6858ace41`. Same detections as infer_v1; softmax 1-1 transformer scores. Holdout refused until a documented 44b6 lock. Do not overwrite locked slugs.

## Rejected (do not retune)

- `configs/complementary_v1.yaml`: locked baseline_v1 primary DoG + complementary local-max (6.0 µm isolation, 0.45 intensity ratio) + locked association_v2 linker. Hash `2c4866928d7262b6`. Tuning 0.633135 vs association_v2 0.632460; holdout **0.591384 vs 0.591570**. Status `rejected_on_holdout`. See `docs/complementary_priors.json`.
- `configs/linkable_v1.yaml`: complementary_v1 spatial gates plus temporal link filter (primary neighbor within 12 µm at t±1). Hash `fa975421edb6136e`. Tuning 0.633168 vs association_v2 0.632460; holdout **0.591458 vs 0.591570**. Status `rejected_on_holdout`. See `docs/linkable_priors.json`.
- `configs/learned_unet_division_v1.yaml`: locked `division_v2` gates on locked learned infer graphs. Hash `8761fb66cc2254f3`. Tuning **0.655235 vs learned_unet_infer_v1 0.655751**. Status `rejected_on_tuning`. Frozen `6bba` not scored. See `reports/learned_unet_division_v1.md`.

## Current scoring candidate

- `configs/learned_unet_gap_v1.yaml` is the holdout champion on the official local scorer (**0.701026**). `learned_unet_gap_test_v1` v1 is the latest **public-test** worker CSV (`audit_ok`, locked gap overlay, not submitted). `learned_unet_test_v1` v1 remains the locked no-gap public-test worker. association_v2 v3 remains a previous public-test worker. `learned_unet_link_v1` is a 44b6-only transformer-linker helper and is **not** a scoring candidate until official tuning finishes. No leaderboard score exists.

## Measured results (local official scorer, not leaderboard)

On frozen validation (`6bba_05db0fb1` + `6bba_c328f2fd`):

| candidate | official combined | notes |
|---|---|---|
| A `baseline_v1` | 0.537 | Kaggle kernel v2 ran on public test; not submitted |
| B `association_v2` | 0.591570 | local official on both frozen sequences; Kaggle kernel v3 independently audited |
| C `division_v1` | 0.590 | 0 TP / 3 FP / 5 FN; rejected |
| D `division_v2` | 0.591570 | identical to B; 0 TP / 0 FP / 5 FN; keep as no-op overlay; do not retune |
| E `complementary_v1` | 0.591384 | rejected; recall unchanged; extra T_pred +312; do not retune |
| F `linkable_v1` | 0.591458 | rejected; recall unchanged; extra T_pred +189; do not retune |
| G `learned_unet_holdout_v1` | 0.693605 | one-shot; recall 0.957; 0/0/5; keep_candidate; do not retune |
| H `learned_unet_gap_v1` | **0.701026** | one-shot overlay; recall 0.976; 0/0/5; 16804 extra edges; keep_candidate; do not retune |

Independent official score of the **downloaded** association_v2 kernel v3 CSV vs the train GEFF for public-test overlap `6bba_05db0fb1`: **0.564336**. That is not a leaderboard score. `6bba_c328f2fd` is not in public test.

See `Biohub_Cell_Tracking/reports/abc_comparison.md`, `reports/division_v2.md`, `reports/holdout_pair_division_v2.md`, `reports/complementary_v1.md`, `reports/linkable_v1.md`, `reports/learned_unet_v1.md`, `reports/learned_unet_infer_v1.md`, `reports/learned_unet_holdout_v1.md`, `reports/learned_unet_division_v1.md`, `reports/learned_unet_gap_v1.md`, `reports/learned_unet_gap_test_v1.md`, `reports/learned_unet_link_v1.md`, `docs/learned_detection_estimate.md`, and `experiments/registry.csv`.

## Kaggle

- Private kernel `mahdimzafari/biohub-baseline-v1` version 2 **COMPLETE** on public `test/` (595 s, 1.49 s/frame, peak RAM 435 MB, `audit_ok`).
- Private kernel `mahdimzafari/biohub-association-v2` version 3 **COMPLETE** on public `test/` (593.53 s, 1.48 s/frame, peak RAM 421 MB, `audit_ok`).
- Private helper kernel `mahdimzafari/biohub-geff-export` version 1 **COMPLETE** (203.74 s). Zipped 73 train GEFFs. **Not a submission.**
- Private helper kernel `mahdimzafari/biohub-holdout-zarr-export` version 1 **COMPLETE** (13.43 s). Zipped only the two holdout Zarrs (993 MB). **Not a submission.**
- Private helper kernel `mahdimzafari/biohub-tuning-zarr-export` version 1 **COMPLETE**. Zipped only the two frozen-tuning Zarrs (919 MB). **Not a submission.** Do not overwrite the five kernels above.
- Private helper kernel `mahdimzafari/biohub-learned-unet-v1` version **3 COMPLETE** / `trained`. GPU T4 x2, 11732 s, peak 1793 MB. **Not a submission.** No `submission.csv`. Do not overwrite.
- Private helper kernel `mahdimzafari/biohub-learned-unet-infer-v1` version **3 COMPLETE** / `inferred`. 187.0 s, peak 1136 MB. 44b6 graphs only. **Not a submission.** No `submission.csv`. Do not overwrite.
- Private helper kernel `mahdimzafari/biohub-learned-unet-holdout-v1` version **1 COMPLETE** / `inferred`. 184.4 s, peak 1177 MB. 6bba graphs only. **Not a submission.** No `submission.csv`. Do not overwrite.
- Private GPU test-time kernel `mahdimzafari/biohub-learned-unet-test-v1` version **1 COMPLETE** / `inferred`. 306.5 s, peak 1184 MB, T4. Public test four stems, 231707 nodes / 208547 edges / 0 divisions, `audit_ok`. **Not submitted.** Do not overwrite.
- Private GPU test-time kernel `mahdimzafari/biohub-learned-unet-gap-test-v1` version **1 COMPLETE** / `inferred`. 330.0 s, peak 1184 MB, T4. Public test four stems, 251660 nodes / 242412 edges / 13912 gap links / 19953 interpolated / 0 divisions, `audit_ok`. Independent stream audit `audit_ok`. `6bba_05db0fb1` 103432/100197 matches locked holdout gap overlay. **Not submitted.** Do not overwrite.
- Private GPU helper kernel `mahdimzafari/biohub-learned-unet-link-v1` version **1 running**. Hash `3a7423c6858ace41`. 44b6 transformer linker. **Not a submission.** No `submission.csv`. Do not overwrite locked train/infer/holdout/test/gap-test slugs.
- Competition submit: **not performed by this worker**. No public or private leaderboard score is recorded here.
- `Kaggle` is used only through configured secrets / `KAGGLE_API_TOKEN`. Never print tokens.

## Complete validation harness

`scripts/run_complete_validation.py` and `scripts/run_holdout_pair.py` inventory both frozen validation sequences and score them with the official scorer plus per-event `score_divisions`. Learned training is `notebooks/kaggle_learned_unet_v1.py` (44b6-only; holdout refused). Learned 44b6 inference is `notebooks/kaggle_learned_unet_infer_v1.py`. One-shot holdout inference is `notebooks/kaggle_learned_unet_holdout_v1.py` and `scripts/run_learned_unet_holdout.py` (6bba only; 44b6 refused; refuses a second score). Test-time inference is `notebooks/kaggle_learned_unet_test_v1.py` (no-gap; discovers `test/*.zarr`; writes `submission.csv`) and `notebooks/kaggle_learned_unet_gap_test_v1.py` (locked gap overlay; new slug). Transformer linking is `notebooks/kaggle_learned_unet_link_v1.py` and `scripts/run_learned_unet_link.py` (44b6 only; 6bba refused). Conservative overlay scoring is `scripts/run_learned_unet_division.py` and `scripts/run_learned_unet_gap.py` (44b6 first; holdout refused unless tuning keeps; holdout refuses a second score). COMPLETE handoff is `scripts/collect_learned_unet_v1.py`, `scripts/collect_learned_unet_infer_v1.py`, `scripts/collect_learned_unet_holdout_v1.py`, `scripts/collect_learned_unet_test_v1.py`, `scripts/collect_learned_unet_gap_test_v1.py`, and `scripts/collect_learned_unet_link_v1.py`.

## Tests

`PYTHONPATH=src python -m pytest tests/test_learned_unet_link_v1.py tests/test_learned_collect.py tests/test_kaggle_api.py` on this run. Train, infer, holdout, no-gap test, and completed gap-test slugs stay in `FORBIDDEN_SLUGS`. The new link slug is allowed to push once. Do not retune rejected `learned_unet_division_v1`. Do not retune or rescore `learned_unet_gap_v1` after the holdout lock.

## Privacy

Tracked tree has no competition volumes, credentials, weights, submissions, or run artifacts. Ignore rules cover `data/`, `submission.csv`, `*.pt` / `*.pth`, `weights/`, `artifacts/{runs,tmp,kaggle,cache}`, `.venv/`, `kaggle.json`, `.env*`.
