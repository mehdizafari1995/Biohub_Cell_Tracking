# Submission ledger

This ledger records **actual** Kaggle kernel runs and competition submissions. Local simulations are labelled as such and are not Kaggle scores. Public leaderboard numbers are never used as the sole basis for parameter tuning. Submissions are limited to deliberate experiments.

Machine-readable copy: `reports/submission_ledger.csv`. Rows are append-only. Previous results are never overwritten.

## baseline_v1_kernel_v1

- **Status:** private kernel `mahdimzafari/biohub-baseline-v1` version **1** failed (~13 s).
- **Error:** `ModuleNotFoundError: No module named 'zarr'` with internet off.
- **Fix:** version 2 reads OME-Zarr v3 chunks via `blosc2` (present on the Kaggle image).
- **Kaggle public / private scores:** not returned.

## baseline_v1_kernel_v2

- **Status:** private kernel version **2** **COMPLETE**. Environment label `kaggle_input`. This is a real Kaggle execution on public `test/`.
- **Config hash:** `d4f79f4af9958c85` (anisotropic DoG + 4 µm NMS + greedy 15 µm, no divisions).
- **Datasets:** `44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`, `6bba_05db0fb1` (exact sample_submission stems).
- **Output:** `/kaggle/working/submission.csv` 11 443 248 bytes, 224 788 rows, 117 515 nodes, 107 273 edges, 0 predicted divisions. Local re-audit of the downloaded file: no errors.
- **Worker:** 595.24 s (1.49 s/frame), peak RAM 435 MB. Decoder was `blosc2 4.1.2`; `zarr`/`imagecodecs`/`numcodecs` were missing.
- **Local official score:** 0.537 on the full frozen validation split (`6bba_05db0fb1` + `6bba_c328f2fd`). The earlier 0.498 figure was `6bba_05db0fb1` alone.
- **Competition submit:** not performed. No public or private leaderboard score was returned.
- **Manual submit:** https://www.kaggle.com/code/mahdimzafari/biohub-baseline-v1 → version 2 → **Submit to Competition**.

## association_v2_kernel_v3

- **Status:** private kernel `mahdimzafari/biohub-association-v2` version **3** **COMPLETE**. Environment label `kaggle_input`. This is a real Kaggle execution on public `test/`.
- **Config hash:** `2c6966b5204ba542` (anisotropic DoG + 4 µm NMS + greedy 12 µm gate + 2-frame gap close + intensity 0.5, interpolated consecutive nodes, no divisions).
- **Datasets:** `44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`, `6bba_05db0fb1` (exact sample_submission stems).
- **Output:** `/kaggle/working/submission.csv` 12 948 601 bytes, 253 773 rows, 129 500 nodes, 124 273 edges, 11 985 interpolated nodes, 0 predicted divisions. Worker audit: `audit_ok`. `submission.csv` was not copied into git.
- **Worker:** 593.53 s (1.48 s/frame), peak RAM 421 MB. Decoder was `blosc2 4.1.2`; `zarr`/`imagecodecs`/`numcodecs` were missing.
- **Local official score:** 0.592 on the full frozen validation split (`6bba_05db0fb1` + `6bba_c328f2fd`). That number is local official, not a leaderboard score.
- **Competition submit:** not performed. No public or private leaderboard score was returned.
- **Manual submit:** https://www.kaggle.com/code/mahdimzafari/biohub-association-v2 → version 3 → **Submit to Competition**.

## association_v2_kernel_v3_independent_audit

- **Status:** downloaded worker `submission.csv` independently audited. This is still not a competition submission.
- **Schema / integrity:** `audit_ok`. 253773 rows, 129500 nodes, 124273 edges, 0 predicted divisions, 0 non-consecutive edges. Integer voxel centroids inside typical `(100, 64, 256, 256)`. Counts match the worker manifest. File was kept under `artifacts/kaggle/` and not committed.
- **Official scorer on public-test overlap `6bba_05db0fb1`:** combined **0.564336**, edge Jaccard 0.5627, adj 0.5643, divisions 0/0/3, recall 0.829. Delta vs local 0.564 is +0.0003. Official commit `075fc5f5`.
- **Not holdout:** `44b6_0113de3b` 0.931, `44b6_0b24845f` 0.280, `6bba_05b6850b` 0.476. Do not retune from these. The four-stem aggregate 0.527 is not a frozen-split score.
- **Still missing:** `6bba_c328f2fd` is not in public test, so the worker CSV cannot complete the frozen split.
- **Kaggle public / private scores:** not returned.

## geff_export_kernel_v1

- **Status:** private helper kernel `mahdimzafari/biohub-geff-export` version **1** **COMPLETE**. Internet off. **Not a competition submission.** No `submission.csv`.
- **Purpose:** zip train `44b6` GEFFs plus two frozen-holdout GEFFs after per-file downloads 429'd.
- **Output:** 73 GEFF stores, 1533 files, no image volumes. Zip kept under ignored `artifacts/kaggle/`.
- **Kaggle public / private scores:** not applicable; not submitted.

## zarr_export_kernel_v1

- **Status:** private helper kernel `mahdimzafari/biohub-holdout-zarr-export` version **1** **COMPLETE**. Internet off. **Not a competition submission.** No `submission.csv`.
- **Purpose:** zip only frozen-holdout `6bba_05db0fb1.zarr` and `6bba_c328f2fd.zarr` after per-file downloads 429'd.
- **Output:** 2 Zarr stores, 204 files, 993 198 228 byte inner zip, 13.43 s worker. Zip kept under ignored `artifacts/kaggle/`.
- **Kaggle public / private scores:** not applicable; not submitted.

## holdout_pair_association_v2_vs_division_v2

- **Status:** local official scorer on both frozen validation sequences after the helper kernel. **Not a Kaggle run.**
- **association_v2:** combined 0.591570, 0/0/5 divisions. Matches the locked 0.592 figure.
- **division_v2:** combined 0.591570, 0/0/5 divisions. No regression. No extra false forks. Keep as no-op overlay. Do not retune.
- **Kaggle public / private scores:** not returned.

## association_v2_local / division_v1_local

Local official scores only. association_v2 later received a private kernel (see above). division_v1 did not (0.590, 0/3/5 divisions) and has no Kaggle run.

## complementary_v1_family

- **Status:** experiment family on `cursor/complementary-detection-v1`. Locked from 44b6, scored once on frozen 6bba, **rejected**. Do not retune.
- **Helper:** private kernel `mahdimzafari/biohub-tuning-zarr-export` version **1 COMPLETE**. Zipped only `44b6_12dfb391.zarr` and `44b6_267148e4.zarr`. Worker archive 919 MB under ignored `artifacts/kaggle/`. **Not a submission.**
- **Local official tuning:** complementary_v1 **0.633135** vs locked association_v2 **0.632460** on both frozen-tuning sequences. Decision `keep_candidate_complementary_v1`. Per-sequence mixed. Not a leaderboard score.
- **Kaggle public / private scores:** not returned.

## complementary_v1_holdout

- **Status:** local official scorer on both frozen validation sequences after reusing `biohub-holdout-zarr-export` v1. **Not a Kaggle run.**
- **association_v2:** combined **0.591570**, recall 0.838, 0/0/5 divisions, 96336 pred nodes. Matches the locked 0.591570 figure.
- **complementary_v1:** combined **0.591384**, recall 0.838, 0/0/5 divisions, 96648 pred nodes. Extra T_pred lowered adj-edge Jaccard. Decision `reject_complementary_v1`. Do not retune.
- **Kaggle public / private scores:** not returned.

## linkable_v1_family

- **Status:** experiment family on `cursor/linkable-complementary-v1`. Learned UNet estimate written first (`docs/learned_detection_estimate.md`); GPU blocked. CPU fallback locked from 44b6, scored once on frozen 6bba, **rejected**. Do not retune.
- **Local official tuning:** linkable_v1 **0.633168** vs locked association_v2 **0.632460** on both frozen-tuning sequences. Decision `keep_candidate_linkable_v1`. Hash `fa975421edb6136e`. Not a leaderboard score.
- **Kaggle public / private scores:** not returned.

## linkable_v1_holdout

- **Status:** local official scorer on both frozen validation sequences after reusing `biohub-holdout-zarr-export` v1. **Not a Kaggle run.**
- **association_v2:** combined **0.591570**, recall 0.838, 0/0/5 divisions, 96336 pred nodes. Matches the locked 0.591570 figure.
- **linkable_v1:** combined **0.591458**, recall 0.838, 0/0/5 divisions, 96525 pred nodes. Extra T_pred (+189) lowered adj-edge Jaccard. Decision `reject_linkable_v1`. Do not retune.
- **Kaggle public / private scores:** not returned.

## Policy

1. One new Kaggle submission only when the official frozen-validation score improves, or when a written generalization hypothesis is being tested.
2. Association and division thresholds are chosen on tuning data (`docs/splits/v1.json`), then evaluated once on frozen validation.
3. Public leaderboard is not used as the sole tuning signal.
