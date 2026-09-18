# Submission ledger

This ledger records **actual** Kaggle kernel runs and competition submissions. Local simulations are labelled as such and are not Kaggle scores. Public leaderboard numbers are never used as the sole basis for parameter tuning. Submissions are limited to deliberate experiments.

| Field | Meaning |
|---|---|
| submission_label | Frozen experiment name (`baseline_v1`, `association_v2`, `division_v1`, …) |
| timestamp_utc | Kernel push or competition-submit time |
| git_commit | Short hash of the workspace at that action |
| config_hash | SHA-256 prefix of the kernel/config payload |
| validation_sequence | Frozen local sequences scored with the official scorer |
| official_local_score | Official combined score on that local split (not Kaggle) |
| kaggle_kernel_slug / version | Private kernel identity |
| actual_kaggle_runtime_sec | Wall time reported by the Kaggle worker, if returned |
| public_lb_score / private_lb_score | Only filled when Kaggle returns them |

Machine-readable copy: `reports/submission_ledger.csv`. Rows are append-only. Previous results are never overwritten.

## Entries

### baseline_v1_kernel

- **Status:** private kernel `mahdimzafari/biohub-baseline-v1` version **1** pushed 2026-09-18T12:03:00Z. Competition **submit was not performed**.
- **Config:** anisotropic DoG + 4 µm NMS + greedy 15 µm linking. No divisions.
- **Local official score:** 0.498 on frozen validation sequence `6bba_05db0fb1` (experiment `20260918T053718Z_val_aniso`). Not the full two-sequence split. Not a leaderboard score.
- **Kaggle public / private scores:** not returned. Do not invent them.
- **Authorization:** this repository does not contain an explicit instruction to call `kaggle competitions submit`. Kernel output must be submitted manually from the kernel page if a leaderboard score is desired.

## Policy

1. One new Kaggle submission only when the official frozen-validation score improves, or when a written generalization hypothesis is being tested.
2. Association and division thresholds are chosen on tuning data (`docs/splits/v1.json`), then evaluated once on frozen validation.
3. Public leaderboard is a delayed, hidden-test-contaminated signal. Do not chase it.
