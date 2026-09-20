# Next actions

Use issue #1 as the command channel. Do one highest-priority unblocked task per run. Continue the active experiment draft PR when it exists.

## Now

1. Keep draft PR #2 (`cursor/kaggle-baseline-association-0b83`) open. Do not merge. Do not overwrite locked `association_v2`.
2. Continue the **division_v2** draft PR #4 on `cursor/competition-operational-strategy-3381`. Do not relaunch rejected `division_v1`. Gates are locked from all 71 `44b6` GEFFs: 10.0 / 8.3 / 0.05. Do not retune after holdout.
3. Download only the two frozen-validation **Zarr** volumes (`6bba_05db0fb1`, `6bba_c328f2fd`; ≈1.0 GB) and official-score `association_v2` vs `division_v2` once. Reject `division_v2` on any combined-score regression or extra false forks.
   - Per-file Kaggle downloads 429 at ~90 s/file. Prefer a private internet-off helper kernel that zips **only those two Zarrs** (same path as `mahdimzafari/biohub-geff-export` v1). Do not copy other volumes. Do not submit.
4. Do **not** submit association_v2 or division_v2 from the CLI.

## HUMAN_ACTION_REQUIRED

- If a public leaderboard number is wanted for the association_v2 generalization hypothesis: Mahdi must open https://www.kaggle.com/code/mahdimzafari/biohub-association-v2 version 3 and **Submit to Competition** from the Kaggle UI. This repo does not authorize `kaggle competitions submit`.
- Grant this worker `issues:read` so cron runs can read issue #1.

## Later (blocked or expensive)

- Learned 3D detection / embedding (organizer UNet + transformer is GPU/Kaggle, not the CPU slice).
- Ensembles or complementary candidates, compared on the frozen split against preserved baseline_v1 and association_v2.
- Harden the final internet-off notebook and runtime for hidden test. Worker measurement: 1.48 s/frame, 421 MB.
- Final Kaggle submission: `HUMAN_ACTION_REQUIRED` — Mahdi must submit version output from the Kaggle UI.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, or invent a leaderboard score.
