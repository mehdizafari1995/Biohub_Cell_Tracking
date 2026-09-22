# Next actions

Use issue #1 as the command channel. Do one highest-priority unblocked task per run. Continue the active experiment draft PR when it exists.

## Now

1. Keep draft PR #2 (`cursor/kaggle-baseline-association-0b83`) open. Do not merge. Do not overwrite locked `association_v2`.
2. Keep draft PR #4 (`cursor/competition-operational-strategy-3381`) open. Official holdout is done: combined **0.591570** identical to association_v2, **0 TP / 0 FP / 5 FN**. Keep the overlay; do **not** retune gates. Do not relaunch rejected `division_v1`.
3. Keep draft PR #5 (`cursor/complementary-detection-v1`) open. complementary_v1 was locked from 44b6 then scored **once** on frozen `6bba`: **0.591384 vs association_v2 0.591570**. Decision **`reject_complementary_v1`**. Recall unchanged; extra T_pred. Do **not** retune isolation/intensity gates. association_v2 remains the scoring candidate.
4. Do **not** submit association_v2, division_v2, or complementary_v1 from the CLI.

## Next experiment family (new branch / new draft PR)

PR #5 is concluded. Start a **new** family; do not continue complementary_v1.

Highest remaining: **estimate** organizer learned 3D detection / embedding (UNet + transformer in `third_party/kaggle-cell-tracking-competition`) against Kaggle 12 h, GPU presence, weight size, and internet-off packaging. Write the estimate before any training or kernel push. If GPU/Kaggle weight packaging is blocked, the fallback 44b6-only hypothesis is temporally linkable complementary peaks (add a peak only when it has a neighbor inside the locked 12 µm gate) so T_pred cannot grow from unlinked local-max noise. Tune on `44b6` only. Do not use `6bba` to choose thresholds.

## HUMAN_ACTION_REQUIRED

- If a public leaderboard number is wanted for the association_v2 generalization hypothesis: Mahdi must open https://www.kaggle.com/code/mahdimzafari/biohub-association-v2 version 3 and **Submit to Competition** from the Kaggle UI. This repo does not authorize `kaggle competitions submit`.
- Grant this worker `issues:read` so cron runs can read issue #1.

## Later (blocked or expensive)

- Learned 3D detection / embedding (organizer UNet + transformer is GPU/Kaggle, not the CPU slice) — estimate first.
- New complementary/ensemble family only with a documented 44b6 hypothesis that addresses T_pred. complementary_v1 is closed.
- Optional private kernel of association_v2 + division_v2 overlay under a **new** slug. Do not overwrite `biohub-association-v2`. Overlay was a no-op on holdout.
- Harden the final internet-off notebook and runtime for hidden test. Worker measurement: 1.48 s/frame, 421 MB.
- Final Kaggle submission: `HUMAN_ACTION_REQUIRED` — Mahdi must submit version output from the Kaggle UI.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, or invent a leaderboard score.
