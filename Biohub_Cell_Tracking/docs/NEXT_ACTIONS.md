# Next actions

Use issue #1 as the command channel. Do one highest-priority unblocked task per run. Continue the active experiment draft PR when it exists.

## Now

1. Keep draft PR #2 (`cursor/kaggle-baseline-association-0b83`) open. Do not merge. Do not overwrite locked `association_v2`.
2. Keep draft PR #4 (`cursor/competition-operational-strategy-3381`) open. Official holdout is done: combined **0.591570** identical to association_v2, **0 TP / 0 FP / 5 FN**. Keep the overlay; do **not** retune gates. Do not relaunch rejected `division_v1`.
3. Continue **complementary_v1** on `cursor/complementary-detection-v1`. Tuning official: **0.633135** vs association_v2 **0.632460** (`keep_candidate`). Per-sequence mixed (gain on `44b6_12dfb391`, tiny adj-edge loss on `44b6_267148e4` from extra T_pred). Next: lock this operating point in `docs/complementary_priors.json` and official-score **once** on frozen `6bba` (reuse `biohub-holdout-zarr-export` v1). Do not retune after holdout. Do not overwrite locked association/division gates.
4. Do **not** submit association_v2, division_v2, or complementary_v1 from the CLI.

## HUMAN_ACTION_REQUIRED

- If a public leaderboard number is wanted for the association_v2 generalization hypothesis: Mahdi must open https://www.kaggle.com/code/mahdimzafari/biohub-association-v2 version 3 and **Submit to Competition** from the Kaggle UI. This repo does not authorize `kaggle competitions submit`.
- Grant this worker `issues:read` so cron runs can read issue #1.

## Later (blocked or expensive)

- Learned 3D detection / embedding (organizer UNet + transformer is GPU/Kaggle, not the CPU slice).
- Ensembles or complementary candidates, compared on the frozen split against preserved baseline_v1 and association_v2, only after complementary_v1 is locked or rejected on tuning.
- Optional private kernel of association_v2 + division_v2 overlay under a **new** slug. Do not overwrite `biohub-association-v2`. Overlay was a no-op on holdout.
- Harden the final internet-off notebook and runtime for hidden test. Worker measurement: 1.48 s/frame, 421 MB.
- Final Kaggle submission: `HUMAN_ACTION_REQUIRED` — Mahdi must submit version output from the Kaggle UI.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, or invent a leaderboard score.
