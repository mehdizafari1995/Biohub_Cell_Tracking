# Next actions

Use issue #1 as the command channel. Do one highest-priority unblocked task per run. Continue the active experiment draft PR when it exists.

## Now

1. Keep draft PR #2 (`cursor/kaggle-baseline-association-0b83`) open. Do not merge. Do not overwrite locked `association_v2`.
2. Keep draft PR #4 (`cursor/competition-operational-strategy-3381`) open. Official holdout is done: combined **0.591570** identical to association_v2, **0 TP / 0 FP / 5 FN**. Keep the overlay; do **not** retune gates. Do not relaunch rejected `division_v1`.
3. Keep draft PR #5 (`cursor/complementary-detection-v1`) open. complementary_v1 was locked from 44b6 then scored **once** on frozen `6bba`: **0.591384 vs association_v2 0.591570**. Decision **`reject_complementary_v1`**. Do **not** retune isolation/intensity gates.
4. Keep draft PR on `cursor/linkable-complementary-v1` open. linkable_v1 was locked from 44b6 then scored **once** on frozen `6bba`: **0.591458 vs association_v2 0.591570**. Decision **`reject_linkable_v1`**. Recall unchanged; extra T_pred +189. Do **not** retune the 12 µm / max_dt=1 temporal gate. association_v2 remains the scoring candidate.
5. Do **not** submit association_v2, division_v2, complementary_v1, or linkable_v1 from the CLI.

## Next experiment family (new branch / new draft PR)

This linkable family is concluded. Start a **new** family; do not continue linkable_v1 or complementary_v1.

Highest remaining: **Kaggle GPU** specimen-separated training of organizer `TemporalUNet3D` + `SimpleNodeTransformer` using the written estimate in `docs/learned_detection_estimate.md`. Train on frozen-tuning specimen `44b6` only. Do not use `6bba` to choose hyperparameters or to train. Do not overwrite `biohub-association-v2` / helper slugs. Weights stay out of git. This CPU worker has no GPU — a GPU kernel is required.

Do **not** launch another complementary-peak / local-max overlay. Holdout recall stayed 0.838 for complementary_v1 and linkable_v1; extra T_pred is the failure mode.

## HUMAN_ACTION_REQUIRED

- If a public leaderboard number is wanted for the association_v2 generalization hypothesis: Mahdi must open https://www.kaggle.com/code/mahdimzafari/biohub-association-v2 version 3 and **Submit to Competition** from the Kaggle UI. This repo does not authorize `kaggle competitions submit`.
- Grant this worker `issues:read` so cron runs can read issue #1.
- If the next learned family needs a Kaggle GPU accelerator that this account cannot enable, Mahdi must enable GPU on a new private kernel slug (do not overwrite association_v2).

## Later (blocked or expensive)

- Learned 3D detection / embedding — estimate is written; needs Kaggle GPU + 44b6-only retrain. Public organizer weights are **not** a valid frozen-`6bba` test (random 90/10 split leakage).
- complementary_v1 and linkable_v1 are closed. Do not retune.
- Optional private kernel of association_v2 + division_v2 overlay under a **new** slug. Overlay was a no-op on holdout.
- Harden the final internet-off notebook and runtime for hidden test. Worker measurement: 1.48 s/frame, 421 MB.
- Final Kaggle submission: `HUMAN_ACTION_REQUIRED` — Mahdi must submit version output from the Kaggle UI.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, or invent a leaderboard score.
