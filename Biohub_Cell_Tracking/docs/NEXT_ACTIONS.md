# Next actions

Use issue #1 as the command channel. Do one highest-priority unblocked task per run. Continue the active experiment draft PR when it exists.

## Now

1. Keep draft PR #2 (`cursor/kaggle-baseline-association-0b83`) open. Do not merge. Do not overwrite locked `association_v2`.
2. Keep draft PR #4 (`cursor/competition-operational-strategy-3381`) open. Official holdout is done: combined **0.591570** identical to association_v2, **0 TP / 0 FP / 5 FN**. Keep the overlay; do **not** retune gates. Do not relaunch rejected `division_v1`.
3. Keep draft PR #5 (`cursor/complementary-detection-v1`) open. complementary_v1 was locked from 44b6 then scored **once** on frozen `6bba`: **0.591384 vs association_v2 0.591570**. Decision **`reject_complementary_v1`**. Do **not** retune isolation/intensity gates.
4. Keep draft PR #6 (`cursor/linkable-complementary-v1`) open. linkable_v1 was locked from 44b6 then scored **once** on frozen `6bba`: **0.591458 vs association_v2 0.591570**. Decision **`reject_linkable_v1`**. Recall unchanged; extra T_pred +189. Do **not** retune the 12 µm / max_dt=1 temporal gate. association_v2 remains the scoring candidate.
5. Continue draft PR #7 (`cursor/learned-unet-v1`). Config hash `4c38035d6657f2ad`. Do **not** overwrite locked prediction/helper slugs. Do **not** write `submission.csv`. Do **not** score frozen `6bba` until 44b6 training is locked.
6. Do **not** submit association_v2, division_v2, complementary_v1, linkable_v1, or learned_unet_v1 from the CLI.

## Next (this family)

1. Confirm private GPU kernel `mahdimzafari/biohub-learned-unet-v1` **version 3** is **COMPLETE** (or record a new `train_error`). Do not overwrite locked slugs. Do not push a new version unless v3 failed.
2. If COMPLETE: download weights into ignored `artifacts/kaggle/` only. Do not commit `.pt` / `.pth`.
3. Official-score once on frozen-tuning `44b6_12dfb391` + `44b6_267148e4` against association_v2. Keep or drop the candidate on that score. Do **not** look at `6bba` yet.
4. Only after a documented 44b6 lock: one-shot official holdout on frozen `6bba`. Reject on combined-score regression. Do not retune.

## HUMAN_ACTION_REQUIRED

- If a public leaderboard number is wanted for the association_v2 generalization hypothesis: Mahdi must open https://www.kaggle.com/code/mahdimzafari/biohub-association-v2 version 3 and **Submit to Competition** from the Kaggle UI. This repo does not authorize `kaggle competitions submit`.
- Grant this worker `issues:read` so cron runs can read issue #1.
- If the learned kernel push returns `blocked_no_gpu` or Kaggle refuses the GPU accelerator, Mahdi must enable GPU on https://www.kaggle.com/code/mahdimzafari/biohub-learned-unet-v1 (do not overwrite association_v2).

## Later (blocked or expensive)

- Learned 3D detection inference kernel (internet-off, attach training-kernel output). Do not use public organizer weights on frozen `6bba`.
- complementary_v1 and linkable_v1 are closed. Do not retune.
- Optional private kernel of association_v2 + division_v2 overlay under a **new** slug. Overlay was a no-op on holdout.
- Harden the final internet-off notebook and runtime for hidden test. Worker measurement: 1.48 s/frame, 421 MB.
- Final Kaggle submission: `HUMAN_ACTION_REQUIRED` — Mahdi must submit version output from the Kaggle UI.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, or invent a leaderboard score.
