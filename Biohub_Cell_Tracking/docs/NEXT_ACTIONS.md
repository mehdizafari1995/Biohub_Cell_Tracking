# Next actions

Use issue #1 as the command channel. Do one highest-priority unblocked task per run. Continue the active experiment draft PR when it exists.

## Now

1. Keep draft PR #2 (`cursor/kaggle-baseline-association-0b83`) open. Do not merge. Do not overwrite locked `association_v2`.
2. Keep draft PR #4 (`cursor/competition-operational-strategy-3381`) open. Official holdout is done: combined **0.591570** identical to association_v2, **0 TP / 0 FP / 5 FN**. Keep the overlay; do **not** retune gates. Do not relaunch rejected `division_v1`.
3. Keep draft PR #5 (`cursor/complementary-detection-v1`) open. complementary_v1 was locked from 44b6 then scored **once** on frozen `6bba`: **0.591384 vs association_v2 0.591570**. Decision **`reject_complementary_v1`**. Do **not** retune isolation/intensity gates.
4. Keep draft PR #6 (`cursor/linkable-complementary-v1`) open. linkable_v1 was locked from 44b6 then scored **once** on frozen `6bba`: **0.591458 vs association_v2 0.591570**. Decision **`reject_linkable_v1`**. Recall unchanged; extra T_pred +189. Do **not** retune the 12 µm / max_dt=1 temporal gate.
5. Continue draft PR #7 (`cursor/learned-unet-v1`). Training hash `4c38035d6657f2ad` (v3 COMPLETE / `trained`). Infer hash `d7578b5f64ddac08` (44b6 tune 0.655751). Holdout hash `d374a21f03dddc04`: official frozen `6bba` **0.693605 vs association_v2 0.591570**, decision **`keep_candidate_learned_unet_holdout_v1`**. Do **not** overwrite the completed train/infer/holdout slugs. Do **not** retune. Do **not** score 6bba twice.
6. Do **not** submit association_v2, division_v2, complementary_v1, linkable_v1, learned_unet_v1, learned_unet_infer_v1, or learned_unet_holdout_v1 from the CLI.

## Next (this family)

1. Package and push a **new** private GPU test-time notebook (new slug; do not overwrite `mahdimzafari/biohub-learned-unet-v1`, `...-infer-v1`, or `...-holdout-v1`). Attach the trained weights. Discover every `test/*.zarr` at runtime. Internet off. Write `submission.csv` + `run_manifest.json`.
2. Estimate first: ~90 s/volume from holdout v1. Hidden test is approximately train-sized (~199 volumes) → ~5 h, under the 12 h GPU limit. Peak RAM ~1.2 GB.
3. Audit the returned CSV schema, dataset coverage, and integrity. Treat it as Kaggle evidence only if Kaggle returns COMPLETE. Do **not** submit from the CLI.
4. Do **not** retune from public-test stems or the public leaderboard.

## HUMAN_ACTION_REQUIRED

- If a public leaderboard number is wanted for the learned holdout generalization hypothesis: after a COMPLETE test-time notebook exists, Mahdi must **Submit to Competition** from the Kaggle UI. This repo does not authorize `kaggle competitions submit`.
- Grant this worker `issues:read` so cron runs can read issue #1.

## Later (blocked or expensive)

- complementary_v1 and linkable_v1 are closed. Do not retune.
- Optional private kernel of association_v2 + division_v2 overlay under a **new** slug. Overlay was a no-op on holdout.
- Harden the learned test-time notebook for hidden test (runtime, memory, decoder fallback).
- Final Kaggle submission: `HUMAN_ACTION_REQUIRED` — Mahdi must submit version output from the Kaggle UI.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, or invent a leaderboard score.
