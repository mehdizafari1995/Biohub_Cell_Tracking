# Next actions

Use issue #1 as the command channel. Do one highest-priority unblocked task per run. Continue the active experiment draft PR when it exists.

## Now

1. Keep draft PR #2 (`cursor/kaggle-baseline-association-0b83`) open. Do not merge. Do not overwrite locked `association_v2`.
2. Keep draft PR #4 (`cursor/competition-operational-strategy-3381`) open. Official holdout is done. Do **not** retune.
3. Keep draft PR #5 (`cursor/complementary-detection-v1`) open. `reject_complementary_v1`. Do **not** retune.
4. Keep draft PR #6 (`cursor/linkable-complementary-v1`) open. `reject_linkable_v1`. Do **not** retune.
5. Continue draft PR #7 (`cursor/learned-unet-v1`). Holdout champion is `learned_unet_orgdet_gap_v1` hash `ee31f36151325f9a`: official 6bba **0.722805 vs orgdet 0.705226**. Public-test slug `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1` v1 **COMPLETE / inferred / `audit_ok`**. Hash `1ce4837ae16bee52`. Independent audit `audit_ok`. `6bba_05db0fb1` 88925/86287 matches locked overlay. Hidden projection **4.32 h**. Do **not** overwrite v1. Retrain lanes remain rejected. Do **not** retune. Do **not** attach 199-video public weights. Do **not** score 6bba again.
6. Do **not** submit any learned slug from the CLI.

## Next (this family)

1. `HUMAN_ACTION_REQUIRED` — Mahdi must **Submit to Competition** from the Kaggle UI on `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1` version 1 **before 2026-09-22 23:59 UTC**. Do **not** submit from the CLI. Do **not** overwrite v1. Do **not** submit orgdet-test, prune-test, or infer-family gap-test now that this overlay is the holdout champion and the public-test worker is COMPLETE / `audit_ok`.
2. Keep `learned_unet_orgdet_gap_v1` locked. Hash `ee31f36151325f9a`. Official 44b6 **0.687221 vs 0.679578**. Official 6bba **0.722805 vs 0.705226**. Do **not** retune. Do **not** score 6bba again.
3. Keep `learned_unet_orgdet_gap_test_v1` locked. Hash `1ce4837ae16bee52`. Slug is in `FORBIDDEN_SLUGS`. Last read-only submit poll **2026-09-22T12:42:26Z** still has only association_v2 **56399003** / **0.553**. Next cron may reconfirm that list; do not invent a leaderboard score.
4. Keep `learned_unet_retrain` 44b6 lanes rejected. Do **not** retune. Do **not** open `6bba` for retrain.
5. Keep discussion **740392** closed as `topic_id_not_found`.
6. Keep rejected overlays (link/motion/hung/conflict/orgop/division) closed. Do not retune.

## HUMAN_ACTION_REQUIRED

- Grant this worker `issues:read` so cron runs can read issue #1.
- Mahdi must **Submit to Competition** from the Kaggle UI on `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1` version 1 **before the 2026-09-22 23:59 UTC entry deadline**. This repo does not authorize `kaggle competitions submit`. Do not resubmit association_v2 **56399003**. Do not UI-submit orgdet-test now that orgdet-gap-test v1 is COMPLETE / `audit_ok`.

## Later (blocked or expensive)

- complementary_v1, linkable_v1, learned_unet_division_v1, learned_unet_link_v1, learned_unet_link_v2, learned_unet_motion_v1, learned_unet_hung_v1, learned_unet_conflict_v1, learned_unet_orgop_v1, and the `learned_unet_retrain` 44b6 lanes are closed.
- Copying public 199-video / DeepCenter checkpoints is forbidden for any KEEP_CANDIDATE path.
- Final Kaggle submission: `HUMAN_ACTION_REQUIRED` — Mahdi must submit version output from the Kaggle UI.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, or invent a leaderboard score.
