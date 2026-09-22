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

1. Read-only orgdet-test submit poll **2026-09-22T16:42:10Z** reconfirmed competition ref **56466744**, status still **pending**, public score still **not returned**. Notebook identity remains proven as `mahdimzafari/biohub-learned-unet-orgdet-test-v1` (`scriptVersionId` **351557863**). That is **not** association_v2 **56399003**, **not** prune-test, **not** infer-family gap-test, and **not** `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1`. Kaggle `versionNumber` was not returned. Decision `new_submission_present`. This is **not** a leaderboard score. Do **not** resubmit. Do **not** use a later public score for tuning.
2. Keep `learned_unet_orgdet_gap_v1` locked. Hash `ee31f36151325f9a`. Official 44b6 **0.687221 vs 0.679578**. Official 6bba **0.722805 vs 0.705226**. Do **not** retune. Do **not** score 6bba again.
3. Keep `learned_unet_orgdet_gap_test_v1` locked. Hash `1ce4837ae16bee52`. Slug is in `FORBIDDEN_SLUGS`. It is **not** the pending competition row. Additional competition submission is **not** authorized this cycle. Do **not** overwrite v1.
4. Keep `learned_unet_orgdet_test_v1` locked. Hash `c45c859610d29e90`. Kernel v1 COMPLETE / `audit_ok`. Do **not** overwrite v1. Do **not** resubmit.
5. Keep `learned_unet_retrain` 44b6 lanes rejected. Do **not** retune. Do **not** open `6bba` for retrain. Do **not** overwrite the completed retrain helper.
6. Keep discussion **740392** closed as `topic_id_not_found`.
7. Keep rejected overlays (link/motion/hung/conflict/orgop/division) closed. Do not retune.

## HUMAN_ACTION_REQUIRED

- Grant this worker `issues:read` so cron runs can read issue #1.
- Do **not** resubmit. Mahdi already initiated the Kaggle UI submit of `mahdimzafari/biohub-learned-unet-orgdet-test-v1`. Competition ref **56466744** is **pending**; public score not returned. Poll read-only. This repo does not authorize `kaggle competitions submit`. Do not resubmit association_v2 **56399003**. Do not submit orgdet-gap-test, prune-test, or infer-family gap-test unless a later message separately authorizes that extra UI click.

## Later (blocked or expensive)

- complementary_v1, linkable_v1, learned_unet_division_v1, learned_unet_link_v1, learned_unet_link_v2, learned_unet_motion_v1, learned_unet_hung_v1, learned_unet_conflict_v1, learned_unet_orgop_v1, and the `learned_unet_retrain` 44b6 lanes are closed.
- Copying public 199-video / DeepCenter checkpoints is forbidden for any KEEP_CANDIDATE path.
- Poll **56466744** read-only until Kaggle returns a terminal status and `publicScore`. Do not invent a leaderboard score. Do not retune from any returned public score.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, or invent a leaderboard score.
