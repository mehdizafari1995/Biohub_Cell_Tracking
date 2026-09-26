# Next actions

Use issue #1 as the command channel. Do one highest-priority unblocked task per run. Continue the active experiment draft PR when it exists.

## Now

1. Keep draft PR #2 (`cursor/kaggle-baseline-association-0b83`) open. Do not merge. Do not overwrite locked `association_v2`.
2. Keep draft PR #4 (`cursor/competition-operational-strategy-3381`) open. Official holdout is done. Do **not** retune.
3. Keep draft PR #5 (`cursor/complementary-detection-v1`) open. `reject_complementary_v1`. Do **not** retune.
4. Keep draft PR #6 (`cursor/linkable-complementary-v1`) open. `reject_linkable_v1`. Do **not** retune.
5. Continue draft PR #7 (`cursor/learned-unet-v1`). Holdout champion is `learned_unet_orgdet_gap_prune_v1` hash `862e3e7f7ac0330e`: official 44b6 **0.693437 vs orgdet-gap 0.687221**; official 6bba **0.723018 vs 0.722805**. Preferred public-test worker is `learned_unet_orgdet_gap_prune_test_v1` hash `52a30a5843e0a880` v1 **COMPLETE / inferred / `audit_ok`**. `READY_FOR_MANUAL_SUBMISSION_REVIEW`. Do **not** retune. Do **not** score 6bba again. Do **not** overwrite v1. `learned_unet_orgilp_v1` hash `dbc6134cddf2048d` official 44b6 **0.676811 vs 0.687221** is **rejected**. Retrain lanes remain rejected. Do **not** attach 199-video public weights.
6. Do **not** submit any learned slug from the CLI.

## Next (this family)

1. Keep `learned_unet_orgdet_gap_prune_test_v1` locked. Hash `52a30a5843e0a880`. Slug is in `FORBIDDEN_SLUGS`. `READY_FOR_MANUAL_SUBMISSION_REVIEW`. Additional competition submission is **not** authorized unless a later message separately authorizes that extra UI click. Do **not** overwrite v1. Do **not** push a second version.
2. Keep `learned_unet_orgdet_gap_prune_v1` locked. Hash `862e3e7f7ac0330e`. Official 44b6 **0.693437**. Official 6bba **0.723018**. Do **not** retune. Do **not** score 6bba again.
3. Keep `learned_unet_orgdet_gap_v1` locked. Hash `ee31f36151325f9a`. Official 44b6 **0.687221**. Official 6bba **0.722805**. Do **not** retune. Do **not** score 6bba again.
4. Keep `learned_unet_orgdet_gap_test_v1` locked. Hash `1ce4837ae16bee52`. Slug is in `FORBIDDEN_SLUGS`. UI submit ref **56583783** is **complete** with public API **`0.602`**. Decision `public_score_returned`. Do **not** overwrite v1. Do **not** resubmit. Do **not** re-poll **56583783** as pending. Do **not** retune from **0.602**.
5. Keep `learned_unet_orgdet_test_v1` locked. Hash `c45c859610d29e90`. Kernel v1 COMPLETE / `audit_ok`. UI submit ref **56466744** public API **`0.612`**. Do **not** overwrite v1. Do **not** resubmit. Do **not** retune from **0.612**.
6. Keep `learned_unet_orgilp_v1` rejected. Hash `dbc6134cddf2048d`. Official 44b6 **0.676811**. Frozen `6bba` not opened. Do **not** retune. Do **not** launch `learned_unet_orgilp_gap_v1`.
7. Keep `learned_unet_retrain` 44b6 lanes rejected. Do **not** retune. Do **not** open `6bba` for retrain.
8. Keep rejected overlays (link/motion/hung/conflict/orgop/orgilp/division) closed. Do not retune.
9. If continuing independent improvement on PR #7, pick one new 44b6-first hypothesis that is **not** a rejected overlay and is not another 12 µm assignment. Preregister the hash and rejection threshold before scoring. Open frozen 6bba only after a strict 44b6 win versus **0.693437**. Do **not** use **0.602**, **0.612**, or **0.553** for tuning.

## HUMAN_ACTION_REQUIRED

- Grant this worker `issues:read` so cron runs can read issue #1.
- Do **not** resubmit. Competition list already has three complete rows: association_v2 **56399003** public **`0.553`**, orgdet-test **56466744** public **`0.612`**, orgdet-gap-test **56583783** public **`0.602`**. This repo does not authorize `kaggle competitions submit`. Do not submit orgdet-gap-prune-test unless a later message separately authorizes that extra UI click. Do not use **0.602**, **0.612**, or **0.553** for tuning.

## Later (blocked or expensive)

- complementary_v1, linkable_v1, learned_unet_division_v1, learned_unet_link_v1, learned_unet_link_v2, learned_unet_motion_v1, learned_unet_hung_v1, learned_unet_conflict_v1, learned_unet_orgop_v1, learned_unet_orgilp_v1, and the `learned_unet_retrain` 44b6 lanes are closed.
- Copying public 199-video / DeepCenter checkpoints is forbidden for any KEEP_CANDIDATE path.
- **56466744** and **56583783** are terminal. Public API **`0.612`** and **`0.602`** are recorded. Do not re-poll them as pending. Do not invent extra digits. Do not retune from **0.612**, **0.602**, or **0.553**. Additional UI submit of prune-test remains unauthorized unless a later message separately authorizes that extra click.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, or invent a leaderboard score.
