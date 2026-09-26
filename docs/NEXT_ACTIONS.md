# Next actions

Use issue #1 as the command channel. Do one highest-priority unblocked task per run. Continue the active experiment draft PR when it exists.

## Now

1. Keep draft PR #2 (`cursor/kaggle-baseline-association-0b83`) open. Do not merge. Do not overwrite locked `association_v2`.
2. Keep draft PR #4 (`cursor/competition-operational-strategy-3381`) open. Official holdout is done. Do **not** retune.
3. Keep draft PR #5 (`cursor/complementary-detection-v1`) open. `reject_complementary_v1`. Do **not** retune.
4. Keep draft PR #6 (`cursor/linkable-complementary-v1`) open. `reject_linkable_v1`. Do **not** retune.
5. Continue draft PR #7 (`cursor/learned-unet-v1`). Holdout champion is `learned_unet_orgdet_gap_prune_v1` hash `862e3e7f7ac0330e`: official 44b6 **0.693437 vs orgdet-gap 0.687221**; official 6bba **0.723018 vs 0.722805**. Do **not** retune. Do **not** score 6bba again. New public-test slug `learned_unet_orgdet_gap_prune_test_v1` hash `52a30a5843e0a880` is packaged. Do **not** overwrite `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1` v1. `learned_unet_orgilp_v1` hash `dbc6134cddf2048d` official 44b6 **0.676811 vs 0.687221** is **rejected**. Retrain lanes remain rejected. Do **not** attach 199-video public weights.
6. Do **not** submit any learned slug from the CLI.

## Next (this family)

1. Launch and collect **one** private public-test rehearsal of `mahdimzafari/biohub-learned-unet-orgdet-gap-prune-test-v1`. Hidden-test projection ~4.35 h under 12 h. Independently audit `submission.csv`. Mark `READY_FOR_MANUAL_SUBMISSION_REVIEW` only if every gate passes. Do **not** overwrite orgdet-gap-test v1. Do **not** submit from the CLI.
2. Keep `learned_unet_orgdet_gap_prune_v1` locked. Hash `862e3e7f7ac0330e`. Official 44b6 **0.693437**. Official 6bba **0.723018**. Do **not** retune. Do **not** score 6bba again.
3. Keep `learned_unet_orgdet_gap_v1` locked. Hash `ee31f36151325f9a`. Official 44b6 **0.687221**. Official 6bba **0.722805**. Do **not** retune. Do **not** score 6bba again.
4. Keep `learned_unet_orgilp_v1` rejected. Hash `dbc6134cddf2048d`. Official 44b6 **0.676811**. Frozen `6bba` not opened. Do **not** retune. Do **not** launch `learned_unet_orgilp_gap_v1`.
5. Keep `learned_unet_orgdet_gap_test_v1` locked. Hash `1ce4837ae16bee52`. Slug is in `FORBIDDEN_SLUGS`. Additional competition submission is **not** authorized this cycle. Do **not** overwrite v1.
6. Keep `learned_unet_orgdet_test_v1` locked. Hash `c45c859610d29e90`. Kernel v1 COMPLETE / `audit_ok`. Do **not** overwrite v1. Do **not** resubmit.
7. Keep `learned_unet_retrain` 44b6 lanes rejected. Do **not** retune. Do **not** open `6bba` for retrain.
8. Keep rejected overlays (link/motion/hung/conflict/orgop/orgilp/division) closed. Do not retune.
9. Read-only orgdet-test submit poll **2026-09-22T17:41:09Z** recorded competition ref **56466744** as **complete** with Kaggle-returned public API string **`0.612`**. Private score **not returned**. Decision `public_score_returned`. Do **not** invent extra digits. Do **not** resubmit. Do **not** re-poll **56466744** as pending. Do **not** retune from **0.612** or **0.553**.

## HUMAN_ACTION_REQUIRED

- Grant this worker `issues:read` so cron runs can read issue #1.
- Do **not** resubmit. Mahdi already UI-submitted `mahdimzafari/biohub-learned-unet-orgdet-test-v1`. Competition ref **56466744** is **complete**; public API string **`0.612`**; private not returned. This repo does not authorize `kaggle competitions submit`. Do not resubmit association_v2 **56399003**. Do not submit orgdet-gap-test, prune-test, infer-family gap-test, or orgdet-gap-prune-test unless a later message separately authorizes that extra UI click. Do not use **0.612** or **0.553** for tuning.

## Later (blocked or expensive)

- complementary_v1, linkable_v1, learned_unet_division_v1, learned_unet_link_v1, learned_unet_link_v2, learned_unet_motion_v1, learned_unet_hung_v1, learned_unet_conflict_v1, learned_unet_orgop_v1, learned_unet_orgilp_v1, and the `learned_unet_retrain` 44b6 lanes are closed.
- Copying public 199-video / DeepCenter checkpoints is forbidden for any KEEP_CANDIDATE path.
- **56466744** is terminal. Public API **`0.612`** is recorded. Do not re-poll it as pending. Do not invent extra digits. Do not retune from **0.612** or **0.553**. Additional UI submit of preferred worker remains unauthorized unless a later message separately authorizes that extra click.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, or invent a leaderboard score.
