# Next actions

Use issue #1 as the command channel. Do one highest-priority unblocked task per run. Continue the active experiment draft PR when it exists.

## Now

1. Keep draft PR #2 (`cursor/kaggle-baseline-association-0b83`) open. Do not merge. Do not overwrite locked `association_v2`.
2. Keep draft PR #4 (`cursor/competition-operational-strategy-3381`) open. Official holdout is done. Do **not** retune.
3. Keep draft PR #5 (`cursor/complementary-detection-v1`) open. `reject_complementary_v1`. Do **not** retune.
4. Keep draft PR #6 (`cursor/linkable-complementary-v1`) open. `reject_linkable_v1`. Do **not** retune.
5. Continue draft PR #7 (`cursor/learned-unet-v1`). Holdout champion is `learned_unet_orgdet_gap_v1` hash `ee31f36151325f9a`: official 6bba **0.722805 vs orgdet 0.705226**. Public-test slug `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1` v1 **COMPLETE / inferred / `audit_ok`**. Hash `1ce4837ae16bee52`. Do **not** overwrite v1. `learned_unet_orgilp_v1` hash `dbc6134cddf2048d` official 44b6 **0.676811 vs 0.687221** is **rejected**. Do **not** retune. Do **not** open `6bba` for orgilp. Do **not** launch `orgilp_gap`. Retrain lanes remain rejected. Do **not** attach 199-video public weights. Do **not** score 6bba again.
6. Do **not** submit any learned slug from the CLI.

## Next (this family)

1. Read-only orgdet-test submit poll **2026-09-22T17:41:09Z** recorded competition ref **56466744** as **complete** with Kaggle-returned public API string **`0.612`**. Private score **not returned**. Notebook identity remains proven as `mahdimzafari/biohub-learned-unet-orgdet-test-v1` (`scriptVersionId` **351557863**). Decision `public_score_returned`. Do **not** invent extra digits. Do **not** resubmit. Do **not** re-poll **56466744** as pending. Do **not** retune from **0.612** or **0.553**.
2. Keep `learned_unet_orgdet_gap_v1` locked. Hash `ee31f36151325f9a`. Official 44b6 **0.687221 vs 0.679578**. Official 6bba **0.722805 vs 0.705226**. Do **not** retune. Do **not** score 6bba again.
3. Keep `learned_unet_orgilp_v1` rejected. Hash `dbc6134cddf2048d`. Official 44b6 **0.676811**. Preflight passed; ILP wrote 58805+26045 edges. Below orgdet-gap and below greedy orgdet. Frozen `6bba` not opened. Do **not** retune. Do **not** launch `learned_unet_orgilp_gap_v1` (consecutive ILP already lost to greedy).
4. Keep `learned_unet_orgdet_gap_test_v1` locked. Hash `1ce4837ae16bee52`. Slug is in `FORBIDDEN_SLUGS`. Additional competition submission is **not** authorized this cycle. Do **not** overwrite v1.
5. Keep `learned_unet_orgdet_test_v1` locked. Hash `c45c859610d29e90`. Kernel v1 COMPLETE / `audit_ok`. Do **not** overwrite v1. Do **not** resubmit.
6. Keep `learned_unet_retrain` 44b6 lanes rejected. Do **not** retune. Do **not** open `6bba` for retrain.
7. Keep rejected overlays (link/motion/hung/conflict/orgop/orgilp/division) closed. Do not retune.
8. If nothing new is authorized on PR #7 or issue #1, do nothing rather than create noise.

## HUMAN_ACTION_REQUIRED

- Grant this worker `issues:read` so cron runs can read issue #1.
- Do **not** resubmit. Mahdi already UI-submitted `mahdimzafari/biohub-learned-unet-orgdet-test-v1`. Competition ref **56466744** is **complete**; public API string **`0.612`**; private not returned. This repo does not authorize `kaggle competitions submit`. Do not resubmit association_v2 **56399003**. Do not submit orgdet-gap-test, prune-test, or infer-family gap-test unless a later message separately authorizes that extra UI click. Do not use **0.612** or **0.553** for tuning.

## Later (blocked or expensive)

- complementary_v1, linkable_v1, learned_unet_division_v1, learned_unet_link_v1, learned_unet_link_v2, learned_unet_motion_v1, learned_unet_hung_v1, learned_unet_conflict_v1, learned_unet_orgop_v1, learned_unet_orgilp_v1, and the `learned_unet_retrain` 44b6 lanes are closed.
- Copying public 199-video / DeepCenter checkpoints is forbidden for any KEEP_CANDIDATE path.
- **56466744** is terminal. Public API **`0.612`** is recorded. Do not re-poll it as pending. Do not invent extra digits. Do not retune from **0.612** or **0.553**. Additional UI submit of preferred worker `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1` remains unauthorized unless a later message separately authorizes that extra click.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, or invent a leaderboard score.
