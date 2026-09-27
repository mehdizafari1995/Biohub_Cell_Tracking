# Next actions

Use issue #1 as the command channel. Do one highest-priority unblocked task per run. Continue the active experiment draft PR when it exists.

## Now

1. Keep draft PR #2 (`cursor/kaggle-baseline-association-0b83`) open. Do not merge. Do not overwrite locked `association_v2`.
2. Keep draft PR #4 (`cursor/competition-operational-strategy-3381`) open. Official holdout is done. Do **not** retune.
3. Keep draft PR #5 (`cursor/complementary-detection-v1`) open. `reject_complementary_v1`. Do **not** retune.
4. Keep draft PR #6 (`cursor/linkable-complementary-v1`) open. `reject_linkable_v1`. Do **not** retune.
5. Continue draft PR #7 (`cursor/learned-unet-v1`). Holdout champion remains `learned_unet_orgdet_fnn_v1` hash `95e9300eb84ba71c`: official 44b6 **0.724653 vs ELF 0.709838**; official one-shot 6bba **0.752091 vs ELF 0.737787**. Public-test slug `mahdimzafari/biohub-learned-unet-orgdet-fnn-test-v1` v1 is **COMPLETE / inferred / `audit_ok`** (independent `audit_ok`) but **not** `READY_FOR_MANUAL_SUBMISSION_REVIEW` because hidden projection **11.04 h** trips `fail_closed_approaches_12h`. This cron packaged a new immutable faster slug `mahdimzafari/biohub-learned-unet-orgdet-fnn-fast-test-v1` hash `9b9763e26bc0f037` (same overlay, vectorized NN). Preferred packaged fallback remains elf-test `ad375d39e13098d1` until that rehearsal is COMPLETE and hidden-safe. Do **not** overwrite fnn-test v1, fnn-fast-test after it completes, or elf-test v1. Do **not** score 6bba again. Do **not** retune `nn_tie_eps` or `max_dt1_edge_um`. TCF remains **rejected**. Retrain lanes remain rejected. Do **not** attach 199-video public weights.
6. Do **not** submit any learned slug from the CLI.

## Next (this family)

1. Keep `learned_unet_orgdet_fnn_test_v1` locked. Hash `1af763e6ffc23188`. Slug is in `FORBIDDEN_SLUGS`. v1 **COMPLETE / inferred / `audit_ok`**. Independent `audit_ok`. `6bba_05db0fb1` **87339/81132** matches locked FNN holdout. Hidden **11.04 h** under 12 h but `fail_closed_approaches_12h`. **Not** `READY_FOR_MANUAL_SUBMISSION_REVIEW`. Do **not** overwrite v1. Do **not** push a second version. Do **not** submit.
2. Keep `learned_unet_orgdet_fnn_v1` locked. Hash `95e9300eb84ba71c`. Official 44b6 **0.724653**. Official 6bba **0.752091**. Do **not** retune `nn_tie_eps`. Do **not** score 6bba again.
3. Keep `learned_unet_orgdet_elf_test_v1` locked. Hash `ad375d39e13098d1`. Slug is in `FORBIDDEN_SLUGS`. v1 **COMPLETE / inferred / `audit_ok`**. Independent `audit_ok`. `6bba_05db0fb1` **88322/83959** matches locked ELF holdout. Hidden **4.88 h**. `READY_FOR_MANUAL_SUBMISSION_REVIEW` as fallback. Additional competition submission is **not** authorized unless a later message separately authorizes that extra UI click. Do **not** overwrite v1. Do **not** push a second version.
4. Keep `learned_unet_orgdet_elf_v1` locked. Hash `292c4f1f968f948d`. Official 44b6 **0.709838**. Official 6bba **0.737787**. Do **not** retune `max_dt1_edge_um`. Do **not** score 6bba again.
5. Keep `learned_unet_orgdet_gap_prune_test_v1` locked. Hash `52a30a5843e0a880`. Slug is in `FORBIDDEN_SLUGS`. `READY_FOR_MANUAL_SUBMISSION_REVIEW` as fallback. Additional competition submission is **not** authorized unless a later message separately authorizes that extra UI click. Do **not** overwrite v1. Do **not** push a second version.
6. Keep `learned_unet_orgdet_gap_prune_v1` locked. Hash `862e3e7f7ac0330e`. Official 44b6 **0.693437**. Official 6bba **0.723018**. Do **not** retune. Do **not** score 6bba again.
7. Keep `learned_unet_orgdet_gap_v1` locked. Hash `ee31f36151325f9a`. Official 44b6 **0.687221**. Official 6bba **0.722805**. Do **not** retune. Do **not** score 6bba again.
8. Keep `learned_unet_orgdet_gap_test_v1` locked. Hash `1ce4837ae16bee52`. Slug is in `FORBIDDEN_SLUGS`. UI submit ref **56583783** is **complete** with public API **`0.602`**. Decision `public_score_returned`. Do **not** overwrite v1. Do **not** resubmit. Do **not** re-poll **56583783** as pending. Do **not** retune from **0.602**.
9. Keep `learned_unet_orgdet_test_v1` locked. Hash `c45c859610d29e90`. Kernel v1 COMPLETE / `audit_ok`. UI submit ref **56466744** public API **`0.612`**. Do **not** overwrite v1. Do **not** resubmit. Do **not** retune from **0.612**.
10. Keep `learned_unet_orgilp_v1` rejected. Hash `dbc6134cddf2048d`. Official 44b6 **0.676811**. Frozen `6bba` not opened. Do **not** retune. Do **not** launch `learned_unet_orgilp_gap_v1`.
11. Keep `learned_unet_retrain` 44b6 lanes rejected. Do **not** retune. Do **not** open `6bba` for retrain.
12. Keep rejected overlays (link/motion/hung/conflict/orgop/orgilp/division/tcf) closed. Do not retune.
13. Keep `learned_unet_orgdet_tcf_v1` rejected. Hash `a6b46dfcf410478f`. Official 44b6 **0.695181** vs prune **0.693437**. Frozen `6bba` not opened. Do **not** retune `min_component_nodes`. Do **not** relaunch TCF.
14. This cron packaged `learned_unet_orgdet_fnn_fast_test_v1` hash `9b9763e26bc0f037` (same locked overlay, vectorized NN, new slug). Do **not** overwrite fnn-test v1. Next: push/collect that private rehearsal if not already COMPLETE; mark `READY_FOR_MANUAL_SUBMISSION_REVIEW` only if hidden projection stays under the 90% / 12 h fail-closed gate and audits pass. Otherwise one new 44b6-first hypothesis that beats FNN **0.724653** by **+0.002000** (threshold **0.726653**) and holdout **0.752091**. Do not pick another 8.0 µm dt=1 retune, short-fragment retune, or 12 µm assignment. Do **not** use **0.602**, **0.612**, or **0.553** for tuning. Do **not** overwrite fnn-test v1.

## HUMAN_ACTION_REQUIRED

- Grant this worker `issues:read` so cron runs can read issue #1.
- Do **not** resubmit. Competition list already has three complete rows: association_v2 **56399003** public **`0.553`**, orgdet-test **56466744** public **`0.612`**, orgdet-gap-test **56583783** public **`0.602`**. This repo does not authorize `kaggle competitions submit`. Do not submit orgdet-elf-test, orgdet-fnn-test, or orgdet-gap-prune-test unless a later message separately authorizes that extra UI click. FNN-test is **not** READY. Do not use **0.602**, **0.612**, or **0.553** for tuning.

## Later (blocked or expensive)

- complementary_v1, linkable_v1, learned_unet_division_v1, learned_unet_link_v1, learned_unet_link_v2, learned_unet_motion_v1, learned_unet_hung_v1, learned_unet_conflict_v1, learned_unet_orgop_v1, learned_unet_orgilp_v1, learned_unet_orgdet_tcf_v1, and the `learned_unet_retrain` 44b6 lanes are closed.
- Copying public 199-video / DeepCenter checkpoints is forbidden for any KEEP_CANDIDATE path.
- **56466744** and **56583783** are terminal. Public API **`0.612`** and **`0.602`** are recorded. Do not re-poll them as pending. Do not invent extra digits. Do not retune from **0.612**, **0.602**, or **0.553**. Additional UI submit of elf-test or prune-test remains unauthorized unless a later message separately authorizes that extra click.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, or invent a leaderboard score.
