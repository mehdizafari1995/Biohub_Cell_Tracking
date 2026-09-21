# Next actions

Use issue #1 as the command channel. Do one highest-priority unblocked task per run. Continue the active experiment draft PR when it exists.

## Now

1. Keep draft PR #2 (`cursor/kaggle-baseline-association-0b83`) open. Do not merge. Do not overwrite locked `association_v2`. Complete frozen validation on both `6bba` sequences is already recorded (association_v2 **0.591570**). Do not redo that download/score cycle.
2. Keep draft PR #4 (`cursor/competition-operational-strategy-3381`) open. Official holdout is done: combined **0.591570** identical to association_v2, **0 TP / 0 FP / 5 FN**. Keep the overlay; do **not** retune gates. Do not relaunch rejected `division_v1`.
3. Keep draft PR #5 (`cursor/complementary-detection-v1`) open. complementary_v1 was locked from 44b6 then scored **once** on frozen `6bba`: **0.591384 vs association_v2 0.591570**. Decision **`reject_complementary_v1`**. Do **not** retune isolation/intensity gates.
4. Keep draft PR #6 (`cursor/linkable-complementary-v1`) open. linkable_v1 was locked from 44b6 then scored **once** on frozen `6bba`: **0.591458 vs association_v2 0.591570**. Decision **`reject_linkable_v1`**. Recall unchanged; extra T_pred +189. Do **not** retune the 12 µm / max_dt=1 temporal gate.
5. Continue draft PR #7 (`cursor/learned-unet-v1`). Training helper `mahdimzafari/biohub-learned-unet-v1` v3 remains **COMPLETE / trained** (hash `4c38035d6657f2ad`; no `submission.csv`). Do **not** overwrite it. Infer hash `d7578b5f64ddac08` (44b6 tune 0.655751). Holdout hash `d374a21f03dddc04`: official frozen `6bba` **0.693605**. `learned_unet_orgdet_holdout_v1` hash `912f0c19b572a054`: official frozen `6bba` **0.705226 vs prune 0.701378**, **`keep_candidate_learned_unet_orgdet_holdout_v1`**. Private helper `mahdimzafari/biohub-learned-unet-orgdet-holdout-v1` v1 **COMPLETE / inferred** (173.8 s, 1185 MB, 121810/113640, no `submission.csv`). Do **not** score 6bba again for orgdet/prune/gap. `learned_unet_orgdet_test_v1` hash `c45c859610d29e90` is packaged (public-test GPU notebook at locked 0.99+greedy; **not pushed**). `learned_unet_prune_v1` hash `115753ea16aaf021` remains a locked prior holdout keep (**0.701378**). Prune test-time hash `074a71dec68304e8`: private kernel `mahdimzafari/biohub-learned-unet-prune-test-v1` v1 **COMPLETE / inferred** on public test (348.2 s, 1202 MB, `audit_ok`). **Not submitted.** `READY_FOR_MANUAL_SUBMISSION_REVIEW` until an orgdet test-time slug is COMPLETE. Read-only prune-test submit poll **2026-09-21T10:45:23Z** still `human_action_required_ui_submit` (only association_v2 **56399003** / **0.553**). `learned_unet_orgop_v1` hash `250c5bfacd4a8198` remains rejected on 44b6. `learned_unet_orgdet_v1` hash `80b20589198148b7` remains a 44b6 keep. Do **not** retune. Do **not** attach 199-video public weights. Transformer/motion/hung/conflict/orgop linkers remain rejected. Do **not** overwrite completed slugs. Next specified (packaged, not launched) continuation: `learned_unet_retrain_v1` hash `92a419926fa7e40b` (`LAUNCH_AUTHORIZED=False`). Next orgdet step: push public-test GPU notebook `learned_unet_orgdet_test_v1`. Do **not** launch retrain until a reviewed cycle. Do **not** overwrite the locked train helper.
6. Do **not** submit association_v2, division_v2, complementary_v1, linkable_v1, learned_unet_v1, learned_unet_infer_v1, learned_unet_holdout_v1, learned_unet_test_v1, learned_unet_division_v1, learned_unet_gap_v1, learned_unet_gap_test_v1, learned_unet_link_v1, learned_unet_link_v2, learned_unet_motion_v1, learned_unet_hung_v1, learned_unet_conflict_v1, learned_unet_orgop_v1, learned_unet_orgdet_v1, learned_unet_orgdet_holdout_v1, learned_unet_orgdet_test_v1, learned_unet_prune_v1, learned_unet_prune_test_v1, or learned_unet_retrain_v1 from the CLI.

## Next (this family)

1. Push the packaged public-test GPU notebook `learned_unet_orgdet_test_v1` (locked orgdet holdout operating point 0.99 / pool 3.0 / TTA + greedy 12 µm, hash `c45c859610d29e90`). Discover `test/*.zarr`. Do **not** overwrite prune-test v1, gap-test v1, no-gap test v1, or orgdet-holdout v1. Hidden-test projection ~4.8 h under 12 h. Wait COMPLETE, collect, confirm `audit_ok`.
2. `HUMAN_ACTION_REQUIRED` — Mahdi must **Submit to Competition** from the Kaggle UI on `mahdimzafari/biohub-learned-unet-prune-test-v1` version 1 **before the 2026-09-22 23:59 UTC entry deadline** unless an orgdet-test slug is COMPLETE / `audit_ok` first. Do **not** submit from the CLI. Do **not** overwrite v1.
3. Keep `learned_unet_orgdet_holdout_v1` locked after one-shot holdout. Hash `912f0c19b572a054`. Official 6bba **0.705226 vs prune 0.701378**. Do **not** retune. Do **not** score frozen `6bba` again. Do **not** overwrite `mahdimzafari/biohub-learned-unet-orgdet-holdout-v1` v1.
4. Keep `learned_unet_retrain_v1` packaged, not launched. Hash `92a419926fa7e40b`. `LAUNCH_AUTHORIZED=False`. Helper `notebooks/kaggle_learned_unet_retrain_v1.py` is packaged. Do **not** push `mahdimzafari/biohub-learned-unet-retrain-v1`. Do **not** overwrite `mahdimzafari/biohub-learned-unet-v1`. Wait for a reviewed cycle.
5. Keep discussion **740392** closed as `topic_id_not_found`. Do **not** retry the by-id/hot-page lookup every hour. It was not used as a source.
6. Keep `learned_unet_orgop_v1` rejected. Hash `250c5bfacd4a8198`. Official 44b6 **0.0 vs infer 0.655751**. Do **not** retune threshold/ILP. Do **not** score frozen `6bba` for orgop. Do **not** overwrite `mahdimzafari/biohub-learned-unet-orgop-v1` v1.
7. Keep `learned_unet_orgdet_v1` as a 44b6 keep. Hash `80b20589198148b7`. Official 44b6 **0.679578 vs infer 0.655751**. Do **not** retune. Holdout scored once via `learned_unet_orgdet_holdout_v1`.
8. Keep `learned_unet_conflict_v1` rejected. Hash `8503d804fb9e9f1d`. Do **not** retune. Do **not** score frozen `6bba` for that overlay. Do **not** package another collision-Hungarian variant.
9. Keep `learned_unet_link_v1` / `link_v2` / `motion_v1` / `hung_v1` rejected. Hung hash `f67bc97d0521dc78` was independently reconfirmed on 44b6 at **0.439023 vs infer 0.655751** (2026-09-21T04:12Z). Do **not** retune those hashes. Do **not** score frozen `6bba` for those overlays.
10. Keep `mahdimzafari/biohub-learned-unet-gap-test-v1` v1 locked. Do **not** overwrite v1. Do **not** retune. Do **not** score 6bba again for the gap overlay.
11. Keep `learned_unet_prune_v1` / `learned_unet_prune_test_v1` locked after one-shot holdout and the public-test rehearsal. Hashes `115753ea16aaf021` / `074a71dec68304e8`. Do **not** retune. Do **not** score frozen `6bba` again. Do **not** overwrite prune-test v1.
12. Keep `learned_unet_division_v1` rejected. Do **not** retune gates. Do **not** score frozen `6bba` for that overlay.
13. Keep `mahdimzafari/biohub-learned-unet-test-v1` v1 locked (no-gap). Do **not** overwrite v1.

## HUMAN_ACTION_REQUIRED

- Grant this worker `issues:read` so cron runs can read issue #1.
- Mahdi must **Submit to Competition** from the Kaggle UI on `mahdimzafari/biohub-learned-unet-prune-test-v1` version 1 **before the 2026-09-22 23:59 UTC entry deadline**. This repo does not authorize `kaggle competitions submit`. Do not resubmit association_v2 **56399003**. Do not submit gap-test now that prune-test v1 is COMPLETE / `audit_ok`.

## Later (blocked or expensive)

- complementary_v1, linkable_v1, learned_unet_division_v1, learned_unet_link_v1, learned_unet_link_v2, learned_unet_motion_v1, learned_unet_hung_v1, learned_unet_conflict_v1, and learned_unet_orgop_v1 are closed. Do not retune transformer-softmax, constant-velocity, full-frame Hungarian, shared-nearest collision-Hungarian, or organizer 0.99+ILP on the locked 3-epoch 44b6 weights. `learned_unet_orgdet_v1` is a 44b6 keep (greedy on orgop detections); do not retune it. `learned_unet_orgdet_holdout_v1` is the holdout champion (0.705226); do not score 6bba again. `learned_unet_orgdet_test_v1` is packaged (hash `c45c859610d29e90`); do not overwrite prune-test. `learned_unet_retrain_v1` is packaged (hash `92a419926fa7e40b`); do not launch until review.
- Copying public 199-video / DeepCenter checkpoints is forbidden for any KEEP_CANDIDATE path. Optional public-test-only in-sample replay must be tagged `not_for_tuning` and is not authorized this cycle.
- Optional private kernel of association_v2 + division_v2 overlay under a **new** slug. Overlay was a no-op on holdout.
- Harden the learned gap/prune test-time notebook for hidden test only if a documented runtime/memory issue appears.
- Final Kaggle submission: `HUMAN_ACTION_REQUIRED` — Mahdi must submit version output from the Kaggle UI.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, or invent a leaderboard score.
