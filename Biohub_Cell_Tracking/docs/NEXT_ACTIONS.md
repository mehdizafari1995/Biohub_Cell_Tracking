# Next actions

Use issue #1 as the command channel. Do one highest-priority unblocked task per run. Continue the active experiment draft PR when it exists.

## Now

1. Keep draft PR #2 (`cursor/kaggle-baseline-association-0b83`) open. Do not merge. Do not overwrite locked `association_v2`. Complete frozen validation on both `6bba` sequences is already recorded (association_v2 **0.591570**). Do not redo that download/score cycle.
2. Keep draft PR #4 (`cursor/competition-operational-strategy-3381`) open. Official holdout is done: combined **0.591570** identical to association_v2, **0 TP / 0 FP / 5 FN**. Keep the overlay; do **not** retune gates. Do not relaunch rejected `division_v1`.
3. Keep draft PR #5 (`cursor/complementary-detection-v1`) open. complementary_v1 was locked from 44b6 then scored **once** on frozen `6bba`: **0.591384 vs association_v2 0.591570**. Decision **`reject_complementary_v1`**. Do **not** retune isolation/intensity gates.
4. Keep draft PR #6 (`cursor/linkable-complementary-v1`) open. linkable_v1 was locked from 44b6 then scored **once** on frozen `6bba`: **0.591458 vs association_v2 0.591570**. Decision **`reject_linkable_v1`**. Recall unchanged; extra T_pred +189. Do **not** retune the 12 µm / max_dt=1 temporal gate.
5. Continue draft PR #7 (`cursor/learned-unet-v1`). Training helper `mahdimzafari/biohub-learned-unet-v1` v3 remains **COMPLETE / trained** (hash `4c38035d6657f2ad`; no `submission.csv`). Do **not** overwrite it. Infer hash `d7578b5f64ddac08` (44b6 tune 0.655751). Holdout hash `d374a21f03dddc04`: official frozen `6bba` **0.693605**. No-gap test-time hash `7a8725b818fc609b`: private kernel v1 **COMPLETE** on public test (306.5 s, 1184 MB, `audit_ok`). `learned_unet_division_v1` rejected on 44b6 (**0.655235 vs 0.655751**). `learned_unet_gap_v1` hash `f4d888d190c45a4a` official-scored on 44b6 then once on frozen `6bba`: **0.701026 vs no-gap 0.693605**, **`keep_candidate_learned_unet_gap_v1`**. Gap test-time hash `85379c198ebc7241`: private kernel `mahdimzafari/biohub-learned-unet-gap-test-v1` v1 **COMPLETE / inferred** on public test (330.0 s, 1184 MB, `audit_ok`). Gap-test is **not** submitted. `learned_unet_prune_v1` hash `115753ea16aaf021` official-scored on 44b6 then once on frozen `6bba`: **0.701378 vs gap 0.701026**, **`keep_candidate_learned_unet_prune_v1`**. Prune test-time hash `074a71dec68304e8`: private kernel `mahdimzafari/biohub-learned-unet-prune-test-v1` v1 **COMPLETE / inferred** on public test (348.2 s, 1202 MB, `audit_ok`, 250864/242412, 796 isolates dropped, edges match gap-test). Hidden-test projection **4.81 h**. **Not submitted.** `READY_FOR_MANUAL_SUBMISSION_REVIEW`. Public 0.94 parity audit posted (`reports/public_094_parity_audit.md`). Follow-up topics API retrieved **738217** metadata (focus3d title; body still SPA). **740392** still unread. Read-only prune-test submit poll **2026-09-21T05:47:28Z** still `human_action_required_ui_submit` (only association_v2 **56399003** / **0.553**). `learned_unet_orgop_v1` hash `250c5bfacd4a8198` official-scored on 44b6 only: **0.0 vs infer 0.655751**. Decision **`reject_learned_unet_orgop_v1`**. Softmax>0.5 produced 0 candidate edges; ILP skipped. Frozen `6bba` was not opened. Do **not** retune. Do **not** attach 199-video public weights. Do **not** score 6bba again. Transformer/motion/hung/conflict/orgop linkers remain rejected. Do **not** overwrite completed slugs. Next specified (not launched) experiment: `learned_unet_retrain_v1` hash `92a419926fa7e40b` (`LAUNCH_AUTHORIZED=False`) — +3-epoch 44b6-only continuation of locked `learned_unet_v1_last.pt`. Reserved slug `mahdimzafari/biohub-learned-unet-retrain-v1`. Do **not** launch until a reviewed cycle. Do **not** overwrite the locked train helper.
6. Do **not** submit association_v2, division_v2, complementary_v1, linkable_v1, learned_unet_v1, learned_unet_infer_v1, learned_unet_holdout_v1, learned_unet_test_v1, learned_unet_division_v1, learned_unet_gap_v1, learned_unet_gap_test_v1, learned_unet_link_v1, learned_unet_link_v2, learned_unet_motion_v1, learned_unet_hung_v1, learned_unet_conflict_v1, learned_unet_orgop_v1, learned_unet_prune_v1, learned_unet_prune_test_v1, or learned_unet_retrain_v1 from the CLI.

## Next (this family)

1. `HUMAN_ACTION_REQUIRED` — Mahdi must **Submit to Competition** from the Kaggle UI on `mahdimzafari/biohub-learned-unet-prune-test-v1` version 1 **before the 2026-09-22 23:59 UTC entry deadline**. Do **not** submit from the CLI. Do **not** overwrite v1.
2. Keep `learned_unet_retrain_v1` specified, not launched. Hash `92a419926fa7e40b`. `LAUNCH_AUTHORIZED=False`. Do **not** push `mahdimzafari/biohub-learned-unet-retrain-v1`. Do **not** overwrite `mahdimzafari/biohub-learned-unet-v1`. Wait for a reviewed cycle.
3. Keep `learned_unet_orgop_v1` rejected. Hash `250c5bfacd4a8198`. Official 44b6 **0.0 vs infer 0.655751**. Do **not** retune threshold/ILP. Do **not** score frozen `6bba`. Do **not** overwrite `mahdimzafari/biohub-learned-unet-orgop-v1` v1.
4. Keep `learned_unet_conflict_v1` rejected. Hash `8503d804fb9e9f1d`. Do **not** retune. Do **not** score frozen `6bba` for that overlay. Do **not** package another collision-Hungarian variant.
5. Keep `learned_unet_link_v1` / `link_v2` / `motion_v1` / `hung_v1` rejected. Hung hash `f67bc97d0521dc78` was independently reconfirmed on 44b6 at **0.439023 vs infer 0.655751** (2026-09-21T04:12Z). Do **not** retune those hashes. Do **not** score frozen `6bba` for those overlays.
6. Keep `mahdimzafari/biohub-learned-unet-gap-test-v1` v1 locked. Do **not** overwrite v1. Do **not** retune. Do **not** score 6bba again for the gap overlay.
7. Keep `learned_unet_prune_v1` / `learned_unet_prune_test_v1` locked after one-shot holdout and the public-test rehearsal. Hashes `115753ea16aaf021` / `074a71dec68304e8`. Do **not** retune. Do **not** score frozen `6bba` again. Do **not** overwrite prune-test v1.
8. Keep `learned_unet_division_v1` rejected. Do **not** retune gates. Do **not** score frozen `6bba` for that overlay.
9. Keep `mahdimzafari/biohub-learned-unet-test-v1` v1 locked (no-gap). Do **not** overwrite v1.

## HUMAN_ACTION_REQUIRED

- Grant this worker `issues:read` so cron runs can read issue #1.
- Mahdi must **Submit to Competition** from the Kaggle UI on `mahdimzafari/biohub-learned-unet-prune-test-v1` version 1 **before the 2026-09-22 23:59 UTC entry deadline**. This repo does not authorize `kaggle competitions submit`. Do not resubmit association_v2 **56399003**. Do not submit gap-test now that prune-test v1 is COMPLETE / `audit_ok`.

## Later (blocked or expensive)

- complementary_v1, linkable_v1, learned_unet_division_v1, learned_unet_link_v1, learned_unet_link_v2, learned_unet_motion_v1, learned_unet_hung_v1, learned_unet_conflict_v1, and learned_unet_orgop_v1 are closed. Do not retune transformer-softmax, constant-velocity, full-frame Hungarian, shared-nearest collision-Hungarian, or organizer 0.99+ILP on the locked 3-epoch 44b6 weights. A later linker follow-up needs a new hypothesis, not another assignment on the same 12 µm last-position pairs or another empty softmax>0.5 graph. `learned_unet_retrain_v1` is the specified next detection experiment (hash `92a419926fa7e40b`); do not launch until review.
- Copying public 199-video / DeepCenter checkpoints is forbidden for any KEEP_CANDIDATE path. Optional public-test-only in-sample replay must be tagged `not_for_tuning` and is not authorized this cycle.
- Optional private kernel of association_v2 + division_v2 overlay under a **new** slug. Overlay was a no-op on holdout.
- Harden the learned gap/prune test-time notebook for hidden test only if a documented runtime/memory issue appears.
- Final Kaggle submission: `HUMAN_ACTION_REQUIRED` — Mahdi must submit version output from the Kaggle UI.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, or invent a leaderboard score.
