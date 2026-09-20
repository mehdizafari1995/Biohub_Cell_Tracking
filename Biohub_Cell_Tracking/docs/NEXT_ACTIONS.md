# Next actions

Use issue #1 as the command channel. Do one highest-priority unblocked task per run. Continue the active experiment draft PR when it exists.

## Now

1. Keep draft PR #2 (`cursor/kaggle-baseline-association-0b83`) open. Do not merge. Do not overwrite locked `association_v2`. Complete frozen validation on both `6bba` sequences is already recorded (association_v2 **0.591570**). Do not redo that download/score cycle.
2. Keep draft PR #4 (`cursor/competition-operational-strategy-3381`) open. Official holdout is done: combined **0.591570** identical to association_v2, **0 TP / 0 FP / 5 FN**. Keep the overlay; do **not** retune gates. Do not relaunch rejected `division_v1`.
3. Keep draft PR #5 (`cursor/complementary-detection-v1`) open. complementary_v1 was locked from 44b6 then scored **once** on frozen `6bba`: **0.591384 vs association_v2 0.591570**. Decision **`reject_complementary_v1`**. Do **not** retune isolation/intensity gates.
4. Keep draft PR #6 (`cursor/linkable-complementary-v1`) open. linkable_v1 was locked from 44b6 then scored **once** on frozen `6bba`: **0.591458 vs association_v2 0.591570**. Decision **`reject_linkable_v1`**. Recall unchanged; extra T_pred +189. Do **not** retune the 12 µm / max_dt=1 temporal gate.
5. Continue draft PR #7 (`cursor/learned-unet-v1`). Training hash `4c38035d6657f2ad`. Infer hash `d7578b5f64ddac08` (44b6 tune 0.655751). Holdout hash `d374a21f03dddc04`: official frozen `6bba` **0.693605**. No-gap test-time hash `7a8725b818fc609b`: private kernel v1 **COMPLETE** on public test (306.5 s, 1184 MB, `audit_ok`). `learned_unet_division_v1` hash `8761fb66cc2254f3` was official-scored on 44b6 only: **0.655235 vs 0.655751**, **`reject_learned_unet_division_v1`**. `learned_unet_gap_v1` hash `f4d888d190c45a4a` was official-scored on 44b6 then once on frozen `6bba`: **0.701026 vs no-gap 0.693605**, **`keep_candidate_learned_unet_gap_v1`**. Gap test-time hash `85379c198ebc7241`: private kernel `mahdimzafari/biohub-learned-unet-gap-test-v1` v1 **COMPLETE / inferred** on public test (330.0 s, 1184 MB, `audit_ok`, 251660/242412/0). Do **not** overwrite completed slugs. Do **not** retune. Do **not** score 6bba again.
6. Do **not** submit association_v2, division_v2, complementary_v1, linkable_v1, learned_unet_v1, learned_unet_infer_v1, learned_unet_holdout_v1, learned_unet_test_v1, learned_unet_division_v1, learned_unet_gap_v1, or learned_unet_gap_test_v1 from the CLI.

## Next (this family)

1. Keep `mahdimzafari/biohub-learned-unet-gap-test-v1` v1 locked. Worker `audit_ok`. Independent stream audit `audit_ok`. `6bba_05db0fb1` counts match the locked holdout gap overlay. Do **not** overwrite v1. Do **not** retune. Do **not** score 6bba again.
2. Keep `learned_unet_division_v1` rejected. Do **not** retune gates. Do **not** score frozen `6bba` for that overlay.
3. Keep `mahdimzafari/biohub-learned-unet-test-v1` v1 locked (no-gap). Do **not** overwrite v1.
4. If a public leaderboard number is wanted: `HUMAN_ACTION_REQUIRED` — Mahdi must **Submit to Competition** from the Kaggle UI on `mahdimzafari/biohub-learned-unet-gap-test-v1` version 1, not from the CLI. Prefer this gap slug over submitting the no-gap `biohub-learned-unet-test-v1` v1.
5. Optional next experiment after a human submit or if remaining GPU budget allows: harden hidden-test runtime/memory/decoder fallback on a **new** slug only if a documented failure appears. Do **not** retune from public-test stems or the public leaderboard.

## HUMAN_ACTION_REQUIRED

- Grant this worker `issues:read` so cron runs can read issue #1.
- Mahdi must **Submit to Competition** from the Kaggle UI on `mahdimzafari/biohub-learned-unet-gap-test-v1` version 1 if a public leaderboard number is wanted. This repo does not authorize `kaggle competitions submit`. Do not submit the no-gap `biohub-learned-unet-test-v1` v1 now that the gap slug is COMPLETE / `audit_ok`.

## Later (blocked or expensive)

- complementary_v1, linkable_v1, and learned_unet_division_v1 are closed. Do not retune.
- Optional private kernel of association_v2 + division_v2 overlay under a **new** slug. Overlay was a no-op on holdout.
- Harden the learned gap test-time notebook for hidden test only if a documented runtime/memory issue appears.
- Final Kaggle submission: `HUMAN_ACTION_REQUIRED` — Mahdi must submit version output from the Kaggle UI.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, or invent a leaderboard score.
