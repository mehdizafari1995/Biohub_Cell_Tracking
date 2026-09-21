# Next actions

Use issue #1 as the command channel. Do one highest-priority unblocked task per run. Continue the active experiment draft PR when it exists.

## Now

1. Keep draft PR #2 (`cursor/kaggle-baseline-association-0b83`) open. Do not merge. Do not overwrite locked `association_v2`. Complete frozen validation on both `6bba` sequences is already recorded (association_v2 **0.591570**). Do not redo that download/score cycle.
2. Keep draft PR #4 (`cursor/competition-operational-strategy-3381`) open. Official holdout is done: combined **0.591570** identical to association_v2, **0 TP / 0 FP / 5 FN**. Keep the overlay; do **not** retune gates. Do not relaunch rejected `division_v1`.
3. Keep draft PR #5 (`cursor/complementary-detection-v1`) open. complementary_v1 was locked from 44b6 then scored **once** on frozen `6bba`: **0.591384 vs association_v2 0.591570**. Decision **`reject_complementary_v1`**. Do **not** retune isolation/intensity gates.
4. Keep draft PR #6 (`cursor/linkable-complementary-v1`) open. linkable_v1 was locked from 44b6 then scored **once** on frozen `6bba`: **0.591458 vs association_v2 0.591570**. Decision **`reject_linkable_v1`**. Recall unchanged; extra T_pred +189. Do **not** retune the 12 µm / max_dt=1 temporal gate.
5. Continue draft PR #7 (`cursor/learned-unet-v1`). Training helper `mahdimzafari/biohub-learned-unet-v1` v3 remains **COMPLETE / trained** (hash `4c38035d6657f2ad`; no `submission.csv`). Do **not** overwrite it. Infer hash `d7578b5f64ddac08` (44b6 tune 0.655751). Holdout hash `d374a21f03dddc04`: official frozen `6bba` **0.693605**. No-gap test-time hash `7a8725b818fc609b`: private kernel v1 **COMPLETE** on public test (306.5 s, 1184 MB, `audit_ok`). `learned_unet_division_v1` rejected on 44b6 (**0.655235 vs 0.655751**). `learned_unet_gap_v1` hash `f4d888d190c45a4a` official-scored on 44b6 then once on frozen `6bba`: **0.701026 vs no-gap 0.693605**, **`keep_candidate_learned_unet_gap_v1`**. Gap test-time hash `85379c198ebc7241`: private kernel `mahdimzafari/biohub-learned-unet-gap-test-v1` v1 **COMPLETE / inferred** on public test (330.0 s, 1184 MB, `audit_ok`). Read-only submit poll **2026-09-21T03:59:39Z**: kernel still `complete`, outputs still `run_manifest.json` + `submission.csv`, competition list still exactly one row (association_v2 **56399003** public **`0.553`**). Gap-test is **not** submitted. `learned_unet_prune_v1` hash `115753ea16aaf021` official-scored on 44b6 then once on frozen `6bba`: **0.701378 vs gap 0.701026**, **`keep_candidate_learned_unet_prune_v1`**. Dropped 425/491 degree-0 leftovers after locked gap. Not another 12 µm assignment. Do **not** retune. Do **not** score 6bba again. Transformer/motion/hung/conflict linkers remain rejected. Do **not** overwrite completed slugs.
6. Do **not** submit association_v2, division_v2, complementary_v1, linkable_v1, learned_unet_v1, learned_unet_infer_v1, learned_unet_holdout_v1, learned_unet_test_v1, learned_unet_division_v1, learned_unet_gap_v1, learned_unet_gap_test_v1, learned_unet_link_v1, learned_unet_link_v2, learned_unet_motion_v1, learned_unet_hung_v1, learned_unet_conflict_v1, or learned_unet_prune_v1 from the CLI.

## Next (this family)

1. Package a **new** public-test slug `learned_unet_prune_test_v1` (locked infer + locked gap overlay + drop degree-0). Do **not** overwrite `mahdimzafari/biohub-learned-unet-gap-test-v1` v1. Estimate CPU overlay is cheap on top of the locked 330 s / 1184 MB gap-test worker. Hidden-test budget remains ~5.4 h under the 12 h GPU limit.
2. Keep `learned_unet_conflict_v1` rejected. Hash `8503d804fb9e9f1d`. Do **not** retune. Do **not** score frozen `6bba` for that overlay. Do **not** package another collision-Hungarian variant.
3. Keep `learned_unet_link_v1` / `link_v2` / `motion_v1` / `hung_v1` rejected. Hung hash `f67bc97d0521dc78` was independently reconfirmed on 44b6 at **0.439023 vs infer 0.655751** (2026-09-21T04:12Z). Do **not** retune those hashes. Do **not** score frozen `6bba` for those overlays.
4. Keep `mahdimzafari/biohub-learned-unet-gap-test-v1` v1 locked until a prune-test kernel is COMPLETE / `audit_ok`. Do **not** overwrite v1. Do **not** retune. Do **not** score 6bba again for the gap overlay.
5. Keep `learned_unet_prune_v1` locked after one-shot holdout. Hash `115753ea16aaf021`. Do **not** retune. Do **not** score frozen `6bba` again.
6. Keep `learned_unet_division_v1` rejected. Do **not** retune gates. Do **not** score frozen `6bba` for that overlay.
7. Keep `mahdimzafari/biohub-learned-unet-test-v1` v1 locked (no-gap). Do **not** overwrite v1.
8. If a public leaderboard number is wanted before prune-test exists: `HUMAN_ACTION_REQUIRED` — Mahdi must **Submit to Competition** from the Kaggle UI on `mahdimzafari/biohub-learned-unet-gap-test-v1` version 1, not from the CLI. Prefer packaging prune-test first.

## HUMAN_ACTION_REQUIRED

- Grant this worker `issues:read` so cron runs can read issue #1.
- After `learned_unet_prune_test_v1` is COMPLETE / `audit_ok`, Mahdi must **Submit to Competition** from the Kaggle UI on that new slug version 1. Until then the packaged submit candidate remains `mahdimzafari/biohub-learned-unet-gap-test-v1` version 1. This repo does not authorize `kaggle competitions submit`. Do not resubmit association_v2 **56399003**.

## Later (blocked or expensive)

- complementary_v1, linkable_v1, learned_unet_division_v1, learned_unet_link_v1, learned_unet_link_v2, learned_unet_motion_v1, learned_unet_hung_v1, and learned_unet_conflict_v1 are closed. Do not retune transformer, constant-velocity, full-frame Hungarian, or shared-nearest collision-Hungarian linkers. A later linker follow-up needs a new hypothesis, not another assignment on the same 12 µm last-position pairs.
- Optional private kernel of association_v2 + division_v2 overlay under a **new** slug. Overlay was a no-op on holdout.
- Harden the learned gap/prune test-time notebook for hidden test only if a documented runtime/memory issue appears.
- Final Kaggle submission: `HUMAN_ACTION_REQUIRED` — Mahdi must submit version output from the Kaggle UI.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, or invent a leaderboard score.
