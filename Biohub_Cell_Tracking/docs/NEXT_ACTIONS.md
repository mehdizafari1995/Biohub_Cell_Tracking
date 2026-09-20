# Next actions

Use issue #1 as the command channel. Do one highest-priority unblocked task per run. Continue the active experiment draft PR when it exists.

## Now

1. Keep draft PR `cursor/kaggle-baseline-association-0b83` (PR #2) open. Do not merge. Do not mark ready. Do not overwrite locked `association_v2`.
2. The requested `association_v2` private internet-off kernel cycle is **done**. Live `mahdimzafari/biohub-association-v2` version 3 is `complete`. This worker re-downloaded the output, independently re-audited it (`audit_ok`), and official-scored public-test overlap `6bba_05db0fb1` at 0.564336 (matches local 0.564). Kernel was not re-pushed. Do **not** submit it from the CLI.
3. The frozen split is still incomplete from the worker CSV because `6bba_c328f2fd` is not in public test. The complete-validation harness from PR #3 is now **merged to `main`**. Do not retune `v1` after seeing holdout.
4. `division_v1` already failed frozen validation (0/3/5). Do not relaunch it on this branch. Later experiment families already have their own draft PRs (#4 division_v2, #5 complementary_v1, #6 linkable_v1, #7 learned_unet_v1). Do not absorb those into this PR.

## HUMAN_ACTION_REQUIRED

- If a public leaderboard number is wanted for the association_v2 generalization hypothesis: Mahdi must open https://www.kaggle.com/code/mahdimzafari/biohub-association-v2 version 3 and **Submit to Competition** from the Kaggle UI. This repo does not authorize `kaggle competitions submit`.
- Grant this worker `issues:read` so cron runs can read issue #1.

## Later (blocked or expensive)

- Learned 3D detection / embedding (organizer UNet + transformer is GPU/Kaggle, not the CPU slice). Public-test copy `44b6_0b24845f` scored 0.280 with recall 0.45; that is a same-specimen copy, not holdout, and is not a reason to retune association_v2.
- Ensembles or complementary candidates, compared on the frozen split against preserved baseline_v1 and association_v2.
- Harden the final internet-off notebook and runtime for hidden test. Worker measurement: 1.48 s/frame, 421 MB. Train-size hidden (~16 sequences) ≈ 40 min; a 200-sequence hypothesis at a 2× margin can exceed 12 h.
- Final Kaggle submission: `HUMAN_ACTION_REQUIRED` — Mahdi must submit version output from the Kaggle UI.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, or invent a leaderboard score.
