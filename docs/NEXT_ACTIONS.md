# Next actions

Use issue #1 as the command channel. Do one highest-priority unblocked task per run. Continue the active experiment draft PR when it exists.

## Now

1. Keep draft PR `cursor/kaggle-baseline-association-0b83` open. Do not merge. Do not mark ready. Do not overwrite locked `association_v2`.
2. `association_v2` private internet-off kernel `mahdimzafari/biohub-association-v2` version 3 is **COMPLETE** and independently audited. Worker CSV official-scored 0.564336 on public-test overlap `6bba_05db0fb1` (matches local 0.564). Do **not** submit it from the CLI.
3. The frozen split is still incomplete from the worker CSV because `6bba_c328f2fd` is not in public test. The complete-validation harness lives on draft PR #3. Do not retune `v1` after seeing holdout.
4. `division_v1` already failed frozen validation (0/3/5). Do not relaunch it on this branch. The next experiment family, if any, is a **new** conservative-division PR with priors chosen only on tuning GEFFs.

## HUMAN_ACTION_REQUIRED

- If a public leaderboard number is wanted for the association_v2 generalization hypothesis: Mahdi must open https://www.kaggle.com/code/mahdimzafari/biohub-association-v2 version 3 and **Submit to Competition** from the Kaggle UI. This repo does not authorize `kaggle competitions submit`.
- Grant this worker `issues:read` so cron runs can read issue #1.

## Later (blocked or expensive)

- Learned 3D detection / embedding (organizer UNet + transformer is GPU/Kaggle, not the CPU slice). Public-test copy `44b6_0b24845f` scored 0.280 with recall 0.45; that is a same-specimen copy, not holdout, and is not a reason to retune association_v2.
- Ensembles or complementary candidates, compared on the frozen split against preserved baseline_v1 and association_v2.
- Harden the final internet-off notebook and runtime for hidden test. Worker measurement: 1.48 s/frame, 421 MB. Train-size hidden (~16 sequences) ≈ 40 min; a 200-sequence hypothesis at 2× margin can exceed 12 h.
- Final Kaggle submission: `HUMAN_ACTION_REQUIRED` — Mahdi must submit version output from the Kaggle UI.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, or invent a leaderboard score.
