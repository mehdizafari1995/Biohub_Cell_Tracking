# Next actions

Use issue #1 as the command channel. Do one highest-priority unblocked task per run. Continue the active experiment draft PR when it exists.

## Now

1. Keep draft PR `cursor/kaggle-baseline-association-0b83` open. Do not merge. Do not mark ready.
2. `association_v2` private internet-off kernel `mahdimzafari/biohub-association-v2` version 3 is **COMPLETE** on public `test/` (593.53 s, `audit_ok`). Do **not** submit it to the competition from the CLI.
3. Complete specimen-separated validation harness lives on draft PR #3 (`cursor/issue-1-worker-prompt-7ab9`). Do not retune `v1` after seeing holdout. Train volumes are still absent on this machine.
4. `division_v1` already failed frozen validation (0/3/5). Do not relaunch it. A revised division hypothesis, if any, must be a **new** experiment family/PR and must not loosen gates after holdout.

## HUMAN_ACTION_REQUIRED

- If a public leaderboard number is wanted for the association_v2 generalization hypothesis: Mahdi must open https://www.kaggle.com/code/mahdimzafari/biohub-association-v2 version 3 and **Submit to Competition** from the Kaggle UI. This repo does not authorize `kaggle competitions submit`.
- Grant this worker `issues:read` so cron runs can read issue #1.

## Later (blocked or expensive)

- Learned 3D detection / embedding (organizer UNet + transformer is GPU/Kaggle, not the CPU slice).
- Ensembles or complementary candidates, compared on the frozen split against preserved baseline_v1 and association_v2.
- Harden the final internet-off notebook and runtime for hidden test. Worker measurement: 1.48 s/frame, 421 MB. Train-size hidden (~16 sequences) ≈ 40 min; a 200-sequence hypothesis at 2× margin can exceed 12 h.
- Final Kaggle submission: `HUMAN_ACTION_REQUIRED` — Mahdi must submit version output from the Kaggle UI.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, or invent a leaderboard score.
