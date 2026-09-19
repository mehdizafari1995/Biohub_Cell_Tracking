# Next actions

Use issue #1 as the command channel. Do one highest-priority unblocked task per run. Continue the active experiment draft PR when it exists.

## Now

1. Keep draft PR `cursor/kaggle-baseline-association-0b83` open. Do not merge. Do not mark ready.
2. Package and run **association_v2** as a private internet-off Kaggle kernel (same path as `baseline_v1`). Record the worker run in the submission ledger. Do **not** submit to the competition.
3. Expand complete specimen-separated validation so both frozen validation sequences and the annotated-division events are scored with the official scorer. Do not retune `v1` after seeing holdout.
4. Only after association_v2 has a real Kaggle worker measurement: consider conservative division work as a **new** experiment family. The current `division_v1` failed frozen validation (0/3/5).

## Later (blocked or expensive)

- Learned 3D detection / embedding (organizer UNet + transformer is GPU/Kaggle, not the CPU slice).
- Ensembles or complementary candidates, compared on the frozen split against preserved baseline_v1 and association_v2.
- Harden the final internet-off notebook and runtime for hidden test. Estimate storage, GPU/CPU, and the 12 h limit first.
- Final Kaggle submission: `HUMAN_ACTION_REQUIRED` — Mahdi must submit version output from the Kaggle UI. This repo does not authorize `kaggle competitions submit`.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, or invent a leaderboard score.
