# Next actions

Use issue #1 as the command channel. Do one highest-priority unblocked task per run. Continue the active experiment draft PR when it exists.

## Now

1. Keep draft PR #2 (`cursor/kaggle-baseline-association-0b83`) open. Do not merge. Do not overwrite locked `association_v2`.
2. Keep the **division_v2** draft PR #4 on `cursor/competition-operational-strategy-3381`. Control-gated one-shot is done: baseline_v1 0.537134 and association_v2 0.591570 reproduced; unchanged division_v2 scored **0.591570** (0/0/5). Decision: `KEEP_CANDIDATE_NOT_SUBMISSION_AUTHORIZED`. Do **not** retune gates. Do not relaunch rejected `division_v1`.
3. Next run: open a **new** experiment family for complementary 3D detection (tuning specimen only). Three of five holdout GT forks sit inside locked division gates and were still missed — the bottleneck is predicted topology / detections, not the 10.0 / 8.3 / 0.05 lock. Do not widen those gates.
4. Do **not** submit association_v2 or division_v2 from the CLI. Public `0.553` (submission 56399003) stays unused.

## HUMAN_ACTION_REQUIRED

- If a public leaderboard number is wanted for the association_v2 generalization hypothesis: Mahdi must open https://www.kaggle.com/code/mahdimzafari/biohub-association-v2 version 3 and **Submit to Competition** from the Kaggle UI. This repo does not authorize `kaggle competitions submit`.
- Grant this worker `issues:read` so cron runs can read issue #1.

## Later (blocked or expensive)

- Learned 3D detection / embedding (organizer UNet + transformer is GPU/Kaggle, not the CPU slice).
- Ensembles or complementary candidates, compared on the frozen split against preserved baseline_v1 and association_v2.
- Optional private kernel of association_v2 + division_v2 overlay under a **new** slug. Do not overwrite `biohub-association-v2`. Overlay was a no-op on holdout.
- Harden the final internet-off notebook and runtime for hidden test. Worker measurement: 1.48 s/frame, 421 MB.
- Final Kaggle submission: `HUMAN_ACTION_REQUIRED` — Mahdi must submit version output from the Kaggle UI.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, or invent a leaderboard score.
