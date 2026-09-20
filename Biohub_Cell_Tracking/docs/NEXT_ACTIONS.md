# Next actions

Use issue #1 as the command channel. Do one highest-priority unblocked task per run. Continue the active experiment draft PR when it exists.

## Now

1. Keep draft PR `cursor/kaggle-baseline-association-0b83` (PR #2) open. Do not merge. Do not mark ready. Do not overwrite locked `association_v2`.
2. Submission **56399003** (`mahdimzafari/biohub-association-v2` v3, config hash `2c6966b5204ba542`) is **complete**. Public LB **`0.553`** (API string). Private score was not returned. Do **not** resubmit this candidate. Do **not** retune from 0.553. That number is distinct from local official 0.592 and public-overlap official 0.564336.
3. Kaggle UI `0/2` is **final-submission selection**, not a two-submit cap. Do not select finals yet. Stronger candidates are still expected. Kaggle will auto-select best scoring entries if fewer than two are chosen.
4. Complete frozen-validation re-score is **done** and matches the locked 0.537 / 0.592 figures. See `reports/frozen_validation_complete_20260920.md`. The next experiment family belongs on a **new** draft PR. Do not absorb it here. Do not resubmit 56399003.

## HUMAN_ACTION_REQUIRED

None for this candidate. Do not click Submit again. Do not choose final submissions.

## Later (blocked or expensive)

- New experiment families (detection / division / learned models) on their own draft PRs. Compare them on the frozen split against preserved baseline_v1 and association_v2.
- Hidden-test runtime was not returned by the API. Public-test worker was 593.53 s / 421 MB; hidden output was 209728379 bytes (~16× public CSV).
- Grant this worker `issues:read` so cron runs can read issue #1.

## Never from automation

Merge, make the repo public, spend money, accept competition terms, expose secrets, upload data/weights/submissions, resubmit this locked candidate, select finals, or invent a private/leaderboard score.
