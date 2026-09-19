# Biohub Cell Tracking

This repository is the private Kaggle program for
[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).

Project code, tests, configs, docs, and ledgers live in `Biohub_Cell_Tracking/`. Treat that directory as the working tree for implementation. Use GitHub issue #1 as the persistent command channel.

## Objective

Produce the strongest defensible, rules-compliant submission. Use specimen-separated validation and the official scorer. Do not claim a leaderboard result without a returned Kaggle score.

## Always read first

1. GitHub issue #1 and all comments
2. `docs/PROJECT_STATE.md` and `docs/NEXT_ACTIONS.md` (also mirrored under `Biohub_Cell_Tracking/docs/`)
3. `Biohub_Cell_Tracking/docs/competition_spec.md`
4. Frozen splits: `Biohub_Cell_Tracking/docs/splits/v1.json` (locked) and `v2.json` (sanity only)
5. Experiment ledger: `Biohub_Cell_Tracking/experiments/registry.csv`
6. Submission ledger: `Biohub_Cell_Tracking/reports/submission_ledger.md`
7. Open pull-request reviews

## Hard rules

- Do not merge automatically, make the repository public, expose credentials or competition data, accept competition terms, spend money, or perform the final Kaggle submission.
- Never print or commit Kaggle credentials, `.env` files, or API tokens.
- Never upload competition data, OME-Zarr/GEFF volumes, model weights, `submission.csv`, virtual environments, or local run artifacts to GitHub.
- A local simulation is not a Kaggle run. Do not invent a leaderboard score.
- Keep `configs/baseline.yaml` frozen. Do not overwrite locked `baseline_v1` or `association_v2` operating points after seeing holdout results.
- Continue an existing experiment draft PR when one is active. Otherwise open one branch and one draft PR for one experiment family.
- If the only remaining work requires a human (merge, public repo, paid action, terms, secrets, or final submit), comment `HUMAN_ACTION_REQUIRED` with one exact action.

## Privacy

`.gitignore` must keep out `data/`, `submission.csv`, `artifacts/{runs,tmp,kaggle,cache}`, `.venv/`, `kaggle.json`, and `.env*`. Confirm with `git ls-files` before every push.
