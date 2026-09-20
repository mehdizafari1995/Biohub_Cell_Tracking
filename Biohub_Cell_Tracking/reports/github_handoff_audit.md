# GitHub handoff audit (sanitized)

Local folder `/workspace` / `Biohub_Cell_Tracking/` is the authoritative project. No Kaggle volumes, credentials, model weights, submissions, or run artifacts are included in git.

## Privacy and credentials

- No `kaggle.json`, `.env`, private keys, or API token values in tracked files.
- `KAGGLE_API_TOKEN` is read from the environment or `~/.kaggle/kaggle.json` at runtime and is never printed or committed (`src/biohub_tracking/kaggle_api.py`).
- `docs/environment_snapshot.json` records only `kaggle_token_present: true/false`.
- Kernel metadata is private (`is_private: true`, internet off). Username in the slug is the public Kaggle profile, not a secret.
- Nested `third_party/.../.git` is not present.

## Ignored (must not upload)

`.gitignore` excludes: `data/`, `submission.csv`, `artifacts/runs/`, `artifacts/tmp/`, `artifacts/kaggle/`, `artifacts/cache/`, `.venv/`, `kaggle.json`, `.env*`, codec wheels.

Confirmed ignored: local `data/` (OME-Zarr + GEFF), root `submission.csv`, Kaggle worker output, detection cache.

## Generated files that stay in git

- Docs, configs, tests, notebooks (source only), reports (ledger / comparison / kernel manifest without CSV).
- Vendored official scorer source (BSD-3-Clause), not competition image data.
- `third_party/.../assets/demo.gif` (~5 MB) from the organizer repo, not Biohub volumes.

## Tests

`python -m pytest Biohub_Cell_Tracking/tests` — 49 passed (2026-09-20T23:27Z, public LB recorded + frozen val re-score).

## Official scorer

- Vendored `royerlab/kaggle-cell-tracking-competition` @ `075fc5f5a52d11077f9dc2b074644618f26939e2`
- `tracksdata==0.1.0rc10` @ `7523bb39`
- `polars==1.44.2`, `geff==1.3.1.1.3`
- `official_scorer_available() is True`

## Push policy

Do not upload: Kaggle data, credentials, model weights, GEFF/Zarr volumes, `submission.csv`, or local run artifacts.

## 2026-09-20 re-audit (independent kernel output)

- `pytest` 48 passed.
- Official scorer pin unchanged: `075fc5f5`.
- Worker `submission.csv` was downloaded into ignored `artifacts/kaggle/` only. Sanitized independent numbers: `reports/kaggle_association_v2_v3_independent_audit.json`.
- Train GEFFs for the four public-test stems were downloaded into ignored `data/train/`. No Zarr volumes were downloaded for this audit.
- `KAGGLE_API_TOKEN` present in environment (value not printed). `~/.kaggle/kaggle.json` absent.
- GitHub CLI authenticated as `mehdizafari1995` for the private repo.

## 2026-09-20T15:00Z START-trigger re-verification

- Live `kernels/status` for `mahdimzafari/biohub-association-v2`: `complete`, `hasFailureMessage=false`.
- Re-downloaded version-3 archive (3 658 915 bytes) into ignored `artifacts/kaggle/association_v2_v3/`. CSV 12 948 601 bytes. Not committed.
- Stdlib + official re-audit both `audit_ok`. Overlap score 0.564336 unchanged. Kernel not re-pushed. Competition submit not performed.
- `pytest` 48 passed with official scorer available.
- Issue #1 still 403 (`issues:read` missing).

## 2026-09-20T17:25Z read-only competition status

- Submission `56399003` pending since `2026-09-20T15:12:52.307Z`. No scores, no error. Second slot unused.
- No kernel push, cancel, data download, or `competitions submit` from this worker.
- Sanitized report: `reports/kaggle_association_v2_submission_56399003_status.md`.

## 2026-09-20T23:27Z public score + frozen val

- API `publicScore` string **0.553**. Private score not returned. Not invented.
- Complete frozen volumes downloaded into ignored `data/train/` only (100/100). Not committed.
- Re-score reproduced locked A 0.537 / B 0.592. `pytest` 49 passed.
- Oracle helper now routes `association_v2` through the association linker (eval-only; locked YAML unchanged).
