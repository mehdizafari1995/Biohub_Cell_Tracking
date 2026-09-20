# Project state

Last updated: 2026-09-20T23:28Z (CONTINUE cycle: API re-check still `complete` / public `0.553`; frozen-val evidence already on this branch). Authoritative implementation: `Biohub_Cell_Tracking/`.

## Repository

- Private GitHub repo: `mehdizafari1995/Biohub_Cell_Tracking`
- Base branch: `main` (baseline + official scorer + privacy audit + coordination docs)
- Active experiment branch: `cursor/kaggle-baseline-association-0b83` (association_v2 / baseline_v1 Kaggle kernel family)
- Command channel: issue #1 (this worker still gets 403 on issues)
- Complete-validation harness from PR #3 is merged to `main` (2026-09-19). This experiment branch stays draft and unmerged.

## What is locked

- Frozen embryo-level split `docs/splits/v1.json`: tune on specimen `44b6`, validate on `6bba`. Do not retune after seeing validation.
- `docs/splits/v2.json` only adds sanity sequence `44b6_2a2eff9f` (same specimen prefix as tuning, not a third embryo).
- Official scorer: vendored `royerlab/kaggle-cell-tracking-competition` @ `075fc5f5a52d11077f9dc2b074644618f26939e2`. Dependencies: `tracksdata==0.1.0rc10`, `polars==1.44.2`, `geff==1.3.1.1.3`.
- Frozen greedy baseline: `configs/baseline.yaml`
- Kaggle kernel config (anisotropic DoG + 4 µm NMS + greedy 15 µm): `configs/baseline_v1.yaml`
- Locked association: `configs/association_v2.yaml` (12 µm gate + 2-frame gap close + intensity 0.5, greedy). Kernel config hash `2c6966b5204ba542`.

## Measured results (local official scorer, not leaderboard)

On frozen validation (`6bba_05db0fb1` + `6bba_c328f2fd`):

| candidate | official combined | notes |
|---|---|---|
| A `baseline_v1` | 0.537 | Kaggle kernel v2 ran on public test; not submitted |
| B `association_v2` | **0.592** | local official on both frozen sequences; Kaggle kernel v3 independently audited |
| C `division_v1` | 0.590 | 0 TP / 3 FP / 5 FN; do not claim generalization |

Independent official score of the **downloaded** association_v2 kernel v3 CSV vs the train GEFF for public-test overlap `6bba_05db0fb1`: **0.564336** (821/276/362 edges, 0/0/3 divisions, recall 0.829). That is not a leaderboard score.

2026-09-20 complete-volume re-score (100/100 frames, both frozen sequences) reproduced A **0.537134** and B **0.591570**. Error analysis: B reduces missed annotated cells (366→288), switches (150→135), and broken links (603→483) via gap-close interpolation; 0/5 annotated divisions recovered. See `reports/frozen_validation_complete_20260920.md`. Do not retune.

See `Biohub_Cell_Tracking/reports/abc_comparison.md`, `reports/kaggle_association_v2_v3_independent_audit.md`, and `experiments/registry.csv`.

## Kaggle

- Private kernel `mahdimzafari/biohub-baseline-v1` version 2 **COMPLETE** on public `test/` (595 s, 1.49 s/frame, peak RAM 435 MB, `audit_ok`).
- Private kernel `mahdimzafari/biohub-association-v2` version 3 **COMPLETE** on public `test/` (593.53 s, 1.48 s/frame, peak RAM 421 MB, `audit_ok`, 129500 nodes / 124273 edges / 11985 interpolated / 0 divisions). Decoder: `blosc2` (no `zarr` on the image).
- Independent re-audit of the downloaded worker `submission.csv`: schema/coverage/integrity `audit_ok`; counts match the worker manifest; all edges `dt=1`; 0 predicted divisions; integer voxel centroids inside `(100, 64, 256, 256)`. CSV was not copied into git.
- 2026-09-20T15:00Z worker re-verification: live `kernels/status` = `complete` (no failure); outputs `submission.csv` + `run_manifest.json`; re-downloaded zip 3 658 915 bytes; CSV 12 948 601 bytes. Stdlib + official re-audit both `audit_ok` and match the earlier 0.564336 overlap figure. Kernel was **not** re-pushed (no version 4). Competition submit was **not** performed.
- Competition submit: Mahdi submitted `mahdimzafari/biohub-association-v2` `scriptVersionId=351174414` (version 3) from the Kaggle UI. Submission **56399003** dated `2026-09-20T15:12:52.307Z` is **complete**. Read-only API at `2026-09-20T23:08:00Z` and re-checked `2026-09-20T23:28:19Z`: public score **`0.553`** (API string; no extra digits). Private score **not returned** (`hasPrivateScore=false`). `errorDescription` empty. Exactly one competition submission. Hidden-test output `totalBytes=209728379`. Scoring-rerun wall-clock was not returned by the API (pending at 17:25:30Z; complete by 23:08:00Z). This **0.553** is the first real Kaggle public score and is distinct from local official 0.592 and public-overlap official 0.564336. Do not retune or resubmit this locked candidate. Kaggle UI `0/2` is final-selection slots, not a two-submit cap; do not select finals yet. See `reports/kaggle_association_v2_submission_56399003_status.md`.
- `Kaggle` is used only through configured secrets / `KAGGLE_API_TOKEN`. Never print tokens.

## Tests

`python -m pytest Biohub_Cell_Tracking/tests` — 49 passed (2026-09-20T23:27Z, this branch, official scorer available).

## Privacy

Tracked tree has no competition volumes, credentials, weights, submissions, or run artifacts. Ignore rules cover `data/`, `submission.csv`, `artifacts/{runs,tmp,kaggle,cache}`, `.venv/`, `kaggle.json`, `.env*`. Sanitized audit: `Biohub_Cell_Tracking/reports/github_handoff_audit.md`.
