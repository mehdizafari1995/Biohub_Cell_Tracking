# One-shot frozen v1: unchanged division_v2 after control reproduction

Local official scorer only. **Not a Kaggle run. Not a leaderboard score.**
Public API string `0.553` was not used. No competition submission. No merge.
`division_v1` was not rerun. Thresholds were not changed after opening `6bba`.

## Protocol

Order on complete frozen `docs/splits/v1.json` validation (`6bba_05db0fb1` + `6bba_c328f2fd`):

1. Reproduce `baseline_v1`.
2. Reproduce locked `association_v2`.
3. Score unchanged `division_v2` **once** only after both controls matched.

Match rule: the repository publishes these frozen scores to six decimals and does not declare a separate epsilon. A control matches when `round(score, 6)` equals the published value.

Shared `baseline_v1` detections (cached). Official scorer `tracking_cellmot.metrics` @ `075fc5f5a52d11077f9dc2b074644618f26939e2`.

## Starting identity

| item | value |
|---|---|
| branch | `cursor/competition-operational-strategy-3381` (PR #4 draft) |
| starting HEAD | `7202736e444ce133a0044972d3df5a38205f1598` |
| `44b6` lock commit | `751587fe81ff4f388d43e572d6aab91c7d77f3fe` |
| `division_v2.yaml` blob | `144c56962fbbcae4bac9c1d21e9d0447b07819c9` (identical at lock and HEAD) |
| division_v2 config hash | `0fa756963e5d42d1` |
| association kernel hash | `2c6966b5204ba542` |
| baseline_v1 yaml hash | `6bb069d78d0e31c8` |
| association_v2 yaml hash | `59503ad9bb8d8c22` |
| frozen split `v1.json` sha256 | `e839eefb3d94ab7cd2e89132c7022746a1f9e88a6e195e7bd9ac9143f553a886` |
| official scorer commit | `075fc5f5a52d11077f9dc2b074644618f26939e2` |
| official `metrics.py` sha256 | `cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444` |
| official src-tree sha256 | `bc0685ada038a6af0ffb9d2628b3de5a628b21f126b4b0bf1f3e02d89dd7f9d8` |
| tracksdata / polars / geff | `0.1.0rc10` / `1.44.2` / `1.3.1.1.3` |

Lock provenance: `git diff 751587f HEAD` is empty for `configs/division_v2.yaml`, `src/biohub_tracking/division.py`, `geff_priors.py`, `association.py`, `link.py`, `track.py`, `detect.py`, and `config.py`. No threshold/config/code change after the `44b6` lock.

## Frozen holdout artifacts (sanitized)

Restored from already-complete helper kernel outputs into ignored `data/train/`. Not committed.

| store | files | bytes | tree sha256-16 | completeness |
|---|---|---|---|---|
| `6bba_05db0fb1.zarr` | 102 | 540242928 | `0aa5728bcf033c86` | 100/100 frames |
| `6bba_c328f2fd.zarr` | 102 | 452924734 | `6817a24e212a615d` | 100/100 frames |
| `6bba_05db0fb1.geff` | 21 | 15343 | `ad60eaf40c837318` | 1229 nodes / 1183 edges / 3 GT divisions |
| `6bba_c328f2fd.geff` | 21 | 10781 | `7957a395b33df830` | 511 nodes / 500 edges / 2 GT divisions |

Helper archives (ignored): GEFF worker zip 321996 bytes; Zarr worker zip 1047433543 bytes.

## Control reproduction

| candidate | official combined | published frozen | match |
|---|---|---|---|
| `baseline_v1` | 0.537134309808737 | 0.537134 | yes |
| `association_v2` | 0.5915697467955445 | 0.591570 | yes |

Both controls matched. `division_v2` was therefore scored exactly once.

## Official scores

| candidate | combined | edge J | adj edge J | node recall | official div TP/FP/FN |
|---|---|---|---|---|---|
| `baseline_v1` | 0.537134 | 0.530452 | 0.537134 | 0.803652 | 0/0/5 |
| `association_v2` | **0.591570** | 0.588813 | 0.591570 | 0.838244 | 0/0/5 |
| `division_v2` | **0.591570** | 0.588813 | 0.591570 | 0.838244 | 0/0/5 |

Per-volume official combined:

| dataset | baseline_v1 | association_v2 | division_v2 | official div TP/FP/FN |
|---|---|---|---|---|
| `6bba_05db0fb1` | 0.498062 | 0.564336 | 0.564336 | 0/0/3 |
| `6bba_c328f2fd` | 0.634281 | 0.660194 | 0.660194 | 0/0/2 |

## Sanitized predicted counts

| candidate | nodes (05db / c328) | edges (05db / c328) | predicted forks (05db / c328) |
|---|---|---|---|
| `baseline_v1` | 61809 / 26591 | 57271 / 24680 | 0 / 0 |
| `association_v2` | 67788 / 28548 | 65552 / 27438 | 0 / 0 |
| `division_v2` | 67788 / 28548 | 65555 / 27441 | 3 / 3 |

`division_v2` attached six predicted forks. The official division matcher still scored **0/0/5**. Combined official score is identical to locked `association_v2`. Do not claim improvement from predicted fork counts.

Runtime: baseline 114.8 s (includes detection), association 10.0 s, division 12.6 s. Peak RAM 763 MB.

## Decision

`KEEP_CANDIDATE_NOT_SUBMISSION_AUTHORIZED`

No combined official-score regression versus locked `association_v2`. Combined score did not improve. Do not retune. Do not submit. Do not overwrite `baseline_v1` or `association_v2`.

## Tests and privacy

`PYTHONPATH=src python -m pytest Biohub_Cell_Tracking/tests` — **86 passed**.

Tracked-file audit: 208 paths. No `data/`, `submission.csv`, `artifacts/{runs,tmp,kaggle,cache}`, `.venv/`, `kaggle.json`, or `.env*` in `git ls-files`. No credential values in tracked files.

## Kaggle

This run did not launch a competition submission, helper kernel, or leaderboard rescore. It only downloaded already-complete helper-kernel outputs to restore ignored local volumes.

Locked context: submission `56399003` remains complete with public API score string `0.553`; exactly one submission. That number was not used here. This worker's submissions-list API returned HTTP 404, so the count was not re-listed; no new submission was created.

## Confirmations

- public `0.553` not used
- submission count unchanged at one (no new submit from this run)
- no retuning
- no submission
- no merge
- PR #2 / #6 / #7 / #8 not modified
