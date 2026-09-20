# Read-only status: association_v2 competition submission 56399003

Queried at **2026-09-20T17:25:30Z** via the authenticated Kaggle API. No kernel push, cancel, download of competition volumes, or `competitions submit` was performed. Signed output URLs and credentials are not recorded here.

## Competition submission

Exactly **one** team submission exists. The second daily/final slot was not used.

| field | value |
|---|---|
| ref | `56399003` |
| submitted | `2026-09-20T15:12:52.307Z` |
| status | **pending** |
| fileName | `submission.csv` |
| kernel | `mahdimzafari/biohub-association-v2` `scriptVersionId=351174414` |
| publicScore | empty (`hasPublicScore=false`) |
| privateScore | empty (`hasPrivateScore=false`) |
| errorDescription | empty (`hasErrorDescription=false`) |
| totalBytes | `0` (normal while pending) |

Elapsed at query: **7957 s / 2.21 h**. CPU notebook limit 12 h, remaining **~9.79 h**. Within the allowed runtime.

## Saved kernel (source version, not the scoring session)

| field | value |
|---|---|
| slug | `mahdimzafari/biohub-association-v2` |
| `kernels/status` | `complete` (`hasFailureMessage=false`) |
| `kernels/list` lastRunTime | `2026-09-19T23:36:49.317Z` |
| outputs | `run_manifest.json`, `submission.csv` |

Those fields describe the already-finished public-test version 3. They do **not** flip to `running` while a competition scoring rerun is pending.

## Available kernel logs

The `/kernels/output` log is the **public-test v3** session, not the hidden-test scoring rerun:

- `environment=kaggle_input` `config=association_v2` `n_test=4`
- datasets: `44b6_0113de3b,44b6_0b24845f,6bba_05b6850b,6bba_05db0fb1`
- last meaningful stdout: `runtime_sec=593.528496317`, `peak_memory_mb=421.46484375`, `audit_ok=true` at log t≈607 s
- last event: NbConvert write at t=616.75 s
- no timeout / OOM / traceback / data-mount error in that log
- GPU/TPU were not enabled on the source kernel push

No separate scoring-rerun log is exposed on this endpoint while status is `pending`.

## Interpretation

`Notebook Running` for ~2 h is the hidden-test scoring rerun of submission `56399003`. The 593.53 s figure was public `test/` (4 volumes). Hidden test is expected to be larger (train-size hypothesis ≈ 40 min; a bigger hidden set can take hours). 2.21 h with no error is still inside the 12 h CPU limit.

## Single safest next action

**Continue monitoring.** Do not cancel, retry, push a new kernel version, or use the second submission. Re-check before `2026-09-21T03:12:52Z` (12 h after submit). If it fails or times out, identify the exact error before any retry.
