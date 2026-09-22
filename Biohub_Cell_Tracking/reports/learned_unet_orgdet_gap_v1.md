# learned_unet_orgdet_gap_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or orgdet-holdout/orgdet-test slugs. Do **not** retune. One-shot holdout
is done. Do **not** score frozen `6bba` again.

## Hypothesis

`learned_unet_orgdet_v1` is greedy 12 µm last-position with
`max_gap_frames=0`. The same locked association_v2 gap-close that
improved infer (tune 0.655751 → 0.658452; holdout 0.693605 → 0.701026)
should recover leftover unmatched orgdet nodes without retuning
detection or consecutive dt=1 links.

## Operating point (locked; one-shot holdout done)

Config: `configs/learned_unet_orgdet_gap_v1.yaml`. Hash `ee31f36151325f9a`.
Parent orgdet `80b20589198148b7`. Parent gap gates `f4d888d190c45a4a`.
Training hash `4c38035d6657f2ad`.

- Source: locked orgdet (or empty-edge orgop) 44b6 / 6bba graphs
- Existing dt=1 orgdet edges kept
- Gap-close: dt=2 or 3, gate `12 * sqrt(dt)` µm, greedy 1-1, interpolate
- Intensity unused. No transformer. No ILP.
- Writes no `submission.csv` from this overlay scorer

Do **not** retune these gates. Do **not** score frozen `6bba` again.

## Cost estimate (before score)

- Storage: orgop/orgdet helper graphs + two GEFFs from `biohub-geff-export`.
  No volumes, no weights.
- CPU: 44b6 **10.1 s / 530 MB**; 6bba **7.2 s / 780 MB**.
- No GPU for the overlay itself.

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4` only.
Orgop detections 62480/0 and 28672/0, then locked orgdet relink, then gap.
GEFFs from `mahdimzafari/biohub-geff-export` v1. **Not a Kaggle run.**

| candidate | combined | edge J | adj edge J | recall | divisions | extra edges |
|---|---|---|---|---|---|---|
| learned_unet_orgdet_v1 | **0.679578** | — | 0.67958 | — | 0/0/2 | 0 |
| learned_unet_orgdet_gap_v1 | **0.687221** | 0.70382 | 0.68722 | 0.97667 | 0/0/2 | 9951 |

Decision **`keep_candidate_learned_unet_orgdet_gap_v1`**. Hash locked before
the 6bba look.

## Official one-shot holdout (not leaderboard)

Local official scorer on `6bba_05db0fb1` + `6bba_c328f2fd` once. Graphs from
`mahdimzafari/biohub-learned-unet-orgdet-holdout-v1` v1 (84328/78493 and
37482/35147). **Not a Kaggle run. Not a leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | extra edges |
|---|---|---|---|---|---|---|
| learned_unet_orgdet_holdout_v1 | **0.705226** | — | 0.70523 | 0.939 | 0/0/5 | 0 |
| learned_unet_orgdet_gap_v1 | **0.722805** | 0.74265 | 0.72281 | 0.95762 | 0/0/5 | 10427 |

Per-sequence: `6bba_05db0fb1` 0.739289 (1052/201/131); `6bba_c328f2fd`
0.684139 (414/90/86). Extra nodes 4597 + 1559.

Decision **`keep_candidate_learned_unet_orgdet_gap_v1`**. This is the new
holdout champion. Do **not** score 6bba again.

## Public-test kernel

`mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1` version **1
COMPLETE / inferred** at 2026-09-22T09:05:18Z (private, internet off,
GPU, hash `1ce4837ae16bee52`). Worker `audit_ok`. Independent stream
audit `audit_ok`. `6bba_05db0fb1` 88925/86287 matches locked overlay.
Hidden projection **4.32 h**. **Not submitted.** Do **not** overwrite
v1. Preferred UI submit. Orgdet-test v1 is fallback only if this
kernel is later overwritten (do not overwrite).

## Policy

Public API `0.553` is association_v2 **56399003** only and was not used
for tuning. Label `KEEP_CANDIDATE_NOT_SUBMISSION_AUTHORIZED` until a
human submits the COMPLETE gap-test worker from the Kaggle UI.
