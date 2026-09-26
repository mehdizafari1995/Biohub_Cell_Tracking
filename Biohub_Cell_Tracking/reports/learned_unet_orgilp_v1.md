# learned_unet_orgilp_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or orgdet/orgop slugs. Do **not** retune. Frozen `6bba` was **not**
opened.

## Hypothesis

`learned_unet_orgop_v1` failed because transformer softmax>0.5 wrote
zero candidate edges, so tracksdata ILP skipped. Replacing that
generator with the locked 12 µm spatial gate (distance-derived
`edge_prob`, organizer ILP weights −1.0/0.1/0.1/1.0) on the same
0.99 / pool-3.0 detections would produce a nonempty graph and beat
locked `learned_unet_orgdet_gap_v1` official 44b6 **0.687221**.

## Operating point (rejected on tuning)

Config: `configs/learned_unet_orgilp_v1.yaml`. Hash `dbc6134cddf2048d`.
Parent orgop `250c5bfacd4a8198` (rejected, not overwritten).
Parent orgdet `80b20589198148b7`. Parent orgdet-gap `ee31f36151325f9a`.
Train hash `4c38035d6657f2ad`.

- Source: locked orgop 44b6 graphs (62480 / 28672 nodes, 0 edges)
- Candidates: dt=1 pairs inside 12 µm, max 8 neighbors/source
- `edge_prob = max(1e-6, 1 − dist/12)`
- Official tracksdata ILP (SCIP fallback; no Gurobi license)
- Preflight required: nonzero detections, candidates, and solved edges
- Transformer unused. Detection not retuned. No `submission.csv`

## Preflight (before score)

| stem | nodes | spatial candidates | status |
|---|---|---|---|
| `44b6_12dfb391` | 62480 | 333502 | `preflight_ok` |
| `44b6_267148e4` | 28672 | 133546 | `preflight_ok` |

Rejected orgop had **0 / 0** softmax>0.5 candidates. The spatial gate
is the documented repair.

## Cost

- Storage: locked orgop graphs + two 44b6 GEFFs. No volumes, no weights.
- CPU: **1008.0 s / 4501 MB**. SCIP ILP. Under the 12 h Kaggle limit.
- No GPU. No kernel push.

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4`.
GEFFs from `mahdimzafari/biohub-geff-export` v1. **Not a Kaggle run.
Not a leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes / edges |
|---|---|---|---|---|---|---|
| learned_unet_orgdet_v1 | **0.679578** | — | 0.67958 | — | 0/0/2 | 91152 / 84057 |
| learned_unet_orgdet_gap_v1 | **0.687221** | 0.70382 | 0.68722 | 0.97667 | 0/0/2 | — / +9951 gap |
| learned_unet_orgilp_v1 | **0.676811** | 0.68779 | 0.67681 | 0.97004 | 0/0/2 | 91152 / 84850 |

Per-sequence orgilp: `44b6_12dfb391` 0.767763 (687/116/86);
`44b6_267148e4` 0.468954 (192/112/85). ILP wrote 58805 + 26045 edges
from 333502 + 133546 candidates. Divisions 0/0/2.

Decision **`reject_learned_unet_orgilp_v1`**. Combined is below the
preregistered orgdet-gap threshold **0.687221** and also below greedy
orgdet **0.679578**. Frozen `6bba` was not opened. Do **not** retune.

`learned_unet_orgilp_gap_v1` was **not** launched. Consecutive ILP
already lost to the same-detection greedy linker, so leftover
association_v2 gap-close cannot be expected to beat locked orgdet-gap
without retuning.

Champion remains `learned_unet_orgdet_gap_v1` / orgdet-gap-test v1
`KEEP_CANDIDATE_NOT_SUBMISSION_AUTHORIZED`.
