# 44b6 FNN detector oracle / feasibility audit

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Frozen `6bba` was **not** opened. This is **not** a Kaggle run
and **not** a leaderboard score. Do **not** retune locked FNN
`95e9300eb84ba71c`. Do **not** use public **0.602** / **0.612** /
**0.553**.

## Scope

Official scorer `075fc5f5` on frozen-tuning `44b6_12dfb391` +
`44b6_267148e4` only. Graphs rebuilt from locked
`mahdimzafari/biohub-learned-unet-orgop-v1` detections, then locked
`learned_unet_orgdet_fnn_v1`. GEFFs from `mahdimzafari/biohub-geff-export`
v1. Restore check matched locked FNN **0.724653**. Runtime 29.49 s,
peak 516.5 MB.

GT matching uses the official 7.0 µm assignment. Unmatched predictions
are not node false positives. Candidate envelope is the locked
association_v2 / orgdet gate: **12 µm**, `max_gap_frames=2`,
`sqrt(dt)`. That envelope is diagnostic; it is not retuned.

## Restore check

| candidate | official combined | TP / FP / FN | divisions | recall |
|---|---|---|---|---|
| locked FNN (rebuild) | **0.724653** | 906 / 174 / 144 | 0/0/2 | 0.974 |

Per-sequence FNN: `44b6_12dfb391` **0.811357** (704/86/69);
`44b6_267148e4` **0.520601** (202/88/75).

## Detection vs association

| bucket | count |
|---|---|
| GT edges | 1050 |
| both endpoints detected | 1023 |
| detection-unavoidable FN edges | 27 (2.57%) |
| currently linked (FNN TP-style) | 906 |
| association-fixable | 117 |
| association-fixable and in candidate | 113 |
| both-detected in 12 µm dt=1 | 1019 |
| both-detected in dt=2 gap | 0 |
| both-detected in ELF 8.0 µm dt=1 | 982 |
| candidate-edge recall of both-detected | **0.996** |
| missed GT nodes | 17 |
| unmatched pred nodes | 93968 (sparse GT; not node FPs) |
| GT divisions recoverable with this detector | **2 / 2** |

Dominant remaining loss is **association**, not detection.

## Official upper bounds (same FNN nodes)

Oracle graphs keep the locked FNN detections and replace edges with
GT-true pairs. They are feasibility graphs, not inference.

| graph | official combined | adj edge J | edge J | TP / FP / FN | divisions |
|---|---|---|---|---|---|
| FNN | **0.724653** | 0.72465 | 0.74020 | 906 / 174 / 144 | 0/0/2 |
| detection oracle | **1.052563** | 0.95256 | 0.97429 | 1023 / 0 / 27 | 2/0/0 |
| candidate oracle | **1.048979** | 0.94898 | 0.97048 | 1019 / 0 / 31 | 2/0/0 |

Scores can exceed 1.0 because GT is sparse. Candidate oracle vs FNN is
**+0.324326**, far above the **+0.002000** gate (**0.726653**).

Per-sequence candidate oracle: `44b6_12dfb391` **1.083590** (768/0/5,
division recovered); `44b6_267148e4` **0.952394** (251/0/26, division
recovered). Four of 255 both-detected edges on `44b6_267148e4` sit
outside the 12 µm envelope.

Both annotated 44b6 forks are detection-recoverable and official-score
as division TPs on the oracle graphs. FNN currently predicts **0**
divisions.

## Preregistered hypothesis

`learned_unet_orgdet_fnn_rank_v1`, hash `2748c4a2b2823c0a`. Parent FNN
`95e9300eb84ba71c`. Learned pairwise ranking among candidates already
inside the locked 12 µm / `sqrt(dt)` / max_gap=2 envelope, then
constrained 1-1 selection. 44b6 train only. Hard negatives. Fail closed
unless the link head shows nonzero learning. Not another 12 µm
assignment. Not a short-fragment retune. Not another 8.0 µm dt=1
retune. Strict tune gate: official 44b6 must exceed FNN **0.724653** by
**≥ 0.002000** (threshold **0.726653**). Frozen `6bba` refused until
that keep.

See `configs/learned_unet_orgdet_fnn_rank_v1.yaml` and
`reports/learned_unet_orgdet_fnn_rank_v1.md`.

## Cost estimate (before any ranker launch)

- Storage this cycle: ignored geff-export zip (322 KB) + orgop zip
  (848 KB) + extracted 44b6/6bba GEFFs and two 44b6 graphs. No volumes
  or weights were committed.
- Full appearance ranker would need 44b6 train Zarrs (tens of GB) and a
  new private GPU helper. Do **not** launch that until a link-head
  canary is `canary_ok`.
- A canary can start from locked FNN/orgop graphs plus the two already
  exported tuning Zarrs if appearance features are required.
- No new public-test kernel. Hidden-runtime of FNN-fast-test stays
  **4.43 h**.
