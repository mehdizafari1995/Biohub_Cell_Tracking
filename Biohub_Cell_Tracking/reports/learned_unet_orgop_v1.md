# learned_unet_orgop_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not** merge.
Do not overwrite locked train/infer/holdout/test/gap-test/prune-test slugs.
Do **not** retune. Frozen `6bba` was **not** opened.

## Hypothesis

Organizer inference defaults (`det_threshold=0.99`, `pool_kernel_um=3.0`,
flip TTA, transformer softmax 0.5, tracksdata ILP −1.0/0.1/0.1/1.0) on
locked 44b6-only `learned_unet_v1` weights would beat greedy infer
**0.655751**. Public 199-video checkpoints were refused.

## Operating point (rejected on tuning)

Config: `configs/learned_unet_orgop_v1.yaml`. Hash `250c5bfacd4a8198`.
Train hash `4c38035d6657f2ad`. Infer parent `d7578b5f64ddac08`.
Private helper `mahdimzafari/biohub-learned-unet-orgop-v1` v1
**COMPLETE / inferred**. Weights
`/kaggle/input/notebooks/mahdimzafari/biohub-learned-unet-v1/learned_unet_v1_last.pt`
(checkpoint hash `4c38035d6657f2ad`). No `submission.csv`.

Kaggle GPU: 153.71 s, peak 1227 MB, T4. Internet off.

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4`.
GEFFs from `biohub-geff-export` v1. **Not a Kaggle run. Not a
leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes / edges |
|---|---|---|---|---|---|---|
| learned_unet_infer_v1 | **0.655751** | 0.69001 | 0.65575 | 0.97004 | 0/0/2 | 117692 / 106385 |
| learned_unet_orgop_v1 | **0.0** | 0.0 | 0.0 | 0.0* | 0/0/2 | 91152 / 0 |

\*Official scorer returns score 0.0 when a graph has no edges; node-recall
is therefore the zero short-circuit, not a separate detection audit.

Per-sequence orgop: both **0.0**. `44b6_12dfb391` 62480 nodes / 0 edges /
773 FN; `44b6_267148e4` 28672 / 0 / 277 FN. Candidate edges after
softmax>0.5: **0 / 0**. Solver `ilp_skipped_no_candidates` (Kaggle had
no `tracksdata`; local ILP had nothing to solve). Helper runtime
153.71 s / 1227 MB; local score 0.57 s / 386 MB.

Decision **`reject_learned_unet_orgop_v1`**. Same empty-edge failure as
rejected `learned_unet_link_v1`. Do **not** retune. Do **not** open
`6bba`. Do **not** overwrite the completed slug.

## Candidate

Champion remains `learned_unet_prune_v1` / prune-test v1
`KEEP_CANDIDATE_NOT_SUBMISSION_AUTHORIZED`.
