# learned_unet_orgdet_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or rejected orgop slugs. Do **not** retune. Frozen `6bba` was **not**
opened this cycle.

## Hypothesis

Rejected `learned_unet_orgop_v1` failed because transformer softmax>0.5
wrote zero candidate edges (official 44b6 **0.0** vs infer **0.655751**),
not necessarily because organizer detections (`det_threshold=0.99`,
`pool_kernel_um=3.0`, flip TTA) are worse. Applying the locked infer
greedy 12 µm last-position linker to those locked orgop nodes should
isolate the detection operating point. This is not another assignment
on infer detections and not a retune of orgop ILP.

## Operating point (44b6 keep; holdout not opened)

Config: `configs/learned_unet_orgdet_v1.yaml`. Hash `80b20589198148b7`.
Train hash `4c38035d6657f2ad`. Infer parent `d7578b5f64ddac08`.
Rejected orgop hash `250c5bfacd4a8198`.

- Source graphs: locked `learned_unet_orgop_v1` detections
- Linker: `greedy_gate_12um` (locked infer), last-position, `max_link_um=12.0`,
  `max_gap_frames=0`, no interpolation, no intensity, no transformer, no ILP
- Detection not retuned (`retunes_detection: false`)
- Writes no `submission.csv`
- Helper slug `mahdimzafari/biohub-learned-unet-orgdet-v1` packaged, not
  pushed (local overlay is sufficient for 44b6)

Do **not** retune this hash. Do **not** open `6bba` until a matching
6bba orgop-detection + greedy graph exists.

## Cost estimate (before score)

- Storage: reuse orgop-helper graphs + two `44b6` GEFFs from
  `mahdimzafari/biohub-geff-export` v1. No volumes, no weights.
- CPU: re-link 91152 locked orgop detections. Measured **9.50 s / 501 MB**.
- No GPU. Not a Kaggle run.

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4` only.
Graphs from private helper `mahdimzafari/biohub-learned-unet-orgop-v1` v1.
GEFFs from private helper `mahdimzafari/biohub-geff-export` v1. **Not a
Kaggle run. Not a leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes / edges |
|---|---|---|---|---|---|---|
| learned_unet_infer_v1 | 0.655751 | 0.69001 | 0.65575 | 0.97004 | 0/0/2 | 117692 / 106385 |
| learned_unet_orgop_v1 | 0.0 | 0.0 | 0.0 | 0.0* | 0/0/2 | 91152 / 0 |
| learned_unet_orgdet_v1 | **0.679578** | 0.69055 | 0.67958 | 0.97004 | 0/0/2 | 91152 / 84057 |

\*Orgop short-circuited on empty edges. Orgdet uses the same nodes.

Per-sequence orgdet: `44b6_12dfb391` **0.772105** (infer 0.747072);
`44b6_267148e4` **0.467676** (infer 0.446373). Official matched-edge
TP/FP/FN **877/220/173** vs infer **877/221/173**. Runtime 9.50 s, peak
501 MB. Gate held (0 edges over 12 µm; max 11.94 µm).

Decision **`keep_candidate_learned_unet_orgdet_v1`**. Frozen `6bba`
stayed closed this cycle because holdout needs a new 6bba-only GPU
detection pass at the locked 0.99/pool-3.0 operating point. Do **not**
retune.

## Error analysis

The 0.99 detector recovered the same official TPs as the 0.5 infer
detector while dropping 26540 extra predicted nodes (91152 vs 117692).
Adjusted Jaccard rose because `T_pred` fell, matching the public-0.94
audit claim that extra detections are costly under the official
adjusted metric even when sparse-GT node FPs are free. See
`reports/learned_unet_orgdet_v1_error.json`.

## Next

Keep `learned_unet_orgdet_v1` as a 44b6 keep. Do **not** retune.
Package a **new** 6bba-only GPU helper that reuses the locked orgop
detection path (`0.99` / pool 3.0 / TTA) plus this locked greedy
linker, then official-score once vs champion
`learned_unet_prune_v1` **0.701378**. Do **not** overwrite orgop v1,
prune-test v1, or locked infer/holdout slugs. Do **not** launch
`learned_unet_retrain_v1`. Current packaged public-test worker remains
`learned_unet_prune_test_v1` v1
`READY_FOR_MANUAL_SUBMISSION_REVIEW`.
