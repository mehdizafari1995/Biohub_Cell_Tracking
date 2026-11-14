# learned_unet_orgdet_holdout_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or rejected orgop slugs. Do **not** retune parent orgdet hash
`80b20589198148b7`. Do **not** score this holdout again.

## Hypothesis

Locked `learned_unet_orgdet_v1` kept on 44b6 (**0.679578 vs infer
0.655751**) because organizer detections (`det_threshold=0.99`,
`pool_kernel_um=3.0`, flip TTA) recovered the same official TPs with
26540 fewer predicted nodes. Applying that locked operating point once
to frozen `6bba` (greedy 12 µm, no gap, no ILP) tests whether the
`T_pred` reduction generalizes against champion
`learned_unet_prune_v1` **0.701378**.

## Operating point (locked; one-shot 6bba)

Config: `configs/learned_unet_orgdet_holdout_v1.yaml`. Hash
`912f0c19b572a054`. Parent orgdet hash `80b20589198148b7`. Train hash
`4c38035d6657f2ad`. Rejected orgop hash `250c5bfacd4a8198`.

- Weights: locked `learned_unet_v1` 44b6-only (`learned_unet_v1_last.pt`)
- Detection: `det_threshold=0.99`, `pool_kernel_um=3.0`, flip TTA
- Linker: `greedy_gate_12um`, last-position, `max_link_um=12.0`,
  `max_gap_frames=0`, no interpolation, no transformer, no ILP
- Stems: `6bba_05db0fb1`, `6bba_c328f2fd` only. 44b6 refused.
- Writes no `submission.csv`
- Helper slug `mahdimzafari/biohub-learned-unet-orgdet-holdout-v1` v1
  **COMPLETE / inferred**

Do **not** retune this hash. Do **not** overwrite v1.

## Cost estimate (before score)

- Storage: locked `learned_unet_v1` weights already on the train helper.
  Graphs JSON only.
- GPU: T4 x2 private helper. Measured **173.8 s / 1185 MB**. Hidden-test
  projection 199 × 86.9 s ≈ **4.8 h**, under 12 h.
- Not a competition submission.

## Official frozen-holdout score (not leaderboard)

Local official scorer `075fc5f5` on `6bba_05db0fb1` + `6bba_c328f2fd`
once. Graphs from private helper
`mahdimzafari/biohub-learned-unet-orgdet-holdout-v1` v1. GEFFs from
private helper `mahdimzafari/biohub-geff-export` v1. **Not a Kaggle
run. Not a leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes / edges |
|---|---|---|---|---|---|---|
| learned_unet_holdout_v1 | 0.693605 | 0.72769 | 0.69360 | 0.95713 | 0/0/5 | 149223 / — |
| learned_unet_prune_v1 | 0.701378 | 0.74335 | 0.70138 | 0.97490 | 0/0/5 | 158640 / 153793 |
| learned_unet_orgdet_holdout_v1 | **0.705226** | 0.72006 | 0.70523 | 0.93871 | 0/0/5 | 121810 / 113640 |

Per-sequence orgdet: `6bba_05db0fb1` **0.735103** (prune 0.718354);
`6bba_c328f2fd` **0.634794** (prune 0.661834). Official matched-edge
TP/FP/FN **1407/271/276** vs prune **1480/308/203**. Scoring runtime
5.58 s, peak 674 MB.

Decision **`keep_candidate_learned_unet_orgdet_holdout_v1`**. Do **not**
retune. Do **not** score holdout again.

## Error analysis

The 0.99 detector dropped 36830 predicted nodes vs prune (121810 vs
158640). Adjusted Jaccard rose even though official TPs fell (1407 vs
1480) and recall fell (0.939 vs 0.975). The larger sequence improved;
the smaller sequence regressed; the weight-average still beats prune.
See `reports/learned_unet_orgdet_holdout_v1_error.json`.

## Next

Keep `learned_unet_orgdet_holdout_v1` as the holdout champion
(**0.705226 vs prune 0.701378**). Public-test GPU notebook
`learned_unet_orgdet_test_v1` hash `c45c859610d29e90` v1 **COMPLETE /
`audit_ok`**. Do **not** overwrite prune-test v1, gap-test v1, this
holdout helper, or orgdet-test v1. `HUMAN_ACTION_REQUIRED`: UI-submit
orgdet-test v1 before 2026-09-22 23:59 UTC.
