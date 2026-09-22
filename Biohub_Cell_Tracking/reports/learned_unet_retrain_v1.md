# learned_unet_retrain_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not** merge.
Do **not** overwrite locked train/infer/holdout/test/gap-test/prune-test/orgop
slugs. Do **not** launch until a reviewed cycle. Frozen `6bba` stays closed.

## Hypothesis

After `reject_learned_unet_orgop_v1` (0.0 vs infer 0.655751; empty
softmax>0.5 graph), organizer defaults on the locked 3-epoch 44b6
weights are closed. Public 0.94-class scores used 199-video / ~402-epoch
checkpoints that leak `6bba` and stay forbidden for any KEEP_CANDIDATE
path. Discussion 734604 / 730160 and the 0.94 audit rank a longer
**44b6-only** retrain as the next honest detection lever.

`learned_unet_retrain_v1` continues the locked `learned_unet_v1` v3
weights (`learned_unet_v1_last.pt`, train hash `4c38035d6657f2ad`) for
**3 additional epochs** (6-epoch equivalent). Architecture, lr, split,
and seed stay frozen. After training, a later cycle would official-score
44b6 with the **locked infer operating point** (greedy 12 µm, det 0.5,
pool 5.0) vs **0.655751**. Public 199-video / DeepCenter / organizer-split
weights stay refused.

## Operating point (packaged, not launched)

Config: `configs/learned_unet_retrain_v1.yaml`. Hash `92a419926fa7e40b`.
Reserved slug `mahdimzafari/biohub-learned-unet-retrain-v1`.
`LAUNCH_AUTHORIZED=True`. In `KERNELS`. Private helper v1 **RUNNING**.

- Resume from `learned_unet_v1_last.pt` only
- `additional_epochs=3`, `from_scratch=false`
- Same 44b6 train / internal-val split as parent
- `writes_submission_csv=false`
- Do **not** overwrite `mahdimzafari/biohub-learned-unet-v1`

## Compute (estimate, not a run)

Parent v3: 11732 s / 3 epochs / peak 1793 MB / T4 x2.
Continuation: **3.10–3.75 h** GPU, peak ~1.8 GB, under `max_hours=11`
and the 12 h kernel limit. Hidden-test wall time is **not** projected
until a 44b6 keep and a later infer packaging cycle.

Needs: locked train helper weights (already ignored local artifacts),
train Zarrs/GEFFs for 44b6, official scorer `075fc5f5`. No new public
datasets.

## Acceptance gates (after an authorized launch)

1. Training completes on 44b6 only. No `6bba`. No `submission.csv`.
2. Official-score the two frozen-tuning sequences with locked infer
   defaults vs **0.655751**. Report combined, adj edge Jaccard, division
   Jaccard, recall, node/edge counts, runtime, peak RAM/VRAM.
3. **Reject** if combined is not strictly greater. Do **not** open `6bba`.
   Do **not** retune epochs / lr / architecture.
4. If it beats 0.655751, lock hashes, then **one** frozen `6bba` score vs
   champion `learned_unet_prune_v1` **0.701378**.
5. Reject on holdout combined regression. Else
   `KEEP_CANDIDATE_NOT_SUBMISSION_AUTHORIZED`.
6. Public 0.553 / 0.94 titles are not tune targets. No CLI submit.
   No overwrite of prune-test v1.

## This cycle

Launch authorized. Status-only poll **2026-09-22T05:53:06Z**: private helper
`mahdimzafari/biohub-learned-unet-retrain-v1` version **1 RUNNING**, 0 output
files, no `submission.csv`, no failure message. Did **not** download. Did
**not** push a second version. Does **not** overwrite the parent train slug.
44b6 lanes A/B/C hashes `1db21f87d00ce083` / `594f6fbc03c974cf` /
`f924802e5f716a61` are locked before scores. Fail-closed official scorer
`scripts/run_learned_unet_retrain_lanes.py` is packaged and refuses
incomplete helper / missing graphs / `6bba`. Frozen `6bba` stays closed
until a lane wins. Champion remains `learned_unet_orgdet_holdout_v1`
**0.705226** / orgdet-test v1 `KEEP_CANDIDATE_NOT_SUBMISSION_AUTHORIZED` /
`READY_FOR_MANUAL_SUBMISSION_REVIEW`.
