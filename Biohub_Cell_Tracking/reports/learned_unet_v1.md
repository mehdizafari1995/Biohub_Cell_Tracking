# learned_unet_v1

New experiment family after rejected `linkable_v1`. Do **not** merge. Keep draft.
Do not overwrite `baseline_v1`, `association_v2`, `division_v2`, `complementary_v1`, or `linkable_v1`.
Do not launch another local-max overlay.

## Hypothesis

Organizer `TemporalUNet3D` + `SimpleNodeTransformer`, retrained **specimen-separated on `44b6` only**, can raise node recall above the locked DoG detector (holdout recall 0.838 for association_v2 / complementary / linkable) without the extra-`T_pred` failure mode of complementary peaks.

Public organizer weights are **not** a valid frozen-`6bba` test (random 90/10 split leakage). See `docs/learned_detection_estimate.md`.

## Operating point (not locked)

Config: `configs/learned_unet_v1.yaml`. Hash `4c38035d6657f2ad`.

- Train stems: every train `44b6_*` except the two frozen-tuning sequences
- Internal val (early-stop / monitoring only): `44b6_12dfb391`, `44b6_267148e4`
- Holdout `6bba` is refused in train and internal val. The kernel never opens those volumes
- 3 epochs, batch 1, grad-accum 8, lr `1e-4`, downsample `(1, 4, 4)`, skip full-res temporal attention
- Edge supervision: index UNet features at **sparse GT nodes** (not detect-and-match). Documented as `gt_node_index`
- Kernel writes weights + `train_manifest.json`. It does **not** write `submission.csv`

Sanitized train record: `reports/learned_unet_v1_train.json`. Weights stay out of git.

## Kaggle helper

Private GPU kernel `mahdimzafari/biohub-learned-unet-v1`. Internet off. **Not a submission.**
Do not overwrite `biohub-baseline-v1`, `biohub-association-v2`, or the export helpers.

| version | status | notes |
|---|---|---|
| 1 | error | GPU T4 x2 available. 44b6 split written (69 train / 2 internal val, no 6bba). Uncaught crash. |
| 2 | error | `train_error` `ValueError: unsupported zarr codec 'zstd'` while reading train volumes. Split still 44b6-only. |
| 3 | **COMPLETE / trained** | 3 epochs, 6117 windows, best train loss 0.3604, 2.08M params, 11732 s, peak 1793 MB, T4 x2, torch 2.10+cu128. Collector extracted weights + manifests into ignored `artifacts/kaggle/learned_unet_v1`. No `submission.csv`. Not a leaderboard score. |

This worker has no local GPU. A local simulation is not a Kaggle run. No leaderboard score. Weights stay out of git. The training slug is locked against overwrite.

COMPLETE handoff is `scripts/collect_learned_unet_v1.py`. After **complete** it extracts only `train_manifest.json` / split JSON / `learned_unet_v1_{best,last}.pt` into ignored `artifacts/kaggle/learned_unet_v1`. It refuses `submission.csv`, Zarr/GEFF members, and holdout stems. Infer sibling: `scripts/collect_learned_unet_infer_v1.py`.

## Inference kernel

`learned_unet_infer_v1` (`configs/learned_unet_infer_v1.yaml`, hash `d7578b5f64ddac08`) attaches this training output and infers only `44b6_12dfb391` + `44b6_267148e4`. It does **not** write `submission.csv`. See `reports/learned_unet_infer_v1.md`.

## Decision

Training COMPLETE. Infer v3 COMPLETE. Official 44b6 tune **0.655751 vs association_v2 0.632460** (`keep_candidate_learned_unet_infer_v1`). Do not retune. One-shot frozen `6bba` holdout is next, under a new infer slug.
