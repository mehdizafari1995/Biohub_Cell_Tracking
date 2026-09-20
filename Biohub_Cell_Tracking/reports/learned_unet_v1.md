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

This operating point is **not locked**. Official 44b6 tuning score and one-shot `6bba` holdout are **not run** until a GPU worker finishes and weights exist locally under ignored `artifacts/kaggle/`.

## Kaggle helper

Private GPU kernel `mahdimzafari/biohub-learned-unet-v1`. Internet off. **Not a submission.**
Do not overwrite `biohub-baseline-v1`, `biohub-association-v2`, or the export helpers.

| version | status | notes |
|---|---|---|
| 1 | error | GPU T4 x2 available. 44b6 split written (69 train / 2 internal val, no 6bba). Uncaught crash. |
| 2 | error | `train_error` `ValueError: unsupported zarr codec 'zstd'` while reading train volumes. Split still 44b6-only. |
| 3 | **running** at this poll | zstd decoder added. Still running. Not complete. Not a leaderboard score. |

This worker has no local GPU. A local simulation is not a Kaggle run. No leaderboard score. Weights stay out of git. The training slug is locked against overwrite while v3 runs.

## Inference kernel (packaged, not pushed)

`learned_unet_infer_v1` (`configs/learned_unet_infer_v1.yaml`, hash `d7578b5f64ddac08`) is a private GPU helper that will attach this training output after COMPLETE, infer only `44b6_12dfb391` + `44b6_267148e4`, and write JSON graphs. It does **not** write `submission.csv`. Do not push it until v3 finishes. See `reports/learned_unet_infer_v1.md`.

## Decision

Pending GPU worker. association_v2 remains the scoring candidate until a 44b6-only lock and one-shot frozen holdout.
