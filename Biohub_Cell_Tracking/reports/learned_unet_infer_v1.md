# learned_unet_infer_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not** merge.
Do not overwrite `baseline_v1`, `association_v2`, `division_v2`,
`complementary_v1`, `linkable_v1`, or the running training helper
`mahdimzafari/biohub-learned-unet-v1`.

## Hypothesis

Specimen-separated `TemporalUNet3D` detections, linked with the locked
association_v2 12 µm 1-1 gate (no gap-close), can raise node recall on
frozen-tuning `44b6` without the extra-`T_pred` failure of complementary
local-max overlays. Transformer edge scores are **not** used in this first
infer pass so the comparison isolates detection.

## Operating point (not locked, not scored)

Config: `configs/learned_unet_infer_v1.yaml`. Hash `d7578b5f64ddac08`.
Training hash remains `4c38035d6657f2ad`.

- Infer stems only: `44b6_12dfb391`, `44b6_267148e4`
- Holdout `6bba` is refused
- `det_threshold` 0.5, TTA flip-x/y/xy, `pool_kernel_um` 5.0
- Normalisation matches training: per-window min-max, not organizer quantiles
- Linker: greedy 12 µm, max_children=1, max_parents=1, no gap-close
- Writes JSON graphs + `infer_manifest.json`. Does **not** write `submission.csv`

## Kaggle helper

Private GPU kernel `mahdimzafari/biohub-learned-unet-infer-v1`. Packaged,
**not pushed**. Attach training helper output only after that kernel is
COMPLETE. Internet off. **Not a submission.**

| version | status | notes |
|---|---|---|
| — | not pushed | Waiting on training kernel v3. Do not attach incomplete weights. Collector `scripts/collect_learned_unet_infer_v1.py` is packaged and will poll this slug only. |

Volume I/O now probe-reads native `zarr` and falls back to the custom zstd decoder (the v2 training failure mode) so the first infer push does not die on `unsupported zarr codec 'zstd'`.

Estimate (2 tuning volumes, T4, 4 TTA): tens of minutes, well under 12 h.
This CPU worker cannot run the model. A local simulation is not a Kaggle run.

## Official score

Not run. Graphs do not exist yet. association_v2 remains the candidate
(locked tuning 0.632460). Do **not** look at `6bba`.
