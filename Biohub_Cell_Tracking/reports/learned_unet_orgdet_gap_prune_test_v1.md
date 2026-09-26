# learned_unet_orgdet_gap_prune_test_v1

Same family as `learned_unet_orgdet_gap_prune_v1` (draft PR #7). Do **not**
merge. Do **not** overwrite `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1`
v1, orgdet-test, prune-test, or gap-test. Do **not** retune hash
`52a30a5843e0a880`. Do **not** score frozen `6bba` again.

## Operating point

Config: `configs/learned_unet_orgdet_gap_prune_test_v1.yaml`.
Hash `52a30a5843e0a880`. Parent overlay `862e3e7f7ac0330e` (official
6bba **0.723018 vs orgdet-gap 0.722805**). Parent orgdet-gap-test
`1ce4837ae16bee52`.

- Discover every `test/*.zarr`
- Locked orgdet detections: 0.99 / pool 3.0 / TTA
- Locked greedy 12 µm, then association_v2 gap-close
- Then drop leftover degree-0 nodes
- Writes `submission.csv` + `run_manifest.json`

## Cost estimate (before kernel)

- Storage: locked `learned_unet_v1` weights already on the train helper.
  Public `test/*.zarr` only. No train volumes. No new weights.
- GPU: orgdet-gap-test measured **312.69 s / 4 volumes (78.17 s/volume)**
  plus **0.5 s** isolate drop. Public rehearsal ~5.3 min. Hidden-test
  projection **78.67 s/volume × 199 ≈ 4.35 h**, under 12 h.
- Peak RAM budget ~1.2 GB from orgdet-gap-test.
- New slug. Do **not** overwrite orgdet-gap-test v1.

## Kernel

Private GPU slug `mahdimzafari/biohub-learned-unet-orgdet-gap-prune-test-v1`.
Internet off. Attaches competition + `mahdimzafari/biohub-learned-unet-v1`.
**Not a competition submission.** Version and COMPLETE metrics are recorded
in the submission ledger after the one rehearsal.

## Status

Packaged. One private public-test rehearsal is authorized this cycle.
`READY_FOR_MANUAL_SUBMISSION_REVIEW` only after COMPLETE / `audit_ok` /
independent audit. Additional UI submit is **not** authorized unless a
later message separately authorizes that extra click. Do **not** submit
from the CLI.

## Next

Poll the new slug only. If COMPLETE, download into ignored
`artifacts/kaggle/`, independently audit `submission.csv`, and lock the
worker. Do **not** overwrite orgdet-gap-test v1. Do **not** score 6bba
again. Do **not** retune.
