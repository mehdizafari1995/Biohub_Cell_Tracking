# learned_unet_orgdet_fnn_fast_test_v1

Same family as `learned_unet_orgdet_fnn_v1` (draft PR #7). Do **not**
merge. Do **not** overwrite
`mahdimzafari/biohub-learned-unet-orgdet-fnn-test-v1` v1,
orgdet-elf-test, orgdet-gap-prune-test, orgdet-gap-test, orgdet-test,
prune-test, or gap-test. Do **not** retune hash `9b9763e26bc0f037`. Do
**not** score frozen `6bba` again.

## Hypothesis

The locked FNN overlay is the holdout champion (official 44b6
**0.724653**, official 6bba **0.752091**). The first public-test worker
`learned_unet_orgdet_fnn_test_v1` was **COMPLETE / `audit_ok`** but
hidden projection **11.04 h** tripped `fail_closed_approaches_12h`
because the per-edge Python nearest-neighbor scan was O(edges ×
nodes_at_t+1). A new immutable slug keeps the same keep/drop rule and
vectorizes the distance scan. Not another 12 µm assignment. Not a
short-fragment retune. Not another 8.0 µm dt=1 length retune. Do **not**
retune `nn_tie_eps`.

## Operating point

Config: `configs/learned_unet_orgdet_fnn_fast_test_v1.yaml`.
Hash `9b9763e26bc0f037`. Parent overlay `95e9300eb84ba71c`. Parent
fnn-test `1af763e6ffc23188`.

- Discover every `test/*.zarr`
- Locked orgdet detections: 0.99 / pool 3.0 / TTA
- Locked greedy 12 µm, then association_v2 gap-close
- Then drop leftover degree-0 nodes
- Then drop consecutive dt=1 edges longer than 8.0 µm
- Then keep only forward nearest-neighbor dt=1 edges (vectorized)
- Then drop newly created degree-0 leftovers
- Writes `submission.csv` + `run_manifest.json`

## Cost estimate (before rehearsal)

- Storage: reuse locked `learned_unet_v1` v3 weights. No new training.
- GPU: same orgdet+gap+prune+ELF path as elf-test (**352.87 s / 4
  volumes**) plus ~0.2 s vectorized NN.
- Hidden: **~4.89 h** under 12 h. Peak ~1.2 GB.
- Not a Kaggle run until the private rehearsal is pushed.

## Status

Packaged. Private slug
`mahdimzafari/biohub-learned-unet-orgdet-fnn-fast-test-v1`. Do **not**
overwrite fnn-test v1. Do **not** submit from the CLI.
