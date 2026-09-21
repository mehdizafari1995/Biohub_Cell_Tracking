# Strategy: public 0.94 vs specimen-separated learned U-Net

Authoritative audit: `reports/public_094_parity_audit.md`.
Do not launch a kernel from this note. Do not submit.

## Facts

- Our champion `learned_unet_prune_v1` (**0.701378**) is official frozen
  `6bba` after 3-epoch 44b6-only training. It is not a public LB number.
- Public 0.94-class notebooks share organizer `DET_THRESHOLD=0.99` +
  tracksdata ILP plus **Pilkwang/Reyhan weights trained on all 199
  videos**. Discussion 730160: those checkpoints have seen both embryos.
- 0.9462 was **not** independently returned as an API `publicScore`.
  Verified claims nearby: 0.941 lineage, one knob 0.942, a 0.948 title
  that reproduced at 0.935. Rule-based public ceiling in 716952 is
  **0.826**, driven by detection, not linking.
- External public checkpoints are rules-legal and **invalid** for a
  KEEP_CANDIDATE path that must survive hidden embryo-disjoint test.

## Decision

Copying the 0.947 weight pack is forbidden for the candidate.
The one high-upside, rules-compliant next experiment is
`learned_unet_orgop_v1` (hash `250c5bfacd4a8198`): locked 44b6 weights
+ organizer predict defaults (0.99 + ILP). Specified, not launched.

Tune only on `44b6_12dfb391` + `44b6_267148e4` vs infer **0.655751**.
Open `6bba` only after a documented keep. Compare holdout to prune
**0.701378**. Public 0.553 is association_v2 only.

## Still the packaged public-test worker

`mahdimzafari/biohub-learned-unet-prune-test-v1` v1 remains
`READY_FOR_MANUAL_SUBMISSION_REVIEW`. UI submit only. No CLI submit.
Read-only poll still shows only association_v2 **56399003**.
Discussion **738217** is independently titled focus3d segmentation
and is **not** part of orgop. Do not overwrite v1 while orgop is specified.
