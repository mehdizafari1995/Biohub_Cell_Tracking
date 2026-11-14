# Strategy: public 0.94 vs specimen-separated learned U-Net

Mirror of `Biohub_Cell_Tracking/docs/strategy_public_094.md`.
Authoritative audit: `Biohub_Cell_Tracking/reports/public_094_parity_audit.md`.
Do not launch a kernel from this note. Do not submit.

## Facts

- Our holdout champion is `learned_unet_orgdet_holdout_v1` (**0.705226**
  vs prune **0.701378**) after locked 0.99/pool-3.0 detections plus
  greedy 12 µm. It is not a public LB number. `learned_unet_prune_v1`
  (**0.701378**) remains a locked prior keep.
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
Authorized cycle `learned_unet_orgop_v1` (hash `250c5bfacd4a8198`)
ran organizer predict defaults (0.99 + ILP) on locked 44b6 weights
only. Official 44b6 combined **0.0 vs infer 0.655751**. Softmax>0.5
wrote 0 candidate edges. Decision **`reject_learned_unet_orgop_v1`**.
Zero-edge diagnosis: no silent load defect; transformer near init
(rel 0.11 vs UNet 6.34). Do **not** launch `orgop_fix`. CPU canary
`learned_unet_edge_train_canary_v1` (`1a3d524e4d922701`) is `canary_ok`.
Do **not** retune orgop. Do **not** open `6bba` for orgop. Public 0.553
is association_v2 only.

## Next specified experiment (helper v1 complete; lanes running)

`learned_unet_retrain_v1` hash `92a419926fa7e40b`. Resume locked
`learned_unet_v1_last.pt` for +3 epochs on 44b6 only. Slug
`mahdimzafari/biohub-learned-unet-retrain-v1`. Collected
**2026-09-22T07:40:50Z**: version **1 COMPLETE / trained**. Lane A/B v2
**RUNNING**. Lane C packaged, not pushed. Fail-closed
lane scorer `scripts/run_learned_unet_retrain_lanes.py` is packaged.
Canary
`canary_ok` is evidence the pair head can learn peaked softmax. Do
**not** overwrite the locked train helper. Do **not** push a second
version. See `Biohub_Cell_Tracking/reports/learned_unet_retrain_v1.md`.

Authorized unbundled detection overlay `learned_unet_orgdet_v1` hash
`80b20589198148b7` official-scored on 44b6: **0.679578 vs infer
0.655751**. One-shot holdout `learned_unet_orgdet_holdout_v1` hash
`912f0c19b572a054` is **0.705226 vs prune 0.701378**. Do **not** score
`6bba` again.

## Packaged public-test worker

`mahdimzafari/biohub-learned-unet-orgdet-test-v1` v1 is
`READY_FOR_MANUAL_SUBMISSION_REVIEW`. UI submit only. No CLI submit.
Read-only poll **2026-09-22T06:45:08Z** still shows only
association_v2 **56399003**. Entry deadline **2026-09-22 23:59 UTC**.
Prune-test v1 remains COMPLETE / `audit_ok` but is no longer the
preferred worker. Discussion **738217** is independently titled
focus3d segmentation and is **not** part of orgop or retrain.
Discussion **740392** is closed as `topic_id_not_found` (by-id HTTP
404, kernel search 0). Do not overwrite orgdet-test v1, prune-test v1,
or completed orgop helper v1.
