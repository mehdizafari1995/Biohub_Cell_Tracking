# Pipeline

```
zarr volume (T,Z,Y,X)
        │
        ▼
 per-timepoint normalize + detect centroids
        │
        ▼
 link t → t+1 in physical µm
        │
        ▼
 TrackGraph (nodes + directed edges)
        │
        ├── official evaluator (tracking_cellmot.metrics) if installed
        ├── diagnostic evaluator (biohub_tracking.evaluate)
        └── submission.csv
```

Detection is required: the official data does not include detections or masks.

Default baseline (`configs/baseline.yaml`):

1. Min-max normalize each 3D frame.
2. `skimage.feature.blob_dog` (`min_sigma=2`, `max_sigma=5`, `threshold=0.05`).
3. Greedy nearest-neighbor linking in µm, 15 µm gate, one child per parent.
4. Export integer voxel centroids.

`configs/hungarian.yaml` replaces greedy matching with Hungarian assignment. That is the first measured experiment, not a replacement of the baseline config.

Locked `association_v2` (`configs/association_v2.yaml` and `notebooks/kaggle_association_v2.py`) keeps the same detections as `baseline_v1` and replaces consecutive greedy linking with a 12 µm gated look-back (up to 2 skipped frames), interpolated consecutive nodes, and an intensity cost. `max_children=1`. Do not overwrite `baseline_v1` or retune after holdout. Independently audit a downloaded worker CSV with `scripts/audit_kaggle_output.py`; that official score is not a leaderboard number.

`configs/division_v2.yaml` runs the locked association_v2 linker, then proposes at most one extra child when conservative geometry and topology checks pass. Thresholds are locked from tuning-specimen (`44b6`) GEFFs only. Official holdout is identical to association_v2 (0.591570, 0/0/5). Keep as a no-op overlay. Do not retune. `division_v1` is rejected (0/3/5 on frozen val).

`configs/complementary_v1.yaml` keeps locked baseline_v1 primary detections and the association_v2 linker, then adds isolated complementary local-max peaks (≥ 6.0 µm from any primary, ≥ 0.45 × primary median intensity). Locked from 44b6, scored once on frozen 6bba, **rejected**. Do not retune.

`configs/learned_unet_infer_v1.yaml` loads specimen-separated `learned_unet_v1` weights and infers only the frozen-tuning pair. Detection is UNet local-max; linking is the locked 12 µm 1-1 gate. Holdout `6bba` is refused. Does not write `submission.csv`. Official 44b6 tune **0.655751 vs association_v2 0.632460**. Do not retune.

`configs/learned_unet_link_v1.yaml` keeps those locked detections and replaces the greedy gate with trained `SimpleNodeTransformer` pairwise scores (softmax, threshold 0.5, 1-1). Official 44b6 **0.0 vs infer 0.655751**. **Rejected.** Hash `3a7423c6858ace41`. Softmax+0.5 wrote 0 edges on dense detections. Frozen `6bba` not scored. Do not retune. Does not write `submission.csv`.

`configs/learned_unet_link_v2.yaml` keeps the same locked detections and transformer scores. Assignment is greedy 1-1 on descending softmax with **no 0.5 cutoff**, plus the locked association_v2 12 µm spatial gate. Official 44b6 **0.224746 vs infer 0.655751**. **Rejected.** Hash `1d4015dc499bd604`. Frozen `6bba` not scored. Do not retune. Do not overwrite this slug. Does not write `submission.csv`.

Validation splits are by embryo ID (`44b6` vs `6bba`). Neighboring frames from one specimen are never randomly split. Complete holdout scoring (both frozen validation sequences plus each annotated-division event) is `scripts/run_complete_validation.py`.

Local diagnostic scores use `biohub_tracking.evaluate`. Reported competition scores use the vendored official `tracking_cellmot.metrics` implementation. See `docs/coordinate_audit.md` and `third_party/THIRD_PARTY.md`.
