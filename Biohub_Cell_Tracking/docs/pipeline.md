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

Validation splits are by embryo ID (`44b6` vs `6bba`). Neighboring frames from one specimen are never randomly split. Complete holdout scoring (both frozen validation sequences plus each annotated-division event) is `scripts/run_complete_validation.py`.

Local diagnostic scores use `biohub_tracking.evaluate`. Reported competition scores use the vendored official `tracking_cellmot.metrics` implementation. See `docs/coordinate_audit.md` and `third_party/THIRD_PARTY.md`.
