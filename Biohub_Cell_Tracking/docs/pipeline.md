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

Validation splits are by embryo ID (`44b6` vs `6bba`). Neighboring frames from one specimen are never randomly split.

Local diagnostic scores use `biohub_tracking.evaluate`. Reported competition scores use the vendored official `tracking_cellmot.metrics` implementation. See `docs/coordinate_audit.md` and `third_party/THIRD_PARTY.md`.
