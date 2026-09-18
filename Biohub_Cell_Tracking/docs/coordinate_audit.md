# Coordinate, timing, and evaluation-boundary audit

Checked against organizer code at
https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/src/tracking_cellmot/metrics.py
and `tracksdata` DistanceMatching @ `7523bb39`.

Small synthetic examples live in `tests/test_official_eval.py`.

## Axis order

- Images: OME-Zarr array `0/` is `(T, Z, Y, X)`.
- Official loader `tracking_cellmot.io._parse_scale` returns `tuple(transform["scale"][-3:])`, i.e. **(Z, Y, X)** microns/voxel: `(1.625, 0.40625, 0.40625)`.
- GEFF node properties are named `t,z,y,x`.
- `DistanceMatching` default `attr_keys` are `(z, y, x)` when `z` exists.
- Submission CSV uses the same `t,z,y,x` voxel centroids.

## Voxel coordinates vs physical distances

Matching multiplies centroid vectors by `scale` **before** comparing to `max_distance=7.0` µm:

```
physical = (z, y, x)_voxels * (1.625, 0.40625, 0.40625)
```

A 4-voxel z offset is 6.5 µm (matchable) with the official scale, but 16 µm with `scale=(4,1,1)` (not matchable at 15 µm). Confirmed with `test_scale_is_zyx_physical_microns`.

GT centroids in downloaded GEFFs are **integer voxel indices**. Predictions are rounded to int for CSV. The matcher accepts floats; rounding is a submission-schema requirement, not a matching requirement.

## Time indices vs physical acquisition time

`graph.match` groups nodes by the integer `t` attribute (`tracksdata.metrics._ctc_metrics._matching_data`). The T axis scale in zarr metadata is **not used** by the evaluator.

Co-located nodes at different `t` do not match (`test_matching_uses_integer_time_index_not_physical_time`).

OME-Zarr on the inspected sample stores T scale `1.0` with unit `"second"`. That is **metadata**, not a confirmed 1 Hz acquisition interval. Linking and scoring use frame index `t` and `t+1` only.

## Sparse GT and unmatched predictions

Unmatched predicted nodes are **not** node false positives. Predicted edges among unmatched cells are ignored unless they conflict with a matched GT node’s in/out partner (`metrics.md` and official tests). Node-count adjustment still uses all predicted nodes vs `estimated_number_of_nodes`.

Do not report unmatched detections as false positives in detection diagnostics.

## Edges and divisions at sequence boundaries

- Only edges with `t_target == t_source + 1` are scored. Backward, same-frame, and skip connections are dropped.
- An extra predicted edge **beyond a GT track start or end** is not penalized (official tests `test_extra_edge_at_track_end_not_penalized`, `test_extra_edge_at_track_start_not_penalized`).
- A cropped sequence that cuts a 5-generation division window (grandparent → parent → children → grandchildren) can make an otherwise correct fork unscorable or a GT division a FN. Partial-volume local runs are therefore not comparable to full-sequence official scores.

## Out-degree cap

Predicted nodes may keep at most two outgoing consecutive edges (lowest edge ids). Extra children are dropped, not penalized.
