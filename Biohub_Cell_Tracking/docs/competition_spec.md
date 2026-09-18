# Competition specification

Sources were read on 2026-09-18. Anything not taken from those pages is marked **unverified**.

## Task

Detect zebrafish cells in 3D+time microscopy, link them across frames, and reconstruct lineages including divisions.

Official wording: “develop algorithms to detect, track and link cells across time in 3D microscopy data, including accurate identification of cell divisions and lineage reconstruction.”

Source: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview

## Scoring metric

```
score = adjusted_edge_jaccard + 0.1 × division_jaccard
```

Published details:

- Predicted nodes are matched to ground-truth nodes **per timepoint** by optimal bipartite assignment on scaled centroid distance.
- Maximum match distance: **7.0 µm**.
- Physical scale used by the metric: **z = 1.625 µm/voxel, y = x = 0.40625 µm/voxel**.
- A predicted edge is a true positive when both endpoints match ground-truth nodes that are connected by a ground-truth edge.
- Edge Jaccard is `TP / (TP + FP + FN)`, then adjusted by a penalty on over-predicting the total number of nodes.
- A cell division is a node with two or more outgoing edges.
- Per-sample adjusted edge Jaccards are weight-averaged by `(TP + FP + FN)`.
- Division Jaccards are micro-averaged across samples.
- Ground truth is **sparse**. The metric accounts for that. Scores can exceed 1.0.

The organizer’s metric document adds:

```
adjusted_jaccard = max(0, jaccard · (1 − a · (T_pred − T_true) / T_true))
```

with `a = 0.1`. `T_true` is a coarse estimate of the true node count, including unlabeled cells. Unmatched predicted nodes are not counted as node false positives, but they do affect `T_pred`. Predicted edges among unlabeled cells are ignored unless they conflict with a matched GT node’s in/out partner.

Division matching in the official document uses a local window:

```
grandparent → dividing parent → children → grandchildren
```

A predicted fork can still score a TP one timepoint before or after the GT split if local directed topology is satisfied. Weakly connected components are not enough.

Sources:

- https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview
- https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md
- https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/src/tracking_cellmot/metrics.py

This repo reports competition-style scores with the organizer implementation vendored at `third_party/kaggle-cell-tracking-competition` (`075fc5f5`), via `biohub_tracking.official_eval`. `biohub_tracking.evaluate` remains a **labelled diagnostic** (published edge rules + parent–daughter division proxy). Combined score uses **adjusted** edge Jaccard, not the raw Edge Jaccard column. See `docs/coordinate_audit.md`.

## Input files

From the official Data page:

| Path | Contents |
|---|---|
| `train/` | Paired `{name}.zarr` image and `{name}.geff` tracking graph |
| `test/` | `{name}.zarr` only. Public test samples are copies from train. Hidden test is swapped in on notebook rerun and is approximately the same size as train |
| `sample_submission.csv` | Valid submission demonstrating the schema (890 bytes) |

Image volumes are Zarr v3 directories:

- Array path: `0/`
- Typical shape: `(T, Z, Y, X)` = `(100, 64, 256, 256)`, `uint16`
- Chunks: one timepoint `(1, 64, 256, 256)`, blosc/zstd
- Chunk key: `0/c/{t}/0/0/0`
- Array metadata: `0/zarr.json`

Ground-truth GEFF (training only) contains:

- `nodes/ids`
- `nodes/props/{t,z,y,x}/values` — integer centroid coordinates in voxels
- `edges/ids` — shape `(N, 2)` as `(source_id, target_id)`
- `estimated_number_of_nodes` in GEFF `zarr.json` extras

Folder names follow `{embryo_id}_{field_of_view}`. Train and test are embryo-disjoint.

**Detections are not provided. Segmentation masks are not provided.** The solver must detect cells from images.

Source: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/data

### Verified from a downloaded official sample

Sample `train/44b6_0db75fae`:

- Image `zarr.json` axes: T (second), Z/Y/X (micrometer)
- Scale: `[1.0, 1.625, 0.40625, 0.40625]` for T,Z,Y,X
- Array: shape `(100, 64, 256, 256)`, dtype `uint16`, chunks `(1, 64, 256, 256)`, codecs `bytes` (little-endian) + `blosc` zstd bitshuffle
- GEFF version 1.1, directed
- 157 annotated nodes, 151 edges, t in `[0, 88]`, integer `t,z,y,x`
- `estimated_number_of_nodes = 15335`
- Node IDs are large integers (not 1..N)
- No GT division (`out_degree >= 2`) in this sample

Sample `train/6bba_05b6850b`:

- 861 nodes, 845 edges, t in `[0, 99]`
- `estimated_number_of_nodes = 6362`
- No GT division in this sample

Public `sample_submission.csv` datasets:

- `44b6_0113de3b`
- `44b6_0b24845f`
- `6bba_05b6850b`
- `6bba_05db0fb1`

The overview snippet uses abbreviated names such as `44b6`. The downloaded `sample_submission.csv` uses the full folder stem. **Use the full stem without `.zarr`.**

### Unverified from official pages

- Exact train sample count and total bytes of the full dump (third-party notes say ~199 train samples and ~81 GB zip; not copied here as official).
- Hidden test embryo IDs.
- Whether every volume is exactly `(100, 64, 256, 256)` (the Data page says “typically”).
- Coordinate origin (array `[0,0,0]` vs optical origin). Centroids are integer voxel indices in the published GEFF props.

## Divisions and parent–daughter links

- A division is a node with **two or more outgoing edges**.
- Organizer baseline documentation: a GT division is one node at `t` linked to two nodes at `t+1`.
- Lineage reconstruction is required by the task statement.
- Official division Jaccard uses a local 5-generation window, not a single-frame fork check.

Sources: overview, `metrics.md`, https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/README.md

## Submission format

Notebook code competition. Output file must be named `submission.csv`.

Columns:

```
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
```

Rules:

- Node rows: `row_type=node` with `node_id,t,z,y,x` integer voxel centroids. `source_id=target_id=-1`.
- Edge rows: `row_type=edge` with `source_id,target_id` referencing node IDs. `node_id,t,z,y,x=-1`.
- `id` is a consecutive throwaway index.
- `dataset` must match test folder names without `.zarr`.
- Every test dataset must appear.

Source: overview. Confirmed against downloaded `sample_submission.csv`.

## Inference interface

Submissions must be made through Notebooks:

- CPU notebook ≤ 12 hours
- GPU notebook ≤ 12 hours
- **Internet disabled**
- Freely and publicly available external data is allowed, including pre-trained models
- File name: `submission.csv`

Source: overview “Code Requirements”.

## External data, models, and rules

From https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules

- Winner license: MIT
- Data access: CC0
- Max team size: 5
- 5 submissions per day; 2 final submissions
- External data/models must be public and reasonably accessible at no or minimal cost
- No private code sharing outside a team
- Must accept the rules before the entry deadline
- Competition data may be used commercially (Competition Use and Commercial are checked)

## Timeline (11:59 PM UTC)

| Event | Date |
|---|---|
| Start | 29 June 2026 |
| Entry deadline | 22 September 2026 |
| Team merger deadline | 22 September 2026 |
| Final submission deadline | 29 September 2026 |

Source: overview.

## Organizer starter material

| Resource | Role | Access on 2026-09-18 |
|---|---|---|
| https://github.com/royerlab/kaggle-cell-tracking-competition | Official baseline repo: TemporalUNet3D + transformer linker, GEFF I/O, evaluator | README, metrics.md, scripts, `src/tracking_cellmot` fetched |
| https://www.kaggle.com/code/thibautgoldsborough/unet-baseline-inference-submission | Organizer UNet inference notebook | Linked from official README; notebook body **unverified** (page not fully retrieved) |
| https://www.kaggle.com/code/inversion/cell-tracking-getting-started-w-nearest-neighbor | Kaggle staff getting-started kernel | Page metadata: Apache-2.0, public score **0.143**. Full source **unverified** (fetch timed out). Local baseline follows the documented blob + nearest-neighbor recipe used in that kernel’s public write-ups |

This project’s CPU baseline follows the getting-started method (3D `blob_dog` + nearest-neighbor linking). It does **not** reproduce the organizer UNet, which needs GPU training/weights.

## Manual Kaggle step

This environment already listed and downloaded competition files, so the attached Kaggle account has accepted the rules.

On another machine: open the competition page and click **Join Competition**, then accept the rules. Without that step the API returns 403 and data cannot be downloaded.
