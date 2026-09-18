# Official evaluator (vendored)

Source: https://github.com/royerlab/kaggle-cell-tracking-competition
Commit: `075fc5f5a52d11077f9dc2b074644618f26939e2`
License: BSD 3-Clause (see `third_party/kaggle-cell-tracking-competition/LICENSE`)

Graph matching dependency:

- https://github.com/royerlab/tracksdata @ `7523bb39ed9ab1bf88512911fd6c69e46fcad188` (`tracksdata==0.1.0rc10`)
- `polars==1.44.2`
- `geff==1.3.1.1.3`

Torch is **not** required for `tracking_cellmot.metrics` or `division_metrics`. Do not import `tracking_cellmot.io` in this CPU environment; that module imports torch.

Install extras into the isolated venv:

```bat
python -m pip install -r requirements-official.txt
```

Reported competition scores in this repo use `tracking_cellmot.metrics.evaluate` plus `per_sample_metrics` / `summarise`. `biohub_tracking.evaluate` is a labelled diagnostic only.

## Combined score vs “Edge Jaccard”

The combined score is **lower than raw Edge Jaccard** when the node-count adjustment is a penalty, because:

```
score = adj_edge_jaccard + 0.1 × division_jaccard
adj_edge_jaccard = max(0, edge_jaccard · (1 − 0.1 · (N_pred − N_true) / N_true))
```

The “Edge Jaccard” column is the **raw** `TP / (TP + FP + FN)`. It is not a bug in the local diagnostic: the published combined score uses the **adjusted** term. When there are no divisions, `summarise` drops the division term and `score = adj_edge_jaccard`.

`tracking_cellmot.metrics.evaluate_datasets` is inconsistent with `metrics.md` and with `scripts/evaluate.py`: its `score` adds **raw** edge Jaccard, not the adjusted value. This project follows `summarise` / `scripts/evaluate.py`.

When `N_pred < N_true` (typical for sparse GT vs `estimated_number_of_nodes`), `adj_edge_jaccard` can **exceed** 1.0. That is official behavior.

## Organizer UNet baseline

Inspected from the same commit. TemporalUNet3D + transformer linker, BSD-3-Clause. Public weights: Kaggle dataset `addisonhoward/biohub-unet-weights`, file `unet_baseline.pth` (~1.43 MB), CC0. Training uses sparse edge supervision and does **not** treat unlabeled voxels as exhaustive background. This CPU slice does not train or run that model.
