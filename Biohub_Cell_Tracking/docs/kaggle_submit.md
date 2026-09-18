# Kaggle notebook path

This is the **local preparation** for a notebook submission. No leaderboard score is claimed until a notebook is actually submitted on Kaggle.

## What to upload

- Notebook source: `notebooks/kaggle_inference.py` (paste into a new Kaggle notebook, or upload as a `.py` notebook).
- Competition data: attach **Biohub - Cell Tracking During Development**.
- Extra datasets: none required for the CPU DoG + NN path. Do not `pip install` on the scoring rerun.
- The older `notebooks/kaggle_nn_baseline.py` is the frozen greedy baseline without duplicate suppression.

## Exact steps (on Kaggle)

1. Open the competition → **Code** → **New Notebook**.
2. Confirm the competition dataset is attached.
3. **Turn Internet off** (Settings → Internet).
4. Paste `notebooks/kaggle_inference.py` (or the selected frozen config after local tuning).
5. **Save Version** → **Save & Run All**.
6. When the version finishes, **Submit** the version. The output file must be `submission.csv`.
7. Confirm the scoring rerun uses hidden test (public `test/` names are not the hidden set).

## What the notebook checks

- Discovers `test/*.zarr` by directory listing. It does not hard-code `44b6_0113de3b` or the other public names.
- Dataset stems without `.zarr`.
- CSV columns and dtypes: `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`.
- Node IDs unique per dataset; edges reference those IDs.
- Integer voxel centroids inside the volume bounds.
- Time index increases; backward edges fail validation.

## Local simulation vs Kaggle

```bat
python scripts\simulate_kaggle.py
```

That script blocks `socket.socket` in-process and runs the notebook against local `data/test`. It is labelled `local_simulation`. It is **not** a Kaggle execution: the Kaggle image, hidden test, and 12-hour worker are different.

## Runtime budget

Measured on a **local simulation** of `notebooks/kaggle_inference.py` against one complete train volume used as a fake `test/` dataset (`6bba_05db0fb1`, T=100): **0.583 s/frame**.

If the hidden test has 200 sequences of 100 frames (unverified; Data page only says it is about the same size as train, and the file listing 429’d):

```
hours ≈ 0.583 × 100 × 200 / 3600 = 3.24
with 2× safety = 6.48  (limit 12)
```

The official-eval path on the same volume was 0.79 s/frame including scoring. Using that slower number: 8.8 h with 2× margin. This is a **local simulation**, not a Kaggle worker measurement.

