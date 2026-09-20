# Complete frozen-validation re-score (2026-09-20)

Unchanged locked configs on complete `docs/splits/v1.json` validation volumes `6bba_05db0fb1` + `6bba_c328f2fd` (100/100 frames each). Official scorer `075fc5f5`. Sparse GT: unmatched predictions are not node false positives.

This is **not** a Kaggle leaderboard score. The first real public LB number for locked `association_v2` is **0.553** (submission 56399003). Do not retune from either number.

| candidate | official combined | adj edge J | raw edge J | div J (TP/FP/FN) | recall | runtime s | peak MB |
|---|---|---|---|---|---|---|---|
| A `baseline_v1` | 0.537134 | 0.537134 | 0.530452 | 0.0 (0/0/5) | 0.804 | 146.7 | 1290 |
| B `association_v2` | **0.591570** | 0.591570 | 0.588813 | 0.0 (0/0/5) | 0.838 | 156.3 | 1380 |

Both match the earlier locked val rows (`val_baseline_v1` 0.537, `val_association_v2` 0.592). Operating points were not changed.

## Per-sequence official

| dataset | complete | A score | A edges TP/FP/FN | A recall | B score | B edges TP/FP/FN | B recall |
|---|---|---|---|---|---|---|---|
| `6bba_05db0fb1` (3 GT div) | 100/100 | 0.498062 | 715/269/468 | 0.770 | 0.564336 | 821/276/362 | 0.829 |
| `6bba_c328f2fd` (2 GT div) | 100/100 | 0.634281 | 365/84/135 | 0.838 | 0.660194 | 379/79/121 | 0.847 |

`6bba_05db0fb1` B matches the independently audited Kaggle public-test CSV (0.564336). That is still not the public LB.

## Error analysis (sparse GT)

Unmatched predictions are unlabeled cells or noise, not automatic FPs.

| dataset | config | missed annotated | ID switches | broken / FN edges | recovered / missed divisions | pred nodes |
|---|---|---|---|---|---|---|
| `6bba_05db0fb1` | A | 283 | 102 | 468 | 0 / 3 | 61809 |
| `6bba_05db0fb1` | B | 210 | 95 | 362 | 0 / 3 | 67788 |
| `6bba_c328f2fd` | A | 83 | 48 | 135 | 0 / 2 | 26591 |
| `6bba_c328f2fd` | B | 78 | 40 | 121 | 0 / 2 | 28548 |

Gap-closing (B vs A): interpolated nodes ≈ 5979 on `6bba_05db0fb1` and 1957 on `6bba_c328f2fd`. That recovers annotated detections and reduces switches/fragments. It does not recover any of the 5 annotated divisions (`max_children=1`).

## Runs

- `20260920T231930Z_val_baseline_v1_complete` (artifacts ignored)
- `20260920T232332Z_val_association_v2_complete` (artifacts ignored)

Volumes were downloaded into ignored `data/train/` and were not committed.
