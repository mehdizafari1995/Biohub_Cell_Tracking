# Experiment registry

Each measured run appends one row to `registry.csv`.

Required fields: configuration, seed, code version, validation score, runtime, and peak memory.

Keep `configs/baseline.yaml` unchanged so later experiments stay comparable.

Reported competition-style scores use `biohub_tracking.official_eval` (organizer `tracking_cellmot.metrics`). `biohub_tracking.evaluate` is a labelled diagnostic. They are not a Kaggle leaderboard submission.

Complete frozen-validation inventory and per-event official division scoring:

```
python scripts/run_complete_validation.py --config configs/baseline.yaml
```

Do not pass `--record` unless both validation volumes were actually scored. The script never retunes `v1`.
