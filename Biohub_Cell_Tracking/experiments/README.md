# Experiment registry

Each measured run appends one row to `registry.csv`.

Required fields: configuration, seed, code version, validation score, runtime, and peak memory.

Keep `configs/baseline.yaml` unchanged so later experiments stay comparable.

Reported competition-style scores use `biohub_tracking.official_eval` (organizer `tracking_cellmot.metrics`). `biohub_tracking.evaluate` is a labelled diagnostic. They are not a Kaggle leaderboard submission.
