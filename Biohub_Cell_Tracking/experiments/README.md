# Experiment registry

Each measured run appends one row to `registry.csv`.

Required fields: configuration, seed, code version, validation score, runtime, and peak memory.

Keep `configs/baseline.yaml` unchanged so later experiments stay comparable.

Local scores use the in-repo reimplementation of the published edge metric plus a simpler division proxy. They are not a Kaggle leaderboard submission.
