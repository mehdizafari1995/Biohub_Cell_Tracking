# Biohub Cell Tracking

Independent Kaggle project for
[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).

It is not shared with other Kaggle repos. Work stays in this folder.

## What is here

- Verified competition notes: `docs/competition_spec.md`
- CPU baseline: 3D blob detection + nearest-neighbor linking
- Local evaluator for the published **edge** metric, plus a labelled division proxy
- Submission CSV export and a Kaggle notebook with the same pipeline
- Embryo-level holdout splits (no random frame splits)

Detection is required. Official data does not include detections or masks.

## Windows (Cursor)

Python 3.11+ . No Docker, no WSL, no paid LLM APIs.

```bat
cd Biohub_Cell_Tracking
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest
```

If you have not joined the competition: open the Kaggle page and click **Join Competition**, then accept the rules.

```bat
python scripts\inspect_environment.py
python scripts\list_kaggle_files.py
python scripts\download_sample.py --dataset 44b6_0db75fae --timepoints 0-9,50,99
python scripts\inspect_sample.py
python scripts\run_baseline.py --config configs\baseline.yaml
```

Keep `configs\baseline.yaml` as the frozen baseline. Measured variants use `configs\hungarian.yaml`.

## Kaggle

1. New notebook on this competition.
2. Internet off.
3. Paste `notebooks/kaggle_nn_baseline.py` or upload it.
4. Save Version, then Submit. Output file must be `submission.csv`.

The public getting-started kernel reports a public score of 0.143 for this class of method. That score was read from the kernel page; this repo has not submitted to the leaderboard.

Organizer UNet + transformer code lives at
https://github.com/royerlab/kaggle-cell-tracking-competition
and is intended for GPU Kaggle runs, not this CPU slice.

## Data

Do not commit `data/`. A small official sample is enough to inspect axes, spacing, GEFF tracks, and to run a partial-volume baseline. Full train is large (the Data page describes ~100-frame uint16 volumes).
