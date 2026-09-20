# Biohub Cell Tracking

Independent Kaggle project for
[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).

It is not shared with other Kaggle repos. Work stays in this folder.

## What is here

- Verified competition notes: `docs/competition_spec.md`
- CPU baseline: 3D blob detection + nearest-neighbor linking
- Official scorer vendored from `royerlab/kaggle-cell-tracking-competition` @ `075fc5f5` (optional extra)
- Local diagnostic evaluator kept and labelled as such
- Submission CSV export and Kaggle notebooks with dynamic `test/*.zarr` discovery
- Frozen embryo-level holdout splits (no random frame splits)

Detection is required. Official data does not include detections or masks.

## Windows (Cursor)

Python 3.11+ . No Docker, no WSL, no paid LLM APIs.

```bat
cd Biohub_Cell_Tracking
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-official.txt
python -m pip install -e .
python -m pytest
```

If you have not joined the competition: open the Kaggle page and click **Join Competition**, then accept the rules.

```bat
python scripts\inspect_environment.py
python scripts\list_kaggle_files.py
python scripts\download_sample.py --dataset 44b6_0db75fae --timepoints 0-9,50,99
python scripts\inspect_sample.py
python scripts\run_eval.py --config configs\baseline.yaml --role all_local --name baseline_official
python scripts\run_eval.py --config configs\hungarian.yaml --role all_local --name hungarian_official
```

Keep `configs\baseline.yaml` as the frozen baseline. Measured variants use `configs\hungarian.yaml` and `configs\physical_nms.yaml`. `configs\baseline_v1.yaml` is the Kaggle kernel config.

Official scores need `requirements-official.txt`. Without that extra, `run_eval.py` still records the labelled diagnostic.

## Kaggle

See `docs/kaggle_submit.md`. Internet off. Discover `test/*.zarr` at runtime. A local `scripts\simulate_kaggle.py` run is **not** a Kaggle submission.

Private kernel `mahdimzafari/biohub-baseline-v1` version 2 completed on public test (595 s, audit_ok). Private kernel `mahdimzafari/biohub-association-v2` version 3 completed on public test (594 s, audit_ok) and its downloaded CSV was independently official-scored on the public-test overlap. Neither has been submitted to the competition. No leaderboard score from this repo.

Keep `configs\baseline.yaml` as the frozen original greedy baseline. `configs\baseline_v1.yaml` is anisotropic DoG + 4 µm NMS. Association/division experiments use `configs\association_v2.yaml` and `configs\division_v1.yaml` without overwriting baseline_v1.

Organizer UNet + transformer code lives at
https://github.com/royerlab/kaggle-cell-tracking-competition
and is intended for GPU Kaggle runs, not this CPU slice.

## Data

Do not commit `data/`. A small official sample is enough to inspect axes, spacing, GEFF tracks, and to run a partial-volume baseline. Full train is large (the Data page describes ~100-frame uint16 volumes).
