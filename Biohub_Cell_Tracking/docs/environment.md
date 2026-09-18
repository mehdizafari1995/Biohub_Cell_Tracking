# Environment

Inspected before installing project dependencies or downloading volumes.

## This Cloud Agent machine (2026-09-18)

| Item | Value |
|---|---|
| OS | Linux (Ubuntu) |
| Python | 3.12.3 (`/usr/bin/python3`) |
| CPU | 4 |
| RAM | 16 GiB total; about 6.5 GiB available at inspection |
| Disk | 252 G, ~229 G free |
| GPU | none (`nvidia-smi` missing) |
| CUDA | not installed |
| Docker / WSL | not required |

Project packages live in `Biohub_Cell_Tracking/.venv`. Exact versions are in `requirements.lock`.

Kaggle access used `KAGGLE_API_TOKEN` from the environment via Bearer auth. Credentials are never printed or committed.

## Windows development (Cursor)

This repo is ordinary Python. On Windows:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest
```

Do not use Docker or WSL. GPU is optional; the baseline is CPU-only.

Place Kaggle credentials in `%USERPROFILE%\.kaggle\kaggle.json` or set `KAGGLE_API_TOKEN`. Never commit that file.

## Kaggle execution

Notebook submissions run with internet disabled, 12 hour CPU or GPU limit. Bundle any extra wheels or weights as a public Kaggle dataset. Do not `pip install` during a scoring rerun unless the package is already on the image.

Paid LLM APIs are not used.
