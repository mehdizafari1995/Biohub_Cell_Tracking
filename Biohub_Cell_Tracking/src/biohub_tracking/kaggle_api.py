from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterator


COMPETITION = "biohub-cell-tracking-during-development"
API_ROOT = "https://www.kaggle.com/api/v1"


def _token() -> str:
    env = os.environ.get("KAGGLE_API_TOKEN")
    if env:
        return env.strip()
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        payload = json.loads(kaggle_json.read_text())
        return str(payload.get("key") or payload.get("token") or "")
    raise RuntimeError(
        "No Kaggle credentials found. Set KAGGLE_API_TOKEN or create ~/.kaggle/kaggle.json. "
        "Never print or commit these files."
    )


def _request(url: str) -> Any:
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {_token()}")
    req.add_header("User-Agent", "biohub-cell-tracking/0.1")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def iter_competition_files(competition: str = COMPETITION) -> Iterator[dict[str, Any]]:
    url = f"{API_ROOT}/competitions/data/list/{competition}"
    while url:
        data = _request(url)
        for item in data.get("files") or []:
            yield item
        token = data.get("nextPageToken") if data.get("hasNextPageToken") else None
        if token:
            url = f"{API_ROOT}/competitions/data/list/{competition}?pageToken={urllib.parse.quote(str(token))}"
        else:
            url = None


def download_file(file_name: str, dest: Path, competition: str = COMPETITION) -> bool:
    encoded = urllib.parse.quote(file_name, safe="")
    url = f"{API_ROOT}/competitions/data/download/{competition}/{encoded}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {_token()}")
    req.add_header("User-Agent", "biohub-cell-tracking/0.1")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            dest.write_bytes(resp.read())
        return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise
