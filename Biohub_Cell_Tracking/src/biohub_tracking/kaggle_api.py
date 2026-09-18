from __future__ import annotations

import json
import os
import time
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


def _request(url: str, retries: int = 8) -> Any:
    delay = 2.0
    for attempt in range(retries):
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {_token()}")
        req.add_header("User-Agent", "biohub-cell-tracking/0.1")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code in {429, 500, 502, 503} and attempt < retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            raise
    raise RuntimeError(f"request failed: {url}")


def iter_competition_files(
    competition: str = COMPETITION,
    checkpoint_path: str | Path | None = None,
    page_sleep: float = 0.4,
) -> Iterator[dict[str, Any]]:
    """Yield competition files. Optionally resume from a JSON checkpoint of the page token."""
    token = None
    n_seen = 0
    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            payload = json.loads(ckpt.read_text())
            token = payload.get("next_page_token")
            n_seen = int(payload.get("n_seen") or 0)
    url = f"{API_ROOT}/competitions/data/list/{competition}"
    if token:
        url = f"{url}?pageToken={urllib.parse.quote(str(token))}"
    while url:
        data = _request(url)
        items = data.get("files") or []
        for item in items:
            n_seen += 1
            yield item
        next_token = data.get("nextPageToken") if data.get("hasNextPageToken") else None
        if checkpoint_path is not None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            Path(checkpoint_path).write_text(
                json.dumps(
                    {
                        "n_seen": n_seen,
                        "next_page_token": next_token,
                        "complete": not bool(next_token),
                    }
                )
            )
        if next_token:
            url = f"{API_ROOT}/competitions/data/list/{competition}?pageToken={urllib.parse.quote(str(next_token))}"
            time.sleep(page_sleep)
        else:
            url = None


def download_file(file_name: str, dest: Path, competition: str = COMPETITION) -> bool:
    encoded = urllib.parse.quote(file_name, safe="")
    url = f"{API_ROOT}/competitions/data/download/{competition}/{encoded}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    delay = 4.0
    for attempt in range(10):
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {_token()}")
        req.add_header("User-Agent", "biohub-cell-tracking/0.1")
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                dest.write_bytes(resp.read())
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            if exc.code in {429, 500, 502, 503} and attempt < 9:
                time.sleep(delay)
                delay = min(delay * 2, 90.0)
                continue
            raise
    return False
