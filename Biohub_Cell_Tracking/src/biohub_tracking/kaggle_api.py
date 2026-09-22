from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Iterator
from urllib.parse import parse_qs, urlparse


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


def _sleep_backoff(delay: float) -> float:
    """Exponential backoff with jitter. Never retry 429 immediately."""
    jitter = random.uniform(0.0, 0.4 * delay)
    time.sleep(delay + jitter)
    return min(delay * 2.0, 120.0)


def _request(url: str, retries: int = 8) -> Any:
    delay = 3.0
    for attempt in range(retries):
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {_token()}")
        req.add_header("User-Agent", "biohub-cell-tracking/0.1")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                delay = max(delay, 15.0)
            if exc.code in {429, 500, 502, 503} and attempt < retries - 1:
                delay = _sleep_backoff(delay)
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
            time.sleep(page_sleep + random.uniform(0.0, 0.4 * page_sleep))
        else:
            url = None


def kernel_status(slug: str) -> dict[str, Any]:
    """Return private-kernel status. Never prints the token."""
    user, name = slug.split("/", 1)
    url = (
        f"{API_ROOT}/kernels/status?"
        f"userName={urllib.parse.quote(user)}&kernelSlug={urllib.parse.quote(name)}"
    )
    data = _request(url)
    return data if isinstance(data, dict) else {"raw": data}


def kernel_output_list(slug: str) -> list[dict[str, Any]]:
    """List output files for a private kernel. Never prints the token."""
    user, name = slug.split("/", 1)
    url = (
        f"{API_ROOT}/kernels/output?"
        f"userName={urllib.parse.quote(user)}&kernelSlug={urllib.parse.quote(name)}"
    )
    data = _request(url)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        files = data.get("files") or data.get("outputFiles") or data.get("data") or []
        if isinstance(files, list):
            return [item for item in files if isinstance(item, dict)]
    return []


def kernel_output_names(files: Sequence[Mapping[str, Any]] | None) -> list[str]:
    """Return output file names only. Drops signed URLs and other secrets."""
    names: list[str] = []
    seen: set[str] = set()
    for item in files or []:
        name = item.get("fileName") or item.get("name") or item.get("file")
        if not name:
            continue
        label = str(name)
        if label in seen:
            continue
        seen.add(label)
        names.append(label)
    return names


def parse_submission_notebook_identity(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract notebook slug + scriptVersionId from relative ``/code/`` paths.

    Absolute URLs, query tokens, and author fields are ignored. A missing
    relative path leaves identity unproven.
    """
    item = dict(raw or {})
    version = item.get("scriptVersionId") or item.get("kernelVersion")
    slug = item.get("kernelSlug") or item.get("slug")
    if isinstance(slug, str):
        slug = slug.strip() or None
    else:
        slug = None
    for key in ("urlNullable", "relativeUrl"):
        text = item.get(key)
        if not isinstance(text, str):
            continue
        if text.startswith("http") or "token=" in text.lower():
            continue
        parsed = urlparse(text)
        if parsed.scheme or parsed.netloc:
            continue
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) != 3 or parts[0] != "code":
            continue
        slug = f"{parts[1]}/{parts[2]}"
        if version in (None, ""):
            vids = parse_qs(parsed.query).get("scriptVersionId") or []
            if vids:
                try:
                    version = int(vids[0])
                except ValueError:
                    version = vids[0]
        break
    if version in (None, ""):
        version = None
    elif isinstance(version, str) and version.isdigit():
        version = int(version)
    return {"kernelSlug": slug, "scriptVersionId": version}


def sanitize_submission(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep ledger-safe competition-submission fields. Never copies tokens or URLs."""
    item = dict(raw or {})
    public = item.get("publicScore")
    if public is not None:
        public = str(public)
    private_present = bool(item.get("hasPrivateScore")) and bool(str(item.get("privateScore") or "").strip())
    error = item.get("errorDescription")
    if isinstance(error, str) and not error.strip():
        error = None
    identity = parse_submission_notebook_identity(item)
    return {
        "ref": item.get("ref") or item.get("id"),
        "status": item.get("status"),
        "publicScore": public,
        "private_score_returned": private_present,
        "date": item.get("date") or item.get("submittedDate") or item.get("dateSubmitted"),
        "fileName": item.get("fileName"),
        "errorDescription": error,
        "scriptVersionId": identity["scriptVersionId"],
        "kernelSlug": identity["kernelSlug"],
    }


def sanitize_topic(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep ledger-safe discussion-topic fields. Drops authors, emails, and URLs."""
    item = dict(raw or {})
    topic_url = item.get("topicUrl") or item.get("url")
    rel = None
    if isinstance(topic_url, str) and topic_url.startswith("/"):
        rel = topic_url
    return {
        "id": item.get("id") or item.get("topicId"),
        "title": item.get("title") or item.get("topicTitle") or item.get("name"),
        "topicUrl": rel,
        "commentCount": item.get("commentCount"),
        "votes": item.get("votes") or item.get("totalVotes"),
        "postDate": item.get("postDate") or item.get("date"),
        "lastCommentPostDate": item.get("lastCommentPostDate"),
        "isSticky": bool(item.get("isSticky")),
    }


def list_competition_topics(competition: str = COMPETITION) -> dict[str, Any]:
    """Return sanitized competition discussion topics. Never prints the token.

    The public API currently returns the first hot page only (20 of many).
    Pagination query params are ignored by the host; do not invent extra pages.
    """
    url = f"{API_ROOT}/competitions/{urllib.parse.quote(competition)}/topics"
    data = _request(url)
    items: list[Any]
    total = None
    if isinstance(data, dict):
        items = data.get("topics") or data.get("items") or data.get("data") or []
        total = data.get("totalCount") or data.get("total")
    elif isinstance(data, list):
        items = data
    else:
        items = []
    topics = [sanitize_topic(item) for item in items if isinstance(item, dict)]
    return {
        "topics": topics,
        "n_returned": len(topics),
        "total_count": total,
        "first_page_only": True,
    }


def get_competition_topic(topic_id: int, competition: str = COMPETITION) -> dict[str, Any]:
    """Look up one sanitized topic by numeric id. Never prints the token.

    Direct ``/topics/{id}`` currently returns HTTP 404 HTML even for
    known hot-page ids. Treat 404 as a negative retrieval, not a body.
    """
    tid = int(topic_id)
    url = (
        f"{API_ROOT}/competitions/{urllib.parse.quote(competition)}/topics/{tid}"
    )
    delay = 3.0
    for attempt in range(8):
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {_token()}")
        req.add_header("User-Agent", "biohub-cell-tracking/0.1")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode())
            topic_raw = payload
            if isinstance(payload, dict) and isinstance(payload.get("topic"), dict):
                topic_raw = payload["topic"]
            if not isinstance(topic_raw, dict):
                topic_raw = {}
            return {
                "id": tid,
                "retrieved": True,
                "http_status": 200,
                "kind": "topic_by_id",
                "topic": sanitize_topic(topic_raw),
                "used_as_source": False,
                "body_retrieved": False,
            }
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {
                    "id": tid,
                    "retrieved": False,
                    "http_status": 404,
                    "kind": "topic_id_not_found",
                    "topic": None,
                    "used_as_source": False,
                    "body_retrieved": False,
                }
            if exc.code == 429:
                delay = max(delay, 15.0)
            if exc.code in {429, 500, 502, 503} and attempt < 7:
                delay = _sleep_backoff(delay)
                continue
            raise
    raise RuntimeError(f"topic lookup failed: {tid}")


def search_kernels(query: str, page_size: int = 5) -> dict[str, Any]:
    """Return sanitized kernel titles for a search query. Never prints the token."""
    encoded = urllib.parse.quote(str(query))
    size = max(1, min(int(page_size), 20))
    url = f"{API_ROOT}/kernels/list?search={encoded}&pageSize={size}"
    data = _request(url)
    items = data if isinstance(data, list) else []
    names: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        label = item.get("title") or item.get("ref") or item.get("kernelSlug")
        if not label:
            continue
        text = str(label)
        if text in seen:
            continue
        seen.add(text)
        names.append(text)
    return {"query": str(query), "n": len(names), "names": names}


def list_competition_submissions(competition: str = COMPETITION) -> list[dict[str, Any]]:
    """Return sanitized competition submissions. Never submits. Never prints the token."""
    url = f"{API_ROOT}/competitions/submissions/list/{urllib.parse.quote(competition)}"
    data = _request(url)
    rows: list[Any]
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("submissions") or data.get("items") or data.get("data") or []
    else:
        rows = []
    return [sanitize_submission(item) for item in rows if isinstance(item, dict)]


def download_kernel_output_zip(
    slug: str,
    dest: Path,
    min_interval_sec: float = 1.5,
) -> Path:
    """Download the kernel output archive to an ignored local path.

    Never prints credentials. Callers must keep ``dest`` out of git.
    """
    user, name = slug.split("/", 1)
    url = f"{API_ROOT}/kernels/output/download/{urllib.parse.quote(user)}/{urllib.parse.quote(name)}"
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    delay = 5.0
    for attempt in range(8):
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {_token()}")
        req.add_header("User-Agent", "biohub-cell-tracking/0.1")
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("wb") as fh:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
            tmp.replace(dest)
            time.sleep(min_interval_sec + random.uniform(0.0, 0.5))
            if dest.exists() and dest.stat().st_size > 0:
                return dest
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                delay = max(delay, 20.0)
            if exc.code in {429, 500, 502, 503} and attempt < 7:
                delay = _sleep_backoff(delay)
                continue
            raise
    raise RuntimeError(f"kernel output download failed for {slug}")


_cool_until = 0.0
_n_rate_limited = 0


def _respect_rate_limit() -> None:
    """Sleep if a previous 429 asked for a cool-down. Never prints the token."""
    remaining = _cool_until - time.time()
    if remaining > 0:
        time.sleep(remaining)


def _note_rate_limited(retry_after: str | None = None) -> float:
    """Record a 429. Returns the cool-down used (seconds)."""
    global _cool_until, _n_rate_limited
    _n_rate_limited += 1
    wait = 45.0
    if retry_after:
        try:
            wait = float(retry_after)
        except ValueError:
            wait = 45.0
    wait = min(max(wait, 30.0), 120.0)
    _cool_until = max(_cool_until, time.time() + wait)
    print(f"rate_limited n={_n_rate_limited} cool={wait:.0f}s", flush=True)
    return wait


def download_file(
    file_name: str,
    dest: Path,
    competition: str = COMPETITION,
    expected_bytes: int | None = None,
    min_interval_sec: float = 1.5,
) -> bool:
    encoded = urllib.parse.quote(file_name, safe="")
    url = f"{API_ROOT}/competitions/data/download/{competition}/{encoded}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        if expected_bytes is None or dest.stat().st_size == int(expected_bytes):
            return True
    delay = 8.0
    for attempt in range(10):
        _respect_rate_limit()
        existing = dest.stat().st_size if dest.exists() else 0
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {_token()}")
        req.add_header("User-Agent", "biohub-cell-tracking/0.1")
        if existing > 0:
            req.add_header("Range", f"bytes={existing}-")
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                status = getattr(resp, "status", 200)
                payload = resp.read()
                content_len = resp.headers.get("Content-Length")
                if status == 206 and existing > 0:
                    with dest.open("ab") as fh:
                        fh.write(payload)
                else:
                    dest.write_bytes(payload)
                actual = dest.stat().st_size
                if expected_bytes is not None and actual != int(expected_bytes):
                    if content_len is not None and existing == 0 and actual != int(content_len):
                        dest.unlink(missing_ok=True)
                        delay = _sleep_backoff(delay)
                        continue
                time.sleep(min_interval_sec + random.uniform(0.0, 0.4 * min_interval_sec))
            return dest.exists() and dest.stat().st_size > 0
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            if exc.code == 416 and dest.exists() and dest.stat().st_size > 0:
                return True
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
                delay = max(delay, _note_rate_limited(retry_after))
            if exc.code in {429, 500, 502, 503} and attempt < 9:
                delay = _sleep_backoff(delay)
                continue
            raise
    return False
