#!/usr/bin/env python3
"""Push a packaged private kernel. Never submits to the competition."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from biohub_tracking.kaggle_api import API_ROOT, _token  # noqa: E402
from make_kaggle_kernel import KERNELS  # noqa: E402

FORBIDDEN_SLUGS = {"mahdimzafari/biohub-baseline-v1"}


def _load_notebook_text(kernel_name: str) -> tuple[str, dict]:
    spec = KERNELS[kernel_name]
    kernel_dir = ROOT / spec["kernel_dir"]
    meta = json.loads((kernel_dir / "kernel-metadata.json").read_text())
    notebook_path = kernel_dir / spec["code_file"]
    nb = json.loads(notebook_path.read_text())
    if meta.get("id") in FORBIDDEN_SLUGS:
        raise SystemExit(f"refusing to overwrite locked kernel {meta.get('id')}")
    if meta.get("is_private") is not True:
        raise SystemExit("refusing to push a non-private kernel")
    if meta.get("enable_internet") is not False:
        raise SystemExit("refusing to push a kernel with internet enabled")
    if "biohub-cell-tracking-during-development" not in (meta.get("competition_sources") or []):
        raise SystemExit("kernel must attach the competition dataset")
    for cell in nb.get("cells") or []:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
    return json.dumps(nb), meta


def push_kernel(kernel_name: str) -> dict:
    if kernel_name not in KERNELS:
        raise SystemExit(f"unknown kernel {kernel_name!r}")
    text, meta = _load_notebook_text(kernel_name)
    payload = {
        "slug": meta["id"],
        "newTitle": meta["title"],
        "text": text,
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": True,
        "enableGpu": False,
        "enableTpu": False,
        "enableInternet": False,
        "competitionDataSources": ["biohub-cell-tracking-during-development"],
        "datasetDataSources": [],
        "kernelDataSources": [],
        "modelDataSources": [],
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{API_ROOT}/kernels/push",
        data=body,
        method="POST",
    )
    req.add_header("Authorization", f"Bearer {_token()}")
    req.add_header("User-Agent", "biohub-cell-tracking/0.1")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise SystemExit(f"kernels/push HTTP {exc.code}: {detail}") from exc


def kernel_status(slug: str) -> dict:
    user, name = slug.split("/", 1)
    url = f"{API_ROOT}/kernels/status?userName={user}&kernelSlug={name}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {_token()}")
    req.add_header("User-Agent", "biohub-cell-tracking/0.1")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", default="association_v2", nargs="?", help="Packaged kernel name")
    parser.add_argument("--status-only", action="store_true", help="Only print kernel status")
    args = parser.parse_args()
    spec = KERNELS[args.kernel]
    slug = spec["kernel_id"]
    if slug in FORBIDDEN_SLUGS:
        raise SystemExit(f"refusing to overwrite locked kernel {slug}")
    if args.status_only:
        status = kernel_status(slug)
        print(json.dumps({k: status.get(k) for k in ("status", "failureMessage", "hasFailureMessage")}, indent=2))
        return
    result = push_kernel(args.kernel)
    safe = {
        "ref": result.get("ref") or result.get("slug") or slug,
        "versionNumber": result.get("versionNumber") or result.get("currentVersionNumber"),
        "error": result.get("error") or result.get("message"),
        "invalidTags": result.get("invalidTags"),
        "keys": sorted(result.keys()),
    }
    print(json.dumps(safe, indent=2, default=str))
    print("NOTE: this is a private kernel push, not a competition submission.")


if __name__ == "__main__":
    main()
