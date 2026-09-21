from __future__ import annotations

import json
import time

from biohub_tracking import kaggle_api


def test_push_script_refuses_locked_helper_and_prediction_kernels():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "push_kaggle_kernel",
        Path(__file__).resolve().parents[1] / "scripts" / "push_kaggle_kernel.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "mahdimzafari/biohub-baseline-v1" in mod.FORBIDDEN_SLUGS
    assert "mahdimzafari/biohub-association-v2" in mod.FORBIDDEN_SLUGS
    assert "mahdimzafari/biohub-geff-export" in mod.FORBIDDEN_SLUGS
    assert "mahdimzafari/biohub-holdout-zarr-export" in mod.FORBIDDEN_SLUGS
    assert "mahdimzafari/biohub-tuning-zarr-export" in mod.FORBIDDEN_SLUGS
    assert "mahdimzafari/biohub-learned-unet-v1" in mod.FORBIDDEN_SLUGS
    assert "mahdimzafari/biohub-learned-unet-infer-v1" in mod.FORBIDDEN_SLUGS
    assert "mahdimzafari/biohub-learned-unet-holdout-v1" in mod.FORBIDDEN_SLUGS
    assert "mahdimzafari/biohub-learned-unet-test-v1" in mod.FORBIDDEN_SLUGS
    assert "mahdimzafari/biohub-learned-unet-gap-test-v1" in mod.FORBIDDEN_SLUGS
    assert "mahdimzafari/biohub-learned-unet-link-v1" in mod.FORBIDDEN_SLUGS
    assert "mahdimzafari/biohub-learned-unet-link-v2" in mod.FORBIDDEN_SLUGS
    assert "mahdimzafari/biohub-learned-unet-prune-test-v1" in mod.FORBIDDEN_SLUGS


def test_fetch_kernel_output_allows_helper_slugs_only():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "fetch_kernel_output",
        Path(__file__).resolve().parents[1] / "scripts" / "fetch_kernel_output.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "mahdimzafari/biohub-holdout-zarr-export" in mod.ALLOWED_SLUGS
    assert "mahdimzafari/biohub-geff-export" in mod.ALLOWED_SLUGS
    assert "mahdimzafari/biohub-tuning-zarr-export" in mod.ALLOWED_SLUGS
    assert "mahdimzafari/biohub-learned-unet-v1" in mod.ALLOWED_SLUGS
    assert "mahdimzafari/biohub-learned-unet-infer-v1" in mod.ALLOWED_SLUGS
    assert "mahdimzafari/biohub-learned-unet-holdout-v1" in mod.ALLOWED_SLUGS
    assert "mahdimzafari/biohub-learned-unet-test-v1" in mod.ALLOWED_SLUGS
    assert "mahdimzafari/biohub-learned-unet-gap-test-v1" in mod.ALLOWED_SLUGS
    assert "mahdimzafari/biohub-learned-unet-link-v1" in mod.ALLOWED_SLUGS
    assert "mahdimzafari/biohub-learned-unet-link-v2" in mod.ALLOWED_SLUGS
    assert "mahdimzafari/biohub-learned-unet-motion-v1" in mod.ALLOWED_SLUGS
    assert "mahdimzafari/biohub-learned-unet-hung-v1" in mod.ALLOWED_SLUGS
    assert "mahdimzafari/biohub-learned-unet-prune-test-v1" in mod.ALLOWED_SLUGS


def test_rate_limit_cooldown_is_bounded_and_silent_of_secrets():
    kaggle_api._cool_until = 0.0
    kaggle_api._n_rate_limited = 0
    wait = kaggle_api._note_rate_limited("60")
    assert wait == 60.0
    assert kaggle_api._n_rate_limited == 1
    assert kaggle_api._cool_until > time.time()
    capped = kaggle_api._note_rate_limited("999")
    assert capped == 120.0
    floor = kaggle_api._note_rate_limited("1")
    assert floor == 30.0
    kaggle_api._cool_until = 0.0


def test_sanitize_submission_drops_urls_and_empty_private_score():
    raw = {
        "ref": 56399003,
        "status": "complete",
        "publicScore": "0.553",
        "privateScore": "",
        "hasPrivateScore": False,
        "date": "2026-09-20T15:12:52.307Z",
        "fileName": "submission.csv",
        "errorDescription": "",
        "url": "https://www.kaggle.com/example?token=SECRET",
        "submittedBy": "mahdimzafari",
    }
    safe = kaggle_api.sanitize_submission(raw)
    assert safe["ref"] == 56399003
    assert safe["publicScore"] == "0.553"
    assert safe["private_score_returned"] is False
    assert "url" not in safe
    assert "submittedBy" not in safe
    blob = json.dumps(safe)
    assert "SECRET" not in blob
    assert "token=" not in blob
    names = kaggle_api.kernel_output_names(
        [
            {"fileName": "run_manifest.json", "url": "https://signed.example/a"},
            {"fileName": "submission.csv", "url": "https://signed.example/b?token=SECRET"},
            {"fileName": "submission.csv"},
            {"name": None},
        ]
    )
    assert names == ["run_manifest.json", "submission.csv"]
    assert "token" not in json.dumps(names)
