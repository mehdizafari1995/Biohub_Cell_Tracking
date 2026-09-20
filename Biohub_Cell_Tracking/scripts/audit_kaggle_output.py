#!/usr/bin/env python3
"""Independently audit a private Kaggle kernel output.

Downloads the worker archive into ``artifacts/kaggle/`` (gitignored),
validates ``submission.csv``, and official-scores public-test stems whose
train GEFFs are present. Never submits to the competition. Never writes
the CSV into git. A local official score is not a leaderboard score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from biohub_tracking.experiments import git_version, write_run_json  # noqa: E402
from biohub_tracking.kaggle_api import (  # noqa: E402
    download_kernel_output_zip,
    download_file,
    kernel_output_list,
    kernel_status,
)
from biohub_tracking.kaggle_output_audit import (  # noqa: E402
    PUBLIC_TEST_DATASETS,
    assemble_kaggle_output_report,
    extract_kernel_output,
    independent_audit,
    score_graphs_against_geffs,
)
from biohub_tracking.official_eval import official_dependency_versions, official_scorer_available  # noqa: E402
from biohub_tracking.submit import read_submission, submission_to_graphs  # noqa: E402
from resume_downloads import download_geff  # noqa: E402


def _safe_status(status: dict) -> dict:
    return {
        "status": status.get("status"),
        "failureMessage": status.get("failureMessage") if status.get("hasFailureMessage") else None,
        "hasFailureMessage": status.get("hasFailureMessage"),
    }


def _write_markdown(report: dict, dest: Path) -> None:
    holdout = report.get("frozen_validation_overlap") or []
    holdout_lines = []
    for row in holdout:
        official = row.get("official") or {}
        holdout_lines.append(
            f"- `{row.get('dataset')}` role=`{row.get('role')}` status=`{row.get('status')}` "
            f"official_combined={official.get('score')} "
            f"div={official.get('division_tp')}/{official.get('division_fp')}/{official.get('division_fn')} "
            f"delta_vs_local={row.get('delta_vs_local_reference')}"
        )
    if not holdout_lines:
        holdout_lines = ["- no frozen-validation overlap scored"]
    extra_lines = []
    for row in report.get("official_rows") or []:
        if row.get("role") == "frozen_validation_overlap":
            continue
        official = row.get("official") or {}
        extra_lines.append(
            f"- `{row.get('dataset')}` role=`{row.get('role')}` status=`{row.get('status')}` "
            f"official_combined={official.get('score')}"
        )
    dest.write_text(
        "\n".join(
            [
                "# Independent association_v2 Kaggle output audit",
                "",
                "This is a local official-scorer audit of a downloaded private kernel CSV.",
                "It is **not** a competition submission and reports **no** leaderboard score.",
                "",
                f"- Kernel: `{report.get('kernel_slug')}` version {report.get('kernel_version')}",
                f"- Config hash: `{report.get('config_hash')}`",
                f"- Code: `{report.get('code_version')}`",
                f"- Worker runtime / peak RAM (from worker manifest, if present): "
                f"{report.get('worker_runtime_sec')} s / {report.get('worker_peak_memory_mb')} MB",
                f"- Independent audit_ok: `{report.get('audit_ok')}`",
                f"- Public / private LB: `{report.get('public_lb_score')}` / `{report.get('private_lb_score')}`",
                f"- Retune after holdout: `{report.get('retune_after_holdout')}`",
                "",
                "## Frozen validation overlap",
                "",
                "Public test includes `6bba_05db0fb1`, which is one locked validation sequence.",
                "`6bba_c328f2fd` is not in public test, so the frozen split is still incomplete.",
                "The four-stem official aggregate mixes tuning-specimen copies and is not holdout.",
                "Do not retune locked association_v2 from those extra stems.",
                "",
                *holdout_lines,
                "",
                "## Other public-test stems (not holdout)",
                "",
                *(extra_lines or ["- none scored"]),
                "",
                "## Audit errors / warnings",
                "",
                f"- errors: {report.get('audit_errors')}",
                f"- warnings: {report.get('audit_warnings')}",
                "",
                report.get("note") or "",
                "",
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default="mahdimzafari/biohub-association-v2")
    parser.add_argument("--version", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "kaggle" / "association_v2_v3")
    parser.add_argument("--geff-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--datasets", nargs="*", default=list(PUBLIC_TEST_DATASETS))
    parser.add_argument("--skip-download-output", action="store_true")
    parser.add_argument("--skip-download-geffs", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "kaggle_association_v2_v3_independent_audit.md",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if "artifacts/kaggle" not in out_dir.as_posix() and "artifacts\\kaggle" not in out_dir.as_posix():
        raise SystemExit("refusing to write kernel output outside artifacts/kaggle/")

    status = {}
    output_files = []
    if not args.skip_download_output:
        status = _safe_status(kernel_status(args.slug))
        output_files = [
            {"fileName": item.get("fileName") or item.get("name"), "totalBytes": item.get("totalBytes")}
            for item in kernel_output_list(args.slug)
        ]
        archive = download_kernel_output_zip(args.slug, out_dir / "kernel_output.zip")
        extract_kernel_output(archive, out_dir)

    csv_path = out_dir / "submission.csv"
    if not csv_path.exists():
        report = {
            "kernel_slug": args.slug,
            "kernel_version": args.version,
            "blocked_reason": "submission.csv not present locally; official score not invented",
            "kernel_status": status,
            "output_files": output_files,
            "public_lb_score": None,
            "private_lb_score": None,
            "audit_ok": False,
        }
        write_run_json(out_dir / "independent_audit.json", report)
        print(json.dumps(report, indent=2, default=str))
        raise SystemExit("submission.csv missing; not inventing a score")

    df = read_submission(csv_path)
    worker_manifest = {}
    manifest_path = out_dir / "run_manifest.json"
    if manifest_path.exists():
        worker_manifest = json.loads(manifest_path.read_text())
    worker_manifest["_independent_bytes"] = csv_path.stat().st_size
    audit = independent_audit(
        df,
        expected_datasets=args.datasets,
        worker_manifest=worker_manifest,
    )

    if not args.skip_download_geffs:
        for name in args.datasets:
            download_geff("train", name, ROOT / "data")

    geff_paths = {}
    for name in args.datasets:
        candidate = args.geff_dir / f"{name}.geff"
        if candidate.exists():
            geff_paths[name] = candidate

    official_rows = []
    if official_scorer_available() and geff_paths:
        official_rows = score_graphs_against_geffs(submission_to_graphs(df), geff_paths)
    elif not official_scorer_available():
        official_rows = [
            {
                "dataset": name,
                "role": "unscored",
                "status": "official_scorer_unavailable",
                "official": None,
                "note": "official scorer dependencies missing; score not invented",
            }
            for name in args.datasets
        ]
    else:
        official_rows = [
            {
                "dataset": name,
                "role": "unscored",
                "status": "missing_geff",
                "official": None,
                "note": "train GEFF absent; official score not invented",
            }
            for name in args.datasets
        ]

    report = assemble_kaggle_output_report(
        audit,
        official_rows,
        kernel_slug=args.slug,
        kernel_version=args.version,
        config_hash=worker_manifest.get("config_hash"),
        code_version=git_version(ROOT.parent),
        worker_runtime_sec=worker_manifest.get("runtime_sec"),
        worker_peak_memory_mb=worker_manifest.get("peak_memory_mb"),
    )
    report["kernel_status"] = status
    report["output_files"] = output_files
    report["official_available"] = official_scorer_available()
    report["official_versions"] = official_dependency_versions() if official_scorer_available() else {}
    report["geffs_present"] = sorted(geff_paths)
    report["submission_bytes"] = csv_path.stat().st_size
    write_run_json(out_dir / "independent_audit.json", report)
    sanitized = {k: report[k] for k in report if k != "submission"}
    write_run_json(ROOT / "reports" / "kaggle_association_v2_v3_independent_audit.json", sanitized)
    _write_markdown(sanitized, args.report)
    print(
        json.dumps(
            {
                "audit_ok": report["audit_ok"],
                "n_holdout_scored": report["n_holdout_scored"],
                "frozen_validation_overlap": report["frozen_validation_overlap"],
                "official_summary_all_scored": report["official_summary_all_scored"],
                "geffs_present": report["geffs_present"],
                "public_lb_score": report["public_lb_score"],
                "report": str(args.report),
                "note": report["note"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
