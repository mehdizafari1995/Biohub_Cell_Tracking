"""Independent audit of a Kaggle worker submission.csv.

This is not a competition submission path and does not invent a leaderboard
score. Official numbers produced here are local official-scorer results on
the downloaded worker CSV versus train GEFFs (public test copies).
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .graphs import TrackGraph
from .official_eval import OfficialSampleResult, evaluate_official, summarise_official
from .submit import COLUMNS, submission_to_graphs, validate_submission

PUBLIC_TEST_DATASETS = (
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
)
FROZEN_VALIDATION = ("6bba_05db0fb1", "6bba_c328f2fd")
TUNING_EMBRYO = "44b6"
VALIDATION_EMBRYO = "6bba"
TYPICAL_VOLUME_SHAPE = (100, 64, 256, 256)
ALLOWED_OUTPUT_NAMES = frozenset({"submission.csv", "run_manifest.json"})
LOCAL_REFERENCE_SCORES = {
    "6bba_05db0fb1": {
        "candidate": "association_v2",
        "official_combined": 0.564,
        "source": "reports/abc_comparison.md",
        "note": "local official on frozen validation; not a leaderboard score",
    }
}


def embryo_id(dataset: str) -> str:
    return str(dataset).split("_", 1)[0]


def classify_dataset(dataset: str) -> str:
    """Label how a public-test stem relates to the frozen split.

    Public test copies are not a third embryo. Scoring a 44b6 public-test
    copy is not holdout evidence. Scoring 6bba_05db0fb1 is the only frozen
    validation overlap in the published sample_submission stems.
    """
    name = str(dataset)
    if name in FROZEN_VALIDATION:
        return "frozen_validation_overlap"
    if embryo_id(name) == VALIDATION_EMBRYO:
        return "validation_specimen_not_in_locked_split"
    if embryo_id(name) == TUNING_EMBRYO:
        return "tuning_specimen_public_test_copy"
    return "unknown"


def count_predicted_divisions(graph: TrackGraph) -> int:
    return int(sum(len(children) >= 2 for children in graph.out_edges().values()))


def consecutive_edge_stats(graph: TrackGraph) -> dict[str, int]:
    index = graph.id_to_index()
    n_consec = 0
    n_gap = 0
    n_backward = 0
    n_unknown = 0
    for src, dst in graph.edges:
        if int(src) not in index or int(dst) not in index:
            n_unknown += 1
            continue
        dt = int(graph.t[index[int(dst)]]) - int(graph.t[index[int(src)]])
        if dt == 1:
            n_consec += 1
        elif dt > 1:
            n_gap += 1
        else:
            n_backward += 1
    return {
        "n_consecutive_dt1": n_consec,
        "n_gap_dt_gt1": n_gap,
        "n_nonpositive_dt": n_backward,
        "n_unknown_endpoint": n_unknown,
    }


def summarize_submission(df: pd.DataFrame) -> dict[str, Any]:
    graphs = submission_to_graphs(df)
    per_dataset: list[dict[str, Any]] = []
    n_div = 0
    n_gap = 0
    for graph in graphs:
        divs = count_predicted_divisions(graph)
        n_div += divs
        edges = consecutive_edge_stats(graph)
        n_gap += int(edges["n_gap_dt_gt1"])
        coords = graph.coords_zyx()
        per_dataset.append(
            {
                "dataset": graph.dataset,
                "role": classify_dataset(graph.dataset),
                "embryo": embryo_id(graph.dataset),
                "n_nodes": graph.n_nodes,
                "n_edges": graph.n_edges,
                "n_predicted_divisions": divs,
                "coord_min_zyx": [int(coords.min(axis=0)[i]) for i in range(3)] if graph.n_nodes else None,
                "coord_max_zyx": [int(coords.max(axis=0)[i]) for i in range(3)] if graph.n_nodes else None,
                "t_min": int(graph.t.min()) if graph.n_nodes else None,
                "t_max": int(graph.t.max()) if graph.n_nodes else None,
                **edges,
            }
        )
    return {
        "n_rows": int(len(df)),
        "n_nodes": int((df["row_type"] == "node").sum()) if len(df) else 0,
        "n_edges": int((df["row_type"] == "edge").sum()) if len(df) else 0,
        "n_predicted_divisions": n_div,
        "n_gap_edges": n_gap,
        "datasets": [row["dataset"] for row in per_dataset],
        "per_dataset": per_dataset,
    }


def independent_audit(
    df: pd.DataFrame,
    expected_datasets: list[str] | tuple[str, ...] = PUBLIC_TEST_DATASETS,
    volume_shapes: dict[str, tuple[int, int, int, int]] | None = None,
    worker_manifest: dict[str, Any] | None = None,
    require_no_divisions: bool = True,
    require_consecutive_edges: bool = True,
) -> dict[str, Any]:
    """Schema, coverage, integrity, and coordinate-unit checks.

    Does not call Kaggle and does not produce a leaderboard score.
    """
    shapes = volume_shapes or {name: TYPICAL_VOLUME_SHAPE for name in expected_datasets}
    errors = validate_submission(df, required_datasets=list(expected_datasets), volume_shapes=shapes)
    warnings: list[str] = []
    summary = summarize_submission(df) if df is not None and len(df) else {
        "n_rows": 0,
        "n_nodes": 0,
        "n_edges": 0,
        "n_predicted_divisions": 0,
        "n_gap_edges": 0,
        "datasets": [],
        "per_dataset": [],
    }
    if require_no_divisions and summary["n_predicted_divisions"]:
        errors.append(
            f"association_v2 must not predict divisions; found {summary['n_predicted_divisions']}"
        )
    if require_consecutive_edges and summary["n_gap_edges"]:
        errors.append(
            f"association_v2 interpolates gaps so every edge should have dt=1; "
            f"found {summary['n_gap_edges']} gap edges"
        )
    if worker_manifest:
        for key in ("n_nodes", "n_edges", "rows"):
            expected = worker_manifest.get(key)
            actual_key = "n_rows" if key == "rows" else key
            actual = summary.get(actual_key)
            if expected is not None and actual is not None and int(expected) != int(actual):
                warnings.append(f"worker manifest {key}={expected} != independent {actual_key}={actual}")
        worker_bytes = worker_manifest.get("submission_bytes")
        if worker_bytes is not None and worker_manifest.get("_independent_bytes") is not None:
            if int(worker_bytes) != int(worker_manifest["_independent_bytes"]):
                warnings.append(
                    f"worker manifest submission_bytes={worker_bytes} != "
                    f"downloaded {worker_manifest['_independent_bytes']}"
                )
    return {
        "errors": errors,
        "warnings": warnings,
        "audit_ok": not errors,
        "summary": summary,
        "expected_datasets": list(expected_datasets),
        "volume_shapes": {k: list(v) for k, v in shapes.items()},
        "leaderboard_score": None,
        "note": (
            "Independent local audit of a downloaded Kaggle worker CSV. "
            "This is not a competition submission and reports no leaderboard score."
        ),
    }


def extract_kernel_output(archive: Path, dest_dir: Path) -> dict[str, Path]:
    """Extract only expected worker files into an ignored directory."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    with zipfile.ZipFile(archive) as zf:
        names = [info.filename for info in zf.infolist() if not info.is_dir()]
        basenames = {Path(name).name for name in names}
        unexpected = sorted(basenames - ALLOWED_OUTPUT_NAMES)
        if unexpected:
            raise ValueError(f"refusing unexpected kernel output files: {unexpected}")
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if name not in ALLOWED_OUTPUT_NAMES:
                continue
            dest = dest_dir / name
            dest.write_bytes(zf.read(info.filename))
            written[name] = dest
    return written


def score_graphs_against_geffs(
    graphs: list[TrackGraph],
    geff_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    """Official-score worker graphs where a train GEFF exists.

    Missing GEFFs are reported; they are not invented as scores.
    """
    rows: list[dict[str, Any]] = []
    for graph in graphs:
        geff = geff_paths.get(graph.dataset)
        role = classify_dataset(graph.dataset)
        if geff is None or not Path(geff).exists():
            rows.append(
                {
                    "dataset": graph.dataset,
                    "role": role,
                    "status": "missing_geff",
                    "official": None,
                    "local_reference": LOCAL_REFERENCE_SCORES.get(graph.dataset),
                    "note": "train GEFF absent; official score not invented",
                }
            )
            continue
        result = evaluate_official(graph, Path(geff))
        rows.append(
            {
                "dataset": graph.dataset,
                "role": role,
                "status": "scored",
                "official": result.as_dict(),
                "local_reference": LOCAL_REFERENCE_SCORES.get(graph.dataset),
                "delta_vs_local_reference": _delta_vs_reference(result),
                "note": (
                    "Official scorer on Kaggle worker output vs train GEFF. "
                    "Not a leaderboard score. Do not retune v1 after seeing this."
                ),
            }
        )
    return rows


def jsonable(value: Any) -> Any:
    """Replace NaN/Inf with None so committed reports stay strict JSON."""
    if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        return None
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    return value


def assemble_kaggle_output_report(
    audit: dict[str, Any],
    official_rows: list[dict[str, Any]],
    *,
    kernel_slug: str,
    kernel_version: int | None,
    config_hash: str | None,
    code_version: str | None,
    worker_runtime_sec: float | None = None,
    worker_peak_memory_mb: float | None = None,
) -> dict[str, Any]:
    scored = [row for row in official_rows if row.get("status") == "scored" and row.get("official")]
    summary = None
    if scored:
        official_objs = [OfficialSampleResult(**row["official"]) for row in scored]
        summary = summarise_official(official_objs)
    holdout = [row for row in official_rows if row.get("role") == "frozen_validation_overlap"]
    holdout_scored = [row for row in holdout if row.get("status") == "scored"]
    payload = {
        "kernel_slug": kernel_slug,
        "kernel_version": kernel_version,
        "config_hash": config_hash,
        "code_version": code_version,
        "worker_runtime_sec": worker_runtime_sec,
        "worker_peak_memory_mb": worker_peak_memory_mb,
        "audit_ok": bool(audit.get("audit_ok")),
        "audit_errors": list(audit.get("errors") or []),
        "audit_warnings": list(audit.get("warnings") or []),
        "submission_summary": audit.get("summary"),
        "official_rows": official_rows,
        "official_summary_all_scored": summary,
        "frozen_validation_overlap": holdout,
        "n_holdout_scored": len(holdout_scored),
        "public_test_overlap_scored": any(
            row.get("dataset") == "6bba_05db0fb1" and row.get("status") == "scored" for row in holdout
        ),
        "complete_frozen_split": False,
        "public_lb_score": None,
        "private_lb_score": None,
        "retune_after_holdout": False,
        "note": (
            "Independent audit of a private internet-off kernel output. "
            "Official numbers are local scorer results on the worker CSV. "
            "They are not a Kaggle leaderboard score. Competition submit was not performed."
        ),
    }
    return jsonable(payload)


def _delta_vs_reference(result: OfficialSampleResult) -> float | None:
    ref = LOCAL_REFERENCE_SCORES.get(result.dataset)
    if not ref:
        return None
    expected = float(ref["official_combined"])
    if result.score != result.score:
        return None
    return float(result.score) - expected
