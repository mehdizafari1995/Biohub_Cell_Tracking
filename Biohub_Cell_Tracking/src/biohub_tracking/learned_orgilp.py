"""Spatial-candidate tracksdata ILP on locked organizer detections.

``learned_unet_orgop_v1`` (hash ``250c5bfacd4a8198``) failed because
transformer softmax>0.5 wrote zero candidate edges, so the official
tracksdata ILP skipped. This overlay keeps those locked 0.99 / pool-3.0
detections and replaces the broken softmax candidate generator with the
locked 12 µm spatial gate. Edge probability is ``1 - dist/gate``.
Organizer ILP weights are unchanged. The transformer is not read.

A deterministic preflight must see nonzero detections, nonzero spatial
candidates, a valid schema, and a nonempty solved graph. Fail closed
otherwise. Do not overwrite rejected orgop or locked slugs.

Never opens ``6bba`` during tuning. Does not write ``submission.csv``.
Not a competition submission. A local official score is not a
leaderboard number.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .graphs import TrackGraph
from .learned_infer import (  # noqa: F401
    INFER_KERNEL_SLUG,
    INFER_STEMS,
    LOCKED_INFER_HASH,
    TRAIN_CONFIG_HASH,
    TRAIN_KERNEL_SLUG,
    FROZEN_HOLDOUT_STEMS,
    HOLDOUT_SPECIMENS,
    SCALE_ZYX,
    TUNING_SPECIMEN,
    HoldoutLeakError,
    assert_no_submission_csv,
    graph_from_payload,
    load_graph_json,
    refuse_non_infer_stems,
    write_graph_json,
)
from .learned_orgdet import LOCKED_ORGDET_HASH
from .learned_orgdet_gap import LOCKED_ORGDET_GAP_HASH, ORGDET_LOCKED_TUNE
from .learned_orgop import (  # noqa: F401
    LOCKED_ORGOP_HASH as REJECTED_ORGOP_HASH,
    ORGANIZER_DET_THRESHOLD,
    ORGANIZER_ILP_APPEARANCE_WEIGHT,
    ORGANIZER_ILP_DISAPPEARANCE_WEIGHT,
    ORGANIZER_ILP_DIVISION_WEIGHT,
    ORGANIZER_ILP_EDGE_WEIGHT,
    ORGANIZER_POOL_KERNEL_UM,
    ORGANIZER_USE_ILP,
    ORGOP_KERNEL_SLUG,
    apply_tracksdata_ilp,
)
from .learned_split import specimen_of

ORGILP_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgilp-v1"
ORGDET_GAP_LOCKED_TUNE = 0.6872212709981039
MAX_LINK_UM = 12.0
MAX_NEIGHBORS_PER_SOURCE = 8
MAX_CANDIDATE_EDGES = 2_000_000
EDGE_PROB_FLOOR = 1e-6

LOCKED_GATES = {
    "candidate_source": "spatial_gate_12um",
    "edge_activation": "spatial_prob_1_minus_dist_over_gate",
    "edge_threshold": 0.0,
    "gate_um": MAX_LINK_UM,
    "max_gap_frames": 0,
    "max_neighbors_per_source": MAX_NEIGHBORS_PER_SOURCE,
    "max_candidate_edges": MAX_CANDIDATE_EDGES,
    "assignment": "tracksdata_ilp",
    "interpolate_gaps": False,
    "uses_transformer": False,
}


class PreflightError(RuntimeError):
    """Fail-closed preflight: zero detections, zero candidates, or empty graph."""


def orgilp_config_payload() -> dict[str, Any]:
    """First spatial-ILP operating point. Do not retune from holdout."""
    return {
        "label": "learned_unet_orgilp_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "orgop_label": "learned_unet_orgop_v1",
        "orgdet_label": "learned_unet_orgdet_v1",
        "orgdet_gap_label": "learned_unet_orgdet_gap_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "rejected_orgop_hash": REJECTED_ORGOP_HASH,
        "orgdet_config_hash": LOCKED_ORGDET_HASH,
        "orgdet_gap_config_hash": LOCKED_ORGDET_GAP_HASH,
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "infer_stems": list(INFER_STEMS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "source_graphs": "locked_learned_unet_orgop_graphs",
        "retunes_detection": False,
        "uses_intensity": False,
        "uses_transformer": False,
        "uses_velocity": False,
        "uses_ilp": True,
        "candidate_source": LOCKED_GATES["candidate_source"],
        "edge_activation": LOCKED_GATES["edge_activation"],
        "edge_threshold": LOCKED_GATES["edge_threshold"],
        "linker": "tracksdata_ilp",
        "gate_um": LOCKED_GATES["gate_um"],
        "max_gap_frames": LOCKED_GATES["max_gap_frames"],
        "max_neighbors_per_source": LOCKED_GATES["max_neighbors_per_source"],
        "max_candidate_edges": LOCKED_GATES["max_candidate_edges"],
        "assignment": LOCKED_GATES["assignment"],
        "interpolate_gaps": LOCKED_GATES["interpolate_gaps"],
        "use_ilp": ORGANIZER_USE_ILP,
        "ilp_edge_weight": ORGANIZER_ILP_EDGE_WEIGHT,
        "ilp_appearance_weight": ORGANIZER_ILP_APPEARANCE_WEIGHT,
        "ilp_disappearance_weight": ORGANIZER_ILP_DISAPPEARANCE_WEIGHT,
        "ilp_division_weight": ORGANIZER_ILP_DIVISION_WEIGHT,
        "preflight_required": True,
        "preflight_min_nodes": 1,
        "preflight_min_candidate_edges": 1,
        "preflight_min_solved_edges": 1,
        "det_threshold": ORGANIZER_DET_THRESHOLD,
        "det_tta": True,
        "pool_kernel_um": ORGANIZER_POOL_KERNEL_UM,
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": False,
        "rejection_reference": "learned_unet_orgdet_gap_v1",
        "rejection_threshold": ORGDET_GAP_LOCKED_TUNE,
        "kernel_slug": ORGILP_KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
        "orgop_kernel_slug": ORGOP_KERNEL_SLUG,
    }


def orgilp_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or orgilp_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGILP_HASH = orgilp_config_hash()
assert LOCKED_ORGILP_HASH == "dbc6134cddf2048d"


def _clean_stem(raw: str) -> str:
    stem = Path(raw).name
    for suffix in (".zarr", ".geff", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def refuse_non_tuning_stems(stems: Iterable[str]) -> None:
    for raw in stems:
        stem = _clean_stem(str(raw))
        if specimen_of(stem) != TUNING_SPECIMEN or stem not in INFER_STEMS:
            raise HoldoutLeakError(f"learned_unet_orgilp_v1 tuning refuses stem {stem}")


def refuse_public_lb_tune_targets(score: float | str) -> None:
    text = str(score).strip()
    if text in {"0.553", "0.612"} or abs(float(score) - 0.553) < 1e-9 or abs(float(score) - 0.612) < 1e-9:
        raise ValueError(
            "public leaderboard 0.553 / 0.612 are not learned_unet_orgilp_v1 tune targets"
        )


def spatial_prob(dist_um: float, gate_um: float = MAX_LINK_UM) -> float:
    """Closer pairs score higher. Floor keeps ILP from treating gate-edge as zero."""
    gate = float(gate_um)
    if gate <= 0:
        raise PreflightError("spatial gate must be positive")
    return float(max(EDGE_PROB_FLOOR, 1.0 - float(dist_um) / gate))


def spatial_gate_candidates(
    node_id: Sequence[int],
    t: Sequence[int],
    z: Sequence[int],
    y: Sequence[int],
    x: Sequence[int],
    *,
    gate_um: float = MAX_LINK_UM,
    max_neighbors_per_source: int = MAX_NEIGHBORS_PER_SOURCE,
    scale_zyx: Sequence[float] = SCALE_ZYX,
) -> list[tuple[int, int, float, float]]:
    """All dt=1 pairs inside the locked 12 µm gate, capped per source.

    Returns ``(src_id, tgt_id, prob, dist_um)``. Does not use transformer
    logits or a 0.5 cutoff. That is the orgop zero-edge repair.
    """
    from scipy.spatial import cKDTree

    nids = [int(v) for v in node_id]
    ts = [int(v) for v in t]
    zs = [float(v) for v in z]
    ys = [float(v) for v in y]
    xs = [float(v) for v in x]
    if not nids:
        return []
    by_t: dict[int, list[int]] = defaultdict(list)
    for i, tk in enumerate(ts):
        by_t[tk].append(i)
    times = sorted(by_t)
    scale = np.asarray(scale_zyx, dtype=np.float64)
    gate = float(gate_um)
    k_cap = max(1, int(max_neighbors_per_source))
    out: list[tuple[int, int, float, float]] = []
    for prev, cur in zip(times, times[1:]):
        if cur - prev != 1:
            continue
        idx_a = by_t[prev]
        idx_b = by_t[cur]
        if not idx_a or not idx_b:
            continue
        pa = np.stack(
            [
                np.asarray([zs[i] for i in idx_a], dtype=np.float64),
                np.asarray([ys[i] for i in idx_a], dtype=np.float64),
                np.asarray([xs[i] for i in idx_a], dtype=np.float64),
            ],
            axis=1,
        ) * scale
        pb = np.stack(
            [
                np.asarray([zs[i] for i in idx_b], dtype=np.float64),
                np.asarray([ys[i] for i in idx_b], dtype=np.float64),
                np.asarray([xs[i] for i in idx_b], dtype=np.float64),
            ],
            axis=1,
        ) * scale
        tree = cKDTree(pb)
        k = min(k_cap, pb.shape[0])
        dists, idxs = tree.query(pa, k=k, distance_upper_bound=gate)
        dists = np.atleast_2d(np.asarray(dists, dtype=np.float64))
        idxs = np.atleast_2d(np.asarray(idxs, dtype=np.int64))
        if k == 1:
            dists = dists.reshape(pa.shape[0], 1)
            idxs = idxs.reshape(pa.shape[0], 1)
        for row, src_i in enumerate(idx_a):
            for dist_um, tgt_j in zip(dists[row], idxs[row]):
                if not np.isfinite(dist_um) or float(dist_um) > gate:
                    continue
                if int(tgt_j) < 0 or int(tgt_j) >= len(idx_b):
                    continue
                dst_i = idx_b[int(tgt_j)]
                out.append(
                    (
                        nids[src_i],
                        nids[dst_i],
                        spatial_prob(float(dist_um), gate),
                        float(dist_um),
                    )
                )
        if len(out) > MAX_CANDIDATE_EDGES:
            raise PreflightError(
                f"spatial candidates exceeded locked budget {MAX_CANDIDATE_EDGES}"
            )
    return out


def _schema_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ("dataset", "node_id", "t", "z", "y", "x")
    for key in required:
        if key not in payload:
            errors.append(f"missing_{key}")
    n_nodes = len(payload.get("node_id") or [])
    for key in ("t", "z", "y", "x"):
        if key in payload and len(payload.get(key) or []) != n_nodes:
            errors.append(f"length_mismatch_{key}")
    if n_nodes < 1:
        errors.append("zero_detections")
    dataset = str(payload.get("dataset") or "")
    if dataset and specimen_of(dataset) != TUNING_SPECIMEN:
        errors.append("holdout_stem")
    return errors


def preflight_orgilp(
    payload: Mapping[str, Any],
    *,
    candidates: Sequence[tuple[int, int, float, float]] | None = None,
    solved: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed when detections, candidates, schema, or solved graph are empty."""
    errors = _schema_errors(payload)
    cands = list(candidates) if candidates is not None else []
    if candidates is None and not errors:
        try:
            cands = spatial_gate_candidates(
                payload.get("node_id") or [],
                payload.get("t") or [],
                payload.get("z") or [],
                payload.get("y") or [],
                payload.get("x") or [],
            )
        except PreflightError as exc:
            errors.append(str(exc))
    if len(cands) < 1:
        errors.append("zero_candidate_edges")
    if solved is not None:
        status = str(solved.get("solver_status") or "")
        n_edges = len(solved.get("edges") or [])
        if status == "ilp_skipped_no_candidates":
            errors.append("ilp_skipped_no_candidates")
        if n_edges < 1:
            errors.append("zero_solved_edges")
        if status and status != "tracksdata_ilp":
            errors.append(f"solver_status_{status}")
    report = {
        "status": "preflight_ok" if not errors else "preflight_failed",
        "n_nodes": len(payload.get("node_id") or []),
        "n_candidate_edges": len(cands),
        "n_solved_edges": len((solved or {}).get("edges") or []),
        "solver_status": (solved or {}).get("solver_status"),
        "errors": errors,
    }
    if errors:
        raise PreflightError("learned_unet_orgilp_v1 preflight failed: " + ", ".join(errors))
    return report


def apply_orgilp_payload(
    payload: Mapping[str, Any],
    *,
    refuse_holdout: bool = True,
) -> dict[str, Any]:
    """Build spatial candidates, preflight, then run official tracksdata ILP."""
    if orgilp_config_hash() != LOCKED_ORGILP_HASH:
        raise RuntimeError("learned_unet_orgilp_v1 hash drifted; refusing to retune")
    if refuse_holdout:
        refuse_non_tuning_stems([str(payload["dataset"])])
        refuse_non_infer_stems([str(payload["dataset"])])
    cands = spatial_gate_candidates(
        payload.get("node_id") or [],
        payload.get("t") or [],
        payload.get("z") or [],
        payload.get("y") or [],
        payload.get("x") or [],
    )
    preflight_orgilp(payload, candidates=cands)
    ilp_payload = {
        "dataset": payload["dataset"],
        "node_id": [int(x) for x in payload.get("node_id") or []],
        "t": [int(x) for x in payload.get("t") or []],
        "z": [float(x) for x in payload.get("z") or []],
        "y": [float(x) for x in payload.get("y") or []],
        "x": [float(x) for x in payload.get("x") or []],
        "candidate_edges": [list(row) for row in cands],
        "edges": [[int(s), int(d)] for s, d, _p, _dist in cands],
        "edge_prob": [float(p) for _s, _d, p, _dist in cands],
        "edge_dist": [float(dist) for _s, _d, _p, dist in cands],
    }
    solved = apply_tracksdata_ilp(
        ilp_payload,
        edge_weight=ORGANIZER_ILP_EDGE_WEIGHT,
        appearance_weight=ORGANIZER_ILP_APPEARANCE_WEIGHT,
        disappearance_weight=ORGANIZER_ILP_DISAPPEARANCE_WEIGHT,
        division_weight=ORGANIZER_ILP_DIVISION_WEIGHT,
        refuse_holdout=refuse_holdout,
    )
    preflight_orgilp(payload, candidates=cands, solved=solved)
    out = dict(payload)
    out["candidate_edges"] = [list(row) for row in cands]
    out["edges"] = [[int(s), int(d)] for s, d in solved.get("edges") or []]
    out["edge_prob"] = list(solved.get("edge_prob") or [])
    out["edge_dist"] = list(solved.get("edge_dist") or [])
    out["n_candidate_edges"] = len(cands)
    out["solver_status"] = solved.get("solver_status")
    out["metadata"] = dict(payload.get("metadata") or {})
    out["metadata"].update(
        {
            "overlay": "learned_unet_orgilp_v1",
            "config_hash": LOCKED_ORGILP_HASH,
            "rejected_orgop_hash": REJECTED_ORGOP_HASH,
            "parent_orgdet_hash": LOCKED_ORGDET_HASH,
            "parent_orgdet_gap_hash": LOCKED_ORGDET_GAP_HASH,
            "uses_transformer": False,
            "uses_ilp": True,
            "retunes_detection": False,
            "n_candidate_edges": len(cands),
            "n_ilp_edges": len(out["edges"]),
            "solver_status": solved.get("solver_status"),
        }
    )
    return out


def apply_orgilp(graph: TrackGraph) -> TrackGraph:
    """Replace edges with spatial-candidate tracksdata ILP. Keep detections."""
    payload = {
        "dataset": graph.dataset,
        "node_id": [int(x) for x in graph.node_id],
        "t": [int(x) for x in graph.t],
        "z": [int(x) for x in graph.z],
        "y": [int(x) for x in graph.y],
        "x": [int(x) for x in graph.x],
        "edges": [[int(s), int(d)] for s, d in np.asarray(graph.edges).reshape(-1, 2)]
        if graph.n_edges
        else [],
        "metadata": dict(graph.metadata),
    }
    solved = apply_orgilp_payload(payload)
    edges = solved.get("edges") or []
    out = TrackGraph(
        dataset=graph.dataset,
        node_id=np.asarray(graph.node_id, dtype=np.int64),
        t=np.asarray(graph.t, dtype=np.int64),
        z=np.asarray(graph.z, dtype=np.int64),
        y=np.asarray(graph.y, dtype=np.int64),
        x=np.asarray(graph.x, dtype=np.int64),
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2) if edges else np.zeros((0, 2), dtype=np.int64),
        estimated_n_nodes=graph.estimated_n_nodes,
        metadata=dict(solved.get("metadata") or {}),
    )
    if out.n_nodes != graph.n_nodes:
        raise RuntimeError("orgilp changed detection count; refusing")
    return out


def overlay_summary(before: TrackGraph, after: TrackGraph) -> dict[str, Any]:
    before_edges = (
        {tuple(map(int, row)) for row in np.asarray(before.edges).reshape(-1, 2)} if before.n_edges else set()
    )
    after_edges = {tuple(map(int, row)) for row in np.asarray(after.edges).reshape(-1, 2)} if after.n_edges else set()
    return {
        "dataset": after.dataset,
        "n_nodes_before": before.n_nodes,
        "n_nodes_after": after.n_nodes,
        "n_edges_before": before.n_edges,
        "n_edges_after": after.n_edges,
        "n_shared_edges": len(before_edges & after_edges),
        "n_replaced_edges": len(before_edges - after_edges),
        "n_new_edges": len(after_edges - before_edges),
        "n_candidate_edges": int(after.metadata.get("n_candidate_edges") or 0),
        "n_ilp_edges": int(after.metadata.get("n_ilp_edges") or after.n_edges),
        "solver_status": after.metadata.get("solver_status"),
        "detection_counts_match": before.n_nodes == after.n_nodes,
        "overlay": "learned_unet_orgilp_v1",
        "config_hash": LOCKED_ORGILP_HASH,
    }


def load_role_graphs(graph_dir: str | Path, stems: Sequence[str]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for stem in stems:
        path = Path(graph_dir) / f"{stem}.json"
        if not path.exists():
            nested = Path(graph_dir) / "graphs" / f"{stem}.json"
            path = nested if nested.exists() else path
        if not path.exists():
            raise FileNotFoundError(f"missing orgop/orgdet graph {path}; do not invent a score")
        payload = load_graph_json(path)
        if str(payload.get("dataset")) != stem:
            raise ValueError(f"{path} dataset {payload.get('dataset')!r} != {stem}")
        payloads.append(payload)
    return payloads


def refuse_existing_holdout_score(path: str | Path) -> None:
    dest = Path(path)
    if not dest.exists():
        return
    payload = json.loads(dest.read_text())
    if payload.get("status") == "holdout_scored" and payload.get("learned_combined") is not None:
        raise SystemExit(
            f"refusing to score frozen holdout twice; existing decision={payload.get('decision')}"
        )


def require_tuning_keep(path: str | Path) -> dict[str, Any]:
    dest = Path(path)
    if not dest.exists():
        raise SystemExit("refusing holdout; tuning report missing. Score 44b6 first.")
    payload = json.loads(dest.read_text())
    decision = str(payload.get("decision") or "")
    if not decision.startswith("keep_"):
        raise SystemExit(f"refusing holdout; tuning decision is {decision}")
    return payload


def tuning_decision(learned_combined: float) -> str:
    """Keep only if official 44b6 combined beats locked orgdet-gap."""
    if float(learned_combined) > ORGDET_GAP_LOCKED_TUNE:
        return "keep_candidate_learned_unet_orgilp_v1"
    return "reject_learned_unet_orgilp_v1"


def holdout_decision(learned_combined: float, champion_holdout: float = 0.722805) -> str:
    if float(learned_combined) > float(champion_holdout):
        return "keep_candidate_learned_unet_orgilp_v1"
    return "reject_learned_unet_orgilp_v1"


def runtime_estimate(
    n_volumes: int = 2,
    limit_hours: float = 12.0,
    peak_memory_mb: float = 2500.0,
) -> dict[str, Any]:
    """CPU spatial ILP on locked orgop detections. No new UNet pass."""
    n = int(n_volumes)
    sec = 180.0 * n
    return {
        "n_volumes": n,
        "estimate_sec": sec,
        "estimate_hours": sec / 3600.0,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_budget": float(peak_memory_mb),
        "source": (
            "CPU spatial 12 µm candidates + official tracksdata ILP on locked "
            "orgop detections; no transformer; no new UNet pass"
        ),
    }
