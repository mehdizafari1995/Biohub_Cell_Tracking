"""Proposed organizer operating point on locked 44b6-only weights.

``learned_unet_orgop_v1`` is a specification, not a launched kernel.
It reproduces the organizer inference defaults
(``det_threshold=0.99`` + tracksdata ILP) on
``learned_unet_v1`` weights only.

Do not attach public 199-video / DeepCenter / secondary-seed weights.
Do not open ``6bba`` until a documented 44b6 keep.
Do not write ``submission.csv``.
Do not launch a Kaggle helper until an authorized cycle.
Not a competition submission.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

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
    WEIGHTS_NAME,
    HoldoutLeakError,
    assert_no_submission_csv,
    assert_tuning_only,
    refuse_non_infer_stems,
)
from .learned_split import FALLBACK_WEIGHTS_NAME, specimen_of

ORGOP_KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgop-v1"
INFER_LOCKED_TUNE = 0.6557507105877639
PRUNE_LOCKED_HOLDOUT = 0.7013778258203534
LAUNCH_AUTHORIZED = True

# Organizer ``predict_unet_transformer.py`` CLI / notebook defaults.
ORGANIZER_DET_THRESHOLD = 0.99
ORGANIZER_POOL_KERNEL_UM = 3.0
ORGANIZER_EDGE_ACTIVATION = "softmax"
ORGANIZER_EDGE_THRESHOLD = 0.5
ORGANIZER_USE_ILP = True
ORGANIZER_ILP_EDGE_WEIGHT = -1.0
ORGANIZER_ILP_APPEARANCE_WEIGHT = 0.1
ORGANIZER_ILP_DISAPPEARANCE_WEIGHT = 0.1
ORGANIZER_ILP_DIVISION_WEIGHT = 1.0

BANNED_PUBLIC_WEIGHT_MARKERS = (
    "pilkwang/biohub-tracking-support-pack",
    "pilkwang/biohub-temporal-unet3d",
    "pilkwang/biohub-deepcenter",
    "reyhanksatria/biohub-tracking-support-pack",
    "reyhanksatria/biohub-deepcenter",
    "reyhanksatria/biohub-temporalunet",
    "thibautgoldsborough/cellmot-baseline-artifacts",
    "giorgosi/biohub-divnet",
    "biohub-tracking-support-pack-50ep",
    "biohub-temporal-unet3d-seed314159",
    "biohub-deepcenter-unet3d-center-prior",
    "edge_predictor_best.pth",
    "checkpoint_last.pth",
)


def orgop_config_payload() -> dict[str, Any]:
    """First organizer-default operating point. Do not retune from holdout."""
    return {
        "label": "learned_unet_orgop_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "infer_stems": list(INFER_STEMS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "weights_source": "locked_learned_unet_v1_44b6_only",
        "weights_name": WEIGHTS_NAME,
        "fallback_weights_name": FALLBACK_WEIGHTS_NAME,
        "refuses_public_199_video_weights": True,
        "refuses_deepcenter": True,
        "refuses_secondary_seed_blend": True,
        "uses_organizer_predict_path": True,
        "det_threshold": ORGANIZER_DET_THRESHOLD,
        "det_tta": True,
        "pool_kernel_um": ORGANIZER_POOL_KERNEL_UM,
        "edge_activation": ORGANIZER_EDGE_ACTIVATION,
        "edge_threshold": ORGANIZER_EDGE_THRESHOLD,
        "linker": "tracksdata_ilp",
        "use_ilp": ORGANIZER_USE_ILP,
        "ilp_edge_weight": ORGANIZER_ILP_EDGE_WEIGHT,
        "ilp_appearance_weight": ORGANIZER_ILP_APPEARANCE_WEIGHT,
        "ilp_disappearance_weight": ORGANIZER_ILP_DISAPPEARANCE_WEIGHT,
        "ilp_division_weight": ORGANIZER_ILP_DIVISION_WEIGHT,
        "max_gap_frames": 0,
        "interpolate_gaps": False,
        "downsample": [1, 4, 4],
        "normalize": "window_minmax",
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": False,
        "kernel_slug": ORGOP_KERNEL_SLUG,
        "train_kernel_slug": TRAIN_KERNEL_SLUG,
        "infer_kernel_slug": INFER_KERNEL_SLUG,
    }


def orgop_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or orgop_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGOP_HASH = orgop_config_hash()
assert LOCKED_ORGOP_HASH == "250c5bfacd4a8198"


def assert_kernel_not_launched() -> None:
    if not LAUNCH_AUTHORIZED:
        raise SystemExit(
            "learned_unet_orgop_v1 kernel launch is not authorized; "
            "post the 0.94 audit and wait for a reviewed cycle"
        )


def assert_launch_authorized() -> None:
    if not LAUNCH_AUTHORIZED:
        raise SystemExit("learned_unet_orgop_v1 launch is not authorized")


def apply_tracksdata_ilp(
    payload: Mapping[str, Any],
    *,
    edge_weight: float = ORGANIZER_ILP_EDGE_WEIGHT,
    appearance_weight: float = ORGANIZER_ILP_APPEARANCE_WEIGHT,
    disappearance_weight: float = ORGANIZER_ILP_DISAPPEARANCE_WEIGHT,
    division_weight: float = ORGANIZER_ILP_DIVISION_WEIGHT,
) -> dict[str, Any]:
    """Run official tracksdata ILP on candidate edges. Fail closed if missing."""
    import polars as pl
    import tracksdata as td

    refuse_holdout_design([str(payload["dataset"])])
    node_id = [int(x) for x in payload.get("node_id") or []]
    t = [int(x) for x in payload.get("t") or []]
    z = [float(x) for x in payload.get("z") or []]
    y = [float(x) for x in payload.get("y") or []]
    x = [float(x) for x in payload.get("x") or []]
    raw_edges = list(payload.get("candidate_edges") or payload.get("edges") or [])
    probs = list(payload.get("edge_prob") or [])
    dists = list(payload.get("edge_dist") or [])
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    assigned = (
        graph.bulk_add_nodes(
            [{"t": ti, "z": zi, "y": yi, "x": xi} for ti, zi, yi, xi in zip(t, z, y, x, strict=True)]
        )
        if node_id
        else []
    )
    id_map = dict(zip(node_id, assigned, strict=True))
    parsed: list[tuple[int, int, float, float]] = []
    for idx, edge in enumerate(raw_edges):
        if len(edge) >= 4:
            src, tgt, prob, dist = int(edge[0]), int(edge[1]), float(edge[2]), float(edge[3])
        else:
            src, tgt = int(edge[0]), int(edge[1])
            prob = float(probs[idx]) if idx < len(probs) else 0.0
            dist = float(dists[idx]) if idx < len(dists) else 0.0
        parsed.append((src, tgt, prob, dist))
    if parsed:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        graph.bulk_add_edges(
            [
                {
                    "source_id": id_map[src],
                    "target_id": id_map[tgt],
                    "edge_prob": prob,
                    "edge_dist": dist,
                }
                for src, tgt, prob, dist in parsed
            ]
        )
    if graph.num_edges() == 0:
        out = dict(payload)
        out["edges"] = []
        out["edge_prob"] = []
        out["edge_dist"] = []
        out["solver_status"] = "ilp_skipped_no_candidates"
        out["n_candidate_edges"] = 0
        return out
    solver = td.solvers.ILPSolver(
        edge_weight=float(edge_weight) * td.EdgeAttr("edge_prob"),
        appearance_weight=float(appearance_weight),
        disappearance_weight=float(disappearance_weight),
        division_weight=float(division_weight),
    )
    solved = solver.solve(graph)
    if solved is None:
        raise RuntimeError("tracksdata ILPSolver returned no solution")
    rev = {int(v): int(k) for k, v in id_map.items()}
    kept: list[list[int]] = []
    kept_prob: list[float] = []
    kept_dist: list[float] = []
    edge_df = solved.edge_attrs(attr_keys=["edge_prob", "edge_dist"])
    src_col = "source_id" if "source_id" in edge_df.columns else edge_df.columns[0]
    tgt_col = "target_id" if "target_id" in edge_df.columns else edge_df.columns[1]
    for row in edge_df.iter_rows(named=True):
        kept.append([rev[int(row[src_col])], rev[int(row[tgt_col])]])
        kept_prob.append(float(row.get("edge_prob", 0.0)))
        kept_dist.append(float(row.get("edge_dist", 0.0)))
    out = dict(payload)
    out["candidate_edges"] = [[s, d, p, dist] for s, d, p, dist in parsed]
    out["edges"] = kept
    out["edge_prob"] = kept_prob
    out["edge_dist"] = kept_dist
    out["solver_status"] = "tracksdata_ilp"
    out["n_candidate_edges"] = len(parsed)
    return out


def tuning_decision(combined: float) -> str:
    if float(combined) > INFER_LOCKED_TUNE:
        return "keep_candidate_learned_unet_orgop_v1"
    return "reject_learned_unet_orgop_v1"


def refuse_public_weight_sources(sources: Iterable[str]) -> None:
    """Reject Pilkwang / organizer / DivNet public checkpoints for this slug."""
    cleaned = [str(s).strip().lower() for s in sources if str(s).strip()]
    hits = []
    for src in cleaned:
        for marker in BANNED_PUBLIC_WEIGHT_MARKERS:
            if marker.lower() in src:
                hits.append(src)
                break
    if hits:
        raise HoldoutLeakError(
            "learned_unet_orgop_v1 refuses public 199-video / DeepCenter / "
            "organizer-split / DivNet weights; got " + ", ".join(hits)
        )


def allowed_weight_names(names: Iterable[str]) -> None:
    allowed = {WEIGHTS_NAME, FALLBACK_WEIGHTS_NAME}
    extra = [Path(n).name for n in names if Path(n).name not in allowed]
    if extra:
        raise HoldoutLeakError(
            "learned_unet_orgop_v1 only loads locked 44b6 weights; got "
            + ", ".join(extra)
        )


def refuse_holdout_design(stems: Iterable[str]) -> None:
    refuse_non_infer_stems(stems)
    leaked = [s for s in stems if specimen_of(s) in HOLDOUT_SPECIMENS]
    if leaked:
        raise HoldoutLeakError(
            "learned_unet_orgop_v1 must not inspect 6bba while designing; got "
            + ", ".join(leaked)
        )


def public_lb_not_a_tune_target(score: float | str) -> None:
    text = str(score).strip()
    if text in {"0.553", "0.5530"} or abs(float(score) - 0.553) < 1e-9:
        raise ValueError(
            "public leaderboard 0.553 is association_v2 only and is not a "
            "learned_unet_orgop_v1 tune target"
        )


def runtime_estimate() -> dict[str, Any]:
    """Two frozen-tuning volumes on GPU T4. Not a hidden-test projection."""
    return {
        "accelerator": "gpu_t4_x2",
        "volumes": 2,
        "stems": list(INFER_STEMS),
        "expected_minutes_low": 5,
        "expected_minutes_high": 15,
        "ilp_runtime_multiplier_vs_greedy": 3.8,
        "hidden_test_not_projected": True,
        "launched": True,
    }
