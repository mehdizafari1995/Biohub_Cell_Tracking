"""44b6-only link-head canary for preregistered ``learned_unet_orgdet_fnn_rank_v1``.

Proves the locked transformer pair head can learn from hard negatives that
already sit inside the locked 12 µm / sqrt(dt) / max_gap=2 envelope. Does
not official-score the ranker. Does not open ``6bba``. Does not write
``submission.csv``. Does not push a kernel. Does not attach 199-video
weights. Not a competition submission.

Parent rank hash ``2748c4a2b2823c0a`` stays locked. Do not retune the
12 µm gate, ELF, FNN, or TCF.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import DEFAULT_SCALE_ZYX
from .learned_orgdet_fnn import LOCKED_ORGDET_FNN_HASH
from .learned_orgdet_fnn_oracle import (
    GAP_SCALE,
    LINKER_GATE_UM,
    LOCKED_RANK_HASH,
    MAX_GAP_FRAMES,
    RANK_LABEL,
    pair_in_candidate,
)
from .learned_split import (
    FROZEN_HOLDOUT_STEMS,
    HOLDOUT_SPECIMENS,
    TRAIN_CONFIG_HASH,
    TUNING_SPECIMEN,
    HoldoutLeakError,
    specimen_of,
)

CANARY_LABEL = "learned_unet_orgdet_fnn_rank_canary_v1"
SCALE_ZYX = DEFAULT_SCALE_ZYX
PUBLIC_LB_NOT_TUNE_TARGETS = ("0.553", "0.602", "0.612")
KERNEL_PUSH = False


def canary_config_payload() -> dict[str, Any]:
    """Immutable CPU ranker canary. Do not retune from holdout or public LB."""
    return {
        "label": CANARY_LABEL,
        "parent_label": RANK_LABEL,
        "parent_rank_hash": LOCKED_RANK_HASH,
        "parent_fnn_hash": LOCKED_ORGDET_FNN_HASH,
        "train_config_hash": TRAIN_CONFIG_HASH,
        "tuning_specimen": TUNING_SPECIMEN,
        "holdout_specimens": list(HOLDOUT_SPECIMENS),
        "frozen_holdout": list(FROZEN_HOLDOUT_STEMS),
        "candidate_gate_um": LINKER_GATE_UM,
        "candidate_max_dt": MAX_GAP_FRAMES,
        "candidate_gap_scale": GAP_SCALE,
        "hard_negatives": True,
        "n_parents": 4,
        "n_steps": 80,
        "lr": 0.001,
        "seed": 0,
        "feat_dim": 64,
        "freeze_unet": True,
        "edge_activation": "pairwise_logit",
        "min_true_minus_hard_logit": 0.5,
        "min_ranking_acc": 1.0,
        "refuses_public_199_video_weights": True,
        "writes_submission_csv": False,
        "kernel_push": False,
        "opens_6bba": False,
        "uses_199_video_weights": False,
        "scores_official_44b6": False,
        "launches_ranker_helper": False,
    }


def canary_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(payload or canary_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_CANARY_HASH = canary_config_hash()
assert LOCKED_RANK_HASH == "2748c4a2b2823c0a"
assert LOCKED_CANARY_HASH == "f7dc3d3292ce813d"
assert canary_config_payload()["parent_rank_hash"] == LOCKED_RANK_HASH
assert canary_config_payload()["parent_fnn_hash"] == "95e9300eb84ba71c"


def refuse_holdout_design(stems: Iterable[str]) -> None:
    leaked = [s for s in stems if specimen_of(str(s)) in HOLDOUT_SPECIMENS]
    if leaked:
        raise HoldoutLeakError(
            "learned_unet_orgdet_fnn_rank_canary_v1 must not inspect 6bba; got "
            + ", ".join(leaked)
        )


def public_lb_not_a_tune_target(score: float | str) -> None:
    text = str(score).strip()
    banned = {s: float(s) for s in PUBLIC_LB_NOT_TUNE_TARGETS}
    if text in banned or any(abs(float(score) - val) < 1e-9 for val in banned.values()):
        raise ValueError(
            f"public leaderboard {text} is not a "
            "learned_unet_orgdet_fnn_rank_canary_v1 tune target"
        )


def _dist_um(src_zyx: np.ndarray, tgt_zyx: np.ndarray) -> float:
    scale = np.asarray(SCALE_ZYX, dtype=np.float64)
    return float(np.linalg.norm((np.asarray(tgt_zyx) - np.asarray(src_zyx)) * scale))


def build_envelope_hard_negative_pairs(
    n_parents: int | None = None,
) -> dict[str, Any]:
    """Deterministic 44b6-refusing graph pairs inside the locked envelope.

    Each ordinary parent has one true child and one in-envelope hard
    negative. Parent 0 is a recoverable fork (two true children). One
    dt=2 pair is included. Out-of-envelope decoys are generated and
    excluded from the candidate set.
    """
    refuse_holdout_design(["44b6_canary"])
    n_parents = int(n_parents if n_parents is not None else canary_config_payload()["n_parents"])
    if n_parents < 1:
        raise ValueError("rank canary needs at least one parent")

    src_zyx: list[list[float]] = []
    tgt_zyx: list[list[float]] = []
    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    ranking: list[dict[str, Any]] = []

    def _add_tgt(zyx: tuple[float, float, float]) -> int:
        tgt_zyx.append([float(zyx[0]), float(zyx[1]), float(zyx[2])])
        return len(tgt_zyx) - 1

    def _record(si: int, ti: int, dt: int, label: int | None) -> None:
        dist = _dist_um(src_zyx[si], tgt_zyx[ti])
        inside = pair_in_candidate(dt, dist)
        row = {
            "src": int(si),
            "tgt": int(ti),
            "dt": int(dt),
            "dist_um": dist,
            "label": label,
            "in_candidate": bool(inside),
        }
        if label is None:
            if inside:
                raise RuntimeError("decoy landed inside the locked envelope")
            excluded.append(row)
            return
        if not inside:
            raise RuntimeError("true/hard pair fell outside the locked envelope")
        candidates.append(row)

    for i in range(n_parents):
        parent = (8.0, 32.0, 16.0 + float(i) * 80.0)
        src_zyx.append(list(parent))
        si = len(src_zyx) - 1
        true_a = _add_tgt((parent[0], parent[1], parent[2] + 8.0))
        hard = _add_tgt((parent[0], parent[1], parent[2] + 22.0))
        decoy = _add_tgt((parent[0], parent[1], parent[2] + 50.0))
        _record(si, true_a, 1, 1)
        _record(si, hard, 1, 0)
        _record(si, decoy, 1, None)
        ranking.append({"src": si, "true": [true_a], "hard": [hard], "dt": 1})
        if i == 0:
            true_b = _add_tgt((parent[0], parent[1] + 10.0, parent[2] + 8.0))
            _record(si, true_b, 1, 1)
            ranking[0]["true"].append(true_b)

    gap_parent = (8.0, 120.0, 16.0)
    src_zyx.append(list(gap_parent))
    gap_si = len(src_zyx) - 1
    gap_true = _add_tgt((gap_parent[0], gap_parent[1], gap_parent[2] + 20.0))
    gap_hard = _add_tgt((gap_parent[0], gap_parent[1], gap_parent[2] + 38.0))
    gap_decoy = _add_tgt((gap_parent[0], gap_parent[1], gap_parent[2] + 55.0))
    _record(gap_si, gap_true, 2, 1)
    _record(gap_si, gap_hard, 2, 0)
    _record(gap_si, gap_decoy, 2, None)
    ranking.append({"src": gap_si, "true": [gap_true], "hard": [gap_hard], "dt": 2})

    n_true = sum(1 for r in candidates if r["label"] == 1)
    n_hard = sum(1 for r in candidates if r["label"] == 0)
    two_child = any(len(r["true"]) >= 2 for r in ranking)
    return {
        "dataset": "44b6_canary",
        "src_zyx_vox": src_zyx,
        "tgt_zyx_vox": tgt_zyx,
        "candidates": candidates,
        "excluded": excluded,
        "ranking": ranking,
        "n_src": len(src_zyx),
        "n_tgt": len(tgt_zyx),
        "n_true": n_true,
        "n_hard": n_hard,
        "n_excluded": len(excluded),
        "allows_two_children": two_child,
        "candidate_gate_um": LINKER_GATE_UM,
        "candidate_max_dt": MAX_GAP_FRAMES,
        "candidate_gap_scale": GAP_SCALE,
        "opened_holdout": False,
    }


def pair_batches(pairs: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Split envelope pairs into per-dt dense batches for the pair head."""
    refuse_holdout_design([str(pairs.get("dataset") or "44b6_canary")])
    batches: list[dict[str, Any]] = []
    src_zyx = np.asarray(pairs["src_zyx_vox"], dtype=np.float64)
    tgt_zyx = np.asarray(pairs["tgt_zyx_vox"], dtype=np.float64)
    for dt in (1, 2):
        cand = [r for r in pairs["candidates"] if int(r["dt"]) == dt]
        if not cand:
            continue
        src_ids = sorted({int(r["src"]) for r in cand})
        tgt_ids = sorted({int(r["tgt"]) for r in cand})
        src_map = {old: i for i, old in enumerate(src_ids)}
        tgt_map = {old: i for i, old in enumerate(tgt_ids)}
        index = []
        labels = []
        rank_rows = []
        for row in cand:
            index.append([src_map[int(row["src"])], tgt_map[int(row["tgt"])]])
            labels.append(float(row["label"]))
        for row in pairs["ranking"]:
            if int(row["dt"]) != dt:
                continue
            si = src_map[int(row["src"])]
            for ti_true in row["true"]:
                for ti_hard in row["hard"]:
                    rank_rows.append([si, tgt_map[int(ti_true)], tgt_map[int(ti_hard)]])
        batches.append(
            {
                "dt": dt,
                "src_zyx": src_zyx[src_ids].tolist(),
                "tgt_zyx": tgt_zyx[tgt_ids].tolist(),
                "pair_index": index,
                "labels": labels,
                "ranking_index": rank_rows,
                "n_src": len(src_ids),
                "n_tgt": len(tgt_ids),
            }
        )
    return batches


def _is_finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


def evaluate_pair_scores(
    logits_by_dt: Mapping[int, np.ndarray],
    batches: Sequence[Mapping[str, Any]],
    min_margin: float,
    min_ranking_acc: float,
) -> dict[str, Any]:
    """Measure true-vs-hard ranking. ``logits_by_dt[dt]`` is [n_src, n_tgt]."""
    margins: list[float] = []
    wins = 0
    n_rank = 0
    true_logits: list[float] = []
    hard_logits: list[float] = []
    for batch in batches:
        dt = int(batch["dt"])
        logits = np.asarray(logits_by_dt[dt], dtype=np.float64)
        for si, ti_true, ti_hard in batch["ranking_index"]:
            tval = float(logits[int(si), int(ti_true)])
            hval = float(logits[int(si), int(ti_hard)])
            margin = tval - hval
            margins.append(margin)
            true_logits.append(tval)
            hard_logits.append(hval)
            n_rank += 1
            if margin > 0.0:
                wins += 1
    ranking_acc = float(wins) / float(n_rank) if n_rank else 0.0
    min_seen = min(margins) if margins else float("nan")
    return {
        "n_rank_pairs": n_rank,
        "ranking_acc": ranking_acc,
        "min_true_minus_hard_logit": min_seen,
        "mean_true_logit": float(np.mean(true_logits)) if true_logits else float("nan"),
        "mean_hard_logit": float(np.mean(hard_logits)) if hard_logits else float("nan"),
        "noncollapsed": bool(
            n_rank > 0
            and ranking_acc + 1e-12 >= float(min_ranking_acc)
            and min_seen >= float(min_margin)
        ),
    }
