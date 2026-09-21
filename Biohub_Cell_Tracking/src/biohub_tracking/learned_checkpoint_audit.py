"""Catch silent checkpoint loads and empty transformer-edge graphs.

Used after rejected ``learned_unet_orgop_v1`` (0 candidate edges). Does
not retune that hash. Does not open ``6bba``. Not a competition submission.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .learned_split import HoldoutLeakError, HOLDOUT_SPECIMENS, specimen_of

TRANSFORMER_PREFIX = "transformer."


def refuse_holdout_stems(stems: Iterable[str]) -> None:
    leaked = [s for s in stems if specimen_of(str(s)) in HOLDOUT_SPECIMENS]
    if leaked:
        raise HoldoutLeakError(
            "checkpoint audit must not inspect 6bba; got " + ", ".join(leaked)
        )


def transformer_keys(keys: Iterable[str]) -> list[str]:
    return [k for k in keys if str(k).startswith(TRANSFORMER_PREFIX)]


def refuse_silent_partial_transformer_load(
    missing_keys: Iterable[str],
    unexpected_keys: Iterable[str] | None = None,
) -> None:
    """Fail closed if transformer weights were omitted from a checkpoint load."""
    del unexpected_keys
    missing = transformer_keys(missing_keys)
    if missing:
        raise RuntimeError(
            "silent partial checkpoint load omitted transformer weights: "
            + ", ".join(missing[:12])
        )


def refuse_zero_candidate_edges(
    n_candidate_edges: Sequence[int],
    *,
    n_nodes: Sequence[int] | None = None,
    label: str = "learned_unet_orgop_v1",
) -> None:
    """Fail closed when detections exist but softmax>0.5 produced no pairs."""
    cands = [int(x) for x in n_candidate_edges]
    if cands and all(c == 0 for c in cands):
        nodes = [int(x) for x in (n_nodes or [])]
        raise RuntimeError(
            f"{label} wrote 0 transformer candidate edges "
            f"(nodes={nodes or 'unknown'}); refuse blind ILP/scoring"
        )


def softmax_max_over_sources(logits: np.ndarray) -> float:
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(np.clip(shifted, -40.0, 40.0))
    probs = exp / np.clip(exp.sum(axis=0, keepdims=True), 1e-8, None)
    return float(probs.max())


def dense_softmax_rarely_exceeds_half(n_src: int, n_tgt: int, peak: float = 3.0) -> bool:
    """Organizer threshold 0.5 is unreachable for dense sources with modest peaks."""
    logits = np.zeros((int(n_src), int(n_tgt)), dtype=np.float64)
    logits[0] = float(peak)
    return softmax_max_over_sources(logits) < 0.5


def graph_candidate_counts(payloads: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    stems: list[str] = []
    n_nodes: list[int] = []
    n_cands: list[int] = []
    n_edges: list[int] = []
    for payload in payloads:
        stem = str(payload.get("dataset") or payload.get("stem") or "")
        refuse_holdout_stems([stem] if stem else [])
        stems.append(stem)
        n_nodes.append(len(payload.get("node_id") or payload.get("t") or []))
        n_cands.append(int(payload.get("n_candidate_edges") or len(payload.get("candidate_edges") or [])))
        n_edges.append(len(payload.get("edges") or []))
    return {
        "stems": stems,
        "n_nodes": n_nodes,
        "n_candidate_edges": n_cands,
        "n_edges": n_edges,
        "opened_holdout": False,
    }
