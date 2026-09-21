#!/usr/bin/python3
"""Diagnose rejected learned_unet_orgop_v1 zero-edge graphs.

Reads ignored train checkpoints when present and locked orgop reports.
Does not retune orgop. Does not open 6bba. Does not submit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from biohub_tracking.experiments import git_version, write_run_json  # noqa: E402
from biohub_tracking.learned_checkpoint_audit import (  # noqa: E402
    dense_softmax_rarely_exceeds_half,
    graph_candidate_counts,
    refuse_silent_partial_transformer_load,
    transformer_keys,
)
from biohub_tracking.learned_edge_canary import runtime_projection  # noqa: E402
from biohub_tracking.learned_split import TRAIN_CONFIG_HASH  # noqa: E402

ORGOP_HASH = "250c5bfacd4a8198"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _group_stats(state: dict[str, Any], pred) -> dict[str, float] | None:
    tensors = [v.detach().float() for k, v in state.items() if pred(k) and v.is_floating_point()]
    if not tensors:
        return None
    n = sum(t.numel() for t in tensors)
    l2 = math.sqrt(sum(float(t.square().sum()) for t in tensors))
    return {"n_params": n, "l2": round(l2, 6), "rms": round(l2 / math.sqrt(n), 6)}


def _rel(trained: dict[str, Any], init: dict[str, Any], pred) -> dict[str, float]:
    num = den = 0.0
    n = 0
    for key, value in trained.items():
        if not pred(key) or not value.is_floating_point():
            continue
        delta = value.float() - init[key].float()
        num += float(delta.square().sum())
        den += float(init[key].float().square().sum())
        n += value.numel()
    return {
        "n": n,
        "delta_l2": round(math.sqrt(num), 6),
        "rel": round(math.sqrt(num) / max(math.sqrt(den), 1e-12), 6),
    }


def inspect_checkpoints(train_dir: Path) -> dict[str, Any] | None:
    best_path = train_dir / "learned_unet_v1_best.pt"
    last_path = train_dir / "learned_unet_v1_last.pt"
    if not best_path.exists() or not last_path.exists():
        return None
    import torch

    from kaggle_learned_unet_v1 import _build_models

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    if best.get("config_hash") != TRAIN_CONFIG_HASH:
        raise SystemExit("train checkpoint hash is not locked learned_unet_v1")
    sd_best = best["state_dict"]
    sd_last = last["state_dict"]
    equal = all(torch.equal(sd_best[k], sd_last[k]) for k in sd_best)
    model_cls, _detection_loss, _edge_loss = _build_models(best["config"])
    model = model_cls()
    incompat = model.load_state_dict(sd_best, strict=False)
    refuse_silent_partial_transformer_load(incompat.missing_keys, incompat.unexpected_keys)
    model.load_state_dict(sd_best, strict=True)
    torch.manual_seed(0)
    init_sd = model_cls().state_dict()
    groups = {
        "unet": lambda k: k.startswith("unet."),
        "detect_head": lambda k: k.startswith("detect_head."),
        "transformer": lambda k: k.startswith("transformer."),
        "transformer.pair_mlp": lambda k: k.startswith("transformer.pair_mlp"),
    }
    model.eval()
    torch.manual_seed(1)
    feat_t = torch.randn(1, 400, 64)
    feat_t1 = torch.randn(1, 80, 64)
    coords_t = torch.rand(1, 400, 3) * 32
    coords_t1 = torch.rand(1, 80, 3) * 32
    with torch.no_grad():
        logits = model.transformer(feat_t, feat_t1, coords_t, coords_t1)[0]
        probs = torch.softmax(logits, dim=0)
    return {
        "best_sha16": _file_hash(best_path),
        "last_sha16": _file_hash(last_path),
        "best_equals_last": equal,
        "config_hash": best.get("config_hash"),
        "n_state_tensors": len(sd_best),
        "n_transformer_keys": len(transformer_keys(sd_best)),
        "missing_keys": list(incompat.missing_keys),
        "unexpected_keys": list(incompat.unexpected_keys),
        "strict_load": True,
        "splits_test": (best.get("splits") or {}).get("test"),
        "holdout_in_splits": [
            s
            for s in list((best.get("splits") or {}).get("train") or [])
            + list((best.get("splits") or {}).get("test") or [])
            if str(s).startswith("6bba")
        ],
        "group_rms_ckpt": {name: _group_stats(sd_best, pred) for name, pred in groups.items()},
        "group_rel_to_seed0_init": {name: _rel(sd_best, init_sd, pred) for name, pred in groups.items()},
        "random_400x80": {
            "logit_mean": float(logits.mean()),
            "logit_std": float(logits.std()),
            "prob_max": float(probs.max()),
            "n_gt_half": int((probs > 0.5).sum()),
        },
        "weights_path_used_by_orgop": "learned_unet_v1_last.pt",
        "same_as_best": equal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "learned_unet_orgop_zero_edge_diagnosis.json",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=ROOT / "artifacts" / "kaggle" / "learned_unet_v1",
    )
    parser.add_argument(
        "--orgop-graphs",
        type=Path,
        default=ROOT / "artifacts" / "kaggle" / "learned_unet_orgop_v1" / "graphs",
    )
    args = parser.parse_args()
    tune = json.loads((ROOT / "reports" / "learned_unet_orgop_v1_tune.json").read_text())
    payloads = []
    if args.orgop_graphs.exists():
        for path in sorted(args.orgop_graphs.glob("44b6_*.json")):
            payloads.append(json.loads(path.read_text()))
    counts = graph_candidate_counts(payloads) if payloads else {
        "stems": [row["dataset"] for row in tune["per_sequence"]],
        "n_nodes": [row["num_pred_nodes"] for row in tune["per_sequence"]],
        "n_candidate_edges": tune.get("n_candidate_edges") or [0, 0],
        "n_edges": tune.get("n_edges") or [0, 0],
        "opened_holdout": False,
    }
    ckpt = inspect_checkpoints(args.train_dir)
    unet_rel = (ckpt or {}).get("group_rel_to_seed0_init", {}).get("unet", {}).get("rel")
    trans_rel = (ckpt or {}).get("group_rel_to_seed0_init", {}).get("transformer", {}).get("rel")
    load_ok = bool(ckpt and ckpt["strict_load"] and not ckpt["missing_keys"])
    untrained = bool(ckpt and trans_rel is not None and unet_rel is not None and trans_rel < 0.3 < unet_rel)
    decision = (
        "no_implementation_defect_transformer_near_init"
        if load_ok and untrained
        else "needs_implementation_fix"
        if ckpt and not load_ok
        else "diagnosis_incomplete_missing_weights"
    )
    payload = {
        "status": "diagnosed",
        "decision": decision,
        "orgop_hash": ORGOP_HASH,
        "train_hash": TRAIN_CONFIG_HASH,
        "orgop_tune_combined": tune.get("learned_combined"),
        "orgop_graphs": counts,
        "checkpoint": ckpt,
        "dense_softmax_blocker": {
            n_src: dense_softmax_rarely_exceeds_half(n_src, 80, 3.0)
            for n_src in (8, 64, 400, 800)
        },
        "infer_vs_orgop": {
            "same_fallback_weights": "learned_unet_v1_last.pt",
            "infer_uses_transformer_edges": False,
            "orgop_uses_transformer_softmax_0.5": True,
            "orgop_ilp_skipped": True,
            "silent_partial_load": False,
        },
        "runtime_projection": runtime_projection(),
        "code_version": git_version(ROOT),
        "opened_holdout": False,
        "is_leaderboard_score": False,
        "notes": (
            "Orgop v1 remains rejected. Do not retune 250c5bfacd4a8198. "
            "Do not overwrite mahdimzafari/biohub-learned-unet-orgop-v1 v1."
        ),
    }
    write_run_json(args.report, payload)
    print(json.dumps({"decision": decision, "n_candidate_edges": counts["n_candidate_edges"], "transformer_rel": trans_rel, "unet_rel": unet_rel}, indent=2))


if __name__ == "__main__":
    main()
