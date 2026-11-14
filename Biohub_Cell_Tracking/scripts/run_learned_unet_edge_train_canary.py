#!/usr/bin/python3
"""Run the locked learned_unet_edge_train_canary_v1 on CPU.

Uses the training notebook architecture and synthetic 1-1 targets. Refuses
6bba and 199-video weights. Does not write submission.csv. Does not push a
kernel. Not a competition submission.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from biohub_tracking.experiments import git_version, write_run_json  # noqa: E402
from biohub_tracking.learned_edge_canary import (  # noqa: E402
    LOCKED_CANARY_HASH,
    canary_config_hash,
    canary_config_payload,
    public_lb_not_a_tune_target,
    refuse_holdout_design,
    runtime_projection,
)
from biohub_tracking.learned_split import TRAIN_CONFIG_HASH  # noqa: E402


def _run_canary(cfg: dict) -> dict:
    import torch

    from kaggle_learned_unet_v1 import _build_models

    refuse_holdout_design(["44b6_canary"])
    public_lb_not_a_tune_target(0.6557507105877639)
    train_cfg = {
        "unet_layers": [32, 64, 128],
        "unet_out_channels": 32,
        "skip_fullres_temporal": True,
        "det_neg_weight": 0.01,
    }
    UNetNodeTransformer, detection_loss, edge_loss = _build_models(train_cfg)
    torch.manual_seed(int(cfg["seed"]))
    n_src = int(cfg["n_src"])
    n_tgt = int(cfg["n_tgt"])
    feat_t = torch.randn(1, n_src, int(cfg["feat_dim"]))
    feat_t1 = torch.randn(1, n_tgt, int(cfg["feat_dim"]))
    coords_t = torch.rand(1, n_src, 3) * 16
    coords_t1 = torch.rand(1, n_tgt, 3) * 16
    target = torch.eye(n_src)[:n_src, :n_tgt]
    if n_tgt != n_src:
        target = torch.zeros(n_src, n_tgt)
        for j in range(n_tgt):
            target[j % n_src, j] = 1.0

    det_logits = torch.randn(1, 1, 4, 8, 8)
    det_coords = torch.tensor([[[1.0, 2.0, 3.0]]])
    det_mask = torch.ones(1, 1, dtype=torch.bool)
    dloss0 = float(detection_loss(det_logits, det_coords, det_mask, 0.01))

    model = UNetNodeTransformer()
    if cfg["freeze_unet"]:
        for param in model.unet.parameters():
            param.requires_grad = False
        for param in model.detect_head.parameters():
            param.requires_grad = False
    opt = torch.optim.AdamW(
        [p for p in model.transformer.parameters() if p.requires_grad],
        lr=float(cfg["lr"]),
    )
    history: list[dict] = []
    started = time.time()
    for step in range(int(cfg["n_steps"])):
        logits = model.transformer(feat_t, feat_t1, coords_t, coords_t1)[0]
        loss = edge_loss(logits, target)
        opt.zero_grad()
        loss.backward()
        grad = model.transformer.pair_mlp[5].weight.grad
        gnorm = float(grad.norm()) if grad is not None else 0.0
        opt.step()
        probs = torch.softmax(logits.detach(), dim=0)
        history.append(
            {
                "step": step,
                "edge_loss": float(loss.detach()),
                "grad_l2": gnorm,
                "prob_max": float(probs.max()),
                "n_gt_half": int((probs > float(cfg["edge_threshold"])).sum()),
            }
        )
    model.eval()
    clone = UNetNodeTransformer()
    clone.load_state_dict(model.state_dict(), strict=True)
    clone.eval()
    with torch.no_grad():
        a = model.transformer(feat_t, feat_t1, coords_t, coords_t1)
        b = clone.transformer(feat_t, feat_t1, coords_t, coords_t1)
        reload_max_abs = float((a - b).abs().max())
        final_probs = torch.softmax(a[0], dim=0)
        final_prob_max = float(final_probs.max())
        final_n_gt_half = int((final_probs > float(cfg["edge_threshold"])).sum())

    first = history[0]
    last = history[-1]
    finite = all(
        x == x and abs(x) != float("inf")
        for row in history
        for x in (row["edge_loss"], row["grad_l2"], row["prob_max"])
    )
    nonzero_grad = any(row["grad_l2"] > 0.0 for row in history[:5])
    peaked = (
        final_prob_max >= float(cfg["min_prob_max"])
        and final_n_gt_half >= int(cfg["min_n_gt_half"])
    )
    reload_ok = reload_max_abs < 1e-6
    status = "canary_ok" if finite and nonzero_grad and peaked and reload_ok else "canary_failed"
    return {
        "status": status,
        "detection_loss_finite": dloss0 == dloss0 and abs(dloss0) != float("inf"),
        "detection_loss": dloss0,
        "first": first,
        "last": last,
        "final_prob_max": final_prob_max,
        "final_n_gt_half": final_n_gt_half,
        "reload_max_abs": reload_max_abs,
        "finite_losses": finite,
        "nonzero_edge_grad": nonzero_grad,
        "noncollapsed_softmax": peaked,
        "checkpoint_reload_exact": reload_ok,
        "runtime_sec": time.time() - started,
        "opened_holdout": False,
        "wrote_submission_csv": False,
        "kernel_pushed": False,
        "used_199_video_weights": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "learned_unet_edge_train_canary_v1.json",
    )
    args = parser.parse_args()
    cfg = canary_config_payload()
    if canary_config_hash(cfg) != LOCKED_CANARY_HASH:
        raise SystemExit("canary config hash drifted; do not retune")
    if cfg["train_config_hash"] != TRAIN_CONFIG_HASH:
        raise SystemExit("canary is not pinned to locked learned_unet_v1")
    result = _run_canary(cfg)
    payload = {
        "status": result["status"],
        "decision": result["status"],
        "config_hash": LOCKED_CANARY_HASH,
        "parent_orgop_hash": cfg["parent_orgop_hash"],
        "train_config_hash": TRAIN_CONFIG_HASH,
        "config": cfg,
        "result": result,
        "runtime_projection": runtime_projection(),
        "code_version": git_version(ROOT),
        "is_kaggle_run": False,
        "is_competition_submission": False,
        "is_leaderboard_score": False,
        "notes": (
            "CPU transformer canary on synthetic 1-1 targets. Architecture from "
            "locked learned_unet_v1. 6bba refused. Not a leaderboard score."
        ),
    }
    write_run_json(args.report, payload)
    print(json.dumps({"status": result["status"], "config_hash": LOCKED_CANARY_HASH, "final_n_gt_half": result["final_n_gt_half"], "reload_max_abs": result["reload_max_abs"]}, indent=2))
    if result["status"] != "canary_ok":
        raise SystemExit("learned_unet_edge_train_canary_v1 failed a proof gate")


if __name__ == "__main__":
    main()
