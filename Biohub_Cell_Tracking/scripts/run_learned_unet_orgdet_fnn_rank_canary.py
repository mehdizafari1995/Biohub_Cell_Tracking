#!/usr/bin/python3
"""Run the locked learned_unet_orgdet_fnn_rank_canary_v1 on CPU.

Uses the training-notebook transformer and graph-pair hard negatives
inside the locked 12 µm / sqrt(dt) envelope. Refuses 6bba, public
leaderboard scores, and 199-video weights. Does not official-score the
ranker. Does not write submission.csv. Does not push a kernel.
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

from biohub_tracking.experiments import git_version, peak_memory_mb, write_run_json  # noqa: E402
from biohub_tracking.learned_orgdet_fnn_oracle import LOCKED_RANK_HASH  # noqa: E402
from biohub_tracking.learned_orgdet_fnn_rank_canary import (  # noqa: E402
    LOCKED_CANARY_HASH,
    build_envelope_hard_negative_pairs,
    canary_config_hash,
    canary_config_payload,
    evaluate_pair_scores,
    pair_batches,
    public_lb_not_a_tune_target,
    refuse_holdout_design,
)
from biohub_tracking.learned_split import TRAIN_CONFIG_HASH  # noqa: E402


def _run_canary(cfg: dict) -> dict:
    import torch
    import torch.nn.functional as F

    from kaggle_learned_unet_v1 import _build_models

    refuse_holdout_design(["44b6_canary"])
    public_lb_not_a_tune_target(0.6557507105877639)
    for banned in ("0.553", "0.602", "0.612"):
        try:
            public_lb_not_a_tune_target(banned)
        except ValueError:
            pass
        else:
            raise SystemExit(f"public LB {banned} must stay banned")

    train_cfg = {
        "unet_layers": [32, 64, 128],
        "unet_out_channels": 32,
        "skip_fullres_temporal": True,
        "det_neg_weight": 0.01,
    }
    UNetNodeTransformer, detection_loss, _edge_loss = _build_models(train_cfg)
    torch.manual_seed(int(cfg["seed"]))
    pairs = build_envelope_hard_negative_pairs(int(cfg["n_parents"]))
    batches = pair_batches(pairs)
    feat_dim = int(cfg["feat_dim"])
    tensors = []
    for batch in batches:
        n_src = int(batch["n_src"])
        n_tgt = int(batch["n_tgt"])
        tensors.append(
            {
                "feat_t": torch.randn(1, n_src, feat_dim),
                "feat_t1": torch.randn(1, n_tgt, feat_dim),
                "coords_t": torch.tensor(batch["src_zyx"], dtype=torch.float32).unsqueeze(0),
                "coords_t1": torch.tensor(batch["tgt_zyx"], dtype=torch.float32).unsqueeze(0),
                "pair_index": torch.tensor(batch["pair_index"], dtype=torch.long),
                "labels": torch.tensor(batch["labels"], dtype=torch.float32),
                "ranking_index": torch.tensor(batch["ranking_index"], dtype=torch.long),
                "dt": int(batch["dt"]),
            }
        )

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

    def _link_loss() -> torch.Tensor:
        total = None
        for item in tensors:
            logits = model.transformer(
                item["feat_t"], item["feat_t1"], item["coords_t"], item["coords_t1"]
            )[0]
            cand = logits[item["pair_index"][:, 0], item["pair_index"][:, 1]]
            bce = F.binary_cross_entropy_with_logits(cand, item["labels"])
            rank = cand.new_zeros(())
            for si, ti_true, ti_hard in item["ranking_index"]:
                rank = rank + F.softplus(logits[int(si), int(ti_hard)] - logits[int(si), int(ti_true)])
            rank = rank / max(int(item["ranking_index"].shape[0]), 1)
            loss = bce + rank
            total = loss if total is None else total + loss
        assert total is not None
        return total / float(len(tensors))

    history: list[dict] = []
    started = time.time()
    for step in range(int(cfg["n_steps"])):
        loss = _link_loss()
        opt.zero_grad()
        loss.backward()
        grad = model.transformer.pair_mlp[5].weight.grad
        gnorm = float(grad.norm()) if grad is not None else 0.0
        opt.step()
        history.append(
            {
                "step": step,
                "link_loss": float(loss.detach()),
                "grad_l2": gnorm,
            }
        )

    model.eval()
    clone = UNetNodeTransformer()
    clone.load_state_dict(model.state_dict(), strict=True)
    clone.eval()
    logits_by_dt: dict[int, list] = {}
    reload_max_abs = 0.0
    with torch.no_grad():
        for item in tensors:
            a = model.transformer(
                item["feat_t"], item["feat_t1"], item["coords_t"], item["coords_t1"]
            )
            b = clone.transformer(
                item["feat_t"], item["feat_t1"], item["coords_t"], item["coords_t1"]
            )
            reload_max_abs = max(reload_max_abs, float((a - b).abs().max()))
            logits_by_dt[int(item["dt"])] = a[0].cpu().numpy()

    scores = evaluate_pair_scores(
        logits_by_dt,
        batches,
        min_margin=float(cfg["min_true_minus_hard_logit"]),
        min_ranking_acc=float(cfg["min_ranking_acc"]),
    )
    first = history[0]
    last = history[-1]
    finite = all(
        x == x and abs(x) != float("inf")
        for row in history
        for x in (row["link_loss"], row["grad_l2"])
    ) and (dloss0 == dloss0 and abs(dloss0) != float("inf"))
    nonzero_grad = any(row["grad_l2"] > 0.0 for row in history[:5])
    reload_ok = reload_max_abs < 1e-6
    status = (
        "canary_ok"
        if finite and nonzero_grad and scores["noncollapsed"] and reload_ok
        else "canary_failed"
    )
    return {
        "status": status,
        "detection_loss_finite": dloss0 == dloss0 and abs(dloss0) != float("inf"),
        "detection_loss": dloss0,
        "first": first,
        "last": last,
        "pair_graph": {
            "n_true": pairs["n_true"],
            "n_hard": pairs["n_hard"],
            "n_excluded": pairs["n_excluded"],
            "allows_two_children": pairs["allows_two_children"],
            "n_src": pairs["n_src"],
            "n_tgt": pairs["n_tgt"],
        },
        "scores": scores,
        "reload_max_abs": reload_max_abs,
        "finite_losses": finite,
        "nonzero_edge_grad": nonzero_grad,
        "noncollapsed_pair_scores": scores["noncollapsed"],
        "checkpoint_reload_exact": reload_ok,
        "runtime_sec": time.time() - started,
        "peak_memory_mb": peak_memory_mb(),
        "opened_holdout": False,
        "wrote_submission_csv": False,
        "kernel_pushed": False,
        "used_199_video_weights": False,
        "scored_official_44b6": False,
        "launched_ranker_helper": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "learned_unet_orgdet_fnn_rank_canary_v1.json",
    )
    args = parser.parse_args()
    cfg = canary_config_payload()
    if canary_config_hash(cfg) != LOCKED_CANARY_HASH:
        raise SystemExit("rank canary config hash drifted; do not retune")
    if cfg["parent_rank_hash"] != LOCKED_RANK_HASH:
        raise SystemExit("rank canary is not pinned to locked rank_v1")
    if cfg["train_config_hash"] != TRAIN_CONFIG_HASH:
        raise SystemExit("rank canary is not pinned to locked learned_unet_v1")
    if cfg["launches_ranker_helper"] or cfg["scores_official_44b6"] or cfg["kernel_push"]:
        raise SystemExit("canary must not launch or score the ranker")
    result = _run_canary(cfg)
    payload = {
        "status": result["status"],
        "decision": result["status"],
        "config_hash": LOCKED_CANARY_HASH,
        "parent_rank_hash": LOCKED_RANK_HASH,
        "parent_fnn_hash": cfg["parent_fnn_hash"],
        "train_config_hash": TRAIN_CONFIG_HASH,
        "config": cfg,
        "result": result,
        "code_version": git_version(ROOT),
        "is_kaggle_run": False,
        "is_competition_submission": False,
        "is_leaderboard_score": False,
        "notes": (
            "CPU transformer canary on graph-pair hard negatives inside the "
            "locked 12 µm / sqrt(dt) envelope. Parent rank "
            f"{LOCKED_RANK_HASH}. 6bba refused. Ranker not scored. "
            "Not a leaderboard score."
        ),
    }
    write_run_json(args.report, payload)
    print(
        json.dumps(
            {
                "status": result["status"],
                "config_hash": LOCKED_CANARY_HASH,
                "parent_rank_hash": LOCKED_RANK_HASH,
                "ranking_acc": result["scores"]["ranking_acc"],
                "min_true_minus_hard_logit": result["scores"]["min_true_minus_hard_logit"],
                "reload_max_abs": result["reload_max_abs"],
                "nonzero_edge_grad": result["nonzero_edge_grad"],
            },
            indent=2,
        )
    )
    if result["status"] != "canary_ok":
        raise SystemExit("learned_unet_orgdet_fnn_rank_canary_v1 failed a proof gate")


if __name__ == "__main__":
    main()
