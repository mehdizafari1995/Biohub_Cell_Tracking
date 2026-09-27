# learned_unet_orgdet_fnn_rank_v1 (preregistered, not scored)

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite locked train/infer/holdout/test/gap-test/prune-test
or orgdet/elf/fnn slugs. Do **not** retune locked FNN. This is **not**
another 12 µm assignment, **not** a short-fragment retune, and **not**
another 8.0 µm dt=1 length retune. **Not yet official-scored.**

## Hypothesis

Official 44b6 FNN oracle (same detections, official scorer `075fc5f5`)
showed association-dominated residue:

- FNN **0.724653**, official 906 / 174 / 144, divisions 0/0/2
- Detection-unavoidable GT edges **27 / 1050** (2.57%)
- Both-endpoint-detected GT edges **1023**; **1019 / 1023** sit inside
  the locked 12 µm dt=1 envelope (candidate-edge recall **0.996**)
- Association-fixable and in-candidate **113**
- Candidate-oracle official combined **1.048979** vs FNN **0.724653**
- Detection-oracle official combined **1.052563**
- Both annotated 44b6 forks are recoverable with this detector

A learned pairwise ranker that only sees candidates already inside the
locked envelope, trained with hard negatives on 44b6 only, can in
principle recover those 113 edges and drop conflict FPs without
rewriting the 12 µm linker, gap gates, ELF length, or FNN keep rule.
Previous transformer linkers (`link_v1`, `link_v2`, `orgop`) failed
because the link head stayed near init. This family is fail-closed
until a canary shows nonzero link-head learning.

Hash `2748c4a2b2823c0a` is locked before any ranker score. Official 44b6
must exceed FNN by **≥ 0.002000** (threshold **0.726653**). Frozen
`6bba` stays closed unless that gate keeps, then one-shot versus
**0.752091**.

## Operating point (preregistered)

Config: `configs/learned_unet_orgdet_fnn_rank_v1.yaml`.
Hash `2748c4a2b2823c0a`. Parent FNN `95e9300eb84ba71c`.

- Source: locked FNN / orgop detections
- Candidate set: dt ∈ {1, 2}, 12 µm · √dt, locked
- Learned pairwise rank + constrained 1-1 selection
- Hard negatives. 44b6 train only
- `verify_nonzero_link_head=true`
- Intensity unused. No ILP. No velocity. No 12 µm retune
- Writes no `submission.csv`

Do **not** retune `nn_tie_eps`, `max_dt1_edge_um`, or the 12 µm gate.
Do **not** score 6bba until a 44b6 keep. Do **not** launch a full
ranker helper until the canary is `canary_ok`.

## Cost estimate (before launch)

- Storage: reuse orgop graphs + two 44b6 GEFFs. Appearance features
  need 44b6 Zarrs; prefer the existing two-volume tuning export first.
  Full 69-volume 44b6 train is tens of GB and is not authorized this
  cycle.
- GPU: new private helper only after canary_ok. Estimate ≤ 11 h T4x2.
- CPU canary: graph-pair hard negatives + finite link loss + nonzero
  grads + reload exact. No `submission.csv`. No `6bba`.

## Status

`preregistered_not_scored`. Not a Kaggle run. Not a leaderboard score.
Public **0.602 / 0.612 / 0.553** were not used.
