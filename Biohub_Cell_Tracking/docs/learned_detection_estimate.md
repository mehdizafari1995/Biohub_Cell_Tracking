# Estimate: organizer learned 3D detection / embedding

Written 2026-09-20 **before** any training run or kernel push. This is not a
Kaggle run and not a leaderboard score. Source: vendored
`third_party/kaggle-cell-tracking-competition` @ `075fc5f5`, plus this worker's
hardware probe.

## Decision

**Blocked on this worker.** Do not train. Do not push a learned-detection kernel
from this run.

| gate | result |
|---|---|
| Local GPU | **None** (`nvidia-smi` missing; 4 CPU / 16 GB RAM) |
| Specimen-separated weights | **Missing.** Organizer default split is a random 90/10 over all stems, not frozen `v1.json` |
| Weight packaging for internet-off Kaggle | **Not done.** Public notebook fetch timed out; no local `.pth` |
| Kaggle 12 h CPU inference | **Not viable.** 3D UNet + TTA is a GPU workload |
| Kaggle GPU inference of public weights | Possible later, but **would leak** if used to judge frozen `6bba` holdout |

Fallback implemented in this family: `linkable_v1` (temporally linkable complementary peaks). See `docs/linkable_priors.json`.

## Method (what we would run)

End-to-end detection + linking from the organizer baseline:

1. `TemporalUNet3D` — 3D U-Net with per-voxel temporal attention, layers `(32, 64, 128)`, 32 output channels. Full-resolution temporal attention is skipped (`skip_fullres_temporal=True`).
2. Detection head `Conv3d(32 → 1)` plus max-pool local-max (`pool_kernel_um` 3–5 µm). Default predict threshold 0.99 because GT is sparse.
3. `SimpleNodeTransformer` — 4 cross-attention blocks, hidden 128, pairwise MLP. Feature dim = 32 UNet channels + 32 sinusoidal positional embeddings.
4. Default spatial downsample **`(1, 4, 4)`** so a `(100, 64, 256, 256)` volume is processed as `(T, 64, 64, 64)`. Centroids are scaled back to original voxels.
5. Window size 2, sliding stride 1. Detection TTA: original + flip-x + flip-y + flip-xy (no Z flip).
6. Optional tracksdata ILP (`--use-ilp`) is off by default; greedy max-children 2 / max-parents 1.

Dependencies: `torch>=2.9.1`, `tracksdata`, `zarr`, `scipy`. Not on this CPU image.

## Size and compute

Parameter count (analytic, float32):

- U-Net encoder/decoder + two temporal-attention stages: **~1.5 M**
- Detection head: 33
- Transformer (`feat_dim=64`, 4 blocks, pair MLP): **~0.8–1.0 M**
- **Total ~2.5 M parameters ≈ 10 MB** `.pth` ( comfortably attachable as a Kaggle dataset)

Training defaults (`scripts/train_unet_transformer.py`):

- 50 epochs, batch 16, AdamW, `lr` CLI default `1e-4`
- Window `(B, 2, 1, 64, 64, 64)` after downsample — fits a T4 16 GB
- ~180 train videos × ~99 windows / 16 ≈ **1.1 k steps/epoch**; 50 epochs ≈ 55 k steps
- Organizer README: the public [UNet baseline inference notebook](https://www.kaggle.com/code/thibautgoldsborough/unet-baseline-inference-submission) was trained with **`--epochs 3`**, “not trained to convergence”
- Full train dump ~81 GB (third-party; unverified). This disk has 247 GB free; the blocker is GPU, not disk
- This worker: 4 CPU, no CUDA. One 3D U-Net forward on CPU would be orders of magnitude slower than the locked DoG baseline (1.48 s/frame on Kaggle CPU). **Local training is not feasible in one cron run.**

Kaggle inference (public test 4 volumes, hidden test ≈ train size):

- ~200 volumes × ~99 windows × 4 TTA encodes
- On GPU T4: likely **tens of minutes to a few hours**, inside 12 h if weights are attached and `torch` is on the image
- On Kaggle **CPU**: likely **over 12 h**. Do not package a CPU learned kernel
- Peak activation for `(1, 2, 32, 64, 64, 64)` is small versus 16 GB; OOM risk is low at the official downsample and high if someone trains at full XY

Internet-off packaging:

- Weights must be a public Kaggle dataset or committed into the kernel (10 MB is fine; do **not** commit competition volumes)
- `torch` is present on Kaggle GPU images; it is not on this worker and may be missing on the CPU image used by `biohub-association-v2`
- External public weights are allowed by the rules if reasonably accessible

## Validation leakage

Organizer `train()` with no splits file shuffles **all** dataset stems (seed 0) into 90/10. That is **not** specimen-separated. Public notebook weights may have seen `6bba` during training.

Using those weights to score frozen `docs/splits/v1.json` holdout would not be a valid generalization test. A learned family must **retrain on `44b6` only** (or on all train except `6bba`) before a one-shot `6bba` score. That retrain needs a Kaggle GPU notebook, the competition `train/` mount, and a new private slug. Do not overwrite `biohub-association-v2`, `biohub-baseline-v1`, or the helper export kernels.

## What this run will not do

- Train UNet / transformer
- Download organizer weights into git
- Push a GPU kernel
- Claim a public-notebook number as our score

Next learned attempt (later run, expensive): private GPU kernel that trains on frozen-tuning specimen only, writes weights to kernel output (not git), then a separate internet-off inference kernel. Estimate GPU hours first; 3 organizer-style epochs is the lower bound, 50 epochs may not fit 12 h plus data load.

## Fallback chosen here

`linkable_v1`: same complementary spatial gates as rejected `complementary_v1`, plus a sequence filter that drops extras without a primary neighbor inside 12 µm at `t±1`. Tune on `44b6` only. CPU, official scorer, no kernel push this run.

## Follow-up (2026-09-20, learned_unet_v1 family)

`linkable_v1` was rejected on holdout. Specimen-separated `learned_unet_v1` training helper v3 is **COMPLETE / trained** (hash `4c38035d6657f2ad`, 11732 s). Infer helper v3 is **COMPLETE / inferred**. Official 44b6 tune **0.655751 vs association_v2 0.632460** (`keep_candidate_learned_unet_infer_v1`, hash `d7578b5f64ddac08`). One-shot frozen `6bba` later kept the no-gap and gap overlays. Transformer edge scores were unused in those infer passes; `learned_unet_link_v1` (hash `3a7423c6858ace41`) isolated that linker on 44b6 only and was **rejected** (official **0.0 vs 0.655751**, 0 edges from softmax+0.5). `learned_unet_link_v2` (hash `1d4015dc499bd604`) is the next packaged helper: same locked detections, no 0.5 cutoff, locked 12 µm gate. Official 44b6 score is not recorded yet. Weights stay out of git. Do not use public organizer weights on frozen `6bba`.
