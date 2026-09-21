# Public 0.94 benchmark parity audit

Read-only strategy audit on draft PR #7. Do **not** merge.
Do **not** overwrite locked `learned_unet_prune_v1` / prune-test / gap-test
/ infer / train slugs. Do **not** launch a reproduction kernel in this
cycle. Do **not** submit. Public leaderboard **0.553** is association_v2
ref **56399003** only and was **not** used for tuning.

Prune-test rehearsal was already **COMPLETE** (`074a71dec68304e8`,
348.18 s, `audit_ok`). That kernel was **not** duplicated.

## Why 0.701378 is not a 0.24 deficit versus 0.94

Locked `learned_unet_prune_v1` **0.701378** is official frozen `6bba`
holdout after 3-epoch **44b6-only** training. Public 0.94-class numbers
are public-test leaderboard claims, mostly from checkpoints trained on
**all 199 annotated videos**. Those are different tasks. This audit does
not treat 0.701378 − 0.94 as a single linker bug.

Independently verified public-test claims this cycle:

| claim | evidence | kind |
|---|---|---|
| 0.947 | `zhincez/biohub-0-947-lb-runnable-with-public-datasets` cites submission **56207450** | title + author-reported LB; not returned by our competition-list API |
| 0.942 | `busyaprime` submission **56053765** vs same pipeline **56053023** at 0.941 | author-reported scored pair |
| 0.941 | four public notebooks in that lineage | author census |
| 0.948 title | `busyaprime` resubmitted one such notebook as **56045620** → **0.935** | title is a claim |
| 0.9462 | requested; **not** independently returned as an API `publicScore` | unresolved numeric |
| 0.826 | discussion **716952** table (rule-based, CV≈LB) | CV + public LB, not 0.94 |
| 0.553 | our API list, association_v2 **56399003** | our only competition row |

## Inventory (requested discussion IDs)

| ID | URL | Author / title (retrieved) | Date / votes | Score claim | License / reuse | CV / LB / oracle |
|---|---|---|---|---|---|---|
| 716952 | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716952 | Rule-based gold-zone write-up | table at time of writing | CV 0.824 / LB **0.826** max | public notebook, rule-based DoG + Hungarian | CV≈LB; **not** 0.9462 |
| 730160 | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/730160 | Public-weight leakage | comments retrieved | public ckpt trained on **199/199** videos; 40 “test” names sit inside that train list | public discussion | in-sample / embryo leak |
| 734604 | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734604 | Public ckpt post-process wall | comments retrieved | “retrain; decompose detection vs linking vs post-process” | public discussion | method advice, no 0.94 number |
| 735352 | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/735352 | Possible big leaderboard shakeup | comments retrieved | plugin “0.940 untuned” on a public model; top-10 gaps <0.001 | public discussion | author plugin claim, not a scored notebook we ran |
| 738217 | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/738217 | topics API title **focus3d : one of the best 3d cell segmentation** (author field empty) | posted 2026-08-30; 36 votes; 50 comments; last comment 2026-09-20 | no numeric score in the topics metadata | public discussion; HTML is an SPA shell | metadata only; **not used** as a source; not the 0.94 ILP stack |
| 740392 | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/740392 | **absent** from authenticated topics first page (20 of 112); kernel search 0 hits; HTML SPA shell | — | **not independently readable** | unresolved body | unresolved; **not used** |
| 742064 / 742266 | public 0.947 stack comments (web index) | 199-video weights are in-sample | — | 0.947 lineage | public | in-sample |

## Inventory (public notebooks / datasets)

| Slug | Author | Datasets | Stated score | License notes | Verdict for our candidate |
|---|---|---|---|---|---|
| `thibautgoldsborough/unet-baseline-inference-submission` | Thibgolds (organizer) | `cellmot-baseline-artifacts` | comment: ILP ~0.73→0.79; `DET_THRESHOLD=0.99`, `USE_ILP=True` | artifacts license **Unknown**; method is official predict path | **defaults are the reproduction target**; organizer-split weights are **not** attached |
| `zhincez/biohub-0-947-lb-runnable-with-public-datasets` | Lê Quang Cảnh; copy of Reyhan Ksatria v5 | Pilkwang TemporalUNet seed 314159 + DeepCenter + support pack | 0.947 / 56207450 | Pilkwang datasets advertised **CC0** | rules-allowed public data; **method-rejected** (199-video leak) |
| `busyaprime/biohub-0-942-lb-one-knob-past-the-public-line` | Akmal Xodarev | same Pilkwang trio + knob provenance | 0.941 lineage, one knob → 0.942 | CC0 weights | same leak; documents title inflation (0.948→0.935) |
| `beraterolelk/0-947-lb-biohub-deepcenter-ilp-tracker` | Berat Erol ÇELİK | same Pilkwang trio | title 0.947 / header 0.951+ | CC0 weights | same stack; title ≠ verified score |
| `haideptry/biohub-sota-0-948-density-adaptive-2xt4-22m` | haideptry | Pilkwang trio + `giorgosi/biohub-divnet-v2` | title 0.948+ | DivNet **CC BY 4.0**; density knobs name public-test stems (`6bba_05b6850b`, …) | reject density-by-public-stem overrides; DivNet later |
| `andnyu/biohub-density-adaptive-0-948-reproduction` | Andrew Yu | same as haideptry | title 0.948 | same | clone of the density-adaptive notebook |

Rules (`docs/competition_spec.md`): freely and publicly available external
data and pre-trained models are allowed if public and low-cost. Winner
license MIT. **No private labels or hidden-test GEFF leakage was found
in the retrieved sources.** Using the 199-video public checkpoints is
therefore **rules-legal** and **scientifically invalid** for a
specimen-separated candidate: discussion **730160** and the
`split_manifest.json` report train=199 with the 40 “test” names inside
that set.

## 0.94-class pipeline (reconstructed)

Shared stack behind the public 0.94 titles, in order:

1. **Weights.** `TemporalUNet3D` + node transformer from
   `pilkwang/biohub-tracking-support-pack-50ep-v1`. Dataset name says
   50 epochs; public write-ups (`m9h/biohub-starter`) report the file is
   **epoch 402**. Plus `DeepCenter` UNet3D and a second TemporalUNet seed
   314159 logit blend. Trained on **all 199** train videos.
2. **Detection.** `DET_THRESHOLD = 0.99` (organizer: GT is sparse, 0.5 is
   too low). Flip TTA. Peak pool on the heatmap.
3. **Linking.** Transformer edge probabilities, then **tracksdata ILP**
   (`USE_ILP=True`; edge −1.0, appear/disappear 0.1, division 1.0).
   Organizer comment: greedy ~0.73 → ILP ~0.79 on *their* baseline.
   Independent public write-up: `--use-ilp` +0.042 on the leaky 402-epoch
   pack (0.8596 → 0.9012) at 3.8× runtime.
4. **Post-process (lineage extras).** Gap-close env (`MAX_GAP` 1–2),
   motion relink, min-track filters, optional DivNet mitosis, optional
   density-adaptive gates on **public-test stem names**.
5. **Not required for 0.94 titles.** Cellpose / focus3d were not
   extracted from the notebooks we flattened. Discussion **738217** is
   independently titled “focus3d : one of the best 3d cell
   segmentation” (topics API; 36 votes). The HTML body is still an SPA
   shell, so focus3d remains **not** part of the locked orgop proposal.

Discussion **716952** is a different pipeline: multi-scale DoG +
Hungarian, no deep model, public LB **0.826**. Detection was the lever;
gap-close was neutral; division edges hurt. That is the public ceiling
for a pure rule-based method, not the 0.94 stack.

## Field-by-field vs locked `learned_unet_prune_v1`

| Field | Public 0.94 stack | Locked prune_v1 (`115753ea16aaf021`) |
|---|---|---|
| Train specimen | all 199 videos / both embryos | `44b6` only (hash `4c38035d6657f2ad`) |
| Epochs | 402 in the “50ep” pack (reported) | 3 |
| Extra models | DeepCenter + seed 314159 blend | none |
| `det_threshold` | **0.99** | **0.5** |
| `pool_kernel_um` | organizer default **3.0** | **5.0** |
| Linker | transformer `edge_prob` + **ILP** | greedy last-position **12 µm** |
| Edge scores | softmax transformer | unused at infer (geometry only) |
| Gap close | often on | locked overlay, `max_gap_frames=2` |
| Isolates | min-track / filters | drop degree-0 after gap |
| Divisions | ILP weight 1.0 (often 0 TP); optional DivNet | 0/0/5 on holdout; overlay rejected |
| Score meaning | public LB, in-sample weights | official **6bba** holdout **0.701378** |

## Ranked gaps for the honest ~0.24

1. **Train coverage / length (largest, not copyable for the candidate).**
   199-video 402-epoch weights vs our 44b6 3-epoch weights. Copying
   Pilkwang/Reyhan checkpoints would leak `6bba` and is **rejected** for
   any KEEP_CANDIDATE path. Optional later public-test-only replay may
   be tagged `not_for_tuning` after review.
2. **Inference path: transformer edges + ILP vs greedy 12 µm (highest
   rules-compliant upside on our weights).** Our infer never consumes
   `SimpleNodeTransformer` edge logits. Organizer and every 0.94 notebook
   do, then ILP. This is a new inference operating point, not another
   12 µm Hungarian (already rejected at 0.439023).
3. **Detection threshold 0.5 vs 0.99.** Organizer: 0.5 is too low because
   GT is sparse. Extra T_pred hurts adjusted Jaccard; prune only removed
   isolates. Bundled into the organizer default rather than swept.
4. **Pool kernel 5.0 vs 3.0 µm.** Bundled into organizer default.
5. **DeepCenter / second seed.** Later. Requires extra public weights
   and a leak decision.
6. **Longer 44b6-only retrain.** Discussion 734604 / 730160: the honest
   way to beat public ckpts. Expensive (multi-hour GPU). After orgop.
7. **Density-adaptive public-test stem overrides.** Reject. Those knobs
   name `6bba_05b6850b` / `44b6_0b24845f` and overfit public test.
8. **Division plugins / DivNet.** Our `learned_unet_division_v1` already
   lost on 44b6 (0.655235 vs 0.655751). 716952 also saw divisions hurt.

Rejected overlays stay closed: link_v1 0.0, link_v2 0.224746, motion
0.583722, hung 0.439023, conflict 0.645100.

## Selected experiment (not launched)

**Slug / config:** `learned_unet_orgop_v1`  
**Hash:** `250c5bfacd4a8198`  
**Reserved kernel id:** `mahdimzafari/biohub-learned-unet-orgop-v1`  
**Status:** specified, tests locked, **`LAUNCH_AUTHORIZED=False`**.
Not in `KERNELS`. Not pushed.

Organizer `predict_unet_transformer.py` defaults on **locked 44b6-only
weights**:

- `det_threshold=0.99`
- `pool_kernel_um=3.0`
- `det_tta=true`
- transformer `edge_activation=softmax`, `edge_threshold=0.5`
- `use_ilp=true`
- ILP weights `−1.0 / 0.1 / 0.1 / 1.0`
- no gap overlay, no DeepCenter, no secondary seed, no public pack

### Compute

GPU T4 x2, internet off, two frozen-tuning volumes only
(`44b6_12dfb391`, `44b6_267148e4`). Expected **5–15 min** (ILP ~3.8×
greedy). Hidden-test wall time is **not** projected until a 44b6 keep
and a later packaging cycle. Needs `pyscipopt` / tracksdata ILP on the
worker.

### Required assets (already local / ignored)

- Weights from `mahdimzafari/biohub-learned-unet-v1` v3:
  `learned_unet_v1_best.pt` / `learned_unet_v1_last.pt` (ignored
  `artifacts/kaggle/`; hash family `4c38035d6657f2ad`)
- Vendored `scripts/predict_unet_transformer.py` @ scorer repo
  `075fc5f5`
- Tuning Zarrs from `biohub-tuning-zarr-export` v1
- Tuning GEFFs from `biohub-geff-export` v1
- Official scorer `075fc5f5` / `tracksdata==0.1.0rc10`

Do **not** download Pilkwang / Reyhan / DivNet / organizer-split
`edge_predictor_best.pth` for this slug.

### Acceptance gates

1. Official-score the two `44b6` tuning sequences vs locked infer
   **0.655751**. Report combined, adj edge Jaccard, division Jaccard,
   recall, node/edge counts, runtime, peak RAM/VRAM.
2. **Reject** if combined does not beat 0.655751. Do **not** open `6bba`.
   Do **not** retune threshold / ILP / pool.
3. If it beats 0.655751, lock code/config/hash, prove no post-lock
   edits, then **one** frozen `6bba` score vs prune **0.701378**.
4. Reject on holdout combined regression. Else
   `KEEP_CANDIDATE_NOT_SUBMISSION_AUTHORIZED`.
5. Public 0.553 / 0.94 titles are not tune targets. No CLI submit.
   No overwrite of prune-test v1.

## Follow-up inventory (2026-09-21T04:48:29Z)

Authenticated topics API independently retrieved **738217** metadata
(title, 36 votes, 50 comments). **740392** is still absent from the
first hot page (20 of 112); pagination query params are ignored by the
host. HTML bodies remain SPA shells. Neither ID was used as a source.
`learned_unet_orgop_v1` stays specified, not launched.

Read-only prune-test submit poll: kernel still **COMPLETE** with
`run_manifest.json` + `submission.csv`. Competition list still exactly
association_v2 **56399003** / public API **0.553**. Decision
`human_action_required_ui_submit`. Did not download CSV. Did not submit.

## Authorized cycle outcome (2026-09-21T05:09:14Z)

The later reviewed cycle launched reserved slug
`mahdimzafari/biohub-learned-unet-orgop-v1` v1 on locked 44b6 weights
only. Official 44b6 combined **0.0 vs infer 0.655751**. Softmax>0.5
wrote 0 candidate edges. Decision **`reject_learned_unet_orgop_v1`**.
Frozen `6bba` was not opened. See `reports/learned_unet_orgop_v1.md`.
This audit body above remains the pre-launch specification.

## Next specified experiment (2026-09-21T05:47:28Z)

After orgop reject, the honest detection lever from this audit
(ranked #6, “after orgop”) is specified as `learned_unet_retrain_v1`
hash `92a419926fa7e40b`: resume locked `learned_unet_v1_last.pt` for
+3 epochs on 44b6 only. Reserved slug
`mahdimzafari/biohub-learned-unet-retrain-v1`.
`LAUNCH_AUTHORIZED=False`. Not in `KERNELS`. Not pushed.
See `reports/learned_unet_retrain_v1.md`.

Read-only prune-test submit poll at the same timestamp: kernel still
**COMPLETE**; competition list still association_v2 **56399003** /
**0.553**. Decision still `human_action_required_ui_submit`. Entry
deadline **2026-09-22 23:59 UTC**.

## What the audit cycle did not do

- Did not launch `learned_unet_orgop_v1` during the audit itself
- Did not attach public 199-video weights
- Did not official-score `6bba`
- Did not submit or merge
- Did not treat 0.9462 as a measured API score
- Did not retune prune / hung / motion / conflict
- Did not launch `learned_unet_retrain_v1`
