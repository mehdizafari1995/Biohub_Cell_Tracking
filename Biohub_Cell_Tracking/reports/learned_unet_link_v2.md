# learned_unet_link_v2

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite `baseline_v1`, `association_v2`, `division_v2`,
`complementary_v1`, `linkable_v1`, `learned_unet_division_v1`, rejected
`learned_unet_link_v1`, or locked learned train/infer/holdout/test/gap-test
slugs. Do **not** retune locked gates. Frozen `6bba` was **not** scored.

## Hypothesis

Rejected `learned_unet_link_v1` wrote **0 edges** because softmax over
hundreds of sources stays below the locked 0.5 probability cutoff.
Keeping the same locked detections and transformer scores, greedy 1-1
assignment on **descending scores with no 0.5 cutoff**, plus the locked
association_v2 **12 µm** spatial gate, can write plausible edges and
raise the official frozen-tuning `44b6` score versus locked
`learned_unet_infer_v1` **0.655751**. Frozen `6bba` is refused until
that 44b6 comparison keeps the candidate.

## Operating point (44b6 only; rejected on tuning)

Config: `configs/learned_unet_link_v2.yaml`. Hash `1d4015dc499bd604`.
Training hash `4c38035d6657f2ad`. Parent infer `d7578b5f64ddac08`.
Rejected parent `learned_unet_link_v1` hash `3a7423c6858ace41` stays
locked.

- Infer stems only: `44b6_12dfb391`, `44b6_267148e4`
- Holdout `6bba` is refused
- Detection unchanged: `det_threshold` 0.5, TTA flip-x/y/xy, `pool_kernel_um` 5.0
- Normalisation matches training: per-window min-max
- Linker: transformer softmax, **no probability cutoff**, greedy 1-1,
  `max_link_um=12.0`
- Pair MLP is row-chunked (score-preserving) to bound GPU memory
- Writes JSON graphs + `infer_manifest.json`. Does **not** write `submission.csv`
- Slug `mahdimzafari/biohub-learned-unet-link-v2` v1 **COMPLETE / inferred**

Do **not** retune this hash. Do **not** overwrite this completed slug.

## Kaggle helper

Private GPU kernel `mahdimzafari/biohub-learned-unet-link-v2`. Internet
off. Attaches `mahdimzafari/biohub-learned-unet-v1`. **Not a submission.**

| version | status | notes |
|---|---|---|
| 1 | **COMPLETE / inferred** | 135.1 s, peak 1251 MB, T4 x2. Graphs: `44b6_12dfb391` 80741/72532, `44b6_267148e4` 36951/31819. No `submission.csv`. |

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4` only.
Graphs from the COMPLETE helper. GEFFs from `mahdimzafari/biohub-geff-export`
v1. **Not a Kaggle run. Not a leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes / edges |
|---|---|---|---|---|---|---|
| learned_unet_infer_v1 | **0.655751** | 0.69001 | 0.65575 | 0.97004 | 0/0/2 | 117692 / 106385 |
| learned_unet_link_v2 | **0.224746** | 0.23608 | 0.22475 | 0.97004 | 0/0/2 | 117692 / 104351 |

Per-sequence linker: `44b6_12dfb391` 0.256194 (infer 0.747072),
`44b6_267148e4` 0.143610 (infer 0.446373). Detection counts match
locked infer exactly (80741 / 36951). Official matched-edge
TP/FP/FN = 475/962/575 vs infer 877/221/173. Runtime 2.41 s, peak 553 MB.

Decision **`reject_learned_unet_link_v2`**. Do **not** retune. Frozen `6bba`
was refused because tuning did not keep the candidate.

## Error analysis

The no-cutoff assignment wrote 1-1 consecutive edges inside the locked
12 µm gate (max 11.94 µm; 0 edges over the gate; 0 predicted divisions).
That fixes the v1 0-edge failure. Transformer ranking among those gated
candidates is worse than greedy nearest-neighbor: fewer official TPs and
more conflicting FPs on the same detections. See
`reports/learned_unet_link_v2_error.json`.

## Next

Keep `learned_unet_gap_v1` / `learned_unet_gap_test_v1` as the candidate.
Do **not** package another transformer-threshold variant. A later run may
start a **new** slug that keeps locked learned detections and replaces
transformer scores with a documented motion-aware / constant-velocity
gate. Do **not** overwrite this rejected v2 slug. Do **not** score `6bba`
for this family. If a public leaderboard number is wanted, Mahdi submits
`biohub-learned-unet-gap-test-v1` v1 from the Kaggle UI. Do **not**
submit from the CLI.
