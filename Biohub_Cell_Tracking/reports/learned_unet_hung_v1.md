# learned_unet_hung_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite `baseline_v1`, `association_v2`, `division_v2`,
`complementary_v1`, `linkable_v1`, `learned_unet_division_v1`, rejected
`learned_unet_link_v1` / `learned_unet_link_v2` / `learned_unet_motion_v1`,
or locked learned train/infer/holdout/test/gap-test slugs. Do **not**
retune locked gates. Frozen `6bba` was **not** scored.

## Hypothesis

Rejected transformer and constant-velocity rankers were worse than
last-position greedy on the same locked detections. Hungarian 1-1
assignment on last-position distance, inside the locked association_v2
**12 µm last-position** spatial limit, should resolve greedy
nearest-neighbor collisions without changing detections
(`learned_unet_infer_v1` **0.655751**) on frozen-tuning `44b6`. Frozen
`6bba` is refused until that 44b6 comparison keeps the candidate.

## Operating point (44b6 only; rejected on tuning)

Config: `configs/learned_unet_hung_v1.yaml`. Hash `f67bc97d0521dc78`.
Training hash `4c38035d6657f2ad`. Parent infer `d7578b5f64ddac08`.
Rejected transformer hashes `3a7423c6858ace41` and `1d4015dc499bd604`
and rejected motion hash `66853b6f1445c18b` stay locked.

- Infer stems only: `44b6_12dfb391`, `44b6_267148e4`
- Holdout `6bba` is refused
- Detection unchanged (locked infer nodes)
- Linker: last-position cost, Hungarian 1-1, hard last-position
  `max_link_um=12.0`, `max_gap_frames=0`, no interpolation, no intensity,
  no transformer, no velocity
- Writes no `submission.csv`
- Helper slug `mahdimzafari/biohub-learned-unet-hung-v1` packaged, not
  pushed (local overlay is sufficient)

Do **not** retune this hash. Do **not** score frozen `6bba` for this overlay.

## Cost estimate (before score)

- Storage: reuse infer-helper graphs + two `44b6` GEFFs from
  `mahdimzafari/biohub-geff-export` v1. No volumes, no weights.
- CPU: re-link 117692 locked detections. Measured **10.7 s / 596 MB**.
- No GPU. Not a Kaggle run.

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4` only.
Graphs from private helper `mahdimzafari/biohub-learned-unet-infer-v1` v3.
GEFFs from private helper `mahdimzafari/biohub-geff-export` v1. **Not a
Kaggle run. Not a leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes / edges |
|---|---|---|---|---|---|---|
| learned_unet_infer_v1 | **0.655751** | 0.69001 | 0.65575 | 0.97004 | 0/0/2 | 117692 / 106385 |
| learned_unet_hung_v1 | **0.439023** | 0.46178 | 0.43902 | 0.97004 | 0/0/2 | 117692 / 113664 |

Per-sequence hung: `44b6_12dfb391` 0.500816 (infer 0.747072),
`44b6_267148e4` 0.294111 (infer 0.446373). Detection counts match
locked infer exactly (80741 / 36951). Official matched-edge
TP/FP/FN = 743/559/307 vs infer 877/221/173. Runtime 10.68 s, peak 596 MB.

Decision **`reject_learned_unet_hung_v1`**. Do **not** retune. Frozen
`6bba` was refused because tuning did not keep the candidate.

## Error analysis

The Hungarian linker wrote 1-1 consecutive edges inside the locked 12 µm
last-position gate (max 11.94 µm; 0 edges over the gate; 0 predicted
divisions). Versus last-position greedy it added 7309 extra edges
(113664 vs 106385) and disagreed on 52239 assignments. Full-frame
min-cost assignment reassigns leftover farther-but-allowed detections
after collisions, which raises official FPs and loses TPs on the same
detections. See `reports/learned_unet_hung_v1_error.json`.

## Next

Keep `learned_unet_gap_v1` / `learned_unet_gap_test_v1` as the candidate.
Do **not** retune this Hungarian hash. Do **not** package another
full-frame Hungarian-threshold variant. Do **not** score `6bba` for this
family. If a later run starts a **new** slug, it needs a different
hypothesis than "Hungarian on every last-position 12 µm pair" (for
example, Hungarian only on true shared-nearest collisions). If a public
leaderboard number is wanted, Mahdi submits
`biohub-learned-unet-gap-test-v1` v1 from the Kaggle UI. Do **not**
submit from the CLI.
