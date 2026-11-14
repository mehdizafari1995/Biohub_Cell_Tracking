# learned_unet_conflict_v1

Same experiment family as `learned_unet_v1` (draft PR #7). Do **not**
merge. Do not overwrite `baseline_v1`, `association_v2`, `division_v2`,
`complementary_v1`, `linkable_v1`, `learned_unet_division_v1`, rejected
`learned_unet_link_v1` / `link_v2` / `motion_v1` / `hung_v1`, or locked
learned train/infer/holdout/test/gap-test slugs. Do **not** retune
locked gates. Frozen `6bba` was **not** scored.

## Hypothesis

Rejected full-frame `learned_unet_hung_v1` lost official TPs because
Hungarian reassigned leftover farther-but-allowed pairs after unique
greedy matches. Keeping unique last-position nearest-neighbor matches
and applying Hungarian only when two tracks share the same nearest
neighbor inside the locked association_v2 **12 µm last-position** gate
should resolve true collisions without that leftover rematch
(`learned_unet_infer_v1` **0.655751**) on frozen-tuning `44b6`. Frozen
`6bba` is refused until that 44b6 comparison keeps the candidate.

## Operating point (44b6 only; rejected on tuning)

Config: `configs/learned_unet_conflict_v1.yaml`. Hash `8503d804fb9e9f1d`.
Training hash `4c38035d6657f2ad`. Parent infer `d7578b5f64ddac08`.
Rejected transformer hashes `3a7423c6858ace41` and `1d4015dc499bd604`,
rejected motion hash `66853b6f1445c18b`, and rejected hung hash
`f67bc97d0521dc78` stay locked.

- Infer stems only: `44b6_12dfb391`, `44b6_267148e4`
- Holdout `6bba` is refused
- Detection unchanged (locked infer nodes)
- Linker: keep unique last-position NNs; Hungarian 1-1 only on
  shared-nearest collisions; hard last-position `max_link_um=12.0`,
  `max_gap_frames=0`, no interpolation, no intensity, no transformer,
  no velocity, no full-frame Hungarian
- Writes no `submission.csv`
- Helper slug `mahdimzafari/biohub-learned-unet-conflict-v1` packaged,
  not pushed (local overlay is sufficient)

Do **not** retune this hash. Do **not** score frozen `6bba` for this overlay.

## Cost estimate (before score)

- Storage: reuse infer-helper graphs + two `44b6` GEFFs from
  `mahdimzafari/biohub-geff-export` v1. No volumes, no weights.
- CPU: re-link 117692 locked detections. Measured **19.8 s / 590 MB**.
- No GPU. Not a Kaggle run.

## Official frozen-tuning score (not leaderboard)

Local official scorer `075fc5f5` on `44b6_12dfb391` + `44b6_267148e4` only.
Graphs from private helper `mahdimzafari/biohub-learned-unet-infer-v1` v3.
GEFFs from private helper `mahdimzafari/biohub-geff-export` v1. **Not a
Kaggle run. Not a leaderboard score.**

| candidate | combined | edge J | adj edge J | recall | divisions | pred nodes / edges |
|---|---|---|---|---|---|---|
| learned_unet_infer_v1 | **0.655751** | 0.69001 | 0.65575 | 0.97004 | 0/0/2 | 117692 / 106385 |
| learned_unet_conflict_v1 | **0.645100** | 0.67882 | 0.64510 | 0.97004 | 0/0/2 | 117692 / 108105 |

Per-sequence conflict: `44b6_12dfb391` 0.735753 (infer 0.747072),
`44b6_267148e4` 0.438423 (infer 0.446373). Detection counts match
locked infer exactly (80741 / 36951). Official matched-edge
TP/FP/FN = 875/239/175 vs infer 877/221/173. Runtime 19.80 s, peak 590 MB.

Decision **`reject_learned_unet_conflict_v1`**. Do **not** retune. Frozen
`6bba` was refused because tuning did not keep the candidate.

## Error analysis

The conflict linker kept 88486 unique last-position NNs and ran
Hungarian on 27402 shared-nearest collision sources (19619 collision
links). Gate held (max 11.94 µm; 0 edges over the gate; 0 predicted
divisions). Versus last-position greedy it added 1720 extra edges
(108105 vs 106385) and disagreed on 10320 assignments. Resolving
shared-nearest collisions still lost 2 official TPs and added 18
conflicting FPs on the same detections. Full-frame hung remains worse
(0.439023; 113664 edges). See `reports/learned_unet_conflict_v1_error.json`.

## Next

Keep `learned_unet_gap_v1` / `learned_unet_gap_test_v1` as the candidate.
Do **not** retune this conflict hash. Do **not** package another
collision-Hungarian or full-frame Hungarian-threshold variant. Do
**not** score `6bba` for this family. If a public leaderboard number is
wanted, Mahdi submits `biohub-learned-unet-gap-test-v1` v1 from the
Kaggle UI. Do **not** submit from the CLI.
