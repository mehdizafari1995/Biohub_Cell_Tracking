# Submission ledger

This ledger records **actual** Kaggle kernel runs and competition submissions. Local simulations are labelled as such and are not Kaggle scores. Public leaderboard numbers are never used as the sole basis for parameter tuning. Submissions are limited to deliberate experiments.

Machine-readable copy: `reports/submission_ledger.csv`. Rows are append-only. Previous results are never overwritten.

## baseline_v1_kernel_v1

- **Status:** private kernel `mahdimzafari/biohub-baseline-v1` version **1** failed (~13 s).
- **Error:** `ModuleNotFoundError: No module named 'zarr'` with internet off.
- **Fix:** version 2 reads OME-Zarr v3 chunks via `blosc2` (present on the Kaggle image).
- **Kaggle public / private scores:** not returned.

## baseline_v1_kernel_v2

- **Status:** private kernel version **2** **COMPLETE**. Environment label `kaggle_input`. This is a real Kaggle execution on public `test/`.
- **Config hash:** `d4f79f4af9958c85` (anisotropic DoG + 4 µm NMS + greedy 15 µm, no divisions).
- **Datasets:** `44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`, `6bba_05db0fb1` (exact sample_submission stems).
- **Output:** `/kaggle/working/submission.csv` 11 443 248 bytes, 224 788 rows, 117 515 nodes, 107 273 edges, 0 predicted divisions. Local re-audit of the downloaded file: no errors.
- **Worker:** 595.24 s (1.49 s/frame), peak RAM 435 MB. Decoder was `blosc2 4.1.2`; `zarr`/`imagecodecs`/`numcodecs` were missing.
- **Local official score:** 0.537 on the full frozen validation split (`6bba_05db0fb1` + `6bba_c328f2fd`). The earlier 0.498 figure was `6bba_05db0fb1` alone.
- **Competition submit:** not performed. No public or private leaderboard score was returned.
- **Manual submit:** https://www.kaggle.com/code/mahdimzafari/biohub-baseline-v1 → version 2 → **Submit to Competition**.

## association_v2_kernel_v3

- **Status:** private kernel `mahdimzafari/biohub-association-v2` version **3** **COMPLETE**. Environment label `kaggle_input`. This is a real Kaggle execution on public `test/`.
- **Config hash:** `2c6966b5204ba542` (anisotropic DoG + 4 µm NMS + greedy 12 µm gate + 2-frame gap close + intensity 0.5, interpolated consecutive nodes, no divisions).
- **Datasets:** `44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`, `6bba_05db0fb1` (exact sample_submission stems).
- **Output:** `/kaggle/working/submission.csv` 12 948 601 bytes, 253 773 rows, 129 500 nodes, 124 273 edges, 11 985 interpolated nodes, 0 predicted divisions. Worker audit: `audit_ok`. `submission.csv` was not copied into git.
- **Worker:** 593.53 s (1.48 s/frame), peak RAM 421 MB. Decoder was `blosc2 4.1.2`; `zarr`/`imagecodecs`/`numcodecs` were missing.
- **Local official score:** 0.592 on the full frozen validation split (`6bba_05db0fb1` + `6bba_c328f2fd`). That number is local official, not a leaderboard score.
- **Competition submit:** not performed. No public or private leaderboard score was returned.
- **Manual submit:** https://www.kaggle.com/code/mahdimzafari/biohub-association-v2 → version 3 → **Submit to Competition**.

## association_v2_kernel_v3_independent_audit

- **Status:** downloaded worker `submission.csv` independently audited. This is still not a competition submission.
- **Schema / integrity:** `audit_ok`. 253773 rows, 129500 nodes, 124273 edges, 0 predicted divisions, 0 non-consecutive edges. Integer voxel centroids inside typical `(100, 64, 256, 256)`. Counts match the worker manifest. File was kept under `artifacts/kaggle/` and not committed.
- **Official scorer on public-test overlap `6bba_05db0fb1`:** combined **0.564336**, edge Jaccard 0.5627, adj 0.5643, divisions 0/0/3, recall 0.829. Delta vs local 0.564 is +0.0003. Official commit `075fc5f5`.
- **Not holdout:** `44b6_0113de3b` 0.931, `44b6_0b24845f` 0.280, `6bba_05b6850b` 0.476. Do not retune from these. The four-stem aggregate 0.527 is not a frozen-split score.
- **Still missing:** `6bba_c328f2fd` is not in public test, so the worker CSV cannot complete the frozen split.
- **Kaggle public / private scores:** not returned.

## geff_export_kernel_v1

- **Status:** private helper kernel `mahdimzafari/biohub-geff-export` version **1** **COMPLETE**. Internet off. **Not a competition submission.** No `submission.csv`.
- **Purpose:** zip train `44b6` GEFFs plus two frozen-holdout GEFFs after per-file downloads 429'd.
- **Output:** 73 GEFF stores, 1533 files, no image volumes. Zip kept under ignored `artifacts/kaggle/`.
- **Kaggle public / private scores:** not applicable; not submitted.

## zarr_export_kernel_v1

- **Status:** private helper kernel `mahdimzafari/biohub-holdout-zarr-export` version **1** **COMPLETE**. Internet off. **Not a competition submission.** No `submission.csv`.
- **Purpose:** zip only frozen-holdout `6bba_05db0fb1.zarr` and `6bba_c328f2fd.zarr` after per-file downloads 429'd.
- **Output:** 2 Zarr stores, 204 files, 993 198 228 byte inner zip, 13.43 s worker. Zip kept under ignored `artifacts/kaggle/`.
- **Kaggle public / private scores:** not applicable; not submitted.

## holdout_pair_association_v2_vs_division_v2

- **Status:** local official scorer on both frozen validation sequences after the helper kernel. **Not a Kaggle run.**
- **association_v2:** combined 0.591570, 0/0/5 divisions. Matches the locked 0.592 figure.
- **division_v2:** combined 0.591570, 0/0/5 divisions. No regression. No extra false forks. Keep as no-op overlay. Do not retune.
- **Kaggle public / private scores:** not returned.

## association_v2_local / division_v1_local

Local official scores only. association_v2 later received a private kernel (see above). division_v1 did not (0.590, 0/3/5 divisions) and has no Kaggle run.

## complementary_v1_family

- **Status:** experiment family on `cursor/complementary-detection-v1`. Locked from 44b6, scored once on frozen 6bba, **rejected**. Do not retune.
- **Helper:** private kernel `mahdimzafari/biohub-tuning-zarr-export` version **1 COMPLETE**. Zipped only `44b6_12dfb391.zarr` and `44b6_267148e4.zarr`. Worker archive 919 MB under ignored `artifacts/kaggle/`. **Not a submission.**
- **Local official tuning:** complementary_v1 **0.633135** vs locked association_v2 **0.632460** on both frozen-tuning sequences. Decision `keep_candidate_complementary_v1`. Per-sequence mixed. Not a leaderboard score.
- **Kaggle public / private scores:** not returned.

## complementary_v1_holdout

- **Status:** local official scorer on both frozen validation sequences after reusing `biohub-holdout-zarr-export` v1. **Not a Kaggle run.**
- **association_v2:** combined **0.591570**, recall 0.838, 0/0/5 divisions, 96336 pred nodes. Matches the locked 0.591570 figure.
- **complementary_v1:** combined **0.591384**, recall 0.838, 0/0/5 divisions, 96648 pred nodes. Extra T_pred lowered adj-edge Jaccard. Decision `reject_complementary_v1`. Do not retune.
- **Kaggle public / private scores:** not returned.

## linkable_v1_family

- **Status:** experiment family on `cursor/linkable-complementary-v1`. Learned UNet estimate written first (`docs/learned_detection_estimate.md`); GPU blocked. CPU fallback locked from 44b6, scored once on frozen 6bba, **rejected**. Do not retune.
- **Local official tuning:** linkable_v1 **0.633168** vs locked association_v2 **0.632460** on both frozen-tuning sequences. Decision `keep_candidate_linkable_v1`. Hash `fa975421edb6136e`. Not a leaderboard score.
- **Kaggle public / private scores:** not returned.

## linkable_v1_holdout

- **Status:** local official scorer on both frozen validation sequences after reusing `biohub-holdout-zarr-export` v1. **Not a Kaggle run.**
- **association_v2:** combined **0.591570**, recall 0.838, 0/0/5 divisions, 96336 pred nodes. Matches the locked 0.591570 figure.
- **linkable_v1:** combined **0.591458**, recall 0.838, 0/0/5 divisions, 96525 pred nodes. Extra T_pred (+189) lowered adj-edge Jaccard. Decision `reject_linkable_v1`. Do not retune.
- **Kaggle public / private scores:** not returned.

## learned_unet_v1_kernel

- **Status:** private GPU helper `mahdimzafari/biohub-learned-unet-v1`. Internet off. **Not a competition submission.** No `submission.csv`.
- **Version 1:** error after writing a 44b6-only split (69 train / 2 frozen-tuning internal val, no 6bba). GPU T4 x2 and torch 2.10+cu128 were present.
- **Version 2:** `train_error` `ValueError: unsupported zarr codec 'zstd'` on train volumes. Split unchanged.
- **Version 3:** **COMPLETE / trained**. Collector downloaded into ignored `artifacts/kaggle/learned_unet_v1`. 3 epochs, 6117 windows, best train loss 0.3604, 2.08M params, 11732 s, peak 1793 MB, T4 x2, torch 2.10+cu128. Split 69/2 `44b6` only. Hash `4c38035d6657f2ad`. No `submission.csv`. Weights stay out of git. Not a leaderboard score.
- **2026-09-20T23:52:33Z status-only reconfirm:** still **COMPLETE / trained**. Five outputs unchanged (`dataset_splits.json`, `learned_unet_v1_config.json`, `train_manifest.json`, best/last weights). Did not re-download. Did not push a replacement. Not a submission.
- **Kaggle public / private scores:** not applicable; not submitted.

## learned_unet_infer_v1_kernel

- **Status:** private GPU helper `mahdimzafari/biohub-learned-unet-infer-v1`. Internet off. **Not a competition submission.** No `submission.csv`.
- **Config hash:** `d7578b5f64ddac08`. UNet local-max + 12 µm greedy 1-1 on frozen-tuning `44b6` only. Holdout refused.
- **Version 1:** COMPLETE / `blocked_no_weights` (5.08 s). Hard-coded training mount missed.
- **Version 2:** COMPLETE / `blocked_no_weights` (3.74 s). `/kaggle/input` is `{competitions, notebooks}`. Push `invalidKernelSources` empty.
- **Version 3:** **COMPLETE / inferred**. 187.0 s, peak 1136 MB, T4 x2. Graphs for `44b6_12dfb391` (80741 nodes) and `44b6_267148e4` (36951 nodes). Weights path `/kaggle/input/notebooks/mahdimzafari/biohub-learned-unet-v1/learned_unet_v1_last.pt`.
- **Local official tuning:** learned_unet_infer_v1 **0.655751** vs locked association_v2 **0.632460**. Decision `keep_candidate_learned_unet_infer_v1`. Recall 0.970, edge Jaccard 0.690, 0/0/2 divisions. Not a leaderboard score. Do not look at `6bba` yet. Do not overwrite this slug.
- **Kaggle public / private scores:** not applicable; not submitted.

## learned_unet_holdout_v1_kernel

- **Status:** private GPU helper `mahdimzafari/biohub-learned-unet-holdout-v1` version **1 COMPLETE** / `inferred`. Internet off. **Not a competition submission.** No `submission.csv`.
- **Config hash:** `d374a21f03dddc04`. Locked infer operating point `d7578b5f64ddac08` on frozen-holdout `6bba` only. `44b6` volumes refused.
- **Worker:** 184.4 s, peak RAM 1177 MB, T4 x2. Graphs: `6bba_05db0fb1` 97062 nodes / 89394 edges; `6bba_c328f2fd` 52161 / 47595. Weights path `/kaggle/input/notebooks/mahdimzafari/biohub-learned-unet-v1/learned_unet_v1_last.pt`.
- **Local official holdout:** learned_unet_holdout_v1 **0.693605** vs locked association_v2 **0.591570**. Decision `keep_candidate_learned_unet_holdout_v1`. Recall 0.957, 0/0/5 divisions. Per-sequence mixed. Not a leaderboard score. Do not retune. Do not score `6bba` again. Do not overwrite this slug.
- **Kaggle public / private scores:** not applicable; not submitted.

## learned_unet_test_v1_kernel

- **Status:** private GPU test-time kernel `mahdimzafari/biohub-learned-unet-test-v1` version **1 COMPLETE** / `inferred`. Internet off. Environment `kaggle_input`. Writes `submission.csv`. **Not a competition submission.**
- **Config hash:** `7a8725b818fc609b`. Locked infer `d7578b5f64ddac08` / holdout `d374a21f03dddc04` / train `4c38035d6657f2ad`. Discovers every `test/*.zarr`. Never opens `train/` volumes.
- **Datasets:** `44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`, `6bba_05db0fb1` (exact sample_submission stems).
- **Output:** `/kaggle/working/submission.csv` 23 025 421 bytes, 440 254 rows, 231 707 nodes, 208 547 edges, 0 predicted divisions. Worker audit `audit_ok`. Independent stream audit `audit_ok`. File kept under ignored `artifacts/kaggle/`.
- **Worker:** 306.53 s (0.77 s/frame), peak RAM 1184 MB. Weights `/kaggle/input/notebooks/mahdimzafari/biohub-learned-unet-v1/learned_unet_v1_last.pt`.
- **Consistency:** `6bba_05db0fb1` 97062/89394 matches the locked holdout helper graphs. Frozen `6bba` was not official-scored again.
- **Estimate vs measured:** public 4 volumes estimated 368.8 s; measured 306.5 s. Hidden-test estimate remains ~5.1 h.
- **Kaggle public / private scores:** not returned. Do **not** submit from the CLI.
- **Manual submit:** Prefer the gap slug after it completes. If submitting the no-gap kernel: https://www.kaggle.com/code/mahdimzafari/biohub-learned-unet-test-v1 → version 1 → **Submit to Competition**.

## learned_unet_gap_test_v1_kernel

- **Status:** private GPU test-time kernel `mahdimzafari/biohub-learned-unet-gap-test-v1` version **1 COMPLETE** / `inferred`. Internet off. Environment `kaggle_input`. Writes `submission.csv`. **Not a competition submission.**
- **Config hash:** `85379c198ebc7241`. Locked infer `d7578b5f64ddac08` / holdout `d374a21f03dddc04` / gap overlay `f4d888d190c45a4a` / no-gap test parent `7a8725b818fc609b` / train `4c38035d6657f2ad`. Discovers every `test/*.zarr`. Never opens `train/` volumes.
- **Datasets:** `44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`, `6bba_05db0fb1` (exact sample_submission stems).
- **Output:** `/kaggle/working/submission.csv` 25 891 562 bytes, 494 072 rows, 251 660 nodes, 242 412 edges, 13 912 gap links, 19 953 interpolated nodes, 0 predicted divisions. Worker audit `audit_ok`. Independent stream audit `audit_ok`. File kept under ignored `artifacts/kaggle/`.
- **Worker:** 330.02 s (0.83 s/frame), peak RAM 1184 MB. Weights `/kaggle/input/notebooks/mahdimzafari/biohub-learned-unet-v1/learned_unet_v1_last.pt`.
- **Consistency:** `6bba_05db0fb1` 103432/100197 matches the locked holdout gap overlay (97062+6370 / 89394+10803). Frozen `6bba` was not official-scored again.
- **Estimate vs measured:** public 4 volumes estimated 388 s; measured 330.0 s. Hidden-test estimate remains ~5.4 h.
- **Do not overwrite:** this slug or `mahdimzafari/biohub-learned-unet-test-v1` v1.
- **Kaggle public / private scores:** not returned. Do **not** submit from the CLI.
- **Manual submit:** https://www.kaggle.com/code/mahdimzafari/biohub-learned-unet-gap-test-v1 → version 1 → **Submit to Competition**.

## learned_unet_division_v1_tune

- **Status:** local official scorer on locked learned infer graphs + locked `division_v2` gates. **Not a Kaggle run.** No new kernel. Frozen `6bba` was **not** scored.
- **Config hash:** `8761fb66cc2254f3`. Parent infer `d7578b5f64ddac08`. Division gates `0fa756963e5d42d1`. Intensity unused.
- **Local official tuning:** learned_unet_division_v1 **0.655235** vs locked learned_unet_infer_v1 **0.655751**. Extra edges 282. Divisions 0/1/2 vs 0/0/2. Recall unchanged. Decision `reject_learned_unet_division_v1`. Do not retune. Do not score holdout.
- **Kaggle public / private scores:** not returned.

## learned_unet_gap_v1_tune

- **Status:** local official scorer on locked learned infer graphs + locked `association_v2` gap-close overlay. **Not a Kaggle run.** No new kernel. Frozen `6bba` was **not** scored in this row.
- **Config hash:** `f4d888d190c45a4a`. Parent infer `d7578b5f64ddac08`. Existing dt=1 links kept. Intensity unused.
- **Local official tuning:** learned_unet_gap_v1 **0.658452** vs locked learned_unet_infer_v1 **0.655751**. Extra edges 16852. Extra interpolated nodes 9879. Recall 0.97667 vs 0.97004. Divisions 0/0/2. Per-sequence mixed (`12dfb391` 0.745296 vs 0.747072; `267148e4` 0.460171 vs 0.446373). Decision `keep_candidate_learned_unet_gap_v1`. Do not retune.
- **Kaggle public / private scores:** not returned.

## learned_unet_gap_v1_holdout

- **Status:** local official scorer on locked learned holdout graphs + locked `association_v2` gap-close overlay. **Not a Kaggle run.** No new kernel. One-shot only.
- **Config hash:** `f4d888d190c45a4a`. Parent holdout `d374a21f03dddc04`. Existing dt=1 links kept. Intensity unused.
- **Local official holdout:** learned_unet_gap_v1 **0.701026** vs locked learned_unet_holdout_v1 **0.693605**. Extra edges 16804. Extra interpolated nodes 9908. Recall 0.97588 vs 0.95713. Divisions 0/0/5. Per-sequence mixed (`6bba_05db0fb1` 0.718133 vs 0.719310; `6bba_c328f2fd` 0.661175 vs 0.633142). Decision `keep_candidate_learned_unet_gap_v1`. Do not retune. Do not score holdout again.
- **Kaggle public / private scores:** not returned.

## learned_unet_link_v1_kernel

- **Status:** private GPU helper `mahdimzafari/biohub-learned-unet-link-v1` version **1 COMPLETE / inferred**. Internet off. **Not a competition submission.** No `submission.csv`.
- **Config hash:** `3a7423c6858ace41`. Locked learned detections + trained transformer softmax 1-1 on frozen-tuning `44b6` only. Holdout refused.
- **Worker:** 162.1 s, peak RAM 1210 MB, T4 x2. Graphs `44b6_12dfb391` 80741/0 and `44b6_267148e4` 36951/0. Detection counts match locked infer.
- **Official 44b6 score:** **0.0** vs locked `learned_unet_infer_v1` **0.655751**. Decision `reject_learned_unet_link_v1`. Softmax+0.5 wrote 0 edges. Frozen `6bba` not opened. Not a leaderboard score.
- **Kaggle public / private scores:** not applicable; not submitted.

## learned_unet_link_v2_packaged

- **Status:** private GPU helper `mahdimzafari/biohub-learned-unet-link-v2` version **1 running** at package time. Later completed; see `learned_unet_link_v2_tune`. Internet off. **Not a competition submission.** No `submission.csv`.
- **Config hash:** `1d4015dc499bd604`. Locked learned detections + transformer greedy 1-1 on descending softmax with no 0.5 cutoff and the locked 12 µm gate. 44b6 only. Holdout refused.
- **Estimate:** ~5.1 min / 2 volumes; ~8.5 h if later promoted to ~199 hidden volumes. Peak RAM budget 2.5 GB. Under the 12 h GPU limit.
- **Kaggle public / private scores:** not applicable; not submitted.

## learned_unet_link_v2_tune

- **Status:** private GPU helper `mahdimzafari/biohub-learned-unet-link-v2` version **1 COMPLETE / inferred**. Internet off. **Not a competition submission.** No `submission.csv`.
- **Config hash:** `1d4015dc499bd604`. Locked learned detections + transformer greedy 1-1 on descending softmax with no 0.5 cutoff and the locked 12 µm gate. 44b6 only. Holdout refused.
- **Worker:** 135.1 s, peak RAM 1251 MB, T4 x2. Graphs `44b6_12dfb391` 80741/72532 and `44b6_267148e4` 36951/31819. Detection counts match locked infer.
- **Official 44b6 score:** **0.224746** vs locked `learned_unet_infer_v1` **0.655751**. Decision `reject_learned_unet_link_v2`. Transformer ranking among 12 µm candidates lost official TPs (475 vs 877) and added conflicting FPs (962 vs 221). Frozen `6bba` not opened. Not a leaderboard score.
- **Kaggle public / private scores:** not applicable; not submitted.

## learned_unet_motion_v1_tune

- **Status:** local official scorer on locked learned infer graphs + constant-velocity 1-1 inside the locked last-position 12 µm gate. **Not a Kaggle run.** Helper packaged, not pushed. Frozen `6bba` was **not** scored.
- **Config hash:** `66853b6f1445c18b`. Parent infer `d7578b5f64ddac08`. Intensity unused. Transformer unused.
- **Local official tuning:** learned_unet_motion_v1 **0.583722** vs locked learned_unet_infer_v1 **0.655751**. Detection counts match (80741 / 36951). Official TP/FP/FN 828/300/222 vs infer 877/221/173. Divisions 0/0/2. Recall unchanged 0.970. Decision `reject_learned_unet_motion_v1`. Do not retune. Do not score holdout.
- **Kaggle public / private scores:** not returned.

## learned_unet_hung_v1_tune

- **Status:** local official scorer on locked learned infer graphs + Hungarian 1-1 inside the locked last-position 12 µm gate. **Not a Kaggle run.** Helper packaged, not pushed. Frozen `6bba` was **not** scored.
- **Config hash:** `f67bc97d0521dc78`. Parent infer `d7578b5f64ddac08`. Intensity unused. Transformer unused. Velocity unused.
- **Local official tuning:** learned_unet_hung_v1 **0.439023** vs locked learned_unet_infer_v1 **0.655751**. Detection counts match (80741 / 36951). Official TP/FP/FN 743/559/307 vs infer 877/221/173. Extra leftover-allowed edges +7309. Divisions 0/0/2. Recall unchanged 0.970. Decision `reject_learned_unet_hung_v1`. Do not retune. Do not score holdout.
- **Independent 44b6 reconfirm (2026-09-21T04:12Z):** combined **0.4390229473498808** matches the locked tune JSON exactly. Did not overwrite that report. Frozen `6bba` stayed closed. Helper still packaged, not pushed.
- **Kaggle public / private scores:** not returned.

## learned_unet_conflict_v1_tune

- **Status:** local official scorer on locked learned infer graphs + unique last-position NNs and Hungarian only on shared-nearest collisions inside the locked last-position 12 µm gate. **Not a Kaggle run.** Helper packaged, not pushed. Frozen `6bba` was **not** scored.
- **Config hash:** `8503d804fb9e9f1d`. Parent infer `d7578b5f64ddac08`. Intensity unused. Transformer unused. Velocity unused. Full-frame Hungarian unused.
- **Local official tuning:** learned_unet_conflict_v1 **0.645100** vs locked learned_unet_infer_v1 **0.655751**. Detection counts match (80741 / 36951). Official TP/FP/FN 875/239/175 vs infer 877/221/173. Extra collision second-choice edges +1720. Divisions 0/0/2. Recall unchanged 0.970. Decision `reject_learned_unet_conflict_v1`. Do not retune. Do not score holdout.
- **Kaggle public / private scores:** not returned.

## association_v2_kernel_v3_lb_reconfirm

- **Status:** read-only competition submission list at **2026-09-21T02:50:39Z**. Did not submit. Did not download a CSV.
- **Ref:** **56399003**. Kernel `mahdimzafari/biohub-association-v2` version 3. Submitted **2026-09-20T15:12:52.307Z**. Status **complete**.
- **Public score:** API string **`0.553`**. Private score **not returned**. Do not invent extra digits. Do not use 0.553 to retune. This is **not** a `learned_unet_gap_test_v1` score.
- **Config hash:** `2c6966b5204ba542`.

## learned_unet_gap_test_v1_submit_poll

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-gap-test-v1` at **2026-09-21T02:50:39Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit.
- **Config hash:** `85379c198ebc7241`.
- **Competition list:** still exactly one row (association_v2 **56399003**). Gap-test lastRunTime **2026-09-20T18:41:51Z** is after that submission, so 56399003 cannot be the gap worker.
- **Decision:** `human_action_required_ui_submit`.
- **Kaggle public / private scores for this slug:** not returned.
- **2026-09-21T03:59:39Z status-only reconfirm:** kernel still complete; competition list still exactly one row. Did not download CSV. Did not submit.

## learned_unet_prune_v1_tune

- **Status:** local official scorer on locked learned infer graphs + locked gap overlay + drop degree-0 leftovers. **Not a Kaggle run.** No new kernel. Frozen `6bba` was **not** scored in this row.
- **Config hash:** `115753ea16aaf021`. Parent gap `f4d888d190c45a4a`. Parent infer `d7578b5f64ddac08`.
- **Local official tuning:** learned_unet_prune_v1 **0.662081** vs locked learned_unet_gap_v1 **0.658452**. Dropped 425 isolates. Divisions 0/0/2. Decision `keep_candidate_learned_unet_prune_v1`. Not a leaderboard score.

## learned_unet_prune_v1_holdout

- **Status:** local official scorer on locked learned holdout graphs + locked gap overlay + drop degree-0 leftovers. **Not a Kaggle run.** No new kernel. One-shot only.
- **Config hash:** `115753ea16aaf021`. Parent holdout `d374a21f03dddc04`. Parent gap `f4d888d190c45a4a`.
- **Local official holdout:** learned_unet_prune_v1 **0.701378** vs locked learned_unet_gap_v1 **0.701026**. Dropped 491 isolates. Edge Jaccard unchanged 0.74335. Divisions 0/0/5. Decision `keep_candidate_learned_unet_prune_v1`. Do not retune. Do not score holdout again. Not a leaderboard score.

## learned_unet_prune_test_v1_kernel

- **Status:** private GPU test-time kernel `mahdimzafari/biohub-learned-unet-prune-test-v1` version **1 COMPLETE** / `inferred`. Internet off. Environment `kaggle_input`. Writes `submission.csv`. **Not a competition submission.**
- **Config hash:** `074a71dec68304e8`. Locked infer `d7578b5f64ddac08` / holdout `d374a21f03dddc04` / gap overlay `f4d888d190c45a4a` / prune `115753ea16aaf021` / gap-test parent `85379c198ebc7241` / no-gap test parent `7a8725b818fc609b` / train `4c38035d6657f2ad`. Discovers every `test/*.zarr`. Never opens `train/` volumes.
- **Datasets:** `44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`, `6bba_05db0fb1` (exact sample_submission stems).
- **Output:** `/kaggle/working/submission.csv` 25 849 389 bytes, 493 276 rows, 250 864 nodes, 242 412 edges, 13 912 gap links, 19 953 interpolated nodes, **796 dropped isolates**, 0 predicted divisions. Worker audit `audit_ok`. Independent stream audit `audit_ok`. File kept under ignored `artifacts/kaggle/`.
- **Worker:** 348.18 s (0.87 s/frame), peak RAM 1202 MB. T4 x2, torch 2.10+cu128. Weights `/kaggle/input/notebooks/mahdimzafari/biohub-learned-unet-v1/learned_unet_v1_last.pt`.
- **Vs gap-test:** edges unchanged (242412). Nodes 251660 → 250864. Per-stem edges match locked gap-test, including `6bba_05db0fb1` 100197. Frozen `6bba` was not official-scored again.
- **Estimate vs measured:** public 4 volumes estimated 390 s; measured 348.2 s. Hidden-test projection from rehearsal **4.81 h** (199 × 87.05 s), under 12 h.
- **Do not overwrite:** this slug, `mahdimzafari/biohub-learned-unet-gap-test-v1` v1, or `mahdimzafari/biohub-learned-unet-test-v1` v1.
- **Kaggle public / private scores:** not returned. Competition list still exactly association_v2 **56399003**. Do **not** submit from the CLI.
- **Manual submit:** https://www.kaggle.com/code/mahdimzafari/biohub-learned-unet-prune-test-v1 → version 1 → **Submit to Competition**.

## learned_unet_orgop_v1_tune

- **Status:** private GPU helper `mahdimzafari/biohub-learned-unet-orgop-v1` version **1 COMPLETE / inferred**, then local official scorer on frozen-tuning `44b6` only. **Not a competition submission.** Frozen `6bba` was not opened.
- **Config hash:** `250c5bfacd4a8198`. Train hash `4c38035d6657f2ad`. Weights `learned_unet_v1_last.pt` from the locked train helper. No public 199-video checkpoints.
- **Helper:** 153.71 s, peak 1227 MB, T4. Internet off. No `submission.csv`. 91152 nodes / 0 edges / 0 softmax>0.5 candidates.
- **Local official tuning:** combined **0.0 vs infer 0.655751**. Divisions 0/0/2. Solver `ilp_skipped_no_candidates`. Decision `reject_learned_unet_orgop_v1`. Not a leaderboard score.

## learned_unet_orgop_zero_edge_diagnosis

- **Status:** read-only checkpoint/graph diagnosis. **No new kernel. No competition submission.** Frozen `6bba` was not opened.
- **Finding:** `best.pt` tensor-equals `last.pt`. Strict load, 0 missing transformer keys. UNet rel-to-init **6.34** vs transformer **0.11**. Random 400×80 `prob_max=0.012`. Decision `no_implementation_defect_transformer_near_init`. Do **not** launch `learned_unet_orgop_fix_v1`.
- **Kaggle public / private scores:** not returned.

## learned_unet_edge_train_canary_v1

- **Status:** local CPU canary hash `1a3d524e4d922701`. **No kernel pushed.** Frozen `6bba` was not opened. No `submission.csv`.
- **Result:** `canary_ok`. Finite det/edge losses, nonzero edge grads, 12/12 pairs >0.5, reload max abs 0.0, 0.60 s.
- **Projection:** 10-epoch 44b6 retrain **10.86 h** under 12 h; 50-epoch organizer scale 54.31 h over. Next family `learned_unet_edge_train_v1` is **not launched**.
- **Kaggle public / private scores:** not returned.

## public_094_parity_audit

- **Status:** read-only strategy audit. **No kernel launched. No competition submission.**
- **Proposed config (audit-time):** `learned_unet_orgop_v1` hash `250c5bfacd4a8198` (organizer `det_threshold=0.99` + official ILP on locked 44b6-only weights). Later authorized cycle rejected on 44b6 at **0.0**; see `learned_unet_orgop_v1_tune`.
- **Public 0.94 stack:** organizer 0.99 + ILP plus Pilkwang/Reyhan 199-video TemporalUNet / DeepCenter / seed-314159 pack. Rules-legal public data; **method-rejected** for the candidate (discussion 730160 leak).
- **0.9462:** not independently returned as an API `publicScore`. Nearby author-reported claims: 0.941 lineage, 0.942 one-knob, 0.948 title reproduced at 0.935. Rule-based 716952 ceiling 0.826.
- **Public 0.553:** association_v2 **56399003** only. Not used for tuning.
- **Decision:** specify orgop; do not launch this cycle. Champion remains `learned_unet_prune_v1` / prune-test v1 `READY_FOR_MANUAL_SUBMISSION_REVIEW`.
- **Kaggle public / private scores:** not returned.

## learned_unet_prune_test_v1_submit_poll

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-prune-test-v1` at **2026-09-21T04:48:29Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit.
- **Config hash:** `074a71dec68304e8`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**). Prune-test last recorded complete poll **2026-09-21T04:28:27Z** is after that submission, so 56399003 cannot be the prune worker.
- **Decision:** `human_action_required_ui_submit`.
- **Kaggle public / private scores for this slug:** not returned.

## public_094_discussion_inventory

- **Status:** read-only topics API follow-up. **No kernel launched. No competition submission.**
- **738217:** independently retrieved title **focus3d : one of the best 3d cell segmentation**; 36 votes; 50 comments; posted 2026-08-30. HTML body still an SPA shell. **Not used as a source.** Not part of `learned_unet_orgop_v1`.
- **740392:** still absent from the first hot page (20 of 112). Pagination params ignored. Kernel search 0 hits. **Not used.**
- **Orgop (inventory-time):** hash `250c5bfacd4a8198` was still specified. Later authorized cycle rejected on 44b6 at **0.0**.
- **Kaggle public / private scores:** not returned.

## learned_unet_prune_test_v1_submit_poll (2026-09-21T05:47:28Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-prune-test-v1` at **2026-09-21T05:47:28Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit.
- **Config hash:** `074a71dec68304e8`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**.
- **Kaggle public / private scores for this slug:** not returned.

## learned_unet_retrain_v1_specified

- **Status:** specified, not launched. **No kernel pushed. No competition submission.**
- **Config hash:** `92a419926fa7e40b`. Parent train `4c38035d6657f2ad`. Reserved slug `mahdimzafari/biohub-learned-unet-retrain-v1`. `LAUNCH_AUTHORIZED=False`.
- **Hypothesis:** +3-epoch 44b6-only continuation of locked `learned_unet_v1_last.pt` after orgop reject. Public 199-video weights refused.
- **Estimate:** 3.10–3.75 h GPU T4 x2 from parent 11732 s / 3 epochs. Under 12 h. Hidden-test not projected.
- **Acceptance (after authorized launch):** official 44b6 vs infer **0.655751**; reject and keep `6bba` closed unless strictly better; then one-shot vs prune **0.701378**.
- **Kaggle public / private scores:** not returned. Public 0.553 unused.

## learned_unet_prune_test_v1_submit_poll (2026-09-21T06:54:02Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-prune-test-v1` at **2026-09-21T06:54:02Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit.
- **Config hash:** `074a71dec68304e8`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**.
- **Kaggle public / private scores for this slug:** not returned.

## learned_unet_retrain_v1_packaged

- **Status:** packaged, not launched. **No kernel pushed. No competition submission.**
- **Config hash:** `92a419926fa7e40b`. Parent train `4c38035d6657f2ad`. Reserved slug `mahdimzafari/biohub-learned-unet-retrain-v1`. `LAUNCH_AUTHORIZED=False`.
- **Notebook:** `notebooks/kaggle_learned_unet_retrain_v1.py`. In `KERNELS`. `push_kaggle_kernel.py` refuses until launch is authorized. Does not overwrite `mahdimzafari/biohub-learned-unet-v1`.
- **Hypothesis:** +3-epoch 44b6-only continuation of locked `learned_unet_v1_last.pt` after orgop reject. Public 199-video weights refused.
- **Estimate:** 3.10–3.75 h GPU T4 x2 from parent 11732 s / 3 epochs. Under 12 h. Hidden-test not projected.
- **Acceptance (after authorized launch):** official 44b6 vs infer **0.655751**; reject and keep `6bba` closed unless strictly better; then one-shot vs prune **0.701378**.
- **Kaggle public / private scores:** not returned. Public 0.553 unused.

## public_094_discussion_inventory (2026-09-21T07:51:24Z)

- **Status:** read-only topics API plus by-id lookup. **No kernel launched. No competition submission.**
- **740392:** `GET /topics/740392` HTTP **404** (`topic_id_not_found`). Absent from hot page 20/112. Kernel search 0. HTML SPA generic title. **Not used as a source.** Do not retry hourly.
- **738217:** still independently titled focus3d; unused.
- **Kaggle public / private scores:** not returned.

## learned_unet_prune_test_v1_submit_poll (2026-09-21T07:51:05Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-prune-test-v1` at **2026-09-21T07:51:05Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit.
- **Config hash:** `074a71dec68304e8`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**.
- **Kaggle public / private scores for this slug:** not returned.

## learned_unet_prune_test_v1_submit_poll (2026-09-21T08:47:44Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-prune-test-v1` at **2026-09-21T08:47:44Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** retry 740392. Did **not** launch retrain.
- **Config hash:** `074a71dec68304e8`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**.
- **Kaggle public / private scores for this slug:** not returned.

## learned_unet_prune_test_v1_submit_poll (2026-09-21T09:45:57Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-prune-test-v1` at **2026-09-21T09:45:57Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** retry 740392. Did **not** launch retrain.
- **Config hash:** `074a71dec68304e8`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**.
- **Kaggle public / private scores for this slug:** not returned.

## learned_unet_orgdet_v1_tune

- **Status:** local official scorer on frozen-tuning `44b6_12dfb391` + `44b6_267148e4`. **Not a Kaggle run. Not a leaderboard score.**
- **Config hash:** `80b20589198148b7`. Parent infer `d7578b5f64ddac08`. Rejected orgop `250c5bfacd4a8198`.
- **Hypothesis:** orgop failed on empty transformer edges; greedy 12 µm on the same 0.99 detections should isolate detection.
- **Official 44b6:** **0.679578 vs infer 0.655751**. Recall 0.97004. TP/FP/FN **877/220/173** vs infer **877/221/173**. Nodes **91152** vs **117692**. 0/0/2 divisions. Per-sequence 0.772105 / 0.467676.
- **Decision:** `keep_candidate_learned_unet_orgdet_v1`. Frozen `6bba` later scored once via `learned_unet_orgdet_holdout_v1` (0.705226). Helper packaged, not pushed.
- **Kaggle public / private scores:** not returned.

## learned_unet_orgdet_holdout_v1_kernel_v1

- **Status:** private helper kernel `mahdimzafari/biohub-learned-unet-orgdet-holdout-v1` version **1 COMPLETE / inferred**. Environment label `kaggle_input`. GPU T4 x2, internet off. **Not a competition submission.** No `submission.csv`.
- **Config hash:** `912f0c19b572a054` (locked orgdet 0.99 / pool 3.0 / TTA + greedy 12 µm on frozen 6bba). Parent orgdet `80b20589198148b7`. Train `4c38035d6657f2ad`.
- **Worker:** 173.80 s, peak RAM 1185 MB. Weights `learned_unet_v1_last.pt`. Graphs `6bba_05db0fb1` 84328/78493 and `6bba_c328f2fd` 37482/35147.
- **Local official holdout:** **0.705226 vs prune 0.701378**. Recall 0.939, 0/0/5 divisions. Nodes 121810 vs prune 158640. Decision `keep_candidate_learned_unet_orgdet_holdout_v1`. Not a leaderboard score. Do not retune. Do not score `6bba` again. Do not overwrite this slug.
- **Kaggle public / private scores:** not returned.

## learned_unet_prune_test_v1_submit_poll (2026-09-21T10:45:23Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-prune-test-v1` at **2026-09-21T10:45:23Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** retry 740392. Did **not** launch retrain.
- **Config hash:** `074a71dec68304e8`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**.
- **Kaggle public / private scores for this slug:** not returned.

## learned_unet_orgdet_test_v1_packaged

- **Status:** packaged private GPU public-test notebook `mahdimzafari/biohub-learned-unet-orgdet-test-v1`. **Not pushed.** **Not a competition submission.**
- **Config hash:** `c45c859610d29e90`. Parent orgdet holdout `912f0c19b572a054`. Locked 0.99 / pool 3.0 / TTA + greedy 12 µm. Discovers `test/*.zarr`. Writes `submission.csv`.
- **Estimate:** 86.9 s/volume from orgdet-holdout v1. Public 4 volumes ~5.8 min. Hidden ~199 volumes **~4.8 h** under 12 h.
- **Kaggle public / private scores:** not returned.

## learned_unet_orgdet_test_v1_kernel_v1

- **Status:** private GPU test-time kernel `mahdimzafari/biohub-learned-unet-orgdet-test-v1` version **1 COMPLETE / inferred**. Environment label `kaggle_input`. GPU T4 x2, internet off. Worker `audit_ok`. Independent stream `audit_ok`. **Not submitted.**
- **Config hash:** `c45c859610d29e90` (locked orgdet holdout 0.99 / pool 3.0 / TTA + greedy 12 µm on discovered `test/*.zarr`). Parent holdout `912f0c19b572a054`.
- **Worker:** 334.23 s, peak RAM 1183 MB. Weights `learned_unet_v1_last.pt`. Public test 188509 nodes / 172148 edges / 0 divisions. `6bba_05db0fb1` 84328/78493 matches locked orgdet holdout. Hidden projection **4.62 h** under 12 h.
- **Kaggle public / private scores:** not returned. Competition list still association_v2 **56399003** / **0.553** only.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-21T11:48:44Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-21T11:48:44Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** retry 740392. Did **not** launch retrain.
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-test v1, not prune-test or gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-21T12:44:57Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-21T12:44:57Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** retry 740392. Did **not** launch retrain.
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-test v1, not prune-test or gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-21T13:51:02Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-21T13:51:02Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** retry 740392. Did **not** launch retrain.
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-test v1, not prune-test or gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-21T14:53:35Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-21T14:53:35Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** retry 740392. Did **not** launch retrain.
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-test v1, not prune-test or gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-21T15:40:47Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-21T15:40:47Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** retry 740392. Did **not** launch retrain.
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-test v1, not prune-test or gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-21T16:50:55Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-21T16:50:55Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** retry 740392. Did **not** launch retrain. Did **not** duplicate the GitHub SCM agent's orgop diagnosis (already landed: `no_implementation_defect_transformer_near_init`, canary `1a3d524e4d922701` `canary_ok`).
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-test v1, not prune-test or gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-21T17:49:37Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-21T17:49:37Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** retry 740392. Did **not** launch retrain.
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-test v1, not prune-test or gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-21T18:40:39Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-21T18:40:39Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** retry 740392. Did **not** launch retrain.
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-test v1, not prune-test or gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-21T19:42:57Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-21T19:42:57Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** retry 740392. Did **not** launch retrain.
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-test v1, not prune-test or gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-21T20:41:02Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-21T20:41:02Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** retry 740392. Did **not** launch retrain.
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-test v1, not prune-test or gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-21T21:51:07Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-21T21:51:07Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** retry 740392. Did **not** launch retrain.
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-test v1, not prune-test or gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-21T22:56:15Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-21T22:56:15Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** retry 740392. Did **not** launch retrain.
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-test v1, not prune-test or gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-21T23:43:20Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-21T23:43:20Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** retry 740392. Did **not** launch retrain.
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-test v1, not prune-test or gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-22T00:49:23Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-22T00:49:23Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** retry 740392. Did **not** launch retrain.
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-test v1, not prune-test or gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-22T01:48:51Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-22T01:48:51Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** retry 740392. Did **not** launch retrain.
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-test v1, not prune-test or gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-22T02:47:03Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-22T02:47:03Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** retry 740392. Did **not** launch retrain.
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-test v1, not prune-test or gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-22T03:43:58Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-22T03:43:58Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** retry 740392. Did **not** launch retrain.
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-test v1, not prune-test or gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.


## learned_unet_orgdet_test_v1_submit_poll (2026-09-22T04:50:14Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-22T04:50:14Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** retry 740392. Did **not** push a second retrain version.
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-test v1, not prune-test or gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.

## learned_unet_retrain_v1_kernel

- **Status:** private GPU helper `mahdimzafari/biohub-learned-unet-retrain-v1` version **1 RUNNING**. **Not a competition submission.** Frozen `6bba` stays closed until a 44b6 lane wins.
- **Config hash:** `92a419926fa7e40b`. Parent train `4c38035d6657f2ad`. Infer `d7578b5f64ddac08`.
- **Launch:** internet off, T4 x2, resume `learned_unet_v1_last.pt` for +3 epochs. Does **not** overwrite `mahdimzafari/biohub-learned-unet-v1`.
- **Preregistered 44b6 lanes (locked before scores):** A infer `1db21f87d00ce083` vs 0.655751; B orgdet `594f6fbc03c974cf` vs 0.679578; C orgop `f924802e5f716a61` vs lane B.
- **Kaggle public / private scores:** not returned.

## learned_unet_retrain_v1_kernel_poll (2026-09-22T04:50:13Z)

- **Status:** status-only poll of `mahdimzafari/biohub-learned-unet-retrain-v1` at **2026-09-22T04:50:13Z**. Kernel still **running**. Output files **0**. No `submission.csv`. No failure message. Did **not** download. Did **not** push a second version.
- **Config hash:** `92a419926fa7e40b`. Decision `status_only`.
- **Kaggle public / private scores:** not returned. Not a competition submission.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-22T05:53:59Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-22T05:53:59Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** retry 740392. Did **not** push a second retrain version.
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-test v1, not prune-test or gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.

## learned_unet_retrain_v1_kernel_poll (2026-09-22T05:53:06Z)

- **Status:** status-only poll of `mahdimzafari/biohub-learned-unet-retrain-v1` at **2026-09-22T05:53:06Z**. Kernel still **running**. Output files **0**. No `submission.csv`. No failure message. Did **not** download. Did **not** push a second version.
- **Config hash:** `92a419926fa7e40b`. Decision `status_only`.
- **Local prep:** fail-closed 44b6 lane scorer `scripts/run_learned_unet_retrain_lanes.py` packaged. Refuses incomplete helper, missing graphs, and `6bba`. Not run. Not a score.
- **Kaggle public / private scores:** not returned. Not a competition submission.

## learned_unet_retrain_v1_kernel_poll (2026-09-22T06:11:33Z)

- **Status:** status-only poll of `mahdimzafari/biohub-learned-unet-retrain-v1` at **2026-09-22T06:11:33Z**. Kernel still **running**. Output files **0**. No `submission.csv`. No failure message. Did **not** download. Did **not** push a second version. 44b6 lanes not launched.
- **Config hash:** `92a419926fa7e40b`. Decision `status_only`.
- **Kaggle public / private scores:** not returned. Not a competition submission.

## learned_unet_retrain_v1_kernel_poll (2026-09-22T06:45:07Z)

- **Status:** status-only poll of `mahdimzafari/biohub-learned-unet-retrain-v1` at **2026-09-22T06:45:07Z**. Kernel still **running**. Output files **0**. No `submission.csv`. No failure message. Did **not** download. Did **not** push a second version. 44b6 lanes not launched.
- **Config hash:** `92a419926fa7e40b`. Decision `status_only`.
- **Kaggle public / private scores:** not returned. Not a competition submission.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-22T06:45:08Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-22T06:45:08Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** retry 740392. Did **not** push a second retrain version.
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-test v1, not prune-test or gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.

## learned_unet_retrain_v1_kernel_complete (2026-09-22T07:40:50Z)

- **Status:** private GPU helper `mahdimzafari/biohub-learned-unet-retrain-v1` version **1 COMPLETE / trained**. Downloaded to ignored `artifacts/kaggle/`. **Not a competition submission.** No `submission.csv`.
- **Config hash:** `92a419926fa7e40b`. Parent train `4c38035d6657f2ad`. Infer `d7578b5f64ddac08`.
- **Training:** 69 `44b6` train + 2 internal-val. 3 extra epochs. Best train loss **0.2629**. Transformer grad **2.64–4.76**. Candidate edges 12190 → 17989. Runtime **10461.4 s**, peak RAM **1816.6 MB**, T4 x2. Resume `learned_unet_v1_last.pt`. Reload exact. Frozen `6bba` closed.
- **Kaggle public / private scores:** not returned. Not a competition submission.

## learned_unet_retrain_infer_v1_kernel

- **Status:** private GPU helper `mahdimzafari/biohub-learned-unet-retrain-infer-v1` version **2 COMPLETE / inferred**. **Not a competition submission.**
- **Config hash:** `1db21f87d00ce083`. Checkpoint `92a419926fa7e40b`. 146.3 s, 1118 MB. Graphs 69997/65214 and 35471/31289.
- **v1 ERROR:** parent-hash-only load. **v2** accepts retrain hash.
- **Official 44b6:** **0.626370 vs infer 0.655751**. Decision `reject_lane`. Frozen `6bba` not opened.
- **Kaggle public / private scores:** not returned.

## learned_unet_retrain_orgdet_v1_kernel

- **Status:** private GPU helper `mahdimzafari/biohub-learned-unet-retrain-orgdet-v1` version **2 COMPLETE / inferred**. **Not a competition submission.**
- **Config hash:** `594f6fbc03c974cf`. 150.5 s, 1181 MB. Graphs 63356/59236 and 30896/27391.
- **Official 44b6:** **0.642049 vs orgdet 0.679578**. Decision `reject_lane`. Frozen `6bba` not opened.
- **Kaggle public / private scores:** not returned.

## learned_unet_retrain_orgop_v1_kernel

- **Status:** private GPU helper `mahdimzafari/biohub-learned-unet-retrain-orgop-v1` version **2 COMPLETE / inferred**. **Not a competition submission.**
- **Config hash:** `f924802e5f716a61`. v1 errored (parent-orgop hash guard). v2 used locked lane hash. 157.1 s, 1230 MB. Softmax>0.5 candidates 0 / 1. Local ILP applied.
- **Official 44b6:** **0.0 vs lane B 0.642049**. Decision `reject_lane`. Frozen `6bba` not opened.
- **Kaggle public / private scores:** not returned.

## learned_unet_retrain_lanes_v1_tune

- **Status:** official local scorer on frozen-tuning 44b6 only. Not a Kaggle run. Not a leaderboard score.
- **Hashes:** A `1db21f87d00ce083` **0.626370**; B `594f6fbc03c974cf` **0.642049**; C `f924802e5f716a61` **0.0**.
- **Decision:** `reject_learned_unet_retrain_lanes_v1`. Winner none. `opens_6bba=false`. Champion remains orgdet holdout **0.705226**.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-22T07:45:56Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-22T07:45:56Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** retry 740392. Did **not** push a second retrain version.
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-test v1, not prune-test or gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-22T08:52:44Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-22T08:52:44Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1.
- **Config hash:** `c45c859610d29e90`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit` as fallback only. Preferred worker is now orgdet-gap-test v1 once COMPLETE / `audit_ok`. Entry deadline **2026-09-22 23:59 UTC**.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-test leaderboard score.

## learned_unet_orgdet_gap_v1_tune

- **Status:** local official scorer on frozen-tuning `44b6_12dfb391` + `44b6_267148e4`. **Not a Kaggle run.** Not a leaderboard score.
- **Config hash:** `ee31f36151325f9a`. Parent orgdet `80b20589198148b7`. Parent gap gates `f4d888d190c45a4a`.
- **Official 44b6:** **0.687221 vs orgdet 0.679578**. extra_edges **9951**. 10.1 s, 530 MB. Decision `keep_candidate_learned_unet_orgdet_gap_v1`.
- **Kaggle public / private scores:** not returned.

## learned_unet_orgdet_gap_v1_holdout

- **Status:** one-shot local official scorer on frozen `6bba_05db0fb1` + `6bba_c328f2fd`. **Not a Kaggle run.** Not a leaderboard score. Do **not** score 6bba again.
- **Config hash:** `ee31f36151325f9a`. Graphs from `mahdimzafari/biohub-learned-unet-orgdet-holdout-v1` v1.
- **Official 6bba:** **0.722805 vs orgdet 0.705226**. extra_edges **10427**. recall 0.95762. 0/0/5 divisions. 7.2 s, 780 MB. Per-seq `6bba_05db0fb1` 0.739289; `6bba_c328f2fd` 0.684139.
- **Decision:** `keep_candidate_learned_unet_orgdet_gap_v1`. New holdout champion. Label `KEEP_CANDIDATE_NOT_SUBMISSION_AUTHORIZED`.
- **Kaggle public / private scores:** not returned.

## learned_unet_orgdet_gap_test_v1_kernel

- **Status:** private GPU test-time kernel `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1` version **1 COMPLETE / inferred** at **2026-09-22T09:05:18Z**. Internet off. T4 x2. Discover `test/*.zarr`. **Not submitted.**
- **Config hash:** `1ce4837ae16bee52`. Parent overlay `ee31f36151325f9a`. Does **not** overwrite orgdet-test v1.
- **Output:** `/kaggle/working/submission.csv` 20 791 351 bytes, 397 414 rows, 202 138 nodes, 195 276 edges, 9 499 gap links, 13 629 interpolated, 0 predicted divisions. Worker audit `audit_ok`. Independent stream audit `audit_ok`. File kept under ignored `artifacts/kaggle/`.
- **Worker:** 312.69 s (0.78 s/frame), peak RAM 1176 MB. T4 x2, torch 2.10+cu128.
- **Holdout overlap:** `6bba_05db0fb1` 88925/86287 matches locked orgdet-gap holdout overlay. Frozen `6bba` was not official-scored again.
- **Hidden-test projection:** 78.17 s/volume × 199 ≈ **4.32 h**, under 12 h.
- **Competition submit:** not performed. No public or private leaderboard score was returned.
- **Manual submit:** https://www.kaggle.com/code/mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1 → version 1 → **Submit to Competition**. `HUMAN_ACTION_REQUIRED` before 2026-09-22 23:59 UTC.

## learned_unet_orgdet_gap_test_v1_kernel_poll (2026-09-22T09:01:22Z)

- **Status:** status-only poll of `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1` at **2026-09-22T09:01:22Z**. Kernel still **running**. Output files **0**. No `submission.csv`. No failure message. Did **not** download. Did **not** overwrite orgdet-test v1.
- **Config hash:** `1ce4837ae16bee52`. Decision `status_only`.
- **Kaggle public / private scores:** not returned. Not a competition submission.

## learned_unet_orgdet_gap_test_v1_submit_poll (2026-09-22T09:54:43Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1` at **2026-09-22T09:54:43Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** push a kernel.
- **Config hash:** `1ce4837ae16bee52`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-gap-test v1, not orgdet-test, prune-test, or infer-family gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-gap-test leaderboard score.

## learned_unet_orgdet_gap_test_v1_submit_poll (2026-09-22T10:44:18Z)

- **Status:** read-only reconfirm of `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1` at **2026-09-22T10:44:18Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** push a kernel.
- **Config hash:** `1ce4837ae16bee52`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-gap-test v1, not orgdet-test, prune-test, or infer-family gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-gap-test leaderboard score.

## learned_unet_orgdet_gap_test_v1_submit_poll (2026-09-22T11:40:00Z)

- **Status:** read-only reconfirm of `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1` at **2026-09-22T11:40:00Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** push a kernel.
- **Config hash:** `1ce4837ae16bee52`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-gap-test v1, not orgdet-test, prune-test, or infer-family gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-gap-test leaderboard score.

## learned_unet_orgdet_gap_test_v1_submit_poll (2026-09-22T12:42:26Z)

- **Status:** read-only reconfirm of `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1` at **2026-09-22T12:42:26Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** push a kernel.
- **Config hash:** `1ce4837ae16bee52`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-gap-test v1, not orgdet-test, prune-test, or infer-family gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-gap-test leaderboard score.

## learned_unet_orgdet_gap_test_v1_submit_poll (2026-09-22T13:46:09Z)

- **Status:** read-only reconfirm of `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1` at **2026-09-22T13:46:09Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** push a kernel.
- **Config hash:** `1ce4837ae16bee52`.
- **Competition list:** still exactly one row (association_v2 **56399003**, public API string **`0.553`**).
- **Decision:** `human_action_required_ui_submit`. Entry deadline **2026-09-22 23:59 UTC**. Preferred worker is orgdet-gap-test v1, not orgdet-test, prune-test, or infer-family gap-test.
- **Kaggle public / private scores for this slug:** not returned. `0.553` is **not** an orgdet-gap-test leaderboard score.
## learned_unet_orgdet_test_v1_submit_poll (2026-09-22T14:23:04Z)

- **Status:** read-only poll after `[USER_ACTION_COMPLETED]` UI submit notice. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** touch `mahdimzafari/biohub-learned-unet-retrain-v1` (already COMPLETE / trained).
- **Config hash:** `c45c859610d29e90`.
- **Competition list (Kaggle-returned, sanitized):** two rows.
  - association_v2 **56399003**, status **complete**, date **2026-09-20T15:12:52.307Z**, public API string **`0.553`**, `scriptVersionId` **351174414**, notebook `mahdimzafari/biohub-association-v2`.
  - new row **56466744**, status **pending**, date **2026-09-22T14:13:11.68Z**, `fileName` `submission.csv`, public score **not returned**, private score **not returned**, `scriptVersionId` **351557863**, notebook `mahdimzafari/biohub-learned-unet-orgdet-test-v1`.
- **Identity:** proven orgdet-test from the relative `/code/` path. Not association_v2, not prune-test, not infer-family gap-test, and **not** `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1`. Kaggle kernel `versionNumber` was **not** returned.
- **Decision:** `new_submission_present`. `orgdet_test_identity_proven=true`.
- **Not a leaderboard score.** Empty `publicScore` is not used for tuning. Do **not** resubmit. Additional UI submit of orgdet-gap-test is **not** authorized this cycle.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-22T15:48:59Z)

- **Status:** read-only reconfirm of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-22T15:48:59Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** touch `mahdimzafari/biohub-learned-unet-retrain-v1`.
- **Config hash:** `c45c859610d29e90`.
- **Competition list (Kaggle-returned, sanitized):** still two rows.
  - association_v2 **56399003**, status **complete**, date **2026-09-20T15:12:52.307Z**, public API string **`0.553`**, `scriptVersionId` **351174414**, notebook `mahdimzafari/biohub-association-v2`.
  - orgdet-test **56466744**, status still **pending**, date **2026-09-22T14:13:11.68Z**, `fileName` `submission.csv`, public score **not returned**, private score **not returned**, `scriptVersionId` **351557863**, notebook `mahdimzafari/biohub-learned-unet-orgdet-test-v1`.
- **Identity:** still proven orgdet-test from the relative `/code/` path. Not association_v2, not prune-test, not infer-family gap-test, and **not** `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1`. Kaggle kernel `versionNumber` was **not** returned.
- **Decision:** `new_submission_present`. `orgdet_test_identity_proven=true`.
- **Not a leaderboard score.** Empty `publicScore` is not used for tuning. Do **not** resubmit. Additional UI submit of orgdet-gap-test is **not** authorized this cycle.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-22T16:42:10Z)

- **Status:** read-only reconfirm of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-22T16:42:10Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** touch `mahdimzafari/biohub-learned-unet-retrain-v1`.
- **Config hash:** `c45c859610d29e90`.
- **Competition list (Kaggle-returned, sanitized):** still two rows.
  - association_v2 **56399003**, status **complete**, date **2026-09-20T15:12:52.307Z**, public API string **`0.553`**, `scriptVersionId` **351174414**, notebook `mahdimzafari/biohub-association-v2`.
  - orgdet-test **56466744**, status still **pending**, date **2026-09-22T14:13:11.68Z**, `fileName` `submission.csv`, public score **not returned**, private score **not returned**, `scriptVersionId` **351557863**, notebook `mahdimzafari/biohub-learned-unet-orgdet-test-v1`.
- **Identity:** still proven orgdet-test from the relative `/code/` path. Not association_v2, not prune-test, not infer-family gap-test, and **not** `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1`. Kaggle kernel `versionNumber` was **not** returned.
- **Decision:** `new_submission_present`. `orgdet_test_identity_proven=true`.
- **Not a leaderboard score.** Empty `publicScore` is not used for tuning. Do **not** resubmit. Additional UI submit of orgdet-gap-test is **not** authorized this cycle.

## learned_unet_orgdet_test_v1_submit_poll (2026-09-22T17:41:09Z)

- **Status:** read-only reconfirm of `mahdimzafari/biohub-learned-unet-orgdet-test-v1` at **2026-09-22T17:41:09Z**. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1. Did **not** touch `mahdimzafari/biohub-learned-unet-retrain-v1`.
- **Config hash:** `c45c859610d29e90`.
- **Competition list (Kaggle-returned, sanitized):** still two rows.
  - association_v2 **56399003**, status **complete**, date **2026-09-20T15:12:52.307Z**, public API string **`0.553`**, `scriptVersionId` **351174414**, notebook `mahdimzafari/biohub-association-v2`.
  - orgdet-test **56466744**, status **complete**, date **2026-09-22T14:13:11.68Z**, `fileName` `submission.csv`, public API string **`0.612`**, private score **not returned**, `scriptVersionId` **351557863**, notebook `mahdimzafari/biohub-learned-unet-orgdet-test-v1`.
- **Identity:** still proven orgdet-test from the relative `/code/` path. Not association_v2, not prune-test, not infer-family gap-test, and **not** `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1`. Kaggle kernel `versionNumber` was **not** returned.
- **Decision:** `public_score_returned`. `orgdet_test_identity_proven=true`.
- **Kaggle-returned public score for this slug:** **`0.612`** (API string; no extra digits). Private score **not returned**. This is **not** a local official holdout score and **not** an orgdet-gap-test score. Do **not** invent extra digits. Do **not** use **0.612** or **0.553** for tuning. Do **not** resubmit. Do **not** re-poll **56466744** as pending. Additional UI submit of orgdet-gap-test is **not** authorized this cycle.

## learned_unet_orgilp_v1_tune (2026-09-26T17:11:42Z)

- **Status:** local official score of `learned_unet_orgilp_v1` on frozen-tuning `44b6` only. **Not a Kaggle run. Not a competition submission. Not a leaderboard score.**
- **Config hash:** `dbc6134cddf2048d`. Parent rejected orgop `250c5bfacd4a8198` (not overwritten).
- **Preflight:** `preflight_ok`. 62480 / 28672 detections; 333502 / 133546 spatial 12 µm candidates; ILP wrote 58805 / 26045 edges (`tracksdata_ilp`, SCIP).
- **Official 44b6:** **0.676811 vs orgdet-gap 0.687221** (also below orgdet **0.679578**). Recall 0.97004. Divisions 0/0/2. 1008.0 s, 4501 MB.
- **Decision:** `reject_learned_unet_orgilp_v1`. Frozen `6bba` not opened. `orgilp_gap` not launched (consecutive ILP already lost to greedy).
- **Kaggle public / private scores:** not applicable; no kernel push and no submit.

## learned_unet_orgdet_gap_prune_v1_tune (2026-09-26T17:48:23Z)

- **Status:** local official score of `learned_unet_orgdet_gap_prune_v1` on frozen-tuning `44b6` only. **Not a Kaggle run. Not a competition submission. Not a leaderboard score.**
- **Config hash:** `862e3e7f7ac0330e`. Parent orgdet-gap `ee31f36151325f9a` (not overwritten).
- **Official 44b6:** **0.693437 vs orgdet-gap 0.687221**. Recall 0.97603. Divisions 0/0/2. Dropped 227 isolates. Extra edges still 9951. 10.40 s, 528 MB.
- **Decision:** `keep_candidate_learned_unet_orgdet_gap_prune_v1`. Hash locked before the 6bba look.
- **Kaggle public / private scores:** not applicable; no kernel push and no submit.

## learned_unet_orgdet_gap_prune_v1_holdout (2026-09-26T17:48:51Z)

- **Status:** local official one-shot score of `learned_unet_orgdet_gap_prune_v1` on frozen `6bba`. **Not a Kaggle run. Not a competition submission. Not a leaderboard score.**
- **Config hash:** `862e3e7f7ac0330e`.
- **Official 6bba:** **0.723018 vs orgdet-gap 0.722805**. Recall 0.95762. Divisions 0/0/5. Dropped 296 isolates. Edge TP/FP/FN match locked orgdet-gap. 7.64 s, 776 MB.
- **Decision:** `keep_candidate_learned_unet_orgdet_gap_prune_v1`. New holdout champion. Do **not** score 6bba again.
- **Kaggle public / private scores:** not applicable; no kernel push and no submit.

## learned_unet_orgdet_gap_prune_test_v1_packaged

- **Status:** private GPU public-test notebook packaged as a **new** slug `mahdimzafari/biohub-learned-unet-orgdet-gap-prune-test-v1`. Internet off. Attaches competition + `mahdimzafari/biohub-learned-unet-v1`. **Not submitted.**
- **Config hash:** `52a30a5843e0a880`. Parent overlay `862e3e7f7ac0330e`. Parent orgdet-gap-test `1ce4837ae16bee52`. Discovers `test/*.zarr`. Locked orgdet 0.99 / pool 3.0 / TTA + greedy 12 µm + gap-close + drop degree-0.
- **Estimate:** orgdet-gap-test 312.69 s / 4 volumes + 0.5 s isolate drop. Hidden-test projection **4.35 h** under 12 h.
- **Guards:** does **not** overwrite orgdet-gap-test / orgdet-test / prune-test / gap-test / no-gap test. New slug is not in `FORBIDDEN_SLUGS` until v1 COMPLETE.
- **Competition submit:** not performed. No public or private leaderboard score. Do **not** submit from the CLI.

## learned_unet_orgdet_gap_test_v1_submit_poll (2026-09-26T19:55:35Z)

- **Status:** read-only poll of `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1`. Kernel still **complete**. Outputs still `run_manifest.json` + `submission.csv`. Did **not** download `submission.csv`. Did **not** submit. Did **not** overwrite v1.
- **Config hash:** `1ce4837ae16bee52`.
- **Competition list (Kaggle-returned, sanitized):** three rows.
  - new row **56583783**, status **complete**, date **2026-09-26T16:09:48.773Z**, `fileName` `submission.csv`, public API string **`0.602`**, private score **not returned**, `scriptVersionId` **351808956**, notebook `mahdimzafari/biohub-learned-unet-orgdet-gap-test-v1`.
  - orgdet-test **56466744**, status **complete**, public API string **`0.612`**, `scriptVersionId` **351557863**.
  - association_v2 **56399003**, status **complete**, public API string **`0.553`**, `scriptVersionId` **351174414**.
- **Identity:** proven orgdet-gap-test from the relative `/code/` path. Not association_v2, not orgdet-test, not prune-test, not infer-family gap-test, and **not** `mahdimzafari/biohub-learned-unet-orgdet-gap-prune-test-v1`.
- **Decision:** `public_score_returned`.
- **Kaggle-returned public score for this slug:** **`0.602`** (API string; no extra digits). Private score **not returned**. This is **not** a local official holdout score and **not** an orgdet-test score. Do **not** invent extra digits. Do **not** use **0.602**, **0.612**, or **0.553** for tuning. Do **not** resubmit. Do **not** re-poll **56583783** as pending.

## learned_unet_orgdet_gap_prune_test_v1_kernel

- **Status:** private GPU test-time kernel `mahdimzafari/biohub-learned-unet-orgdet-gap-prune-test-v1` version **1 COMPLETE / inferred** at **2026-09-26T19:54:59Z**. Internet off. Discover `test/*.zarr`. **Not submitted.**
- **Config hash:** `52a30a5843e0a880`. Parent overlay `862e3e7f7ac0330e`. Does **not** overwrite orgdet-gap-test v1.
- **Output:** `/kaggle/working/submission.csv` 20 766 993 bytes, 396 947 rows, 201 671 nodes, 195 276 edges, 9 499 gap links, 13 629 interpolated, **467 dropped isolates**, 0 predicted divisions. Worker audit `audit_ok`. Independent stream audit `audit_ok`. File kept under ignored `artifacts/kaggle/`.
- **Worker:** 331.08 s (0.83 s/frame), peak RAM 1164 MB.
- **Holdout overlap:** `6bba_05db0fb1` 88777/86287 matches locked orgdet-gap-prune holdout overlay. Frozen `6bba` was not official-scored again.
- **Versus locked orgdet-gap-test:** edges stayed **195276**; nodes **202138 → 201671**.
- **Hidden-test projection:** 82.77 s/volume × 199 ≈ **4.58 h**, under 12 h.
- **Competition submit:** not performed. This slug is **not** on the competition list. Public **0.602** is orgdet-gap-test **56583783** only.
- **Manual submit:** `READY_FOR_MANUAL_SUBMISSION_REVIEW`. Additional UI click is **not** authorized unless a later message separately authorizes it.

## learned_unet_orgdet_error_decomp_44b6 / learned_unet_orgdet_tcf_v1_tune

- **Status:** local official-scorer 44b6-only error decomposition, then one preregistered short-fragment TCF overlay. **Not a Kaggle run. Not a leaderboard score.** Frozen `6bba` was **not** opened.
- **Decomp:** locked orgdet / orgdet-gap / orgdet-gap-prune combined scores matched **0.679578 / 0.687221 / 0.693437**. Prune residue official **908 / 229 / 142**. Detection FN edges **27**, association FN edges **115**. Unsupported t±1 leftovers **0**. Size≤7 fragments **1954** nodes with **0** sparse-GT matches.
- **Config hash:** `a6b46dfcf410478f`. Parent prune `862e3e7f7ac0330e`. Gate official 44b6 **> 0.693437 + 0.002000** = **0.695437**.
- **Official 44b6:** **0.695181** vs prune **0.693437** (delta **+0.001743**). Edge Jaccard unchanged. Dropped 1954 nodes. Runtime 10.63 s, peak 512 MB.
- **Decision:** `reject_learned_unet_orgdet_tcf_v1`. Do **not** retune fragment length. Do **not** score 6bba. Do **not** use public **0.602** / **0.612** / **0.553**.

## learned_unet_orgdet_elf_v1_tune / holdout

- **Status:** local official scores of `learned_unet_orgdet_elf_v1` (long-dt1 edge-length filter after locked orgdet+gap+prune). **Not a Kaggle run. Not a competition submission. Not a leaderboard score.**
- **Config hash:** `292c4f1f968f948d`. Parent prune `862e3e7f7ac0330e`. Gate official 44b6 **> 0.693437 + 0.002000** = **0.695437**. Not another 12 µm assignment. Not a short-fragment retune.
- **Official 44b6:** **0.709838** vs prune **0.693437** (delta **+0.016401**). Edge Jaccard 0.72626. Recall 0.976. Divisions 0/0/2. Dropped 2247 long dt=1 edges. Runtime 11.26 s, peak 529 MB. Decision `keep_candidate_learned_unet_orgdet_elf_v1`. Hash locked before the 6bba look.
- **Official one-shot 6bba:** **0.737787** vs prune **0.723018**. Edge Jaccard 0.75737. Recall 0.956. Divisions 0/0/5. Dropped 2888 long dt=1 edges. Runtime 7.67 s, peak 771 MB. Decision `keep_candidate_learned_unet_orgdet_elf_v1`. New holdout champion. Do **not** score 6bba again.
- **Public 0.602 / 0.612 / 0.553** were not used.

## learned_unet_orgdet_elf_test_v1_packaged / complete

- **Status:** private GPU public-test notebook packaged, pushed, and **COMPLETE / inferred**. New slug `mahdimzafari/biohub-learned-unet-orgdet-elf-test-v1`. Hash `ad375d39e13098d1`. Parent overlay `292c4f1f968f948d` (official 44b6 **0.709838**; official 6bba **0.737787**). Does **not** overwrite orgdet-gap-prune-test v1. **Not a competition submission. Not a leaderboard score.**
- **Output:** `/kaggle/working/submission.csv` 20 279 535 bytes, 387 812 rows, 199 977 nodes, 187 835 edges, 9 499 gap links, 13 629 interpolated, **2 161 dropped isolates**, **7 441 long dt=1**, 0 predicted divisions. Worker audit `audit_ok`. Independent stream audit `audit_ok`. File kept under ignored `artifacts/kaggle/`.
- **Holdout overlap:** `6bba_05db0fb1` **88322/83959** matches locked orgdet-elf holdout overlay. Gap links match prune-test (9499).
- **Hidden-test projection:** 88.22 s/volume × 199 ≈ **4.88 h**, under 12 h.
- **Competition submit:** not performed. This slug is **not** on the competition list.
- **Manual submit:** `READY_FOR_MANUAL_SUBMISSION_REVIEW`. Additional UI click is **not** authorized unless a later message separately authorizes it.
- **Guards:** slug is now in `FORBIDDEN_SLUGS`. Do **not** overwrite v1. Do **not** submit from the CLI. Public **0.602 / 0.612 / 0.553** were not used.

## learned_unet_orgdet_elf_error_decomp_44b6 / learned_unet_orgdet_fnn_v1_tune / holdout

- **Status:** local official-scorer 44b6 ELF residue, then one preregistered forward-NN keep overlay. **Not a Kaggle run. Not a competition submission. Not a leaderboard score.**
- **Decomp:** locked ELF combined matched **0.709838**. Official residue **910 / 203 / 140**. Predicted divisions **0**. 30/203 conflict FPs were not the source NN at t+1 versus 10/910 TPs. Turning / degree / all-interp / another 8.0 µm dt=1 filter stayed closed.
- **Config hash:** `95e9300eb84ba71c`. Parent ELF `292c4f1f968f948d`. Gate official 44b6 **> 0.709838 + 0.002000** = **0.711838**. Hash locked before the 44b6 score.
- **Official 44b6:** **0.724653** vs ELF **0.709838** (delta **+0.014815**). Edge Jaccard 0.74020. Recall 0.974. Divisions 0/0/2. Dropped 2949 non-forward-NN dt=1 edges. Runtime 72.14 s, peak 530 MB. Decision `keep_candidate_learned_unet_orgdet_fnn_v1`.
- **Official one-shot 6bba:** **0.752091** vs ELF **0.737787**. Edge Jaccard 0.77110. Recall 0.955. Divisions 0/0/5. Dropped 3566 non-forward-NN dt=1 edges. Runtime 114.96 s, peak 734 MB. Decision `keep_candidate_learned_unet_orgdet_fnn_v1`. New holdout champion. Do **not** score 6bba again.
- **Public-test worker:** packaged and collected this cycle as `learned_unet_orgdet_fnn_test_v1`.
- **Public 0.602 / 0.612 / 0.553** were not used.

## learned_unet_orgdet_fnn_test_v1_packaged / complete

- **Status:** private GPU public-test notebook packaged, pushed, and **COMPLETE / inferred**. New slug `mahdimzafari/biohub-learned-unet-orgdet-fnn-test-v1`. Hash `1af763e6ffc23188`. Parent overlay `95e9300eb84ba71c` (official 44b6 **0.724653**; official 6bba **0.752091**). Does **not** overwrite orgdet-elf-test v1. **Not a competition submission. Not a leaderboard score.**
- **Output:** `/kaggle/working/submission.csv` 19 713 671 bytes, 377 183 rows, 197 058 nodes, 180 125 edges, 9 499 gap links, 13 629 interpolated, **5 080 dropped isolates**, **7 441 long dt=1**, **7 710 non-forward-NN**, 0 predicted divisions. Worker audit `audit_ok`. Independent stream audit `audit_ok`. File kept under ignored `artifacts/kaggle/`.
- **Holdout overlap:** `6bba_05db0fb1` **87339/81132** matches locked orgdet-fnn holdout overlay. Gap links and long-dt1 counts match elf-test (9499 / 7441).
- **Hidden-test projection:** 199.70 s/volume × 199 ≈ **11.04 h**, under 12 h but `fail_closed_approaches_12h`.
- **Competition submit:** not performed. This slug is **not** on the competition list.
- **Manual submit:** **not** `READY_FOR_MANUAL_SUBMISSION_REVIEW` because of the 90% hidden-runtime gate. Preferred packaged fallback remains elf-test.
- **Guards:** slug is now in `FORBIDDEN_SLUGS`. Do **not** overwrite v1. Do **not** submit from the CLI. Public **0.602 / 0.612 / 0.553** were not used.

## learned_unet_orgdet_fnn_fast_test_v1_packaged / complete

- **Status:** private GPU public-test notebook packaged, pushed, and **COMPLETE / inferred**. New slug `mahdimzafari/biohub-learned-unet-orgdet-fnn-fast-test-v1`. Hash `9b9763e26bc0f037`. Parent overlay `95e9300eb84ba71c` (official 44b6 **0.724653**; official 6bba **0.752091**). Parent fnn-test `1af763e6ffc23188`. Vectorized forward-NN scan. Does **not** overwrite fnn-test v1. **Not a competition submission. Not a leaderboard score.**
- **Output:** `/kaggle/working/submission.csv` 19 713 671 bytes, 377 183 rows, 197 058 nodes, 180 125 edges, 9 499 gap links, 13 629 interpolated, **5 080 dropped isolates**, **7 441 long dt=1**, **7 710 non-forward-NN**, 0 predicted divisions. Worker audit `audit_ok`. Independent stream audit `audit_ok`. File kept under ignored `artifacts/kaggle/`. Counts match locked fnn-test.
- **Holdout overlap:** `6bba_05db0fb1` **87339/81132** matches locked orgdet-fnn holdout overlay. Gap links and long-dt1 counts match elf-test (9499 / 7441).
- **Hidden-test projection:** 80.06 s/volume × 199 ≈ **4.43 h**, under 12 h and the 90% fail-closed gate. Runtime **320.23 s** vs parent fnn-test **798.80 s**. Peak RAM 1186 MB.
- **Competition submit:** not performed. This slug is **not** on the competition list.
- **Manual submit:** `READY_FOR_MANUAL_SUBMISSION_REVIEW`. Preferred packaged public-test worker for the FNN holdout champion. Additional UI click is **not** authorized unless a later message separately authorizes it.
- **Guards:** slug is now in `FORBIDDEN_SLUGS`. Do **not** overwrite v1. Do **not** overwrite fnn-test v1 or elf-test v1. Do **not** submit from the CLI. Do **not** retune `nn_tie_eps`. Public **0.602 / 0.612 / 0.553** were not used.

## learned_unet_orgdet_fnn_oracle_44b6 / learned_unet_orgdet_fnn_rank_v1_preregistered

- **Status:** local official-scorer 44b6 FNN detector oracle, then one preregistered learned-ranking hypothesis. **Not a Kaggle run. Not a competition submission. Not a leaderboard score.**
- **Restore:** ignored `biohub-geff-export` + `biohub-learned-unet-orgop-v1` outputs. Locked FNN rebuild official **0.724653** (exact match). Frozen `6bba` was not opened.
- **Oracle:** GT edges 1050; both endpoints detected 1023; detection-unavoidable 27 (2.57%); association-fixable in-candidate 113; candidate-edge recall **0.996**. Official candidate-oracle **1.048979** (1019/0/31, divisions 2/0/0). Official detection-oracle **1.052563** (1023/0/27, divisions 2/0/0). Runtime 29.49 s, peak 516.5 MB.
- **Config hash:** `2748c4a2b2823c0a`. Parent FNN `95e9300eb84ba71c`. Gate official 44b6 **> 0.724653 + 0.002000** = **0.726653**. Hash locked before any ranker score. Later official-scored and rejected (see below).
- **Public-test worker:** unchanged. FNN-fast-test v1 stays COMPLETE / READY / not submission-authorized.
- **Public 0.602 / 0.612 / 0.553** were not used.

## learned_unet_orgdet_fnn_rank_canary_v1

- **Status:** local CPU canary hash `f7dc3d3292ce813d`. Parent rank `2748c4a2b2823c0a`. **No kernel pushed.** Frozen `6bba` was not opened. Ranker was not official-scored. No `submission.csv`.
- **Result:** `canary_ok`. Finite link loss 1.361 → 8.50e-5, nonzero pair-head grads 0.547, ranking **6 / 6**, min true−hard logit **20.677**, reload max abs **0.0**, 1.12 s, 375.4 MB.
- **Graph pairs:** 6 true / 5 hard-negative / 5 out-of-envelope excluded; two-child fork allowed; dt ∈ {1, 2}.
- **Guards:** does **not** overwrite fnn-fast-test v1. Does **not** launch a ranker helper. Public **0.602 / 0.612 / 0.553** were not used.

## learned_unet_orgdet_fnn_rank_v1_tune

- **Status:** local official scorer on both frozen-tuning `44b6` sequences. **Not a Kaggle run. Not a competition submission. Not a leaderboard score.** Frozen `6bba` was not opened.
- **Config hash:** `2748c4a2b2823c0a`. Parent FNN `95e9300eb84ba71c`. Helper packaged as `mahdimzafari/biohub-learned-unet-orgdet-fnn-rank-v1`. **Not pushed.**
- **Local official tuning:** learned_unet_orgdet_fnn_rank_v1 **0.006663** vs locked FNN **0.724653**. Official TP/FP/FN **8 / 124 / 1042**. Divisions **0/309/2**. Recall 0.974. 1751832 envelope candidates; 1019 true / 1750813 hard. Runtime 405.9 s, peak 3814 MB. Decision `reject_learned_unet_orgdet_fnn_rank_v1`. Do not retune. Do not open `6bba`.
- **Cutoff defect:** a first pass with an extra logit>0 cutoff wrote 0 edges (official **0.0**). That cutoff is not in the locked payload. Rank-order + constraints was the scored operating point.
- **Kaggle public / private scores:** not applicable; not submitted. Public **0.602 / 0.612 / 0.553** were not used.

## learned_unet_orgdet_fnn_div_v1_tune

- **Status:** local official scorer on both frozen-tuning `44b6` sequences. **Not a Kaggle run. Not a competition submission. Not a leaderboard score.** Frozen `6bba` was not opened.
- **Config hash:** `e184e6ce5bd90e46`. Parent FNN `95e9300eb84ba71c`. Division gates `0fa756963e5d42d1`. No kernel.
- **Local official tuning:** learned_unet_orgdet_fnn_div_v1 **0.724653** vs locked FNN **0.724653**. Official TP/FP/FN **906 / 174 / 144** identical to FNN. Divisions **0/0/2**. Extra edges **51** (16 + 35 unmatched forks). Official event scorer recovered **0 / 2** annotated forks. Runtime 16.1 s, peak 514 MB. Decision `reject_learned_unet_orgdet_fnn_div_v1`. Do not retune. Do not open `6bba`.
- **Kaggle public / private scores:** not applicable; not submitted. Public **0.602 / 0.612 / 0.553** were not used.

## learned_unet_orgdet_fnn_acf_v1_tune_and_holdout

- **Status:** local official scorer on both frozen-tuning `44b6` sequences, then one-shot frozen `6bba`. **Not a Kaggle run. Not a competition submission. Not a leaderboard score.**
- **Config hash:** `a66075085ac64400`. Parent FNN `95e9300eb84ba71c`. Appearance from the two exported tuning Zarrs, then holdout Zarrs after the 44b6 keep. No kernel.
- **Local official tuning:** learned_unet_orgdet_fnn_acf_v1 **0.734620** vs locked FNN **0.724653**. Official TP/FP/FN **907 / 160 / 143**. Divisions **0/0/2**. Dropped **1657** inconsistent dt=1 edges. Runtime 15.11 s, peak 562 MB. Decision `keep_candidate_learned_unet_orgdet_fnn_acf_v1`.
- **Local official holdout:** learned_unet_orgdet_fnn_acf_v1 **0.755503** vs locked FNN **0.752091**. Official TP/FP/FN **1462 / 205 / 221**. Divisions **0/0/5**. Dropped **879** inconsistent dt=1 edges. Runtime 13.63 s, peak 767 MB. Decision `keep_candidate_learned_unet_orgdet_fnn_acf_v1`. New holdout champion. Do not score 6bba again.
- **Kaggle public / private scores:** not applicable; not submitted. Public **0.602 / 0.612 / 0.553** were not used.

## learned_unet_orgdet_fnn_acf_test_v1_packaged / complete

- **Status:** private GPU public-test notebook packaged, pushed, and **COMPLETE / inferred**. New slug `mahdimzafari/biohub-learned-unet-orgdet-fnn-acf-test-v1`. Hash `2d13d0e3c90df093`. Parent overlay `a66075085ac64400` (official 44b6 **0.734620**; official 6bba **0.755503**). Parent fnn-fast-test `9b9763e26bc0f037`. Vectorized FNN then 3×3×3-mean appearance filter. Does **not** overwrite fnn-fast-test v1, fnn-test v1, or elf-test v1. **Not a competition submission. Not a leaderboard score.**
- **Output:** `/kaggle/working/submission.csv` 19 585 900 bytes, 374 763 rows, 196 290 nodes, 178 473 edges, 9 499 gap links, 13 629 interpolated, **5 848 dropped isolates**, **7 441 long dt=1**, **7 710 non-forward-NN**, **1 652 inconsistent dt=1**, 0 predicted divisions. Worker audit `audit_ok`. Independent stream audit `audit_ok`. File kept under ignored `artifacts/kaggle/`.
- **Holdout overlap:** `6bba_05db0fb1` **87109/80506** matches locked orgdet-fnn-acf holdout overlay. Gap / long-dt1 / non-forward-NN counts match fnn-fast-test (9499 / 7441 / 7710).
- **Hidden-test projection:** 80.99 s/volume × 199 ≈ **4.48 h**, under 12 h and the 90% fail-closed gate. Runtime **323.98 s** vs parent fnn-fast-test **320.23 s**. Peak RAM 1187 MB.
- **Competition submit:** not performed. This slug is **not** on the competition list.
- **Manual submit:** `READY_FOR_MANUAL_SUBMISSION_REVIEW`. Preferred packaged public-test worker for the ACF holdout champion. Additional UI click is **not** authorized unless a later message separately authorizes it.
- **Guards:** slug is now in `FORBIDDEN_SLUGS`. Do **not** overwrite v1. Do **not** overwrite fnn-fast-test v1, fnn-test v1, or elf-test v1. Do **not** submit from the CLI. Do **not** retune `max_rel_appearance`. Public **0.602 / 0.612 / 0.553** were not used.

## learned_unet_orgdet_fnn_acf_bnn_v1_tune

- **Status:** local official scorer on both frozen-tuning `44b6` sequences. **Not a Kaggle run. Not a competition submission. Not a leaderboard score.** Frozen `6bba` was not opened.
- **Config hash:** `bdb38cf13e226ced`. Parent ACF `a66075085ac64400`. No kernel.
- **Local official tuning:** learned_unet_orgdet_fnn_acf_bnn_v1 **0.734152** vs locked ACF **0.734620**. Official TP/FP/FN **901 / 154 / 149**. Divisions **0/0/2**. Dropped **1148** non-backward-NN dt=1 edges. Runtime 16.83 s, peak 557 MB. Decision `reject_learned_unet_orgdet_fnn_acf_bnn_v1`. Missed the preregistered **+0.002000** gate (**0.736620**). Do not retune. Do not open `6bba`.
- **Kaggle public / private scores:** not applicable; not submitted. Public **0.602 / 0.612 / 0.553** were not used.

## learned_unet_orgdet_fnn_acf_ipc_v1_tune

- **Status:** local official scorer on both frozen-tuning `44b6` sequences. **Not a Kaggle run. Not a competition submission. Not a leaderboard score.** Frozen `6bba` was not opened.
- **Config hash:** `7302409595bee523`. Parent ACF `a66075085ac64400`. Rejected BNN `bdb38cf13e226ced`. No kernel.
- **Local official tuning:** learned_unet_orgdet_fnn_acf_ipc_v1 **0.734620** vs locked ACF **0.734620**. Official TP/FP/FN **907 / 160 / 143**. Divisions **0/0/2**. Dropped **0** conflict incoming dt=1 edges. Conflict children **0**. Runtime 19.62 s, peak 552 MB. Decision `reject_learned_unet_orgdet_fnn_acf_ipc_v1`. After locked ACF no child has two incoming dt=1 edges, so the overlay is a no-op and missed the preregistered **+0.002000** gate (**0.736620**). Do not retune. Do not open `6bba`.
- **Kaggle public / private scores:** not applicable; not submitted. Public **0.602 / 0.612 / 0.553** were not used.

## learned_unet_orgdet_fnn_acf_gacf_v1_tune

- **Status:** local official scorer on both frozen-tuning `44b6` sequences. **Not a Kaggle run. Not a competition submission. Not a leaderboard score.** Frozen `6bba` was not opened.
- **Config hash:** `5ce49a6d175c8032`. Parent ACF `a66075085ac64400`. Rejected BNN `bdb38cf13e226ced`. Rejected IPC `7302409595bee523`. No kernel.
- **Local official tuning:** learned_unet_orgdet_fnn_acf_gacf_v1 **0.734745** vs locked ACF **0.734620**. Official TP/FP/FN **907 / 160 / 143**. Divisions **0/0/2**. Dropped **160** inconsistent gap-close edges / **61** chains. Runtime 19.01 s, peak 581 MB. Decision `reject_learned_unet_orgdet_fnn_acf_gacf_v1`. Missed the preregistered **+0.002000** gate (**0.736620**). Official matched edges unchanged; leftover unmatched gap links only. Do not retune. Do not open `6bba`.
- **Kaggle public / private scores:** not applicable; not submitted. Public **0.602 / 0.612 / 0.553** were not used.

## learned_unet_orgdet_fnn_acf_gelf_v1_diag

- **Status:** local 44b6 ACF gap-chain length diagnostic. **Not a Kaggle run. Not a competition submission. Not a leaderboard score. Not an official overlay score.** Frozen `6bba` was not opened.
- **Config hash:** `59812f90385ed4d2`. Parent ACF `a66075085ac64400`. Rejected GACF `5ce49a6d175c8032`. `max_gap_edge_um` stays null. No kernel.
- **Local diagnostic:** 2253 gap chains; 14 TP max **8.751 µm**; 4 conflict FPs max **5.629 µm**; 2235 leftovers not used as negatives. Proposed 8.8 µm gate drops **0** conflict FPs. Decision `reject_learned_unet_orgdet_fnn_acf_gelf_v1` (`no_conflict_fp_above_gate`). Overlay apply remains refuse-closed. Do not sweep leftovers. Do not retune. Do not open `6bba`.
- **Kaggle public / private scores:** not applicable; not submitted. Public **0.602 / 0.612 / 0.553** were not used.

## learned_unet_orgdet_fnn_acf_tac_v1_diag

- **Status:** local 44b6 ACF unique-predecessor dt=1 turning-angle diagnostic. **Not a Kaggle run. Not a competition submission. Not a leaderboard score. Not an official overlay score.** Frozen `6bba` was not opened.
- **Config hash:** `4dfc5514c65496e5`. Parent ACF `a66075085ac64400`. Rejected GELF `59812f90385ed4d2`. `max_turn_deg` stays null. No kernel.
- **Local diagnostic:** 80168 unique-pred dt=1 two-hops; 563 measurable TP max **180.0 deg** (cosine **-1.0**); 112 conflict FPs also 0–180; 53564 leftovers not used as negatives; 25929 degenerate zero-length hops not gated. Decision `reject_learned_unet_orgdet_fnn_acf_tac_v1` (`tp_covers_full_angle`). Overlay apply remains refuse-closed. Do not sweep leftovers. Do not retune. Do not open `6bba`. Runtime 15.50 s, peak 230 MB.
- **Kaggle public / private scores:** not applicable; not submitted. Public **0.602 / 0.612 / 0.553** were not used.

## learned_unet_orgdet_fnn_acf_scf_v1_diag

- **Status:** local 44b6 ACF unique-predecessor dt=1 speed-change diagnostic. **Not a Kaggle run. Not a competition submission. Not a leaderboard score. Not an official overlay score.** Frozen `6bba` was not opened.
- **Config hash:** `a259d184c9f99046`. Parent ACF `a66075085ac64400`. Rejected TAC `4dfc5514c65496e5`. `max_speed_change_um` stays null. No kernel.
- **Local diagnostic:** 80168 unique-pred dt=1 speed-changes; 855 TP max **5.997 µm/frame**; 145 conflict FPs max **4.596 µm/frame**; 79168 leftovers not used as negatives. Proposed 6.0 µm/frame gate drops **0** conflict FPs. Decision `reject_learned_unet_orgdet_fnn_acf_scf_v1` (`no_conflict_fp_above_gate`). Overlay apply remains refuse-closed. Do not sweep leftovers. Do not retune. Do not open `6bba`. Runtime 13.14 s, peak 241 MB.
- **Kaggle public / private scores:** not applicable; not submitted. Public **0.602 / 0.612 / 0.553** were not used.

## Policy

1. One new Kaggle submission only when the official frozen-validation score improves, or when a written generalization hypothesis is being tested.
2. Association and division thresholds are chosen on tuning data (`docs/splits/v1.json`), then evaluated once on frozen validation.
3. Public leaderboard is not used as the sole tuning signal.
