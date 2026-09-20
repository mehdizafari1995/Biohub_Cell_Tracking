# Independent association_v2 Kaggle output audit

This is a local official-scorer audit of a downloaded private kernel CSV.
It is **not** a competition submission and reports **no** leaderboard score.

- Kernel: `mahdimzafari/biohub-association-v2` version 3
- Config hash: `2c6966b5204ba542`
- Code: `16601cc`
- Live Kaggle status (2026-09-20T15:00Z re-query): `complete`, no failure message. Outputs: `submission.csv`, `run_manifest.json`.
- Worker runtime / peak RAM (from worker manifest, if present): 593.528496317 s / 421.46484375 MB
- Independent audit_ok: `True`
- Public / private LB: `None` / `None`
- Retune after holdout: `False`
- Kernel re-pushed this cycle: `False` (no version 4)

## Frozen validation overlap

Public test includes `6bba_05db0fb1`, which is one locked validation sequence.
`6bba_c328f2fd` is not in public test, so the frozen split is still incomplete.
The four-stem official aggregate mixes tuning-specimen copies and is not holdout.
Do not retune locked association_v2 from those extra stems.

- `6bba_05db0fb1` role=`frozen_validation_overlap` status=`scored` official_combined=0.5643362235389078 div=0/0/3 delta_vs_local=0.00033622353890783163

## Other public-test stems (not holdout)

- `44b6_0113de3b` role=`tuning_specimen_public_test_copy` status=`scored` official_combined=0.9309656078730045
- `44b6_0b24845f` role=`tuning_specimen_public_test_copy` status=`scored` official_combined=0.2795015869935827
- `6bba_05b6850b` role=`validation_specimen_not_in_locked_split` status=`scored` official_combined=0.47628644727699737

## Audit errors / warnings

- errors: []
- warnings: []

Independent audit of a private internet-off kernel output. Official numbers are local scorer results on the worker CSV. They are not a Kaggle leaderboard score. Competition submit was not performed.

## 2026-09-20T15:00Z worker re-verification

This START-trigger cycle re-queried live kernel status, re-downloaded the version-3 archive into ignored `artifacts/kaggle/`, and re-ran stdlib schema/integrity plus official scoring. Counts, `audit_ok`, and the 0.564336 overlap score are unchanged. CSV was not copied into git.
