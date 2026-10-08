# Which result files are current

One measurement fix and one reporting choice on 2026-10-07/08 (see sprint page):

1. **Answer criteria (maths), a choice, reported both ways:** strict = the correct result answers the question (Claude
   judge) with no working shown (`gepa.maths_checks`); lenient = the correct number appears however it is used, labels
   and working allowed (`gepa.lenient_maths_present`). Reports carry both (`*_lenient` fields).
2. **Calibration (a bug fix):** the no-concept replies that set each probe's 1% line are cleaned of replies that show the concept
   anyway (`gepa.clean_negatives`; maths: lookups the model computed; chemistry: replies Claude finds any chemistry in).
   `gepa.CALIBRATION_VERSION` records the rules; `calibration.json` and `revalidated.json` carry the version they used.

Current, per run directory:
- `revalidated.json` (with `calibration_version`): held-out test arms re-scored from saved logits under current rules.
- `ladder_<probe>.json`, `ladder_<probe>_fresh_test.json` written after 2026-10-07 23:50: saved answers re-scored under other probes.
- `large_test.json`, `fresh_test.json` written after the re-threshold step (`followups.rethreshold`).

Not current:
- `summary.json`: the numbers as the search saw them during the run (its own calibration and answer check at the time).
  Kept as the record of the search; use `revalidated.json` for evaluation.
- `superseded/`: files computed before the fixes, kept for reference only.
