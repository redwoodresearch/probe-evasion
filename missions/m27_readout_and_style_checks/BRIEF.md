# Mission m27_readout_and_style_checks: is the prompted score drop real, or a readout artifact or a style shift?

## Why

Across the score-only searches (m22 maths 27B, m23 maths 9B/122B/Gemma, m25 program output, m26 ordering), prompting
with feedback moved the median correct answer's probe score a quarter to a half of the way to the 1% line. Two readings
would deflate this for the post:

1. **Readout artifact.** Probes are read as the peak of a running average (EMA, alpha 0.5, starting at 0) of per-token
   log-odds. m26 found that filler before the first name drags the average down so the peak never climbs; m19 found
   evading replies need a strongly negative opener. If the drop disappears under other reasonable readouts, the attack
   exploits the readout, not what the probe represents.
2. **Style shift.** If non-task replies written in the attack's style drop just as much, and the probe still ranks task
   above non-task within that style, prompting shifts the whole distribution rather than hiding the task.

## Part A: readouts (saved data only, no GPU)

Per-token log-odds are saved for every reply (search rounds, final tests, plain, calibration) in the runs folders of
m22, m23, m25, m26 (and m19 for the white-box reference). For each, recompute the reply score under:
- the deployed readout (EMA peak), as a check that you reproduce saved peaks;
- per-token max (no smoothing);
- mean over reply tokens;
- EMA peak computed only from the first answer token onwards (find the answer span: the correct result for maths, the
  printed output for code, the first name for ordering);
- max over the answer span only.
For each readout: re-set the 1% line on the same calibration pool, and recompute the plain median, the frozen-test (or
best-round) median change from plain, the share of the gap closed, and counts under the line. Also per-round medians
for the Opus-with-scores and no-scores chains, so the per-round figure can be redrawn per readout.

Deliverable: one table (task × readout: plain median, line, best attack's change, gap closed, under-line count) and a
small-multiples figure of per-round medians under each readout. Say plainly which drops survive which readouts.

## Part B: within-style twins (needs generation; keep it small)

For each task's frozen best instruction(s) per writer (m22 once its frozen test exists, else the best search round; m23
9B; m25 neutral-description Claude run; m26 Opus-with-scores and no-scores), generate non-task twins: the same
instruction applied to that task's non-task requests (maths: no-maths lookups where the answer is given; code: the
literal-print programs; ordering: "order already given" requests, all in the same one-sentence format). Check what
already exists first: m19 has `twin_*`, m25 has literal-print controls per round, m26 may have no-ordering controls.
Generate only what's missing. Score with the same probes, instruction removed.

Report per task: (i) how far styled non-task replies move vs plain non-task replies; (ii) within-style AUROC (attack
replies vs styled non-task replies); (iii) where the 1% line would sit if calibrated on styled non-task replies, and how
many attack replies fall under it.

## Report (RESULT.md, plain language)

The two verdicts (readout artifact? style shift?) per task, the table and figure, and what the post can safely claim.

## Rules

- Modal names `probe-evasion_nabil-baugher_m27-<label>_kill-2026-11-30`; stop only your own apps, by app id. GPU cap
  about $60 (Part B only; batch generation and scoring). Claude API cap about $40 (validity judging of twins if needed).
- Keys via `load_dotenv('.env')`, never printed. No core-module edits; code in
  `src/probe_evasion/missions/m27_readout_and_style_checks.py`. Import from m22/m25/m26 modules; don't edit them.
- Write only in `missions/m27_readout_and_style_checks/`, that module, and `data/concepts/m27_*`.
- No repo-changing git. One critic check-in on the analysis before the verdict.
- Save every recomputed score table and every generated twin with per-token scores.
