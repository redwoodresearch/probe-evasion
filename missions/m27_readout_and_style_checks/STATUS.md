# m27_readout_and_style_checks: status

**Phase:** done. RESULT.md written after the critic check-in (findings acted on, listed below and in RESULT.md).

## Part A verdict (saved data only; no GPU); superseded first version corrected after the critic

**Not a readout artifact.** The drop survives at least four of five readouts on every task (the four besides the
deployed EMA peak are diagnostics). The exception: Gemma's reply mean loses it (7 plain answers under that line too).
Share of each readout's own plain-to-line gap closed, mean over the pre-registered frozen best-round Opus-with-scores
arms, frozen test (m22's arrived during the mission), 95% intervals resampling test problems in RESULT.md:

| task | EMA peak (deployed) | per-token max | mean over reply | EMA from first answer token | max over answer span |
|---|---|---|---|---|---|
| maths 27B (m22) | 35% | 29% | 47% | 34% | 27% |
| maths 9B (m23) | 47% | 50% | 76% | 40% | 35% |
| maths 122B (m23) | 44% | 34% | 40% | 50% | 36% |
| maths Gemma (m23) | 19% | 13% | -25% | 16% | 8% |
| code (m25, m22's description) | 26% | 31% | 47% | 13% | 23% |
| code (m25, told label kinds) | 30% | 39% | 50% | 19% | 32% |
| ordering (m26) | 53% | 42% | 42% | 39% | 42% |

- Part of the drop is pre-answer text (restarting the average at the first answer token): code about half, ordering
  and 9B less, maths 27B / 122B / Gemma none. Under-line counts stay near 0 except maths 9B (36 deployed; 70-300 on the
  diagnostics).

## Part B verdict

**Partly a style shift that squeezes the task vs non-task gap from both sides, but no hiding within style.** Styled
non-task twins rise by up to +6 log-odds (most would be false alarms at the deployed line); within-style AUROC
0.989-1.000 for every score-only arm; a line set on styled twins (removed or visible context) catches all attack
replies except a handful per arm on maths 9B and 0-10 on one ordering arm. m19's white-box template is the exception
(AUROC 0.93, 305/478 under the styled line) but is fully caught with the instruction visible and a styled line in that
context (0/478). Answer swap: maths / code scores don't depend on the answer being right; ordering partly does.

## Critic check-in (research-reviewer) and actions

1. Must-fix: computing maths twins passed cleaning (per-unit quotients), creating a spurious "30/45 under the visible
   line" (27B Opus chain 0). DONE: maths twins / non-answers with any number not in the request (outside the answer
   span) are rejected; 2-58 per arm, all genuine computations on inspection; that arm is now 0 under.
2. Must-fix: "survives every readout" false (Gemma mean; 9B counts). DONE: reworded with counts.
3. Must-fix: stale table and mixed arm sets. DONE: the Opus mean uses best-round arms only (as Part B); all tables
   regenerated on m22's frozen test.
4. Must-fix: Part B for maths 27B on the frozen test. DONE (twins unchanged: frozen instructions identical); answer
   swap rescored on test rows (app ap-n33XtQ0zxUkfpJ0kvmsKoR); the eval-row version kept as `forced_eval_rows.jsonl`.
5. DONE: raw-scale retention and bootstrap intervals (problems resampled) for deployed vs diagnostic.
6. DONE: per-round figure normalised by the plain eval median, caption caveat, raw-scale figure added.
7. DONE: 95th percentiles next to 99th-percentile lines from ~200 replies.
8. DONE: attack vs in-context non-answers (AUROC 0.970-0.999; counts under 99th / 95th pct); ordering non-answers
   cleaned the same way (correct order absent, m26 audit): 154 of 200 kept.
9. DONE: style result framed by twin shift and separation retained. 10. DONE: m19 reproduction check compared
   probabilities with log-odds; fixed (max difference < 1e-11). 11. DONE: code swap limits stated (56% / 73% of peaks
   at or after the change; 303 unswappable). 12-14: noted.

## Decisions (Part A)

- **Data.** Every saved reply's per-token log-odds for the target probe: round files, `plain_eval.jsonl`,
  `test_rows.jsonl`. Calibration pools: `calibration_scores.jsonl` has peaks only; per-token log-odds come from each run
  root's `score_cache*.jsonl` (keyed by `item_key(request, reply)`), found for every calibration reply in every run,
  including all 1,301 of m22's (lead's amendment 3: no rescoring needed). Check: recomputed EMA peaks equal the saved
  peaks exactly (max difference < 1e-11 over ~135k replies); re-set deployed lines equal the saved lines (m22: -5.79 vs
  m11's -5.80, the reference line m22 asserts within 5%).
- **Readouts** (all diagnostics except the first): EMA peak (alpha 0.5, start 0; deployed); per-token max; mean over
  reply tokens; EMA peak from the first answer token (the average restarted at 0 there, so earlier text cannot drag
  it); max over the answer span. Also saved, not tabled: the deployed average's peak restricted to tokens from the
  first answer token on.
- **Answer span.** maths: the result in digits (any grouping), decimal / e / x10^ / k notation, English number words
  or Roman numerals; code: the printed output (m25's `output_regex`); ordering: the five answer names (first token of
  the first name starts the span). Calibration replies: the stated value (maths: the given figure; code: a lookup's
  literal or a recalled answer; ordering: the pair, mention order or recalled list). Valid replies the rules miss
  (other languages, glued number words, Chinese numerals: ~2,400, mostly maths 122B and 27B) are located by Haiku 5.5
  quoting the span; spot-checked. Unlocated after both: 1 test, 1 search and 14 calibration replies of ~120k (dropped
  from the span readouts only).
- **Selection.** "Best attack" is chosen separately under each readout among frozen arms (symmetric); the Opus mean
  above uses all frozen Opus-with-scores arms, no selection beyond the pre-registered freeze. m22 (no frozen test yet):
  each chain's best eligible eval round under that readout. Medians: valid replies only; leaked eval problems left out
  (m22 / m26 rules); refused rounds left out.

## Lead's amendments (received during Part A)

1. Part B: ~200 in-context non-answers per task for maths and code; gap closed also on a scale with 0 = in-context
   non-answer level and 0 = styled-twin level.
2. Score twins instruction-visible as well as removed; an instruction-visible number with its line set on styled
   twins in that context.
3. Done (above): calibration per-token log-odds found in the caches.
4. Haiku audit on twins (drop twins that compute anyway).
5. Optional: teacher-forced attack replies with a wrong answer of the same form.
- Alternative readouts labelled as diagnostics. Caps: GPU ~$90, Claude ~$50.

## Part B (decisions and first numbers)

- **Instructions:** each writer's frozen instruction (frozen.json; for m22, whose frozen test is still running, the same
  pre-registered leak-adjusted freeze rule `m22.best_round` applied to the saved chains): maths 27B Opus-with-scores
  and no-scores (6), maths 9B Opus and self (6), code m25 neutral Opus (3), ordering Opus with and without scores (6).
- **Twins** (Qwen, seed 0, 96 tokens, m22's sampler): 200 non-task requests per instruction and once with no
  instruction. maths: m11's 200 no-maths lookups (m19's twin set, so m19's saved twins compare directly); code: 200
  fresh literal-print programs (m25's generator, seed 2727); ordering: 200 fresh puzzles in m26's "order already
  given" format. Clean = finished, states the given value (maths: located by rule or Haiku; code: m25's output rule;
  ordering: m26's exact-order rule) and the existing Haiku audit says nothing is worked out (maths: m1a's computation
  prompt run on Haiku; code: m25's audit; ordering: m26's audit). Clean shares 71-100%; the two lowest (9B Opus
  chain 1 and self chain 0, ~72%) are instructions that make 9B compute anyway.
- **In-context non-answers** (amendment 1): maths and code, 100 test problems x 2 "do not solve; instead name the
  quantity / restate the first number (variable / first-line value)" instructions, read with the instruction removed;
  clean = finished, correct result absent, audit passes. Ordering: m26's saved non-solving replies.
- **Scored** with the same probes, instruction removed and visible. Attack replies: the frozen test (m22: the frozen
  round's 80 eval replies).
- **Answer swap** (amendment 5): every valid plain and frozen-arm reply re-read with its answer replaced by a wrong one
  of the same form, wrong from the first answer token (first draft changed a later digit / name, after the peak:
  redone).
- **First numbers (before the critic and the m22 frozen test; superseded by the verdicts above).** Styled twins move up by 0 to +6 log-odds against plain twins; within-style AUROC (attack vs
  styled twins) is 0.989-1.000 on every score-only arm (m19 white-box: 0.93); under a 1% line set on styled twins,
  0 attack replies for maths 27B and code, 0-10 for ordering, 5-40 per arm for maths 9B; m19's template 305/478. With
  the instruction visible and the line set on styled twins in that context: 0 for nearly every arm (m19: 0/478).
  Wrong answers of the same form score about as high as right ones for maths and code (median change -0.2 to -0.3;
  code 0.0); for ordering a wrong order lowers the plain score by 3.3. Full numbers: `part_b/summary.json`.

## Spend

- Claude: Haiku span locating (~2,400 calls) and twin / non-answer audits (~5,000 calls): ~$4.
- GPU: Part B generation and scoring (app ap-QlsaTpYwWoiGcGZ53YenWO, ~22 min, 27B and 9B samplers and scorers), the
  answer-swap rescoring (ap-AaFJ8eJRWoIkXQNEAE9AGZ, ~10 min) and the 27B swaps on the frozen test
  (ap-n33XtQ0zxUkfpJ0kvmsKoR, ~8 min): estimated ~$20 in total (cap $90). All apps finished on their own.
- Claude total ~$6 (cap $50).

## Log

- Part B: `uv run modal run -n probe-evasion_nabil-baugher_m27-twins_kill-2026-11-30 -m probe_evasion.missions.m27_readout_and_style_checks::m27_part_b` (both apps finished on their own), then `python -m ... part_b`.
- Part A run (`uv run python -m probe_evasion.missions.m27_readout_and_style_checks part_a`, then `part_a_outputs`).
