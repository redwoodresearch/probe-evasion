# m31_more_chains: status

**Question.** With one pinned writer model (Claude Opus 4.8, no fallback), does seeing the probe's scores help the
prompt-writer push Qwen3.5-27B's own-reply probe scores down on valid answers, and by how much? m28 ran 3 + 3 chains
per task; m31 adds 5 + 5 per task (maths, ordering) so each arm has 8 chains.

**Phase:** done. Searches, tests, analysis, critic 2 and RESULT.md are final. All m31 Modal apps stopped.

## Entry points

Module `src/probe_evasion/missions/m31_more_chains.py` (imports m22's runner and m28's pinned writer; edits neither).
Out dir `missions/m31_more_chains/runs/` (roots `qwen35_27b/` maths, `qwen35_27b_ordering/` ordering).

```
# 1. local, no GPU: copy calibration scores + plain eval replies (m28.prepare_root), start the 20 chain states
#    (asserting each system prompt equals m28's chain of the same writer), write the chain files
uv run python -m probe_evasion.missions.m31_more_chains prepare maths   # and ordering
# 2. push the roots, launch the searches detached (m22's driver and run_search, unchanged)
uv run modal run -m probe_evasion.remote_search::push --paths missions/m31_more_chains/runs/qwen35_27b
uv run modal run --detach -n probe-evasion_nabil-baugher_m31-search-maths_kill-2026-11-30 \
    -m probe_evasion.missions.m22_score_only_iterations::m22_launch --chains-file missions/m31_more_chains/search_chain_maths.json
# 3. pull, freeze locally (writes <root>/frozen.json before any test reply), push, launch m22's run_frozen_test
uv run modal run -m probe_evasion.remote_search::pull --paths missions/m31_more_chains/runs/qwen35_27b
uv run python -m probe_evasion.missions.m31_more_chains freeze maths
uv run modal run --detach -n probe-evasion_nabil-baugher_m31-test-maths_kill-2026-11-30 \
    -m probe_evasion.missions.m22_score_only_iterations::m22_launch --chains-file missions/m31_more_chains/test_chain_maths.json
# 4. analysis + figure (saved files only; reads m28's runs read-only)
uv run python -m probe_evasion.missions.m31_more_chains report
```

## Design (written before any m31 chain ran)

Everything about a chain is identical to m28's pinned arms:
- **Writer:** m28's `pinned_opus` (`claude-opus-4-8` called directly, adaptive thinking, effort high, 1-hour cache, no
  server-side fallback, asserts the answering model) wrapped in m28's `refusal_checked` (Haiku refusal check; a refusal
  is a failed attempt). Monitor = `m28.pinned_monitor(task)` (registered via `pinned_maths` / `pinned_ordering`).
- **Writer prompt:** the task's m22 / m26 writer prompt (`m22.WRITER_SYSTEM` for maths, `m26.WRITER_SYSTEM` for
  ordering) with the full readout spec. `opus48_pinned` gets m22's scored feedback; `opus48_pinned_noscores` gets m22's
  no-scores prompt (`m22.noscores_system`) and feedback (random sample of 8 valid + 3 invalid replies, no scores). This
  is what m28's pinned arms use (m28 STATUS, "Arm 4"). `prepare` asserts every new chain's system prompt and opening
  message equal m28's chain_0 of the same writer: they do, for both tasks and both writers.
- **Runner:** m22's `run_search` (all 10 chains of a task in lockstep, 16 rounds 0-15, no zero-shot extras), the same
  40 eval problems x seeds 1, 2 per round, scored with the instruction removed (headline) and visible, m22's leak rule
  in the parser (maths answers in digits / words / e / k notation; ordering m26's 2-names-in-order rule), up to 3
  answered attempts per round.
- **Validity:** maths m19's strict rule (full check within 3 log-odds of the line, cheaper rule above, as m22 / m28's
  search); ordering m26's exact-answer rule + blind reader on every reply.
- **Lines and plain reference:** copied from m22 / m26 via `m28.prepare_root`; byte-identical to m28's
  (`lines.json`, `plain_eval.jsonl`, `calibration_scores.jsonl` checked with `cmp`). Maths 1% line -5.80, ordering -7.30.
- **Chain names:** m31's chains are `chain_0..4` in m31's folder; pooled with m28's `chain_0..2` (labelled by source).
  Chain index enters only as the seed for which replies the feedback samples; the writer's own sampling makes chains
  independent.

## Pre-registration (written before any m31 chain ran)

**Unit = one chain.** Pooled sample per task and arm: m28's 3 + m31's 5 = 8 chains (fewer only if a chain fails to
finish; reported).

**Primary measure (per chain), as m28's rule 4:** mean over rounds 1-15 that ran of the round's paired change vs plain
(per (problem, seed) with both replies valid: score minus plain's score; median over pairs; `m22.paired_change` via
`m28.unit_stats`), with each round's leaked eval problems left out under the task's leak rule (`m28.rounds_view`).
Failed rounds (no instruction) are skipped. Selection-free: no round is picked with the probe.

**Primary test (per task, maths and ordering separately):** D = mean over scored chains - mean over unscored chains
(negative = scores help). Exact two-sided permutation test over all C(16, 8) = 12,870 relabellings of the 16 chains.
Effect size: D with a 95% interval resampling chains within each arm (10,000 draws), Hedges' g, and D as a share of
the plain-to-line gap. **Decision rule:** "a detectable benefit from seeing probe scores" for a task if p < 0.05 and
D < 0; reported also with Holm's correction over the two tasks (the stricter p < 0.025 for the smaller p). "No
detectable benefit" otherwise, with D and its interval reported as the size we can rule out.

**Secondary (reported, no decision attached):**
1. Final test, **last round** of every chain (selection-free): per chain the median valid test score minus plain's
   (500 test replies: 100 untouched problems x seeds 10-14); 8 vs 8 permutation test and interval as above. For m28's
   chains, m28's own test rows (its `@last` arms); m31's from m31's test.
2. Final test, **best round** (m22's freeze rule, picked with the probe on the eval set, for both arms): same.
3. Per round: median and mean over chains of the paired change per arm; the figure with every chain.
4. Rounds 1-3 and 13-15 phase means per chain (same test); round 0 (zero-shot, before any feedback; should not differ).
5. Batch check: m31-only (5 vs 5) and m28-only (3 vs 3) differences, to see that the two batches agree.
6. Valid share per arm, writer failures (failed rounds, leaks), distinct instructions per chain (stagnation).

**Final test (pre-registered):** plain + for every new chain its best round (m22's freeze rule: lowest leak-adjusted
median valid score on the 80 eval replies among rounds with at least 40/80 valid; ties -> more valid, then earlier;
a chain with no eligible round gets no best arm, recorded) and its last round that ran (selection-free). 100 untouched
test problems x seeds 10-14. Validity as m22's amended test (maths: full check within 3 log-odds of the line, cheaper
rule above, 5% full-check audit of far-above-line replies; ordering: its full rule on every reply); every valid reply
under the 0.1% line re-judged by Opus (m19 / m26 rejudge). `frozen.json` is written by `freeze` before any test reply;
m22's `run_frozen_test` reads it. No arm re-frozen or re-run after test results are seen.

**Not pooled:** m22's and m26's Opus chains (Opus 5.5 with fallback to 4.8, models mixed by condition); quoted only as
context.

## Critic 1 (design; research-plan-reviewer) and actions: amendments written before any m31 result was seen

The critic found the chain set-up identical to m28's pinned arms (writer, prompts, feedback, leak rule, eval set,
seeds, plain reference, lines; differences only in score-cache file name and which chains share a lockstep batch) and
the test plumbing correct (m22's `freeze` returns m31's `frozen.json` unchanged; arm keys match m28's test rows). It
noted that m28's 3v3 already separates the arms completely on the primary measure (maths about -3.6 vs -0.8, ordering
about -8.5 vs -6.0), so the informative output is the size of the effect, not the yes/no. Amendments:

1. **Headline effect size, named now.** Two numbers, both selection-free, each with its interval:
   (a) the primary eval measure's D (rounds 1-15; this decides "detectable"), and
   (b) **the last-round final-test D** (held-out problems, end of search; the "how big at the end" number).
   The primary averages in the early learning rounds, so it understates the end-state gap if scored chains keep
   improving; both are reported side by side. Best-round test D is secondary (selected with the probe).
2. **Scope.** The verdict is about Claude Opus 4.8 as the writer. m22's no-scores writer (mostly Opus 5.5) reached
   about -3.3 on maths over rounds 1-15, pinned 4.8 without scores about -0.8 (m28), so the no-scores baseline depends
   on the writer model. RESULT.md will not carry "scores help by X" over to Opus 5.5, and will say how this revises
   the post's Result 3 (which was based on the model-mixed m22 / m26 chains).
3. **Batch check:** batch-stratified exact permutation (relabel only within m28's 6 and m31's 10 chains; 5,040
   relabellings) reported next to the pooled test (pooled stays primary).
4. **Stagnation** (unscored chains resubmitting one instruction) is part of what a writer without the signal does, not
   a confound; the mean over rounds penalises exploration, which works against the scored arm, so a detected benefit
   is conservative. Distinct instructions per chain reported.
5. **Round 0 differs by design** (the system prompts say whether scores will be shown). Sensitivity: per chain, rounds
   1-15 mean minus round 0 (separates "the feedback helps" from "the prompt framing helps"); noisy, since round 0 is
   one instruction per chain.
6. **Intervals:** the percentile bootstrap over 8 chains is narrow; Welch's t interval reported alongside.
7. **Under-line counts compared:** per chain share of test replies valid and under the 1% line, both arms (expected
   0 everywhere, as in m22 / m26 / m28).
8. Code: assert every chain has a primary value; Holm-adjusted p computed in `report()`; m31 writer and judge spend
   summed in `analysis.json`; n per arm printed next to p; figure's line labelled approximate.

## Amendment 2 (coordinator, 10-11 ~22:10): pinned Opus 5.5 pair and a 2x2. Written before any 5.5 chain ran

**Why.** m28's pinned Opus 4.8 without scores does far worse than m22's mostly-Opus-5.5 no-scores writer, so "do
scores help?" may depend on the writer model. Coordinator asked for the reconciling comparison.

**Runs.**
- Writer `pinned_opus55` (this module): m28's `pinned_opus` call (adaptive thinking, effort high, 1-hour cache) on
  `claude-opus-5-5`, **no server-side fallback**, asserts the answering model, wrapped in m28's `refusal_checked`
  (Haiku check on the parsed instruction). Registered in `m22.WRITERS` at import as `opus55_pinned` (scored feedback)
  and `opus55_pinned_noscores` (`feedback: "noscores"`), exactly as m28 registers its 4.8 writers. Monitors
  `pinned55_maths` / `pinned55_ordering`.
- Same prompts: `prepare(task, "opus55")` asserts each 5.5 chain's system prompt and opening equal m28's 4.8 chain of the
  same feedback kind (passed for both tasks and both kinds).
- 4 chains with scores + 4 without, 16 rounds, maths and ordering; own root `runs_opus55/` (separate drivers never share
  files), cache tag `search55`. Pre-flight: one round-0 call answered by claude-opus-5-5 (end_turn, $0.04).
- **Refusals:** a refusal (API stop reason "refusal", or the Haiku check flagging the instruction) is a failed attempt;
  3 failed attempts = a failed round, nothing runs, and m22's "no usable instruction" message is sent (m22's rule).
- Final test: best and last round of every 5.5 chain (same rules as above); a chain with no round run has no arms
  (recorded in `frozen_absent.json`).

**Refusal rate (a result in its own right), per arm and task:** share of answered writer attempts that were refusals
(API refusal vs Haiku-flagged, split), failed rounds per chain, and the first round each chain refused; same for the
4.8 arms.

**2x2 analysis (writer model 4.8 / 5.5 x scores / no scores), per task, on the primary per-chain measure:**
- cell means with spread (4.8: 8 + 8 chains; 5.5: 4 + 4);
- the scores effect within each model (exact permutation, Welch interval); the model effect within each feedback kind
  (same);
- the interaction (scores effect for 5.5 minus scores effect for 4.8) with a Welch-type interval;
- the same on the last-round final-test change;
- m22's / m26's mixed Opus arms (5.5 with fallback to 4.8; 3 + 3 chains) shown for reference, not tested.
- **Failed rounds:** the primary averages rounds that ran (as for 4.8). Sensitivity, pre-registered: carry-forward,
  where a failed round takes the change of the chain's last instruction that ran (0 = plain if none ran yet), i.e. the
  attacker keeps using its previous instruction. A chain with no round run is reported but has no primary value.
- Budget: up to +$200 Claude, +$60 GPU (coordinator).

## Critic 2 (before the verdict; fresh research-reviewer, rerun at the lead's request) and actions

The critic recomputed every headline number from the raw files (all match exactly) and found no code bugs. Actions:

1. *Must-fix:* the "asked for number words in 38 of 128" count came from a regex and was wrong. Most matches said
   "never spelled out" or "in plain words". By reading the instructions: Opus 4.8 without scores asked in 5 of 128
   (all early); Opus 5.5 without scores asked in 4 of 60 that ran, plus 15 of 27 leak-rejected attempts (my reading;
   the critic counted 13). Qwen complied every time. The writers' notes show they dropped words after validity dips
   and reasoned from the probe description that words would raise the score. Section 4 and the verdict were rewritten
   around this, using behavioural counts (rounds where most valid replies are in words: 87 / 3 for 4.8, 28 / 4 for 5.5;
   chains settling on words 6 of 8 vs 0 of 8).
2. *Must-fix:* validity-penalised sensitivity added (not pre-registered; an invalid reply counts as no evasion):
   - maths 4.8: eval D -1.63 (p 0.0002), test D -2.02 (p 0.005);
   - maths 5.5: eval -0.88 (p 0.14), test -1.36 (p 0.31);
   - ordering 4.8: eval -0.79 (p 0.067), test -2.80 (p 0.0008).
   Now in the code (`primary_invalid_as_plain`, `change_invalid_as_plain`) and RESULT section 1. Size claims are given
   as ranges.
3. *Must-fix:* rounds with no valid reply counted as plain: ordering 4.8 D -0.94 (p 0.052). Ordering is now described
   as firmest on the last-round test and less robust on the eval set ("beyond doubt" dropped).
4. *Must-fix:* mixed-arm description corrected. Maths unscored was 25 rounds by 5.5 and 23 by 4.8; scored was 30 by
   4.8. m22's maths control used words in 2 of 3 chains, and its word rounds contained 2-5 eval answers that today's
   leak rule would reject. So the leak rule, not the writer model, explains the gap.
5. Should-fix, all done:
   - round 0 described as "not significant; part may be prompt framing";
   - section 4 labelled exploratory, also in the short answer;
   - refusals softened, with a chain-level test (maths p 0.17, ordering 0.14);
   - valid share now over rounds 1-15 that ran, including 0-valid rounds;
   - best-round spreads given, with a note on the maths 5.5 best-round permutation p vs its Welch interval;
   - decision rule stated as pre-registered (p < 0.05, Holm alongside).
6. Batch: m31's ordering scored chains started slowly (round 0 D +0.74, rounds 1-3 D +0.99), and the batches agree
   late (rounds 13-15 -2.32 vs -3.46; last-round test -2.33 vs -3.81). The batch difference is borderline chance
   (p about 0.07); nothing structural differs. Reworded.
7. Minor, done: terms defined; maths test gap 14.7; words-share denominators and the other-numeral forms (19% of 5.5
   scored replies) noted; dotted-line legend added to the summary figure; p-value floors stated (0.1 for 3 v 3);
   test-arm count corrected (36 new arms per task); audit 3 / 858 (m31) and 5 / 1,441 (with m28); Hedges' g, bootstrap
   intervals and distinct instructions reported.

## Spend estimate (before runs)

- Claude: maths judges ~12,800 search replies x ~$0.0045 ~ $58; ordering ~$28; writers (pinned 4.8) ~$18 scored + ~$8
  unscored per task ~ $52; tests: maths ~10,500 replies ~ $38, ordering ~$10. Total ~$190 (cap ~$300). m22's driver
  stops if its judge estimate passes $130 per search (pinned-writer cost is not counted by that guard; tracked here).
- GPU: two search drivers (2 H100 samplers + 2 H200 scorers, ~1.5-2 h) ~ $50; tests ~ $25. Total ~ $75 (cap ~$150).

## Log

- 10-11 21:30 Read brief, m22 / m26 / m28 results, status and modules. m28's pinned chains (both tasks) finished;
  m28's ordering test finished; m28's maths test running (app ap-PAatxtx7NOFq4GyIfximKV).
- `prepare` done for both tasks: roots set up, 20 chain states started, prompts asserted equal to m28's.
- 21:39 Roots pushed (`remote_search::push`). Searches launched detached: maths **ap-tQTTZ7b1yRgG1riU52Itco**
  (`m31-search-maths`), ordering **ap-19DIiJ3bYejyGu9tuGBztE** (`m31-search-ordering`); logs `logs/search_*.log`.
  Launched while the design critic runs: the chain set-up is fixed by the brief (identical to m28's pinned arms), so
  critic findings can only change the analysis and the test, which have not run. Round 0: every writer answered on the
  first attempt with claude-opus-4-8.
- 21:45 Design critic returned; amendments (section "Critic 1") written before any m31 result. Lead told about the
  writer-model dependence; lead asked for a pinned Opus 5.5 pair (Amendment 2).
- 21:47 Opus 5.5 pair prepared (prompts asserted equal to m28's), pre-flight call fine, roots pushed, launched detached:
  maths **ap-SKK9HXZMF6N6tgGNgT2EU7** (`m31-search55-maths`), ordering **ap-C6Ztc091oGiTxyylBMtPPY**
  (`m31-search55-ordering`); logs `logs/search55_*.log`. Round 0: all 16 writers answered on the first attempt.
- ~22:00 Lead: m22's runner gained reasoning-on support for m32 (new Monitor fields default off; behaviour with
  thinking off identical; rows also get `n_reply_tokens`). The running searches use the code uploaded at launch; m31's
  tests will run on the updated runner with thinking off (identical behaviour).
- 22:05 Progress: 4.8 maths round 3, ordering round 3; 5.5 maths round 2-3, ordering round 4. Early 5.5 events: several
  failed rounds from the leak rule in both 5.5 maths arms (the writer copies eval answers into examples), and the first
  5.5 refusal (opus55_pinned/own/chain_3, round 2, 3 refused attempts = failed round). The analysis now splits failed
  rounds by cause (leak / refusal / other). Analysis code dry-run on m28's saved data works (2x2, refusal table,
  reference arms from m22 / m26, test records incl. m26's last rounds from m28's ordering test).
- 22:07 Session handed back before the runs finished (state above).
- 22:10 Resumed (lead). Lead's analysis request: in the 2x2, report the primary both with failed rounds dropped and
  with carry-forward (both already computed: `primary`, `primary_carry_forward`), and the failed-round rate by cause
  per arm (`refusals[arm].failed_round_causes`).
- 22:30 5.5 ordering search finished (all 8 chains, 16 rounds; driver Claude ~$14 judges + $19.3 writers). Pulled,
  frozen locally (16 arms, none absent; `frozen.json` written before any test reply), pushed `frozen.json`; test
  launched detached: **ap-0a8zJRubdbu97JRKA5lrci** (`m31-test55-ordering`, log `logs/test55_ordering.log`). The
  traceback after "finished" in the search log is vLLM's shutdown noise (also in m28's logs).
- 22:50 5.5 maths search finished (8 chains x 16 rounds; driver: ~$40 judges, $18.6 writers). Pulled, frozen (16
  arms, none absent), pushed; test launched: **ap-1wJsNjhVVerHS7D9xxc4Ld** (`m31-test55-maths`, `logs/test55_maths.log`).
  m28's maths test had finished meanwhile (`m28_test_rows.jsonl`, 22:06).
- ~23:00 4.8 ordering search finished (10 chains x 16 rounds). Pulled, frozen (20 arms, none absent), pushed; test
  launched: **ap-cKXzPgbtO7Cjjk3s8YMkEd** (`m31-test-ordering`, `logs/test_ordering.log`).
- ~23:10 4.8 maths search finished (driver: ~$55 judges, $22.7 writers). Pulled, frozen (20 arms, none absent),
  pushed; test launched: **ap-hLXTUgmYw9V3zOJOTvizqz** (`m31-test-maths`, `logs/test_maths.log`). All four searches
  done; all four tests running.
- ~23:30 All four tests finished and pulled. Test arms match `frozen.json` in every root; plain reproduces m22's /
  m26's test (maths 500 valid, 498/500 replies identical to m22's; ordering 478 valid). Test judge calls: 4.8 maths
  15,600 (~$47), 4.8 ordering 1,856 (~$6), 5.5 maths 6,882 (~$21), 5.5 ordering 1,283 (~$4).
- Analysis additions (no change to any pre-registered measure): failed-round causes and rates per arm; leak-rejected
  attempts per arm; maths answer-format split (number words vs digits, m28's `result_in_words`), added after seeing that
  m22's mixed no-scores control beat both pinned no-scores arms (exploratory, labelled so in RESULT).
- RESULT.md drafted; critic 2 launched.
- Critic 2 rerun (fresh) at the lead's request; findings applied (section above). `report` rerun with the added
  sensitivities; figures regenerated. RESULT.md final.

## Spend log

| when | what | estimate |
|---|---|---|
| 22:05 | 4 searches so far (driver logs): 4.8 maths ~$15 judges + ~$4.4 writer; 4.8 ordering ~$4 + ~$5.7; 5.5 maths ~$8 + ~$2.6; 5.5 ordering ~$5 + ~$4.2 | ~$49 Claude so far; projected ~$190 search + ~$80 tests (of $300 + $200) |
| 23:40 | Final Claude (estimates): search judges ~$124, writers $93, test judges ~$78, Haiku + pre-flight ~$1 | ~$295 (cap $500 incl. amendment) |
| 23:40 | GPU: Modal billing for m31 apps $83 through 22:00 PDT (lags); estimate incl. rest of searches + tests | ~$130-150 (cap $210) |
| 22:05 | GPU: 4 drivers (2 samplers + 2 scorers each), ~25 min so far | ~$15 so far; projected ~$90 search + ~$40 tests (of $150 + $60) |
