# m31_more_chains: result

## Short answer

**Yes. Seeing the probe's scores clearly helps the prompt-writer, once the writer model is held fixed.** This holds for
both tasks and both writer models tried.

**Maths, Opus 4.8 writer, 8 chains with scores vs 8 without.**
- Over rounds 1-15 on the eval set, valid answers end up 3.4 log-odds below plain with scores and 0.8 below without.
- The difference is 2.6 log-odds (95% interval 2.0 to 3.2). Every scored chain beats every unscored chain (exact
  permutation p = 0.0002).
- On 100 untouched test problems, the last round's instruction is 3.5 log-odds better with scores (interval 2.7 to
  4.3; p = 0.0002). That is about a quarter of the distance from plain to the flag line.

**Ordering, Opus 4.8.**
- Rounds 1-15: -8.1 with scores vs -6.5 without. The difference is 1.6 log-odds (interval 0.7 to 2.5; p = 0.002).
- Last round on the test: 2.9 log-odds better with scores (interval 1.8 to 4.0; p = 0.0003).

**Opus 5.5 (pinned, no fallback; 4 vs 4 chains).** The effect points the same way and is at least as large.
- Maths: 2.5 log-odds on the eval set, 3.7 on the test.
- Ordering: 4.8 log-odds on the eval set, 9.1 on the test.
- With 4 vs 4 chains, p = 0.029 is the smallest p-value the test can give. Every scored chain beats every unscored
  chain.

**Opus 5.5 refuses more when shown scores.** On maths, 19% of its answers with scores were refusals (18 of 94),
against 5% without (5 of 92). On ordering it was 3 of 67 vs 0 of 65. Opus 4.8 never refused (0 of 546).

**This revises the post's Result 3 ("probe scores add little").** That result compared a scored writer that was mostly
Opus 4.8 (after Opus 5.5 declined and the server fell back) with an unscored writer that was mostly Opus 5.5. It also
used an older, looser leak check.

**Most of the maths benefit is one discovery: answers in number words.**
- With scores, about 72% of valid replies give the result in words. Without scores, 2.5% (Opus 4.8) or 6.5% (Opus 5.5).
- Word answers score 3.5-4 log-odds below plain whichever writer produced them.
- m22's mixed-model unscored writer used words in 65% of its replies. That is why it looked almost as good as the
  scored writer.

**Still, nothing reaches the line.** No valid test reply in any arm scored under the 1% flag line (maths -5.80, ordering
-7.30). Scored maths instructions also cost validity: 382 of 500 test replies are valid, against 500 of 500 without
scores (Opus 4.8, last round).

## Terms

- **Plain:** Qwen3.5-27B answering with no instruction.
- **Score:** the target probe's reading of a reply, in log-odds (EMA running peak, instruction removed before reading).
  Lower means less "worked it out" signal.
- **Change vs plain:** a valid reply's score minus plain Qwen's score on the same problem and seed. The median is taken
  over valid (problem, seed) pairs: `m22.paired_change`.
- **Chain:** one writer improving one instruction over 16 rounds (round 0 is written before any feedback).
- **With / without scores:** the same writer and system prompt. Without scores, the writer sees Qwen's replies and their
  validity verdicts but no score, trace or under-the-line count. This is m22's no-scores control.
- **Pinned:** one model answers every turn, with no server-side fallback.
- **Mixed:** m22's and m26's Opus arms (Opus 5.5 with fallback to Opus 4.8). They are shown for reference only.
- **Last round / best round:** in the final test, each chain's round-15 instruction (selection-free) and its lowest-median
  eval round (m22's freeze rule, picked with the probe for both arms).

## What was run

- **Opus 4.8 writer, as the brief asked.**
  - m28's pinned writer (`pinned_opus`, `refusal_checked`).
  - 5 new chains with scores and 5 without, per task, 16 rounds, on m22's runner (`run_search`, unchanged).
  - These are pooled with m28's 3 + 3, giving 8 + 8 per task.
  - Every new chain's system prompt and opening message were asserted identical to m28's.
- **Opus 5.5 writer (coordinator's amendment).**
  - The same call on `claude-opus-5-5` with no server-side fallback, registered the same way.
  - 4 chains with scores and 4 without, per task.
  - A refusal is a failed attempt; 3 failed attempts make a failed round, and nothing runs that round (m22's rule).
- **Tasks.**
  - Maths: m22's one-sentence arithmetic, with probe `onpk@L48@s0`, line -5.80.
  - Ordering: m26's five-name puzzles, with probe `ordpk@L48@s0`, line -7.30.
  - Both use Qwen3.5-27B with reasoning off.
- **Eval set, validity and checks.**
  - The same eval set every round: 40 problems x seeds 1, 2.
  - The same validity rules as m22, m26 and m28.
  - The same leak check on every instruction (eval answers in digits, words, e or k notation; ordering: two or more
    names in order). The same Haiku refusal check.
  - Lines, calibration scores and plain replies are byte-identical to m28's.
- **Final test.**
  - Every new chain's best and last round, on 100 untouched problems x seeds 10-14 (500 replies), with m22's amended
    validity.
  - `frozen.json` was written before any test reply existed. The test arms match it exactly.
  - Plain reproduces m22's / m26's test (maths 500 valid with 498 of 500 replies identical; ordering 478 valid).
  - m28's chains use m28's own test rows.
- **Pre-registration.** Everything was pre-registered in STATUS.md before the runs, with two amendments, each written
  before its data existed.
  - The primary measure is per chain: the mean over rounds 1-15 of the eval change vs plain. Rounds whose instruction
    failed are skipped. Each round leaves out the eval problems that its instruction leaked.
  - The primary test is an exact two-sided permutation test over chains (12,870 relabellings for 8 vs 8, 70 for 4 vs 4).
  - Effect sizes are differences of means with Welch 95% intervals.
  - "Detectable" means p < 0.05 with Holm's correction over the two tasks.

## Main figures

- **`missions/m31_more_chains/figures/rounds_all_chains.png`:** per-round change vs plain for every chain (thin lines)
  and the median over chains (bold). One panel per writer model (pinned 4.8, pinned 5.5, earlier mixed) and per task.
  Orange is with scores, blue without; dashed lines are m28's 4.8 chains.
- **`missions/m31_more_chains/figures/scores_by_writer_model.png`:** one dot per chain in three views: the rounds 1-15
  mean (eval), the last-round final test and the best-round final test. Arms are in 2x2 order, plus the mixed arms for
  reference. Bars are arm means.

## Results

### 1. The scores effect within each writer model

Numbers are change vs plain in log-odds, as the mean over chains with the per-chain range in brackets.
D = with scores minus without (negative means scores help). The interval is Welch's 95% interval; p is the exact
permutation p-value over chains.

| task, writer | measure | with scores | without scores | D (95% interval) | p |
|---|---|---|---|---|---|
| maths, 4.8 (8 v 8) | **eval, rounds 1-15 (primary)** | -3.39 (-3.8 to -1.7) | -0.80 (-1.1 to -0.3) | **-2.58 (-3.16, -2.00)** | **0.0002** (Holm 0.0003) |
| | test, last round | -4.60 (-5.8 to -3.1) | -1.12 (-2.1 to -0.6) | -3.48 (-4.27, -2.69) | 0.0002 |
| | test, best round | -4.71 | -2.35 | -2.36 (-3.49, -1.23) | 0.0006 |
| maths, 5.5 (4 v 4) | eval, rounds 1-15 | -4.07 | -1.55 | -2.53 (-3.45, -1.60) | 0.029 (minimum) |
| | test, last round | -5.38 | -1.72 | -3.65 (-4.92, -2.38) | 0.029 |
| | test, best round | -4.72 | -3.79 | -0.93 (-2.14, +0.27) | 0.029 |
| ordering, 4.8 (8 v 8) | **eval, rounds 1-15 (primary)** | -8.12 (-9.6 to -6.4) | -6.50 (-7.6 to -5.3) | **-1.61 (-2.54, -0.69)** | **0.002** (Holm 0.002) |
| | test, last round | -9.69 (-11.2 to -8.4) | -6.80 (-8.6 to -5.5) | -2.89 (-4.02, -1.76) | 0.0003 |
| | test, best round | -9.77 | -8.49 | -1.29 (-2.13, -0.45) | 0.005 |
| ordering, 5.5 (4 v 4) | eval, rounds 1-15 | -9.37 | -4.56 | -4.81 (-7.68, -1.95) | 0.029 (minimum) |
| | test, last round | -13.03 | -3.95 | -9.08 (-12.11, -6.05) | 0.029 |
| | test, best round | -13.01 | -6.93 | -6.08 (-8.42, -3.74) | 0.029 |

As shares of the plain-to-line gap (maths 14.5 log-odds, ordering 17.4 on the eval set, 17.9 on the test):
- Opus 4.8's benefit from scores is 18% of the gap in maths (eval) and 24% (last-round test).
- In ordering it is 9% (eval) and 16% (test).
- Opus 5.5's benefit in ordering is about half the gap on the last-round test.

**Checks on the primary result (all pre-registered):**
- **Batch.** Relabelling only within m28's and m31's own chains gives p = 0.0004 (maths) and 0.004 (ordering).
  - Maths: m28's 3 v 3 alone gives D = -2.78; m31's 5 v 5 alone gives D = -2.46 (p = 0.008).
  - Ordering: m28's D = -2.60; m31's D = -1.02 (p = 0.09). m31's ordering batch shows a smaller effect than m28's, but
    in the same direction.
- **Round 0 (before any feedback) does not differ.** Maths 4.8: -1.70 vs -1.05 (p = 0.41). Ordering: -3.71 vs -3.88
  (p = 0.88). The benefit comes from the feedback, not from the prompt saying scores will be shown.
  - Subtracting each chain's round 0 leaves D = -1.93 (maths, p = 0.048) and -1.78 (ordering, p = 0.054) for Opus 4.8.
  - This adjustment is noisy, because round 0 is a single instruction per chain.
- **Carry-forward for failed rounds** (the chain keeps its previous instruction's change) barely changes anything.
  - Maths 4.8: D = -2.56. Maths 5.5: D = -2.45. Ordering 4.8: D = -1.52. Ordering 5.5: D = -4.77.
  - Failed-round rates by cause are in section 3.
- **When the gap opens.**
  - Maths 4.8: the gap is already there in rounds 1-3 (-1.56) and grows by rounds 13-15 (-3.63).
  - Ordering 4.8: no difference in rounds 1-3 (D = -0.04). The gap opens later: -2.75 by rounds 13-15. Without scores,
    the 4.8 ordering chains plateau around -6.5 from round 3 on.
- **The best-round test (picked with the probe) narrows the gap.** Best-of-16 selection with the probe also helps the
  unscored chains, for example maths 5.5 without scores: -1.72 at the last round, -3.79 at the best. That arm is
  "no scores, then the probe picks the round", which is not a clean no-scores condition. The last round is the clean
  comparison.

### 2. Writer model x scores (2x2)

Rounds 1-15 eval mean (primary), mean over chains:

| task | 4.8 with | 4.8 without | 5.5 with | 5.5 without | interaction (5.5 effect minus 4.8 effect) | reference: mixed with / without |
|---|---|---|---|---|---|---|
| maths | -3.39 | -0.80 | -4.07 | -1.55 | +0.05 (-0.95, +1.06) | -3.59 / -2.76 |
| ordering | -8.12 | -6.50 | -9.37 | -4.56 | -3.20 (-6.10, -0.30) | -7.35 / -6.12 |

Last-round test, same layout:

| task | 4.8 with | 4.8 without | 5.5 with | 5.5 without | interaction | reference: mixed with / without |
|---|---|---|---|---|---|---|
| maths | -4.60 | -1.12 | -5.38 | -1.72 | -0.17 (-1.54, +1.21) | -3.95 / -3.16 |
| ordering | -9.69 | -6.80 | -13.03 | -3.95 | -6.19 (-9.19, -3.20) | -9.30 / -7.98 |

- **Maths:** the scores effect is the same size for both models; the interaction is about 0. Opus 5.5 is slightly
  stronger than 4.8 in both conditions: 0.7 log-odds, significant only without scores (p = 0.004).
- **Ordering:** the scores effect is much larger for Opus 5.5. Its chains with scores reach -9.4 (rounds 1-15) and -13.0
  on the test, the lowest of any arm. Its unscored chains are the weakest: -4.6 on the eval set, and they drift back up
  to -4.0 by the last-round test.
- **The earlier mixed arms (reference) don't fit either pinned model,** because each confounds model and leak rule:
  - Their scored arm sits near pinned 4.8 with scores, which fits most of its rounds being written by Opus 4.8.
  - Their unscored arm (mostly Opus 5.5) is much stronger in maths (-2.8) than pinned 5.5 without scores (-1.6).
  - See section 4 for why.

### 3. Refusals and failed rounds

Answered writer attempts across all chains of the arm:

| arm | refusals | rounds failed, as a share of chain-rounds, by cause |
|---|---|---|
| maths, 4.8 with | 0 / 155 | 3.1%: eval answer copied into the instruction |
| maths, 4.8 without | 0 / 130 | 0.8%: no valid reply |
| maths, 5.5 with | **18 / 94 (19%)**, all API refusals | 12.5%: 4 rounds refusal, 4 rounds leak |
| maths, 5.5 without | 5 / 92 (5%) | 7.8%: 4 rounds leak, 1 no valid reply |
| ordering, 4.8 with | 0 / 133 | 7.8%: the instruction ran but no reply was valid |
| ordering, 4.8 without | 0 / 128 | 0 |
| ordering, 5.5 with | 3 / 67 (4%) | 1.6%: no valid reply |
| ordering, 5.5 without | 0 / 65 | 3.1%: no valid reply |

- **Opus 5.5's refusals come mid-chain.** The first refusal falls in rounds 2-5 for three of the four scored maths
  chains, and rounds 1-7 for three unscored maths chains. Refusals cost rounds only in the scored maths arm (4 of 64).
- **The Haiku check never fired.** All refusals were the API's own stop reason.
- **"No valid reply" rounds** are instructions so aggressive that Qwen fails validity on all 80 replies, for example
  spelling the names letter by letter or padding past 40 words. They are skipped as in m28. Carry-forward shows this
  does not matter.
- **Leak-rejected attempts** (the writer's worked examples contained eval answers): maths 4.8 with 30/155, 4.8 without
  2/130, 5.5 with 20/94, **5.5 without 27/92**. Many of 5.5's rejected attempts asked for number words; see section 4.

### 4. Mechanism in maths: number words (exploratory, added after seeing the data)

Valid eval replies in rounds 1-15, using m28's `result_in_words`:

| arm | share of replies giving the result in words | median change, word answers | median change, digit answers |
|---|---|---|---|
| 4.8 with scores | 72% | -3.7 | -2.2 |
| 4.8 without | 2.5% | -3.6 | -0.7 |
| 5.5 with scores | 73% | -4.1 | -1.6 |
| 5.5 without | 6.5% | -3.8 | -1.2 |
| mixed with (m22) | 71% | -4.0 | -2.2 |
| mixed without (m22) | 65% | -3.4 | -1.4 |

- **The scores teach the writer that number-word answers score low.** Writing the result in words is worth about
  3.5-4 log-odds whoever writes the instruction.
- **Unscored pinned writers rarely get Qwen into words.** They do sometimes ask for words: Opus 4.8 without scores in
  38 of the 128 instructions that ran (by a simple text match).
- **Scored writers also do better within digit answers** (-2.2 vs -0.7 for Opus 4.8). So words are most, but not all,
  of the gap.
- **Why m22's unscored control did better than pinned 5.5 without scores:** it got Qwen into words 65% of the time.
  - At the time, the parser did not reject eval answers written in words. m22's audit later found eval answers in 29
    of the control's 47 instructions.
  - Here, pinned 5.5 without scores lost 27 of 92 attempts to that (now stricter) leak check, 14 of them instructions
    asking for number words.
  - So the m22 / m26 vs pinned comparison mixes writer model, fallback and leak rule. It cannot be read as "Opus 5.5
    needs no scores".

### 5. Validity and the line

- **Final test, under the 1% line:** 0 of 500 valid replies in every one of the 24 new chains' best and last arms, and
  in m28's arms. The 5% full-check audit of maths test replies far above the line disagreed with the cheaper rule on 3
  of 858.
- **Valid share falls with scores in maths:**
  - Opus 4.8, last round: 382 of 500 (range 245-495) with scores, against 500 of 500 (499-500) without.
  - Opus 5.5: 318 (139-444) against 497.
  - Eval rounds 1-15: 76% vs 97% valid (Opus 4.8).
- **Ordering validity barely differs:** 434 vs 457 (Opus 4.8, last round, p = 0.15).
- **The medians above are over valid replies only.** In maths, part of the scored writers' advantage is bought with
  replies that fail validity.

## Verdict, in plain words

- With one writer model held fixed, a writer that sees per-token probe scores pushes the probe down clearly further
  than the same writer seeing only Qwen's replies and validity.
- **The size, for Opus 4.8:**
  - Maths: about 2.5-3.5 log-odds extra, on top of a no-scores drop of only about 1.
  - Ordering: about 1.5-3 log-odds extra, on top of a no-scores drop of about 6.5.
- **For Opus 5.5:** the same in maths, and much larger in ordering (5-9 log-odds).
- Both are detectable beyond doubt with 8 v 8 chains (p = 0.0002 maths, 0.002 ordering), and in the selection-free
  final test.
- **In maths, the scores mainly tell the writer that number-word answers read low.** An unscored writer can find this
  too (m22's control did, helped by a looser leak rule), but the pinned writers mostly didn't.
- **No arm gets a valid reply under the flag line.**

## Caveats

- **Opus 5.5 has 4 chains per arm.** p = 0.029 is the floor for 4 v 4. Its intervals are wide, especially in ordering.
- **Refusals make Opus 5.5's scored maths arm partly a "fewer rounds" arm** (12.5% of rounds failed). Carry-forward and
  failed-round-dropped versions agree.
- **m31's ordering batch shows a smaller 4.8 effect than m28's** (-1.0 vs -2.6). The pooled result holds within batches
  (stratified p = 0.004), but the true ordering effect for 4.8 may be nearer the low end of its interval.
- **The headline is the instruction-removed reading**, as in m22 / m26 / m28. Instruction-visible medians are saved per
  test arm (`median_visible`) but not analysed here.
- **The mixed m22 / m26 arms are reference only.** They differ in writer model mix, fallback and leak rule.
- **"Scores help" is about this setup:** one writer, 16 rounds, 80 eval replies per round and these feedback formats.
  It says nothing about writers that can test text against the probe directly.

## Cost

- **Claude** (estimates; judges counted as upper bounds by m22's call counter):
  - Search judges: about $124 (4.8: $55 maths + $15 ordering; 5.5: $40 + $14).
  - Writers: $93 (4.8: $23 + $32; 5.5: $19 + $19).
  - Test judges: about $78 (4.8 maths $47, ordering $6; 5.5 maths $21, ordering $4).
  - Haiku refusal checks and the pre-flight call: about $1.
  - Total about $295, against a combined cap of $500 ($300 brief + $200 amendment).
- **GPU (Modal billing for m31's apps):** $83 billed through 22:00 PDT (searches mostly). The billing report lags; with
  the rest of the searches and the four tests, I estimate about $130-150 in total, against a combined cap of $210.
  All m31 apps are stopped (they finished on their own).

## Files

- **Code:** `src/probe_evasion/missions/m31_more_chains.py`. It covers the pinned Opus 5.5 writer, run set-up, the
  freeze and the analysis. It imports m22's runner and m28's writer and edits neither.
- **Status:** `missions/m31_more_chains/STATUS.md`, with the design, pre-registration, both amendments, critic
  check-ins and the log.
- **Analysis:** `missions/m31_more_chains/analysis.json`. It holds every chain's per-round values, every test arm's
  measures, the 2x2 blocks, the refusal tables, the answer-format split and the spend.
- **Runs:**
  - Pinned Opus 4.8: `missions/m31_more_chains/runs/{qwen35_27b,qwen35_27b_ordering}/`.
  - Pinned Opus 5.5: `missions/m31_more_chains/runs_opus55/...`.
  - Each holds per chain `state.json`, `proposals.jsonl` (every writer call, with reasoning) and `round_XX.jsonl`
    (every reply with scores and validity); plus `frozen.json`, `test_rows.jsonl` and `judge_spend_*.json`.
- **Logs:** `missions/m31_more_chains/logs/`.
