# m31_more_chains: result

## Short answer

**Yes. With the writer model held fixed, seeing the probe's scores helps the prompt-writer.** The benefit passes the
pre-registered test in both tasks, for Opus 4.8 and for Opus 5.5.

**Maths, Opus 4.8 writer (8 chains with scores vs 8 without).**
- Rounds 1-15, eval set: valid answers end 3.4 log-odds below plain with scores and 0.8 below without.
  - Difference D = -2.6 (95% interval -3.2 to -2.0). Every scored chain beats every unscored chain (exact permutation
    p = 0.0002).
- Last-round instruction on 100 untouched test problems: D = -3.5 (-4.3 to -2.7), p = 0.0002. That is about a quarter
  of the distance from plain to the flag line.
- Counting invalid replies as no evasion shrinks the gap to D = -1.6 (eval) and -2.0 (test), still p <= 0.005.
- **So the benefit is about 1.6 to 3.5 log-odds, depending on how invalid replies are treated.**

**Ordering, Opus 4.8.**
- Rounds 1-15: D = -1.6 (-2.5 to -0.7), p = 0.002.
- Last-round test: D = -2.9 (-4.0 to -1.8), p = 0.0003, and robust to how invalid replies are treated.
- The eval-set result is less robust:
  - counting invalid replies as no evasion: D = -0.8 (p = 0.07);
  - counting rounds where no reply was valid as plain: D = -0.9 (p = 0.05);
  - m31's own 5 v 5 batch alone: D = -1.0 (p = 0.09).
- **So the ordering benefit is about 1 to 3 log-odds, firmest at the end of the search.**

**Opus 5.5 (pinned, no fallback; 4 vs 4 chains).** Every scored chain beats every unscored chain in both tasks. With
4 vs 4, p = 0.029 is the smallest p-value possible.
- Maths: about as large as for 4.8 (eval D -2.5, test D -3.7). Counting invalid replies as no evasion it is -0.9 to -1.4
  and not significant (p 0.14 and 0.31).
- Ordering: much larger (eval D -4.8, test D -9.1), unchanged by any of the sensitivity checks.

**Opus 5.5 refuses more often when shown scores, but not detectably so at chain level.**
- Maths: 18 of 94 attempts (19%) with scores vs 5 of 92 (5%) without.
- Ordering: 3 of 67 vs 0 of 65.
- Per chain: maths p = 0.17, ordering p = 0.14.
- Opus 4.8 never refused (0 of 546).

**This revises the post's Result 3 ("probe scores add little").** That result came from m22 / m26 Opus arms that mixed
writer models (Opus 5.5 with server-side fallback to 4.8) and used an older, looser leak check.

**Exploratory: in maths, the scores mainly teach the writer that answers written in number words score low.**
- Scored writers get Qwen to answer in words; the pinned unscored writers almost never ask for it.
- When they did ask, they dropped it after a validity dip, and several reasoned from the probe description that words
  would *raise* the score. Without scores nothing corrects that.

**Nothing reaches the line.** No valid test reply in any arm scored under the 1% flag line (maths -5.80, ordering -7.30).
Scored maths instructions also cost validity: 382 of 500 test replies are valid, against about 500 without scores
(Opus 4.8, last round).

## Terms

- **Plain:** Qwen3.5-27B answering with no instruction.
- **Score:** the target probe's reading of a reply, in log-odds. It is the running peak of a moving average of
  per-token scores, read with the instruction removed. Lower means less "worked the answer out" signal.
- **Change vs plain:** a valid reply's score minus plain Qwen's score on the same problem and seed. Per round, the
  median over valid (problem, seed) pairs.
- **Chain:** one writer improving one instruction over 16 rounds. Round 0 is written before any feedback.
- **With / without scores:** the same writer and system prompt. Without scores, the writer sees Qwen's replies and
  their validity verdicts but no score.
- **Pinned:** one model answers every writer turn (no server-side fallback to another model).
- **Mixed:** m22's and m26's Opus arms, where Opus 5.5 fell back to Opus 4.8 when it declined. Shown for reference
  only.
- **D:** with scores minus without, as a difference of means over chains. Negative means scores help.
- **p:** exact two-sided permutation p over chains. Its floor is 0.0002 for 8 v 8, 0.029 for 4 v 4 and 0.1 for 3 v 3.
- **Interval:** Welch's 95% interval for D.
- **Last round / best round** (final test): each chain's round-15 instruction (selection-free), and its lowest-median
  eval round (m22's freeze rule, picked with the probe for both arms).
- **Leak check:** an instruction that contains an eval answer (in digits, words, e or k notation; ordering: two or
  more names in order) is rejected. 3 rejected attempts make a failed round.

## What was run

- **Opus 4.8, as the brief asked.** m28's pinned writer, with 5 new chains with scores and 5 without per task,
  16 rounds, on m22's runner (unchanged). Pooled with m28's 3 + 3, this gives 8 + 8.
- **Opus 5.5 (coordinator's amendment).** The same call on `claude-opus-5-5` with no fallback; 4 + 4 chains per task.
  A refusal is a failed attempt.
- **Prompts.** Every new chain's system prompt and opening message were asserted identical to m28's.
- **Tasks.** Maths: m22's one-sentence arithmetic (probe `onpk@L48@s0`, line -5.80). Ordering: m26's five-name puzzles
  (`ordpk@L48@s0`, line -7.30). Both on Qwen3.5-27B, reasoning off.
- **Shared with m22 / m26 / m28.** The same 80 eval replies per round, validity rules, leak and refusal checks, lines
  and plain replies (byte-identical to m28's).
- **Final test.** Every new chain's best and last round, on 100 untouched problems x seeds 10-14 (500 replies).
  - `frozen.json` was written before any test reply, and the test arms match it exactly.
  - Plain reproduces the earlier tests: maths 500 valid, 498 of 500 replies identical to m22's; ordering 478 valid.
  - m28's chains use m28's own test rows.
- **Pre-registration (STATUS.md, before the runs, plus two amendments each written before its data).**
  - Primary measure: per chain, the mean over rounds 1-15 of the eval change vs plain. Failed rounds are skipped, and
    each round's leaked eval problems are left out.
  - Decision rule per task: "detectable" if p < 0.05 and D < 0, with Holm's correction over the two tasks reported
    alongside.
  - Effect sizes as D with intervals.

## Main figures

- **`missions/m31_more_chains/figures/rounds_all_chains.png`:** per-round change vs plain, every chain (thin lines) and
  the median over chains (bold). One panel per writer model (pinned 4.8, pinned 5.5, earlier mixed) and per task.
  Orange is with scores, blue without; dashed lines are m28's 4.8 chains.
- **`missions/m31_more_chains/figures/scores_by_writer_model.png`:** one dot per chain for the rounds 1-15 eval mean, the
  last-round test and the best-round test. Arms are in 2x2 order, plus the mixed arms for reference; bars are arm means.

## Results

### 1. The scores effect within each writer model

Change vs plain in log-odds: mean over chains, with the per-chain range in brackets.

| task, writer | measure | with scores | without scores | D (95% interval) | p |
|---|---|---|---|---|---|
| maths, 4.8 (8 v 8) | **eval, rounds 1-15 (primary)** | -3.39 (-3.8 to -1.7) | -0.80 (-1.1 to -0.3) | **-2.58 (-3.16, -2.00)** | **0.0002** (Holm 0.0003) |
| | test, last round | -4.60 (-5.8 to -3.1) | -1.12 (-2.1 to -0.6) | -3.48 (-4.27, -2.69) | 0.0002 |
| | test, best round | -4.71 (-6.1 to -3.3) | -2.35 (-3.8 to -0.9) | -2.36 (-3.49, -1.23) | 0.0006 |
| maths, 5.5 (4 v 4) | eval, rounds 1-15 | -4.07 | -1.55 | -2.53 (-3.45, -1.60) | 0.029 (floor) |
| | test, last round | -5.38 (-6.1 to -4.4) | -1.72 (-2.4 to -1.0) | -3.65 (-4.92, -2.38) | 0.029 |
| | test, best round | -4.72 (-5.8 to -4.0) | -3.79 (-3.9 to -3.5) | -0.93 (-2.14, +0.27) | 0.029* |
| ordering, 4.8 (8 v 8) | **eval, rounds 1-15 (primary)** | -8.12 (-9.6 to -6.4) | -6.50 (-7.6 to -5.3) | **-1.61 (-2.54, -0.69)** | **0.002** (Holm 0.002) |
| | test, last round | -9.69 (-11.2 to -8.4) | -6.80 (-8.6 to -5.5) | -2.89 (-4.02, -1.76) | 0.0003 |
| | test, best round | -9.77 (-11.4 to -9.0) | -8.49 (-9.8 to -7.5) | -1.29 (-2.13, -0.45) | 0.005 |
| ordering, 5.5 (4 v 4) | eval, rounds 1-15 | -9.37 | -4.56 | -4.81 (-7.68, -1.95) | 0.029 (floor) |
| | test, last round | -13.03 (-13.8 to -11.6) | -3.95 (-6.6 to -2.2) | -9.08 (-12.11, -6.05) | 0.029 |
| | test, best round | -13.01 (-13.7 to -11.6) | -6.93 (-9.0 to -5.3) | -6.08 (-8.42, -3.74) | 0.029 |

\* Maths 5.5 best round: the 4 scored chains all rank below the 4 unscored ones, so the permutation p is at its floor,
but the gap is small and the Welch interval crosses 0.

- **Pre-registered extras for the primary:**
  - Bootstrap intervals over chains: maths 4.8 -2.95 to -2.05; ordering 4.8 -2.40 to -0.84.
  - Hedges' g: maths -4.8, ordering -1.8.
  - Batch-stratified permutation (relabelling only within m28's and m31's chains): p = 0.0004 (maths), 0.004 (ordering).
- **Gap shares.** The plain-to-line gap is maths 14.5 (eval) / 14.7 (test) and ordering 17.4 / 17.9. Opus 4.8's benefit
  from scores covers 18% (maths eval) and 24% (maths test) of it, and 9% (ordering eval) and 16% (ordering test).

**Sensitivity checks** (D, with scores minus without):

| task, writer | primary (failed rounds skipped) | failed rounds carried forward | failed / 0-valid rounds = plain | invalid replies = no evasion (eval) | invalid replies = plain score (last-round test) |
|---|---|---|---|---|---|
| maths 4.8 | -2.58 (p 0.0002) | -2.56 | -2.48 (p 0.0002) | -1.63 (p 0.0002) | -2.02 (-3.41, -0.64; p 0.005) |
| maths 5.5 | -2.53 (0.029) | -2.45 | -2.11 (0.029) | -0.88 (p 0.14) | -1.36 (-4.86, +2.14; p 0.31) |
| ordering 4.8 | -1.61 (0.002) | -1.52 | -0.94 (p 0.052) | -0.79 (p 0.067) | -2.80 (-4.06, -1.54; p 0.0008) |
| ordering 5.5 | -4.81 (0.029) | -4.77 | -4.78 (0.029) | -4.75 (0.029) | -9.10 (0.029) |

- **The last three columns were not pre-registered.** They were added at critic 2's request.
- **"Invalid = no evasion" scores an invalid reply as change 0** (eval) or as plain's median score (test). It asks:
  counting every attempt, how much lower does the writer push the probe?
- **Robust:** Opus 4.8 in maths on every treatment, and the last-round test in ordering.
- **Fragile:** ordering's eval-set result for 4.8, and maths for 5.5 once validity is charged.

**Batch check (Opus 4.8).**
- Maths: m28's 3 v 3 gives D = -2.78 (p = 0.1, the 3 v 3 floor); m31's 5 v 5 gives D = -2.46 (p = 0.008). The batches
  agree.
- Ordering: m28's 3 v 3 gives D = -2.60 (p = 0.1); m31's 5 v 5 gives D = -1.02 (p = 0.09).
  - m31's scored ordering chains started slowly: round 0 D +0.74 and rounds 1-3 D +0.99, against m28's -0.79 and -1.76.
  - Late in the search the batches agree better: rounds 13-15 D -2.32 (m31) vs -3.46 (m28); last-round test -2.33
    (m31, p = 0.016) vs -3.81 (m28).
  - The batch difference itself is borderline chance (critic 2: p about 0.07 for relabelling which chains are "m28").
  - Nothing structural differs between the batches: same prompts, lines and plain test, and the same Chain code.

**Round 0 and timing.**
- Round 0 (before any feedback) does not differ significantly: maths 4.8 -1.70 vs -1.05 (p = 0.41); ordering -3.71 vs
  -3.88 (p = 0.88).
- Subtracting each chain's round 0 leaves D = -1.93 (maths, p = 0.048) and -1.78 (ordering, p = 0.054).
- So most of the benefit comes from the feedback. In maths, part may come from the prompt telling the writer that scores
  will be shown, but that part is not resolved.
- Maths 4.8: the gap is there in rounds 1-3 (D -1.56) and grows by rounds 13-15 (-3.63).
- Ordering 4.8: no gap in rounds 1-3 (D -0.04), -2.75 by rounds 13-15. Without scores, the ordering chains plateau
  near -6.5 from round 3.

**Best round.** Best-of-16 selection with the probe also helps unscored chains (maths 5.5 without scores: -1.72 at the
last round, -3.79 at the best). It is "no scores, then the probe picks the round", so the last round is the clean
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

- **Maths:** the scores effect is about the same size for both models. Opus 5.5 is about 0.7 log-odds stronger than
  4.8 in both conditions, significant only without scores (p = 0.004).
- **Ordering:** the scores effect is much larger for 5.5.
  - Its scored chains reach -9.4 on the eval set and -13.0 on the test, the lowest of any arm.
  - Its unscored chains are the weakest arm and stay near -4 to -4.5 to the end (rounds 13-15: -4.26).
- **Mixed arms (reference only)** were written by:
  - maths with scores: 30 of 48 rounds by Opus 4.8;
  - maths without scores: 25 by 5.5 and 23 by 4.8;
  - ordering with scores: 33 of 48 by 4.8;
  - ordering without scores: all 48 by 5.5.
  Their maths unscored arm (-2.8) is far stronger than either pinned unscored arm. Section 4 explains why: the leak
  rule, not the writer model.

### 3. Refusals and failed rounds

| arm | refusals / answered attempts | chains with a refusal | rounds failed, share of chain-rounds, by cause |
|---|---|---|---|
| maths, 4.8 with | 0 / 155 | 0 of 8 | 3.1%: eval answer copied into the instruction |
| maths, 4.8 without | 0 / 130 | 0 of 8 | 0.8%: no valid reply |
| maths, 5.5 with | **18 / 94 (19%)** | 3 of 4 | 12.5%: 4 rounds refusal, 4 rounds leak |
| maths, 5.5 without | 5 / 92 (5%) | 3 of 4 | 7.8%: 4 leak, 1 no valid reply |
| ordering, 4.8 with | 0 / 133 | 0 of 8 | 7.8%: ran, no valid reply (10 rounds) |
| ordering, 4.8 without | 0 / 128 | 0 of 8 | 0 |
| ordering, 5.5 with | 3 / 67 (4%) | 3 of 4 | 1.6%: no valid reply |
| ordering, 5.5 without | 0 / 65 | 0 of 4 | 3.1%: no valid reply |

- **Every refusal was the API's own refusal stop reason.** The Haiku check on instructions never fired.
- **Refusals cost rounds only in the scored maths 5.5 arm** (4 of 64). The first refusal came in rounds 2-5.
- **The per-chain difference in refusal rate is not significant** (maths p = 0.17, ordering p = 0.14).
- **"No valid reply" rounds** are instructions so aggressive that all 80 replies fail validity, for example spelling
  the names letter by letter, or padding past 40 words. The primary skips them (as m28 does); see the sensitivity table.
- **Leak-rejected attempts** (the writer's examples contained eval answers): maths 4.8 with 30 of 155, 4.8 without
  2 of 130, 5.5 with 20 of 94, **5.5 without 27 of 92**.

### 4. Exploratory (added after seeing the data): number words in maths

Valid eval replies in rounds 1-15, using m28's `result_in_words`. The share is out of replies whose result was found
in either English words or digits; for 5.5 with scores, a further 19% of valid replies used other forms, such as
Chinese or Russian numerals.

| arm | result in words | median change, word answers | median change, digit answers |
|---|---|---|---|
| 4.8 with scores | 72% of 6,553 | -3.7 | -2.2 |
| 4.8 without | 2.5% of 9,241 | -3.6 | -0.7 |
| 5.5 with scores | 73% of 2,364 | -4.1 | -1.6 |
| 5.5 without | 6.5% of 4,034 | -3.8 | -1.2 |
| mixed with (m22) | 71% of 2,720 | -4.0 | -2.2 |
| mixed without (m22) | 65% of 3,124 | -3.4 | -1.4 |

- **Word answers sit 3.5-4 log-odds below plain whoever writes the instruction.** Scored writers settle on them:
  - rounds where most valid replies are in words: 87 (4.8 with) and 28 (5.5 with), against 3 and 4 without;
  - chains that settle on words: 6 of 8 scored 4.8 chains, 0 of 8 unscored.
- **Unscored pinned writers rarely ask for words.** This is from reading the instructions; a first regex count
  (38 of 128) was wrong, because most matches said "never spelled out".
  - Opus 4.8 asked in 5 of 128 instructions that ran, all early.
  - Opus 5.5 asked in 4 of 60 that ran. Another 15 of its 27 leak-rejected attempts asked for words (14 English, 1
    Chinese numerals); the rejections came from example numbers that were eval answers.
  - Qwen complied every time (92-100% of replies in words).
- **The writers' notes show why they gave up on words.**
  - Word rounds cost some validity (for example 72/80 and 66/80 valid).
  - Several writers reasoned from the probe description that more number tokens would raise the running peak, so words
    would score *higher*. In fact those rounds scored about -3.5 to -4.
  - Two 4.8 chains then banned spelled-out numbers for the rest of the chain.
  - With scores, the feedback corrects this prior at once. Without scores nothing does.
- **Scored writers also do better within digit answers** (-2.2 vs -0.7 for 4.8). So words are most, but not all, of
  the maths gap.
- **Why m22's mixed unscored control looked strong:** two of its three maths chains used words. Its word rounds
  contained 2-5 eval answers each, which today's leak rule would reject but m22's original parser did not. With about
  half its rounds written by Opus 4.8, the leak rule, not the writer model, explains most of the gap to the pinned
  unscored arms.

### 5. Validity and the line

- **Final test, under the 1% line:** 0 valid replies in every arm: all 36 new test arms per task (18 chains x best /
  last) and m28's 12 pinned arms per task.
- **5% full-check audit of far-above-line maths test replies:** 3 of 858 disagree with the cheaper rule (m31); 5 of
  1,441 including m28.
- **Valid share, maths:**
  - Last round, Opus 4.8: 382 of 500 with scores (range 245-495), against 499.75 without (499-500).
  - Opus 5.5: 318 (139-444) against 497.
  - Eval rounds 1-15 (rounds that ran, including 0-valid rounds): 75% vs 96% (4.8) and 71% vs 94% (5.5).
- **Valid share, ordering:**
  - Last round, Opus 4.8: 434 vs 457 (p = 0.15).
  - Eval: 78% vs 89% (4.8), 87% vs 85% (5.5).
- **Score medians are over valid replies only.** In maths, part of the scored writers' advantage is bought with invalid
  replies. The "invalid = no evasion" columns in section 1 put a number on it.

### 6. Other pre-registered secondaries

- **Distinct instructions per chain** (out of 16): maths 4.8 15.1 with vs 12.2 without; ordering 14.9 vs 12.1. Unscored
  4.8 chains resubmit the same instruction more often. Opus 5.5 hardly does (14-16 in every arm).
- **Per-round medians per arm:** in `analysis.json` (`per_round`) and the first figure.

## Verdict, in plain words

- With one writer model held fixed, a writer that sees per-token probe scores pushes the probe further down than the
  same writer seeing only Qwen's replies and validity verdicts.
- This passes the pre-registered rule in both tasks (p = 0.0002 maths, 0.002 ordering for Opus 4.8; Holm-adjusted
  0.0003 and 0.002).
- **How big, for Opus 4.8:**
  - Maths: about 2.5-3.5 log-odds on valid replies, or 1.6-2.0 if invalid replies count as failures. This is on top of
    a no-scores drop of about 1.
  - Ordering: about 1.6-2.9 log-odds on top of a no-scores drop of about 6.5. It is firmest at the end of the search
    (last-round test p = 0.0003); the eval-set result is fragile to how invalid rounds are treated (p 0.05-0.07).
- **Opus 5.5:** similar in maths, though not significant once invalid replies count as failures, and much larger in
  ordering (5-9 log-odds, robust).
- **Mechanism in maths (exploratory):** the scores show the writer that number-word answers read low, correcting a
  wrong guess the writers draw from the probe description.
- **No arm gets a valid reply under the flag line.**

## Caveats

- **Opus 5.5 has 4 chains per arm.** p = 0.029 is the 4 v 4 floor, and the intervals are wide.
- **Validity cost matters for the size:** the valid-only numbers are the pre-registered ones; the penalised numbers
  are smaller (section 1).
- **m31's ordering 4.8 batch shows a smaller eval effect than m28's** (-1.0 vs -2.6). This is mostly from a slower
  start; the batches agree late in the search.
- **The headline reading has the instruction removed,** as in m22 / m26 / m28. Instruction-visible medians are saved
  per test arm but not analysed.
- **The mixed m22 / m26 arms are reference only.** They differ in model mix, fallback and leak rule.
- **"Scores help" is about this setup:** 16 rounds, 80 eval replies per round, these feedback formats, no direct access
  to the probe.

## Cost

- **Claude** (estimates; judges counted as upper bounds by m22's counter):
  - Search judges: about $124 (4.8: $55 maths + $15 ordering; 5.5: $40 + $14).
  - Writers: $93 (4.8: $23 + $32; 5.5: $19 + $19).
  - Test judges: about $78 (4.8 maths $47, ordering $6; 5.5 maths $21, ordering $4).
  - Haiku checks and the pre-flight call: about $1.
  - Total about $295, against a combined cap of $500 ($300 brief + $200 amendment). The critic subagents' own usage is
    not included.
- **GPU (Modal billing for m31's apps):** $83 billed through 22:00 PDT (the report lags); I estimate $130-150 in total,
  against a combined cap of $210. All m31 apps are stopped.

## Files

- **Code:** `src/probe_evasion/missions/m31_more_chains.py`. It covers the pinned Opus 5.5 writer, run set-up, the
  freeze, the analysis and the figures. It imports m22's runner and m28's writer and edits neither.
- **Status:** `missions/m31_more_chains/STATUS.md`, with the design, pre-registration, amendments, critic check-ins and
  the log.
- **Analysis:** `missions/m31_more_chains/analysis.json`. It holds per-chain per-round values, every test arm, the 2x2
  blocks including the sensitivities, refusal and failure tables, the number-words split and spend.
- **Runs:**
  - Pinned Opus 4.8: `missions/m31_more_chains/runs/{qwen35_27b,qwen35_27b_ordering}/`.
  - Pinned Opus 5.5: `missions/m31_more_chains/runs_opus55/...`.
  - Each holds per chain `state.json`, `proposals.jsonl` (every writer call, with reasoning) and `round_XX.jsonl`
    (every reply with scores and validity); plus `frozen.json`, `test_rows.jsonl` and `judge_spend_*.json`.
- **Logs:** `missions/m31_more_chains/logs/`.
