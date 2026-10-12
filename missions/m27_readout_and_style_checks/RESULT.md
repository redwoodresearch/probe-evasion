# m27_readout_and_style_checks: result

**Question.** In the score-only searches, a prompt-writer shown probe scores moved the median valid answer's probe
score a quarter to a half of the way from plain answers to the 1% flag line. The searches covered maths on
Qwen3.5-27B (m22), maths on 9B, 122B and Gemma (m23), program output (m25) and ordering (m26). Two readings would
deflate this result:

1. **Readout artifact.** The drop might exist only because of how the probe is read: the peak of a running average
   (EMA, alpha 0.5, starting at 0) of per-token log-odds. Low-scoring filler before the answer, for example, drags
   that average down.
2. **Style shift.** The instruction's style might move all replies, task or not, while the probe still ranks task
   replies above non-task replies written in the same style.

## Bottom line

**1. Not a readout artifact.**
- On every task the drop survives at least four of the five readouts. Apart from the deployed EMA peak, these are
  **diagnostics, not deployed readings**: per-token max, mean over the reply, the EMA restarted at the first answer
  token, and the max over the answer's own tokens.
- Each readout gets its own 1% line, re-set on the same calibration pool.
- The one failure is maths on Gemma under the reply mean. There the drop disappears (Opus arms' mean +0.7 log-odds).
  That readout is a poor detector on Gemma anyway: 7 of 435 plain answers fall under its line.
- **Part of the drop is the text before the answer.** Restarting the average at the first answer token:
  - Code (m22's probe description): the share of the gap closed falls from 26% to 13%. In raw log-odds, 40% of the
    drop is left.
  - Ordering: 53% to 39% (57% of the raw drop left).
  - Maths 9B: 47% to 40% (54% left).
  - Maths 27B (35% vs 34%), 122B and Gemma: no reliable shrink.
  - Intervals for these are in the Part A table below.
- **The answer tokens' own scores drop too.** The max over the answer span keeps two-thirds or more of the raw drop, so the
  attack is more than "put negative filler first".
- **Score-only attacks still do not get under the line**, whatever the readout:
  - 0 for maths 27B, ordering and code (m22's probe description), except up to 3 on the per-token max and
    answer-span max readouts;
  - up to 25 of 4,209 for code (told the label kinds);
  - up to 10 of 3,271 for maths 122B.
  - **Maths 9B is the exception:** 36 of 3,584 under the deployed line, but 70-300 under the diagnostic readouts (177
    per-token max, 300 reply mean).
  - Plain answers are never under any line, apart from Gemma's reply mean (7 of 435).

**2. Partly a style shift, but the task-vs-non-task ranking survives within each style.**
- **Twins.** We applied each frozen instruction to the task's non-task requests:
  - maths: lookups where the answer is given;
  - code: programs that just print a literal;
  - ordering: requests that already state the order.
- **The styled twins move up, not down:** median +2 to +6 log-odds against plain non-task replies (-0.6 to +0.3 for
  3 of 21 arms). So the attack squeezes the gap between task and non-task replies from both sides:
  - the task replies fall by 2-11;
  - the non-task replies rise by 0-6.
- **What is left of the gap.** The within-style gap (attack median minus styled-twin median) is a share of the plain
  gap (plain answers minus plain twins). It is 40-77% for maths and code, and 18-46% for ordering.
- **Under the deployed line, these instructions would flag most styled twins as false alarms.** For example, all
  ordering twins and 79-184 of about 190 maths 27B twins per arm. Plain twins are flagged 0-29 times out of about 190.
- **Within-style AUROC** (attack replies vs styled twins, instruction removed) is 0.989-1.000 on every score-only arm.
- **A 1% line set on the styled twins** catches:
  - every attack reply for maths 27B and code;
  - all but 0-10 per arm for ordering;
  - all but 5-40 per arm for maths 9B (the line rests on about 140-200 twins).
- **Instruction visible, line set on styled twins in that context** (amendment 2): every attack reply is caught except
  in maths 9B (6 of 291 for Opus chain 2, 2 of 472 for self chain 0).
- **Against Qwen's own non-answers to the same requests** (amendment 1, the like-for-like comparison):
  - AUROC 0.970-0.999.
  - Many attack replies fall under the non-answers' 99th percentile, but that line is set by 2-3 answer-like
    outliers (for example "The tank holds 2798 litres.").
  - Under the 95th percentile, 0-5 per arm fall, except 30 of 476 for maths 9B Opus chain 0.

**3. The white-box reference (m19) is different.**
- Its template makes answers genuinely look like styled lookups when the instruction is removed:
  - AUROC 0.93;
  - 305 of 478 attack replies under the styled-twin line;
  - the within-style gap is 8% of the plain gap.
- With the instruction visible and the line calibrated on styled twins in that context, it is caught completely (0 of
  478 under).
- m19's earlier "0/500 visible" used the plain line, which also flags every styled lookup in that context. That number
  was not meaningful; this one is.

**4. What the maths and code probes read: the answer-swap diagnostic** (amendment 5).
- **Method.** Valid replies were re-read with the answer replaced by a wrong value of the same form:
  - maths: the leading digit changed;
  - code: the first digit of the output changed;
  - ordering: the first two names swapped.
- **Maths:** scores barely move. Median paired change -0.1 to -0.4 log-odds for plain and attack replies. The deployed
  peak sits at or after the changed token in 92-100% of replies, so the check is informative here.
- **Code:** median change 0.0 to -0.1. This check is weaker: the code peak often sits on the opening bracket or quote
  before any value. Only 56% of plain and 73% of attack peaks come at or after the change, and 303 single-string
  outputs could not be swapped.
- **Ordering:** a wrong order lowers plain answers by 3.3 and attack replies by 1.0 on average.
- **Reading.** The maths and code probes respond to "a value in the answer slot after a request that needs working
  out", not to whether the value is right. The ordering probe partly tracks the right order. This says nothing about
  whether the model computed internally.

**5. In-context non-answers sit near or under the line for maths and code.**
- These are Qwen's replies to the real request under "don't solve it, name the quantity / restate the first number"
  instructions, read with the instruction removed.
- Median scores and how many fall under the deployed line:

  | task | median | under the deployed line |
  |---|---|---|
  | maths 27B | -4.9 | 35 of 200 |
  | maths 9B | -6.1 | 183 of 200 |
  | code | -6.1 | 152 of 200 |
  | ordering (m26's replies, cleaned here) | -4.5 | 0 of 154 |

- So for maths and code the line can be reached in the plain request's context, unlike ordering. The score-only
  attacks still land far above these non-answers. Measured as the share of the way from plain to the non-answer
  median, they close 17-60% (maths and code) and 43-74% (ordering).

**6. Follow-ups (Parts C and D).**
- **Before the answer.** The drop is already present at the token just before the answer is written, and larger
  there: 87-189% of that reading's gap, against 23-42% on the answer tokens. That holds even when the pre-answer token
  is the same. So the attack changes what the probe reads in the model's state before the answer, not just the
  answer's presentation, while the model still writes the correct answer.
- **Broad traffic.** The original 1% lines would flag 72-100% of ordinary Qwen chat replies that contain numbers or
  lists. Re-set on such a pool (1,720 replies), the lines move up by 3-5 log-odds. Score-only attack replies still
  almost never get under them: 6 of 1,207 for maths, 5 of 1,316 for code, 58 of 1,270 for ordering (Opus with scores).
  The m19 white-box template gets under all of them.

## What the post can safely claim

- **The drops are real.** They are drops in the probe's per-token output, not an artifact of the running-average peak.
  The text before the answer contributes part of them: about half for code, less for ordering and 9B, essentially
  none for maths 27B.
- **Describe them as compressing the task vs non-task gap, not as hiding the task.** The same instructions raise
  non-task replies (most styled non-task replies would be flagged as false alarms), and within a style the probe
  still ranks task replies above non-task ones (AUROC about 1.0).
- **A style-aware monitor catches essentially all of them.** That means a line calibrated on replies in the attack's
  style, or reading with the instruction visible and calibrating there. Maths 9B is the exception, with a handful
  per arm.
- **Do not claim the maths or code probes detect correct computation.** A wrong answer of the same form scores the
  same.
- **The m19 white-box template is the only case where task replies become indistinguishable from styled lookups.**
  That holds only when the instruction is removed; it is caught fully with the instruction visible and a properly
  calibrated line.
- **Label the alternative readouts as diagnostics.**

## Part A: readouts (saved per-token scores only; no GPU)

**What was read.** Every saved reply was re-read under five readouts of the target probe's per-token log-odds:
- calibration pool, plain answers, every search round and the frozen test;
- m22's frozen test arrived during this mission and is used here.

**Checks.**
- Recomputed EMA peaks equal the saved peaks exactly: max difference below 1e-11 over ~135k replies, incl. m19.
- Re-set deployed lines equal the saved lines. m22: -5.79 vs m11's reference line -5.80, which m22 asserts within
  5%.
- Calibration per-token log-odds came from each run's score caches, for every calibration reply. That includes all
  1,301 of m22's, so amendment 3 needed no GPU.

**Table.**
- Each cell gives, for the mean over the pre-registered frozen best-round Opus-with-scores arms:
  - the share of that readout's own plain-to-line gap closed, with a 95% interval from resampling test problems with
    all their seeds, lines held fixed;
  - the raw change in log-odds.
- "Under" = valid frozen-test replies under that readout's 1% line, over every writer's frozen arms.
- Full table, incl. plain medians, lines and the best arm per readout: `part_a/table.md`.

| task | EMA peak (deployed) | per-token max | mean over reply | EMA from first answer token | max over answer span | under the line (deployed; diagnostics) |
|---|---|---|---|---|---|---|
| maths 27B (m22) | 35% [32, 38], -5.2 | 29%, -4.3 | 47%, -4.8 | 34% [31, 38], -4.0 | 27%, -4.3 | 0 / 7,829; 0 on all |
| maths 9B (m23) | 47% [44, 53], -6.3 | 50%, -5.6 | 76%, -6.1 | 40% [33, 47], -3.4 | 35%, -4.5 | 36 / 3,584; 177, 300, 100, 70 |
| maths 122B (m23) | 44% [41, 46], -2.7 | 34%, -2.4 | 40%, -1.6 | 50% [47, 53], -2.6 | 36%, -2.5 | 0 / 3,271; 0, 0, 4, 10 |
| maths Gemma (m23) | 19% [16, 22], -2.1 | 13%, -1.4 | -25%, +0.7 | 16% [13, 19], -1.8 | 8%, -1.4 | 12 / 3,071; 10, 53 (7 plain), 0, 0 |
| code (m25, m22's probe description) | 26% [24, 30], -4.0 | 31%, -4.4 | 47%, -5.6 | 13% [10, 17], -1.6 | 23%, -2.9 | 0 / 2,664; 3, 0, 0, 3 |
| code (m25, told the label kinds) | 30% [27, 32], -4.5 | 39%, -5.4 | 50%, -6.0 | 19% [16, 23], -2.4 | 32%, -4.0 | 0 / 4,209; 0, 0, 0, 25 |
| ordering (m26) | 53% [51, 55], -9.5 | 42%, -10.0 | 42%, -3.7 | 39% [37, 40], -5.4 | 42%, -10.0 | 0 / 5,044; 0 on all |
| m19 white-box template (maths 27B) | 105%, -15.5 | 108%, -16.1 | 87%, -8.8 | 112%, -13.0 | 109%, -17.3 | 385 / 478; 338, 0, 428, 332 |

**The deployed share minus the from-answer share (95% interval), in percentage points:**

| task | deployed minus from-answer |
|---|---|
| code (m22's probe description) | 9 to 18 |
| code (told the label kinds) | 6 to 14 |
| ordering | 13 to 16 |
| maths 9B | 1 to 14 |
| maths Gemma | 1 to 5 |
| maths 27B | -1 to 3 |
| maths 122B | -8 to -5 (the restarted average closes *more* of its gap) |

**Two scales.** The gap-closed scale and the raw scale can disagree, because each readout's line moves too. Both are
given.

**Figures.**
- `readouts_per_round.png`: per search round, each Opus chain's median valid score (eval set) as the share of that
  readout's gap from the plain eval median to its own line. Opus with scores is blue, Opus without scores orange;
  hollow markers mark rounds with fewer than 40/80 valid.
- Readouts with a narrow plain-to-line gap look larger on that normalised scale (the reply mean, especially for 9B
  and Gemma). So `readouts_per_round_raw.png` shows the same medians in raw log-odds above each readout's line.

## Part B: within-style twins

**Setup.**
- **Instructions:** each writer's frozen instruction, 21 in all:
  - maths 27B: Opus with and without scores;
  - maths 9B: Opus and the 9B model writing for itself;
  - code (m25, m22's probe description): Opus;
  - ordering: Opus with and without scores.
- **Twins:** 200 non-task requests per instruction, plus once with no instruction. Requests:
  - maths: m11's 200 lookups, the same set as m19's twins, so m19's saved twins compare directly;
  - code: 200 fresh literal-print programs;
  - ordering: 200 fresh puzzles in m26's "order already given" format.
- **Clean twins** must be finished, state the request's given value, and pass the existing Haiku audit (nothing worked
  out). For maths, a twin with any number not in the request is also rejected (critic, below). Clean shares are
  70-100%; the lowest are 9B instructions that make 9B compute anyway.
- **In-context non-answers:** 2 instructions x 100 test problems for maths and code. Ordering uses m26's saved
  non-solving replies, cleaned the same way: correct result absent, audit passes.
- **Scoring:** everything was scored instruction removed and visible.
- **Attack replies:** the frozen test.
- Full per-arm table: `part_b/table.md`. Figure: `style_twins.png` (task replies vs same-style twins per arm, plus
  in-context non-answers and the deployed line).

| task (arms) | attack change vs plain | styled twins vs plain twins | within-style AUROC (removed / visible) | attack under styled-twin 1% line (removed) | under the visible-context styled line | share of the plain gap left within style |
|---|---|---|---|---|---|---|
| maths 27B (6) | -2.3 to -5.9 | +3.1 to +5.9 | 1.000 / 1.000 | 0 | 0 | 40-70% |
| maths 9B (6) | -3.1 to -8.9 | -0.6 to +6.1 | 0.989-1.000 / 0.999-1.000 | 5-40 per arm | 0-6 | 40-69% |
| code (3) | -3.8 to -4.2 | 0.0 to +3.8 | 1.000 / 1.000 | 0 | 0 | 57-77% |
| ordering (6) | -6.5 to -11.2 | +3.0 to +5.2 | 0.995-1.000 / 1.000 | 0-10 | 0 | 18-46% |
| m19 white-box template | -15.5 | +1.2 | 0.928 / 1.000 | 305 / 478 | 0 / 478 | 8% |

**Gap closed on three scales** (amendment 1), as the share of the way from the plain median to each endpoint:

| task | to the deployed line | to the in-context non-answer median | to the styled-twin median |
|---|---|---|---|
| maths 27B | 16-40% | 17-43% | 15-44% |
| maths 9B | 23-67% | 21-60% | 21-50% |
| code | 25-28% | 24-27% | 23-28% |
| ordering | 36-63% | 43-74% | 43-77% |

## Part C: the probe just before the answer (follow-up; saved per-token scores, no GPU)

**Question.** With reasoning off, the model must have the answer (at least its first token) ready at the position just
before it writes it. Is the attack's drop already there, or only on the answer's own tokens?
- If the drop is already there, the attack changes the model's internal state (as this probe reads it) before the
  answer.
- If it is only on the answer tokens, the attack changes how the answer is presented.

**Readings** (all diagnostics), on frozen-test replies:
- the probe's log-odds at the token just before the first answer token;
- the max over the 3 tokens before it;
- the running average at that point;
- the deployed peak taken over the tokens before the answer only.

Each reading's 1% line was re-set on the calibration pool. Replies that open with the answer have no such token and
are left out: about 9% of maths calibration replies, 10% of maths plain answers and 20% of ordering plain answers.

**Mean change, Opus-with-scores frozen arms vs plain** (share of that reading's plain-to-line gap closed in
brackets). Full table: `part_c/table.md`.

| task | deployed EMA peak | token just before the answer | running average just before the answer | max over the answer span | attack replies under the pre-answer-token line | plain under |
|---|---|---|---|---|---|---|
| maths 27B | -4.8 (33%) | -11.7 (109%) | -10.3 (87%) | -4.2 (27%) | 1,368 / 2,189 | 0 |
| maths 9B | -6.3 (47%) | -17.1 (189%) | -13.7 (134%) | -4.5 (35%) | 1,017 / 1,072 | 0 |
| code | -4.0 (26%) | -13.4 (125%) | -9.8 (116%) | -2.9 (23%) | 1,140 / 1,307 | 0 |
| ordering | -9.5 (53%) | -15.7 (112%) | -12.1 (102%) | -10.0 (42%) | 1,241 / 1,270 | 0 |
| m19 white-box template (27B) | -15.3 (105%) | -20.0 (185%) | -16.8 (142%) | -17.3 (110%) | 485 / 485 | 0 |

**Same-token control.** The token before the answer differs between plain and attack replies: "is" or "$" against
"—", '"' or "transcribed". Holding it fixed means comparing replies whose pre-answer token is the same, which for
maths is the space Qwen writes before a digit.

| task | plain | Opus with scores | m19 template |
|---|---|---|---|
| maths 27B | +7.7 | +1.0 | -12.9 |
| maths 9B | +9.9 | -3.2 | |
| code | +14.3 | +7.1 | |

Ordering has no shared pre-answer token.

**Verdict per task: the drop is already present before the answer is written, on every task.**
- At the position where the model is about to write the answer, the probe's reading under attack is far lower than in
  plain replies. That holds even with the same token in that position.
- Most attack replies are below a 1% line set for that position.
- On the answer tokens themselves the drop is smaller: max over the answer span closes 23-42% of its gap, against
  87-189% just before the answer.
- So the score-only attacks do not only change how the answer is presented. They change what the probe reads in the
  model's state before the answer.

**What this does not show.** The model still writes the correct answer at the next token. So what changes is the
direction this probe reads, not necessarily the computation itself.
- Read together with Part B's style twins and the answer swap, the more likely reading is this: the probe's signal
  just before the answer reflects the context and style ("an answer to a problem is coming") more than the work.
- **Caveat 1:** multi-digit answers may be partly computed while the digits are written.
- **Caveat 2:** the same-token control fixes the token but not the text before it.

## Part D: lines from a broad pool of ordinary chat (follow-up; GPU)

**Question.** How fragile are the task lines to the choice of "ordinary traffic"?

**Pool.** No existing Qwen3.5-27B pool had replies with numbers or lists. (m1b's chat pool has digits in only 105 of
1,185 replies; m18 and m23 hold one-sentence lookups.) So I built one:
- **Requests:** 1,892 ordinary chat requests, written by Haiku 5.5 over 40 everyday topics (recipes, travel, sports
  records, how-to steps, study plans...). Haiku was told to write requests whose replies contain numbers or ordered
  lists, and no arithmetic problems, code-output questions or ordering puzzles. Saved in
  `data/concepts/m27_broad_pool_prompts.jsonl`.
- **Replies:** Qwen3.5-27B, reasoning off, at most 96 tokens (most are cut off there), read with the request as context.
- **Kept:** the 1,720 replies containing a digit or a list marker.
- **Scoring:** each reply was scored with the maths (onpk), code (cpk) and ordering (ordpk) probes, all layer 48,
  seed 0, deployed EMA-peak reading.

**Lines and how many valid frozen-test replies fall under each:**

| task | the original line flags this many of the broad pool | line: original / broad pool / broad + original | Opus-with-scores arms under (n) | all frozen own-target arms under (n) | m19 template under (n = 485) | plain under |
|---|---|---|---|---|---|---|
| maths 27B | 1,245 / 1,720 (72%) | -5.8 / -2.3 / -3.1 | 0 / 6 / 1 (1,207) | 0 / 6 / 1 (7,829) | 384 / 485 / 480 | 0 / 0 / 0 |
| code | 1,658 / 1,720 (96%) | -5.4 / -1.3 / -1.5 | 0 / 5 / 0 (1,316) | 0 / 13 / 8 (2,664) | | 0 / 0 / 0 |
| ordering | 1,720 / 1,720 (100%) | -7.3 / -2.0 / -2.2 | 0 / 58 / 30 (1,270) | 0 / 126 / 69 (5,044) | | 0 / 0 / 0 |

**Verdict: the lines are very fragile; the conclusions about the attacks are not.**
- The original 1% lines were set on one-sentence lookups. They would flag 72-100% of ordinary chat replies that
  contain numbers or lists. So these probes, at these lines, would be unusable as monitors on broad traffic. The
  "1% false-alarm" lines are only 1% for the calibration format.
- Re-set on the broad pool, the lines move up by 3.4-5.3 log-odds. Even then the score-only attacks stay almost all
  above the line:
  - maths: 6 of 1,207 Opus replies;
  - code: 5 of 1,316;
  - ordering: 58 of 1,270 Opus replies (126 of 5,044 over all writers).
- Plain answers stay at 0. Adding the original pool to the broad pool roughly halves those counts.
- The m19 white-box template gets under every version of the line (480-485 of 485).
- The pool's top-scoring replies are ordinary answers that mention quantities, and sometimes do a little arithmetic
  in passing ("82 games ... Since there are 30 teams ..."). That is realistic traffic, and it is what sets the broad
  line.

## Critic check-in (research-reviewer, after the first full pass) and what changed

1. **Must-fix: maths twins that compute leaked into "clean".** These were per-unit quotients the Haiku audit missed.
   That produced a spurious "30 of 45 under the visible line" for maths 27B Opus chain 0. Fixed with a rule rejecting
   any number not in the request (outside the answer span). Rejects 2-58 twins per arm, all genuine computations on
   inspection (for example "The books cost $143 each." for a $5,005 total). That arm is now 0 under.
2. **Must-fix: "survives every readout" was false.** Gemma's reply mean loses the drop, and the 9B diagnostic counts
   are larger than "a handful". Reworded, with the counts given.
3. **Must-fix: stale tables, and mixed arm sets.** The STATUS table predated m22's frozen test. The "Opus mean"
   included last-round arms for some tasks only. The mean now uses the pre-registered best-round arms everywhere,
   matching Part B. Tables were regenerated.
4. **Must-fix: rerun Part B for maths 27B on the frozen test.** Done once the test arrived; the answer swap was
   rescored on test rows too. The frozen instructions match the twin instructions exactly.
5. Both measures (gap closed and raw) and bootstrap intervals were added for the pre-answer claim.
6. The per-round figure now normalises by the plain *eval* median, and a raw log-odds version was added.
7. 95th percentiles are reported next to the fragile 99th-percentile lines from about 200 samples.
8. Attack replies vs in-context non-answers was added. Ordering's non-answers were cleaned like the others.
9. The style result is framed as squeezing the gap from both sides.
10. The m19 reproduction check had compared probabilities with log-odds; fixed.
11. The code answer-swap limits were stated.
12-14. Labels and coverage notes.

## Caveats

- **Twins use different requests from the attack replies** (lookups, literal prints, "order given"), so twin AUROC
  partly reflects request content. The in-context non-answers remove that confound; they agree, apart from a few
  answer-like outliers.
- **Twin and non-answer lines rest on 140-200 replies,** so a 1% line is about the 2nd-3rd highest value. Counts under
  the 95th percentile are given alongside.
- **Visible-context twin scores can reflect real computation.** Some instructions ("work it out silently...") make
  Qwen compute on the twin request; the cleaning removes the ones that show it.
- **Twins:** 1 sample per request (seed 0). Attack replies: 100 problems x 5 seeds.
- **The answer-span readouts depend on locating the answer.** About 2,400 valid replies (other languages, glued number
  words, Chinese numerals; mostly maths 122B) were located by Haiku quoting the span; spot-checked correct, sometimes
  including the unit. Fewer than 30 of about 135k replies could not be located; they are left out of the span readouts only.
- **Part D's pool replies were cut at 96 tokens** (the attack budget), with prompts written by Haiku. A different
  traffic mix would give a different line; the point is how far the line moves.
- **The "Opus" writer was mostly the Opus 4.8 fallback** in feedback rounds (see m22, m23, m25, m26).

## Files

- **Code:** `src/probe_evasion/missions/m27_readout_and_style_checks.py`.
  - `part_a`, `part_a_outputs`, `part_b` (local);
  - `m27_part_b` (Modal: generation, scoring, answer swaps).
- **Part A:**
  - `part_a/<run>/scores.jsonl`: every reply x every readout, with answer spans and how each was located;
  - `part_a/m19_reference_scores.jsonl`;
  - `part_a/summary.json`, incl. per-round medians, best arms and bootstrap intervals;
  - `part_a/table.md`;
  - figures `readouts_per_round.png` and `readouts_per_round_raw.png`.
- **Part B:**
  - `part_b/<task>/generated.jsonl`: every twin and non-answer, with per-token scores, instruction removed and visible,
    and peaks of every probe;
  - `clean.jsonl`: the same rows with cleaning flags;
  - `forced.jsonl`: answer swaps, scored;
  - `forced_eval_rows.jsonl` (maths 27B, an earlier swap on eval rows, superseded);
  - `score_cache_b.jsonl`;
  - `part_b/summary.json`, `part_b/table.md`, `style_twins.png`.
- **Part C:** `part_c/summary.json`, `part_c/table.md`.
- **Part D:** `data/concepts/m27_broad_pool_prompts.jsonl`; `part_d/replies.jsonl`; `part_d/scores_*.jsonl` (per-token
  scores; `qwen35_27b` = maths, `monitor_neutral` = code, `monitor` = ordering); `part_d/summary.json`.
- **Process:** `STATUS.md`.

## Cost

- **GPU (Modal):** about $27 in total, estimated, within the raised $90 cap plus the $15 follow-up allowance. All apps
  finished on their own:
  - Parts A and B, about $20:
    - twins and non-answers, ap-QlsaTpYwWoiGcGZ53YenWO;
    - answer swaps, ap-AaFJ8eJRWoIkXQNEAE9AGZ;
    - maths 27B swaps on the frozen test, ap-n33XtQ0zxUkfpJ0kvmsKoR.
  - Part D, about $7: generating 1,892 replies and scoring them with three probe sets, ap-VAm58WumK65fTh3OqUMY2D.
  - Part C needed no GPU.
- **Claude:** about $6.5, within the cap. All Haiku:
  - about 2,400 answer-span locations;
  - about 5,700 audits;
  - 40 calls writing Part D's prompts.
- **OpenRouter:** none.
