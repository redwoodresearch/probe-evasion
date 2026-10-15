# m34_what_the_probe_reads: result

**Question.** The post says prompting moves the task-probe scores but "the probe still sees the task". Five cheap checks
decide how far that claim can be pushed.

## Terms

- **Probe score / deployed reading:** the peak of a running average (alpha 0.5) of a linear probe's per-token log-odds
  over the reply. Higher means "this reply is answering a task".
- **Instruction removed:** the headline reading. The reply is re-read after the problem alone, without the attack
  instruction that produced it. **Visible** means it is read in the context it was generated in (instruction plus
  problem).
- **Attack arms:** the frozen instructions the earlier searches produced, each run on 100 fresh test problems × 5
  seeds.
  - m22: maths, Qwen3.5-27B; m23: maths, Qwen3.5-9B; m25: code; m26: ordering; m28: the same with extra writers.
  - "Own-target" arms attacked the probe trained on Qwen's own replies (`onpk@L48@s0` for maths 27B).
  - "Best" = the round the pre-registered rule froze. "Last" = round 15, with no selection.
- **Writers:**
  - Opus with / without probe scores;
  - "pinned 4.8" = Claude Opus 4.8 on every round (m28);
  - "no readout spec" = a writer not told how the score is read;
  - random styles = probe-blind odd styles (m28);
  - Qwen-27B / Qwen-9B / Qwen-122B writing instructions themselves.
- **m19 template:** the white-box red-team instruction (engineering notation), the one attack that crossed the line.
- **1% line:** the score above which 1% of ordinary replies would be flagged.
- **Twins:** the same instruction applied to non-task requests, i.e. requests whose answer is already given.

## Short answers

1. **Is Qwen still computing where the probe reads low? Yes for 27B and code, mixed for 9B, no for m19's template.**
   These are the replies under m27's pre-answer 1% line in the headline re-read (instruction removed):
   - **27B:** in 83% of the replies we can evaluate, Qwen's next-token distribution at that slot still singles out the
     correct leading digit among the digits that are not just echoes of numbers in the question (median 0.81-0.97).
     Echoes of the question's numbers compete there: the correct digit is the outright top digit only 42-65% of the
     time.
   - **9B:** 40% of the evaluable replies. Answers written in digits mostly do not have the answer ready (27%); answers
     written in words do (98%).
   - **Code:** the first value token after the bracket has median probability 0.67.
   - **m19 template:** 0% on both models. Its "engineering notation" slot reads to Qwen as a quoted given number.
   - **Generation context:** read as generated, Qwen predicts the answer at P ≈ 0.99 everywhere. Even so, the
     pre-answer probe there is 6.0 (27B) and 7.9 (code) log-odds below plain; for 9B the drop is only 2.6.
   - So m27 Part C's pre-answer drop is, for 27B and code, largely a drop in the probe while the answer is still
     computed. This is the "computing while the probe reads low" signal.
2. **Compute without stating the result: mostly an answer detector, with a small but real work component.**
   - When Qwen is asked to say only a property of the result, the replies score near non-answer level. Examples: "is
     the result even?" or "does the output contain '7'?".
   - They sit 84-98% of the way from plain answers to replies where the same question is asked about a result *given*
     in the request.
   - The remaining few log-odds are consistent: +0.4 to +2.6 as asked, +2.2 to +6.6 after the original request.
   - Stating the result in the same sentence restores near-plain scores.
3. **Sister probes: not informative. Probes at other layers: informative, and they move too.**
   - The nine layer-48 maths probes are near-parallel (weight cosine 0.75-0.95; readings correlate 0.89-1.00).
   - New probes trained the same way at layers 32, 40 and 56 move on most arms by shares similar to the deployed
     probe's (per-arm r = 0.73-0.95).
   - Exceptions: layer 32 barely moves for random styles (3% vs 12%) and Qwen-122B's instructions (4% vs 21%), and
     layer 40 moves less for the unscored writers.
   - So the drop is not specific to the attacked layer-48 direction. It is stronger at layers 48-56 than at 32-40.
4. **Broad-pool line positions: the tone-vs-task contrast holds.**
   - On m27's broad pool of ordinary chat replies with numbers or lists, the task lines still sit 80-85% of the way
     from plain to the median ordinary reply. Sarcasm sits at 33-36%.
   - d' (how far plain answers are from ordinary replies, in standard deviations) stays 9.8-13.2 (sarcasm 4.4-5.0).
5. **Within-style gap closed** (the "beyond style" number): searched with scores 48-58% on maths 27B and 75% on
   ordering; random styles 28-42%.
   - About half of that comes from the styled non-task replies rising, not from the task replies falling. On the
     task-falling side alone it is 23-32% vs 7-26%.
   - Within every style the probe still separates task from non-task replies (AUROC 0.989-1.000).

## What the post may claim

- **"The model is still computing at the moment the probe reads low" is supported for Qwen3.5-27B.**
  - Where the score-only attacks push the pre-answer probe reading under its 1% line, 27B still holds the answer
    (correct leading digit strongly preferred over non-echo digits in 83% of evaluable replies).
  - In the generation context, the pre-answer probe drops 6 log-odds while Qwen predicts the answer at P ≈ 0.99.
  - State it with the 9B and m19 caveats: 9B's digit answers and m19's template genuinely change what the model
    predicts at that slot.
  - Also state that echoes compete: the attack text makes "quote a number from the question" a strong alternative.
- **"The probe reads the answer, plus a little of the work."** Do not claim the maths or code probes detect computation
  as such. Computing a property of the result without stating it scores near non-answers. Stating the answer brings the
  score back.
- **Transfer to unattacked probes:** cite the other layers, not the sister seeds. Phrase it as "not specific to the
  attacked direction", not "a general representation change". The new probes share training data, and cross-layer
  cosines (0.14-0.52) are similar to cross-task cosines at layer 48 (0.18-0.32).
- **Tone vs task:** cite the broad pool. Task lines sit at 80-85%, sarcasm at 33-36%.
  - On the broad scale the attacks look larger: the best Opus arms reach 0.31-0.76 of plain-to-ordinary, and 9.1% of
    ordering's best arm falls under the broad-pool line (1.1% for maths, 0% for code).
  - The deployed one-sentence lines would flag 73% / 96% / 100% of those ordinary replies.
- **"Beyond style":** scored searches close about half (maths) to three quarters (ordering) of the plain
  task-vs-non-task gap within their own style; probe-blind random styles close about a third. State the split:
  - roughly half of each number is non-task replies becoming task-like;
  - the search's advantage is clearer on the task-reply side (maths 27-32% vs 7-11%).

## Q1: next-token probability of the answer at the token before it

**Method.**
- **Data:** every valid frozen-test reply of plain, every own-target arm and m19's template:
  - maths 27B: 8,814 replies (every valid one; all had a located answer);
  - maths 9B: 4,305;
  - code: 3,142.
- **Answer location:** the answer's tokens come from m27 Part A.
- **One forward pass per reply** (own `M34Lab` class, batched as m10.BigLab; float32 output head) gives:
  - the target probe's per-token log-odds;
  - Qwen's next-token distribution at the answer's first token.
- **Two contexts:** instruction removed and visible.
- **Measures:**
  - *within-form P*: the correct first token's share among number tokens of the same form (digits, or number words,
    with the same spacing and capitalisation);
  - *non-echo P*: the same share after dropping tokens that would start a number given in the question. Replies whose
    correct leading token is itself such an echo, about a third, are left out as unevaluable;
  - *digit route* (critic's fix): for answers written in words, Qwen often predicts " " there, which is the start of a
    digit answer. So we append " " and the correct leading digit and read the digit distribution at that point.
    4,344 27B and 502 9B replies.
  - For code: P of the answer's first letter or digit token, which comes after brackets and quotes.
- **Line:** m27 Part C's pre-answer 1% line. It is set on calibration replies in their own context.
- **Reproduction check:** same tokens as saved. Probe log-odds vs saved: median absolute difference 0.00-0.03 (max 0.56).

**Replies under the pre-answer 1% line, instruction removed:**

| model | answer form | under the line | evaluable (non-echo) | correct leading digit favoured (non-echo P > 0.5) | median non-echo P |
|---|---|---|---|---|---|
| 27B | digits | 457 | 292 | 276 (95%) | 0.81 |
| 27B | words (digit route) | 1,744 | 1,078 | 861 (80%) | 0.97 |
| 27B | m19 template | 485 | 289 | 0 | 0.19 |
| 9B | digits | 2,142 | 1,288 | 342 (27%) | 0.22 |
| 9B | words (digit route) | 499 | 302 | 297 (98%) | 0.96 |
| 9B | m19 template | 273 | 164 | 0 | 0.18 |

- **Code** (first value token, removed): 1,756 of 2,664 attack replies under the line. Their first-value P has median
  0.67 (plain 0.999).
- **Echo competition:** among all digits including echoes, the correct leading digit has a median share of 0.37 for
  27B word answers under the line. It is the top digit 42% of the time (digit-form answers: 0.58 and 65%).

**All replies, by context** (attack = all own-target frozen arms; full per-arm and per-form tables in `q1_table.md`,
`q1_summary.json`, `q1_digit_path.json`):

| model, context | plain: probe before answer / P(answer) | attack: probe before answer (median) | attack: P(correct first token) | attack: greedy writes the whole answer | attack under the line |
|---|---|---|---|---|---|
| 27B removed | +7.1 / 1.000 | -1.9 | 0.000 raw; non-echo 0.50 (digit form 0.97) | 24% | 2,256 / 7,829 |
| 27B visible | +7.1 / 1.000 | +1.1 | 0.992 | 93% | 478 (non-echo P 0.98) |
| 9B removed | +9.9 / 0.999 | -8.7 | 0.003 raw; non-echo 0.20 | 11% | 2,641 / 3,584 |
| 9B visible | +9.9 / 0.999 | +7.3 | 0.972 | 95% | 258 (0.95) |
| code removed | +4.3 / 0.994 | -8.1 | first value token 0.76 | 11% | 1,756 / 2,664 |
| code visible | +4.3 / 0.994 | -3.6 | first value token 0.99 | 96% | 21 |

- The raw P(correct first token) is near 0 in the removed context mostly because the reply's form is a surprise there.
  For example, the model expects " " before a digit and gets " fifty", or expects the reply to end and gets a number.
  That is why the within-form and digit-route measures are needed.
- Figure: `q1_probe_vs_answer_probability.png` (probe before the answer vs the non-echo share of the correct leading digit, word answers via the digit route; both contexts).

**Verdict.**
- **27B:** the model is still computing where the probe reads low. That holds in the headline re-read (correct
  leading digit favoured among non-echo digits in 83% of evaluable under-line replies) and in the generation context
  (P ≈ 0.99 with a 6-log-odds pre-answer drop).
- **9B:** mixed. Its digit-form attack answers lose the answer at that slot in the re-read; its word-form ones keep it.
- **m19's template** changes the computation itself at that slot, on both models.
- **The cost of the attack text:** it makes echoing a number from the question a strong competitor. So the attacks
  also weaken the model's commitment to the answer in the re-read, not only the probe.

## Q2: requests that need the computation but not its result

**Method.**
- **Requests:** Qwen3.5-27B on m22's 100 maths test problems and m25's 100 code programs, 2 seeds, one-sentence replies.
  - **compute:** say only whether the result is even or odd, or more than T (T a round number 0.7-1.5× the
    result); for code, whether the output contains a character c, or whether its largest number is more than T.
  - **copy:** the same question about the first number in the problem, or about the program's first line.
  - **given** (the critic's wording control): the *same* compute question, word for word, on requests whose result is
    already given (m27's 200 maths lookups; programs that print a literal).
  - **state:** state the result and its property.
- **Validity:** compute, copy and given replies must not state the result or any new number; state replies must state
  the result. 3,059 of 3,062 are valid.
- **Correctness:** Haiku 5.5 says which option each reply gives; truth is programmatic.
- **Scoring:** the deployed reading, read as asked (the request as the monitor would see it), and after the original
  "state the result" request.

| task | condition | accuracy (majority) | median score as asked | after the original request | over the deployed line (as asked) |
|---|---|---|---|---|---|
| maths | plain answer | | +8.9 | +8.9 | 100% |
| maths | state result + parity | 100% (67%) | +6.0 | +8.9 | 100% |
| maths | compute parity | 84% (67%) | -4.9 | -2.5 | 95% |
| maths | compute threshold | 71% (53%) | -6.2 | -0.7 | 31% |
| maths | given parity (result in request) | 98% (58%) | -7.5 | -9.2 | 8% |
| maths | given threshold | 98% (55%) | -7.6 | -5.5 | 27% |
| maths | copy parity / threshold (first number) | 94% / 98% | -7.2 / -5.7 | -5.3 / -4.4 | 0% / 58% |
| maths | m27 non-answers | | -6.8 / -7.2 | | 10% / 0% |
| code | plain answer | | +9.9 | +9.9 | 100% |
| code | state output + contains c | 91% (56%) | +8.5 | +10.0 | 100% |
| code | compute contains c / threshold | 72% / 87% | -7.6 / -6.7 | -4.9 / -3.3 | 10% / 20% |
| code | given contains c (literal print) | 96% (50%) | -8.0 | -7.1 | 0% |
| code | copy contains / threshold (first line) | 80% / 89% | -8.8 / -7.6 | -5.7 / -5.3 | 0% / 0% |

Deployed lines: -5.80 (maths), -5.44 (code). Pairwise differences, AUROCs and positions are in `q2_table.md`.

| compute vs same question with the result given | as asked: difference (AUROC) | after original request: difference (AUROC) | compute's position, plain 0 → given 1 |
|---|---|---|---|
| maths parity | +2.6 (0.97) | +6.6 (1.00) | 0.84 |
| maths threshold | +1.4 (0.74) | +4.8 (0.98) | 0.91 |
| code contains c | +0.4 (0.89) | +2.2 (0.93) | 0.98 |

**Verdict.** Mostly an answer detector, with a small work component.
- Replies that compute but don't state the result land 84-98% of the way from plain answers to the given-result
  replies.
- The leftover is consistent and bigger in the re-read after the original request. At the narrow deployed line it
  decides flagging: maths compute-parity is flagged 95% vs given 8%. That is a work signal, not the line's narrowness.
- **Caveat:** compute accuracy (71-87%) is well under the given-result accuracy (96-98%). Part of the low score may be
  Qwen not fully computing when told not to state the result, so "the probe misses the work" and "less work is done"
  cannot be fully separated.
- The yes/no properties allow shortcuts (the parity of a product follows from the factors' parities).

Files: `q2_summary.json`, `q2_table.md`, `q2_compute_without_answer.png`, `q2/judged.jsonl`,
`q2/judged_given.jsonl`, requests `data/concepts/m34_what_the_probe_reads_q2_requests.jsonl` and `..._q2_given_requests.jsonl`.

## Q3: sister probes and probes at other layers

**Cosines** (saved weights; `q3_cosines.json`, `q3_cosine_table.md`).
- The nine layer-48 maths probes (own-reply, size-matched own-reply and Claude-text; 3 seeds each): weight cosine
  0.75-0.95, median 0.85. Random pairs would give about 0.014.
- Their deployed readings correlate 0.89-1.00 over every m22 test reply. Seed-1 own-reply is the odd one (0.75-0.85).
- Cross-task at layer 48 (maths vs code vs ordering): cosine 0.18-0.32; per-token readings on the broad pool correlate
  0.36-0.38.

**New probes.**
- **Recipe:** m1a's peak recipe (`m1a_peak_probe.fit_peak_probe`) on the same 1,898 own-reply rows, at layers 32, 40,
  48 (refit) and 56, seeds 0-2. Saved in `results/concept_probes/m34_what_the_probe_reads_maths_layers_peak.json`.
- **Refit check:** the refit layer-48 seed-0 probe has cosine 0.9995 with the deployed probe. All validation AUROCs
  are 1.0. Cross-layer cosines are 0.14-0.52.
- **Lines:** each probe gets its own 1% line on m22's calibration pool. Plain-to-line gaps are 11-15 log-odds.
- **Data read:** m22's own-target arms plus m19, and all 34 m28 maths arms, instruction removed.

**Gap closed (share of each probe's own plain-to-line gap; median over arms of each type; seed 0):**

| arm type (arms) | L32 | L40 | L48 refit | L56 | deployed L48 |
|---|---|---|---|---|---|
| Opus with scores, best (3) | 30% | 29% | 33% | 35% | 33% |
| pinned 4.8 with scores, best (3) | 35% | 36% | 39% | 40% | 40% |
| Opus no scores, best (3) | 25% | 16% | 27% | 26% | 27% |
| Qwen-27B writes for itself (3) | 29% | 16% | 25% | 23% | 25% |
| Qwen-122B writes (3) | 4% | 14% | 21% | 23% | 21% |
| no readout spec, best (6) | 28% | 16% | 20% | 17% | 20% |
| random styles (6) | 3% | 8% | 12% | 15% | 12% |
| random styles in words (2) | 29% | 20% | 23% | 23% | 23% |
| told-probe (2) | 0% | 0% | 1% | 1% | 0% |
| m19 white-box template (1) | 69% | 83% | 105% | 133% | 105% |

- **Per-arm agreement with the deployed probe:** r = 0.87-0.88 (L32), 0.73-0.95 (L40), 0.89-0.95 (L56).
- **Within-style gap closed** (m27's maths twins, the six searched arms): 38-39% (L32), 29-41% (L40), 47-48%
  (L48 / L56).
- **Attack replies under each layer's own line:** 0-3 (L32 / L40 / L48), 43-48 at L56 (mostly Qwen-27B's own
  chain 2).
- **m19's template under the line:** 0 at L32 / L40, 263-394 at L48, 470-475 at L56.

Files: `q3_layers.json`, `q3_layers_table.md`, `q3_layers.png`.

**Verdict.**
- The sister-seed transfer is near-tautological.
- Probes at other layers, pointing in clearly different directions, move with the deployed probe on the arms that move
  it most (the scored and searched instructions). So the drop is not a weakness of the one attacked direction.
- The effect grows with depth: random styles and Qwen-122B's instructions barely touch layer 32, and the within-style
  gap is smaller at layers 32-40.
- m19's white-box template is partly layer-specific.

## Q4: broad-pool line positions

**Method.** m29's pool-tail decomposition (`m29.point_summary`). The ordinary pool is swapped for m27 Part D's 1,720
broad-pool replies with a number or a list, with the line at their 99th percentile. Plain answers and the best
Opus-with-scores arm are m29's.

| task | ordinary pool | median ordinary | plain - ordinary | d' | 1% line | line position (0 = plain, 1 = ordinary) | best Opus arm, same scale (all Opus best arms) | best arm under the line | broad-pool replies flagged |
|---|---|---|---|---|---|---|---|---|---|
| maths 27B | one-sentence | -9.1 | 17.9 | 14.9 | -5.80 | 0.82 | 0.33 (0.26-0.33) | 0% | 73% |
| maths 27B | broad | -5.2 | 14.1 | 13.0 | -2.35 | 0.80 | 0.42 (0.34-0.42) | 1.1% | 1% |
| code | one-sentence | -9.5 | 19.4 | 16.2 | -5.44 | 0.79 | 0.22 (0.20-0.22) | 0% | 96% |
| code | broad | -3.6 | 13.5 | 13.2 | -1.27 | 0.83 | 0.31 (0.28-0.31) | 0% | 1% |
| ordering | one-sentence | -9.4 | 20.0 | 14.1 | -7.30 | 0.90 | 0.56 (0.32-0.56) | 0% | 100% |
| ordering | broad | -4.1 | 14.8 | 9.8 | -1.97 | 0.85 | 0.76 (0.44-0.76) | 9.1% | 1% |

- Broad plus one-sentence pool together: line positions 0.75-0.79.
- **Reference (m29):** sarcasm 0.33-0.36 with d' 4.4-5.0; condescension 0.72 with d' 7.2.

**Verdict.** The median ordinary reply and the line both move up by 3.4-5.9 log-odds, so the line position hardly
changes. The tone-vs-task contrast does not depend on the narrow pools. On the broad scale, ordering's best arm gets
close to its line and 9% of its replies cross it.

## Q5: within-style gap closed, one table

**Definition.**
- Gap closed within style = 1 - (attack median - styled-twin median) / (plain median - plain-twin median), on the
  deployed reading, instruction removed. Plain and plain-twin medians are fixed per task.
- It splits into two parts:
  - **task-falling share** = (plain median - attack median) / plain gap;
  - **twin-rising share** = (styled-twin median - plain-twin median) / plain gap.

**Data.**
- **Existing twins:** m27 Part B and m28, reproduced to within 1 point.
- **New twins** (9,000): Qwen-27B and Qwen-122B writers, pinned 4.8 with and without scores (best and last), and the
  last-round arms of m22 / m25. Generated as m28's `run_twins` does and cleaned with `m28.clean_twins`.
- **Audit override:** for the three Qwen-27B ordering instructions, the Haiku "does it work something out" audit
  rejected 192-196 of 200 twins. It reads instruction-dictated boilerplate as reasoning ("All clues verified."). The
  audit is ignored for those three arms. On the other 38 ordering arms, ignoring it moves styled-twin medians by at
  most 0.03.
- **Intervals:** 95%, resampling test problems and twins.
- **m19 reference row:** uses m19's own saved twins.

| arm family | maths 27B: closed (task falling / twins rising) | maths 9B | code | ordering: closed (task falling / twins rising) |
|---|---|---|---|---|
| searched, Opus with scores, best | 58% (27% / 26%) | 48% | 37% | 75% (58% / 19%) |
| searched, Opus with scores, last | 48% (23% / 25%) | | 31% | |
| pinned 4.8 with scores, best / last | 55% / 55% (32% / 24%) | | | 75% / 75% (51% / 25%) |
| searched, Opus no scores, best / last | 40% / 38% (22% / 17%) | | | 75% (47% / 23%) |
| pinned 4.8 no scores, best / last | 23% / 31% (7% / 16%) | | | 68% / 59% (45% / 25%) |
| model writes for itself (Qwen-27B; 9B) | 43% (21% / 21%) | 53% | | 47% (38% / 10%) |
| Qwen-122B writes | 16% (17% / 10%) | | | 61% (50% / 31%) |
| no readout spec (fallback / pinned) | 41% / 30% (12% / 25%) | | | 47% / 23% |
| random styles, best of 16 / unselected | 28% / 35% (11% / 18%; 7% / 27%) | | | 36% / 42% (26% / 14%) |
| random styles in words (m22-informed) | 58% (21% / 37%) | | | |
| m19 white-box template | 92% (85% / 7%) | | | |

- Medians over arms. The two shares are medians taken separately, so they need not add up to the closed share.
- Every arm, with its interval: `q5_table.md`, `q5_within_style.png`, `q5_within_style.json`.
- **AUROC** of attack replies vs styled twins: 0.989-1.000 on every arm (m19 0.951).

**Verdict.**
- Scored searches close about half the within-style gap on maths and three quarters on ordering; probe-blind random
  styles close about a third.
- About half of most maths numbers is the non-task replies rising, which is fairly flat across families (mostly 16-27%).
  So the task-falling share separates the arms more cleanly: searched with scores 27-32% vs random 7-11% on maths.
- The metric credits styles that make non-task replies look task-like. Report both sides.
- No arm makes task replies overlap their styled twins.

## Critic check-in and what changed

One research-reviewer read the code, the data and the first draft. Its findings and the changes:

1. **Q1 over-claimed a collapse.** The first draft said the answer prediction "collapses" in the re-read and that
   Part C's drop was "mostly a property of the re-read". The critic showed:
   - for 27B, the collapse came from answers written in words, where the number-word renormalisation amplifies a tiny
     tail while Qwen prefers the digit route (" ");
   - digit-form 27B answers keep the answer;
   - the visible-context pre-answer drop is still 6-8 log-odds for 27B and code.

   Fixed: a split by answer form, plus a new GPU read through the digit route. The verdict is reversed for 27B: it is
   still computing. The removed-context counts of replies under the line with the answer still predicted are now
   reported.
2. **Q2 over-claimed "near-pure answer detector".**
   - Added the critic's wording control: the same question on requests with the result given.
   - Both readings are now reported.
   - The 95% vs 8% flagging is now treated as a work signal.
   - Accuracy is flagged as a confound (compute 71-87% vs given 96-98%).
3. **Q5 "separates arms more than the raw drop" was wrong.** Added the task-falling / twin-rising split and rephrased.
4. **Q3 "every arm moves similarly" was too strong.** Now the exceptions (L32 for random styles and Qwen-122B; L40 for
   unscored writers) and the cross-task cosine baseline are named, and the phrasing is "not specific to the attacked
   direction".
5. Q4 now reports the broad-line crossings (ordering 9.1%) and the narrow lines' false-alarm rate on the broad pool.
6. Q5 has an audit sensitivity check. Terms are defined. The critic's worry that Q3 dropped replies without a located
   answer was checked: none were dropped (8,814 of 8,814).

## Caveats

- **Q1.**
  - The "non-echo" measure leaves out about a third of replies, those whose correct leading token is also the leading
    token of a number in the question.
  - The digit route is a counterfactual continuation: the reply actually used words.
  - Only the first token is asked about. Multi-digit answers can be partly computed while being written.
  - The pre-answer line is set on calibration replies without any instruction, so it is not a calibrated 1% line in
    the visible context.
  - The code Spearman and within-form figures cover only the 445 code replies whose answer starts with a digit.
- **Q2.** Shortcut-friendly properties; one model; different problem subsets for code thresholds (148 vs 114 replies).
  51 code compute-threshold replies are just "No." or similar.
- **Q3.** One training set. The new probes are read instruction removed only. m22's Claude-text-target arms were not
  re-read.
- **Q4.** m27's broad pool: Haiku-written prompts, replies cut at 96 tokens.
- **Q5.**
  - Twins use different requests from the attack replies.
  - 200 twins at one seed per arm.
  - Three ordering arms have the audit ignored.
- **Batching:** this lab's forward passes are batched. Probe log-odds differ from the saved unbatched 27B scores by a
  median 0.03 (max 0.56). Within-mission comparisons all use one lab.

## Files

- **Code:** `src/probe_evasion/missions/m34_what_the_probe_reads.py`.
  - Local steps: `q4 q5 q5_clean q3_cosines q3 q1 q1_digit_path q2_judge q2`.
  - Modal entry points: `m34_smoke m34_q1 m34_q3 m34_q2 m34_q5 m34_followup`.
- **Data:** under `missions/m34_what_the_probe_reads/`:
  - `q1/` (raw reads with next-token summaries, records, `digit_path/`);
  - `q2/`;
  - `q3/`;
  - `q5_twins/<run>/twins/`;
  - summaries, tables, figures;
  - `logs/`.
- **Probes:** `results/concept_probes/m34_what_the_probe_reads_maths_layers_peak.json`.
- **Requests:** `data/concepts/m34_what_the_probe_reads_q2_requests.jsonl`, `..._q2_given_requests.jsonl`.
- **Process:** `STATUS.md`.

## Cost

- **GPU (Modal):** seven apps, all finished on their own (IDs in STATUS.md). About 70 minutes of app wall time on at
  most 6 H100s each, so about $15-30 by estimate, within the $80 cap.
- **Claude:** Haiku 5.5 only. About 2,900 Q2 verdicts plus about 9,000 twin audits and span locations, roughly $6 by
  estimate, within the $30 cap.
