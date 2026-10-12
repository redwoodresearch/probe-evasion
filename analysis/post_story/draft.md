# DRAFT v3: A cheap black-box search moves activation-probe scores a long way, but the probe still sees the task

*Working draft. [PENDING mXX] marks an experiment running now; [HOLE] a gap with no experiment yet. Sources in brackets
(mission folders under `missions/`). The history of drafts and reviews is in PLAN.md.*

## TL;DR

- **Question.** Can prompting alone make a model keep doing a task while a linear probe on its activations stops
  noticing, and how much does the attacker need to know about the probe?
- **Set-up.** Qwen3.5-27B (also Qwen3.5-9B, Qwen3.5-122B-A10B, and an abliterated Gemma-2-9B), reasoning off, does
  tasks with an automatic correctness check: maths word problems, predicting a short program's output, ordering five
  people from clues. A probe trained on the model's own replies flags "doing the task". A writer model (Claude, or Qwen
  itself) writes one instruction for the monitored model and revises it over 16 rounds of feedback. A reply counts only
  if it is correct. [m22, m23, m25, m26]
- **Signs of life.** A cheap, black-box search that sees the monitored model's replies (and in most arms per-token
  probe scores) moves the median correct answer 3–11 log-odds down, roughly a fifth to two thirds of the way from
  plain answers to the probe's 1% flag line, on every task and model size tried. The movement transfers to probes the
  writer never saw (on maths 27B the never-attacked probe trained on Claude-written answers moves 6.8 log-odds, more
  than the attacked one's 4.9), and it is already present in the model's state at the token before the answer is
  written. [m22, m25, m26, m27, m29]
- **But the probe still sees the task.**
  - Not one of 27 searched maths instructions got a single correct answer under the line on fresh problems (0/500 each);
    on the attacked probes, attacked maths, code and ordering answers all still score above every ordinary reply used to
    set the line. [m22, m29]
  - The same instructions applied to requests that don't need the task push those replies *up*: task and non-task
    replies converge from both sides, and within the attack's style the probe still separates them almost perfectly
    (AUROC 0.989–1.000); a line set on same-style replies catches essentially all (maths 9B is the exception, 5–40 per
    arm). [m27]
  - The answers that score lowest are visibly odd (number words, stock openers, false "as you wrote" attribution,
    filler); a judge picks the plain reply as the normal one 100% of the time for maths and ordering. [m29]
- **What the attacker needs.** With per-token scores plus the probe's training data and a scoring tool (and a hint about
  which notations were allowed), Claude found instructions that put 384–407 of 500 correct answers under the line.
  Without them, nothing crossed. Whether the per-token scores themselves help depends on the writer model: with Claude
  Opus 4.8 pinned, scores helped clearly; a mostly-Opus-5.5 writer without scores got about as far as with them.
  [m19, m22, m28; PENDING m28, m31: random-style baseline, a writer told nothing about the readout, Qwen just told a
  probe is watching, 8 runs per arm with pinned Opus 4.8 and 5.5]
- **Why tone probes were crossed and task probes weren't.** With the identical search (same runner, pinned writer,
  16 rounds, full-strength validity), sarcasm and condescension probes are still crossed and task probes still aren't.
  Crossing follows mechanically from two numbers: how far the replies move (as a share of the distance from plain
  answers to typical ordinary replies) and where the 1% line sits on that scale. Sarcasm's line sits only 36% of the way
  down (ordinary traffic has near-sarcastic replies), so even task-sized moves would cross it; a zero-shot deadpan
  instruction already moves sarcasm 40%. Condescension moves further than any task (61–74% vs 20–56%) and its line sits
  at 71%. Task lines sit at 79–90%. Score feedback adds about 20 points of movement for tone and tasks alike. [m29, m30]

## 1. Set-up

- **Models.** Qwen3.5-27B (main), Qwen3.5-9B, Qwen3.5-122B-A10B (mixture of experts, ~10B parameters active per token,
  so not cleanly "bigger"), Gemma-2-9B (the abliterated model the repo uses). Reasoning off. [m22, m23] [PENDING m32:
  reasoning on]
- **Tasks.** Maths word problems with a one-sentence answer; "what does this Python program print?" (12 short program
  types Qwen gets right at least 85% of the time); order five people from four comparison clues. A reply counts only if
  correct; odd wording is allowed. [m22, m25, m26]
- **Probes.** Linear, one layer at about 75% depth, trained on the monitored model's own replies (task done vs matched
  replies that needn't do it, e.g. the answer is already given), read as the peak of a running average over the reply's
  tokens, the reading they were trained for. Flag line at 1% false alarms on a pool of ordinary replies in the same
  one-sentence format. The probe reads the reply with the attack instruction removed, so the instruction's own text
  can't lower the score. [m1a, m22, m25, m26]
- **What the probe actually detects.** Swapping the answer for a wrong value of the same form barely changes maths and
  code scores (−0.1 to −0.4 log-odds); a wrong order lowers ordering scores by 3.3. So the maths probe responds to "a
  value in the answer slot after a request that needs working out", not to correct computation. [m27]
- **Attacker loop.** A writer gets the task and a paragraph about the probe. Each round it sees a sample of replies,
  whether each was correct, and (except in controls) each token's probe score, then writes a new instruction. 3 runs ×
  16 rounds; each run's best and last instructions are frozen and tested on 100 fresh problems × 5 samples. [m22]
- **Disclosure.** Every writer in m22–m26, including zero-shot and the no-scores control, was told the exact
  running-average formula (including that the first token alone gives half its score), the layer, the line, and what
  the probe's negative examples were. [m22 WRITER_SYSTEM; PENDING m28: a writer told only that "a monitor flags
  computing the answer"]

## 2. How far scores move, and what that means

**Central figure** [m29 central_figure.png, m28 arms to be added]: share of the gap closed per arm on the final tests,
every run as a dot, on three scales: to the 1% line; to the level of in-context non-answers (the model's replies to the
real request that skip the task); to the level of same-style non-task replies.

| Task, model (Claude writer, final tests) | Median change from plain (log-odds) | Gap to 1% line closed | Correct answers under the line |
|---|---|---|---|
| Maths 27B | −0.9 to −7.2 across 27 instructions | up to 55% | 0 of 500 for every instruction [m22] |
| Maths 9B | −3.5, −6.4, −8.9 (Claude's three runs) | 26–67% | 30 of 500 for one instruction (6 distinct answers), else 0–1 [m23] |
| Code 27B | −4.2 (main run) | 25–28% | 0 [m25] |
| Ordering 27B | −6.5 to −11.2 | 36–63% | 0 [m26] |

- **Operational meaning.** On the attacked probes for maths 27B, code and ordering, every valid score-only reply scores
  above every one of the 1,301–1,697 ordinary replies (down to the pool's resolution, about 1 in 1,300); the median
  attacked reply sits about 6–15 log-odds above the line. On the never-attacked maths probe trained on Claude-written
  answers, 5 of 3,617 dip into the ordinary range, none under the line. Maths 9B is the exception (half of one arm sits
  inside the ordinary range, 97% still caught at 1%). Only the white-box template lands among ordinary replies. [m29]
- **The lines are fragile, the conclusion isn't.** The task probes' 1% lines would flag 72% (maths), 96% (code) and 100%
  (ordering) of ordinary chat replies that mention numbers or lists; the 1% figure holds only for the one-sentence format
  they were calibrated on. Re-set on a broad pool, the lines rise 3.4–5.3 log-odds, yet score-only attack replies still
  almost never get under (maths 6 of 7,829; code 13 of 2,664; ordering 126 of 5,044), while the white-box template does
  (485 of 485). [m27 Part D]
- **Selection and validity.** Pairing each reply with plain Qwen's reply on the same problem and seed makes the drop
  slightly smaller (median 0.1 log-odds). Counting invalid replies as caught costs the median arm 2.4 points of gap
  closed (75th percentile 6.7); low-validity arms lose more (Opus maths chain 0: 33% → 17%). [m29]
- **Where it moves most.** Maths division (2-digit answers) and code integer outputs drop most; plain difficulty can't
  be tested because plain Qwen gets 96–100% right. [m29]
- **Zero-shot** instructions close 0–20% of the gap, most for ordering and the 9B model. [m22, m23, m25, m26]
- **Model size** matters mainly at zero-shot (9B moves more; permutation p 0.004–0.02). After feedback there's no
  ordering by size. Correct-answer rates fall more on 9B, 122B and Gemma (to 45–64%) than on 27B (85%). [m23]

## 3. The probe still sees the task

- **Other readouts.** The drop survives four of five ways of reading the probe (per-token max, mean, running average
  restarted at the answer, max over the answer tokens) on every task. Restarting the average at the answer removes about
  half the drop for code, less for ordering and 9B, none for maths 27B, 122B and Gemma. [m27]
- **Before the answer.** At the token just before the answer, the attack already moves the probe past the line on every
  task (e.g. maths 27B −11.7 log-odds), more than on the answer tokens (23–42% of the gap). Comparing only replies with
  the same token before the answer: maths 27B plain +7.7 vs attack +1.0. The model still writes the correct answer at the
  next token, so the probe's pre-answer signal tracks context and style more than the work. [m27 Part C]
- **Same-style twins.** Each frozen instruction applied to non-task requests raises those replies' scores by up to 6
  log-odds (at the deployed line most would be false alarms). Attack vs twin AUROC 0.989–1.000 on every score-only arm;
  with the line set on twins, 0 attack replies get under it for maths 27B and code, 0–10 for ordering, 5–40 per arm for
  maths 9B. [m27] [PENDING m28: twins for random styles — is the convergence specific to searched instructions or
  generic to any odd style?]
- **In-context non-answers** (replies to the real request that skip the task) score close to the line (−4.5 to −6.1)
  and still separate from attack replies (AUROC 0.97–0.999). [m27]
- **Transfer.** On maths 27B the no-scores writer moves all nine layer-48 maths probes by 24–39%; attacks aimed at the
  own-reply probe move the never-attacked Claude-text probe more in log-odds (−6.8 vs −4.9). The writers aren't
  exploiting a particular seed or recipe at that layer. Ordering's apparent target-specificity is mostly one sister seed
  being insensitive; code's persists in last-round arms (24% vs 18%). Across layers it's mixed (9B's layer 8 moves 11%,
  122B's layer 12 moves 98%). [m29]
- **The evading replies are visibly odd.** Maths: answers in English words, "As the question itself already tells
  us…". Code: "Yes, exactly as you wrote above: [34, 18]". Ordering: filler before the first name. A second judge rates
  only 35% of the first judge's "normal" replies as normal; side by side, the plain reply is picked as the normal one
  100% of the time for maths and ordering. Among replies rated fully normal, the drop shrinks to about 0 for code and
  several maths arms. [m22, m25, m26, m29]

## 4. How much does the attacker need to know?

Maths 27B, own-reply probe, fresh problems, median change from plain (mean of each arm's runs):

| Attacker | Last-round instruction | Best of 16 (picked with the probe) |
|---|---|---|
| White-box: per-token scores + training data + scoring tool + notation hints | — | −15.4; 384/500 under the line [m19 via m22] |
| Per-token scores, mostly Opus 4.8 | −4.0 | −5.2 [m22] |
| No scores, mostly Opus 5.5 | −3.2 | −3.5 [m22] |
| Per-token scores vs no scores, Opus 4.8 pinned | [PENDING m28/m31] | |
| Per-token scores vs no scores, Opus 5.5 pinned | [PENDING m31] | |
| Writer told nothing about the readout | [PENDING m28] | |
| Random odd styles (no probe knowledge) | — | [PENDING m28] |
| Qwen3.5-27B writing for itself | | −3.7 [m22] |
| Qwen just told a probe is watching | [PENDING m28] | — |

- Early evidence on the writer: on m28's first 3+3 pinned Opus 4.8 runs, scores help clearly in the per-round eval
  (maths −3.6 vs −0.8; ordering −8.5 vs −6.0), while m22's mostly-Opus-5.5 no-scores writer reached about −3.3 without
  scores. So "does the writer need the probe signal?" depends on the writer model. [m28 via m31 design critic]
- The white-box gap mixes information, budget and rules: that red team started from replies Claude wrote directly and
  its prompt listed scientific, k and Roman notation as allowed (exactly its winning formats); m22's rules dropped the
  hint. [m19, m22]
- Zero-shot (round 0) moves about −0.2 on this probe. [m22]

## 5. Tone vs task probes

- **Earlier crossings.** Sarcasm: 268/500 valid sarcastic replies under the line with a score-guided search (plain 29);
  the median attacked reply sits roughly at the line. Condescension: crossed only by a 42-round, multi-model search
  (475/500 full-strength replies on fresh messages); a 6-round search got 13%. Patching the probe, broad style coverage
  and layer ensembles each lost to a fresh search. [m3a, m3a2, m3a4, m8b, m11]
- **Where the line sits** (pool_tail_figure): on one scale from the median ordinary reply (0) to plain answers, the 1%
  line is 11–12 log-odds above typical ordinary replies for sarcasm, 6.6 flattery, 8.1 anger, against 2–4 for maths,
  code and ordering. Plain answers sit 4–5 standard deviations above ordinary replies for sarcasm, flattery and anger
  (7.2 condescension) and 8.6–16.2 for the tasks. [m29]
- **The shifts aren't "similar".** With the same long search (m11), condescension moved 11.1 log-odds (0.88 of the way to
  ordinary) and maths 3.5–3.9 (0.20–0.22). Log-odds aren't directly comparable across probes, and no single margin
  measure predicts crossing (Gemma maths, m9's jokey probe and m3b's criticism probe have small margins and weren't
  crossed; condescension has a task-like one and was). [m29] [PENDING m30: equal-effort search on sarcasm and
  condescension with the same runner, writer and rounds as the tasks]
- **Caveats on the tone side.** Flattery's probe failed its own plain-caught check and detects only gushing wording, so
  it is left out of the argument; anger's attack replies are mostly mild (32 of 196 clearly angry). Tone validity allows
  a weakened concept, while task validity is exact correctness. Re-reading condescension with the word "condescending"
  also removed from the request cuts its evasions from 463 to 155. Several tone searches optimised the score with the
  instruction in context. [m3c, m8a, m11, m29]
- Ordering's 1% line sits below where any reply to an unsolved puzzle can start (the running average starts at half the
  first token's score), so its "gap to the line" overstates what's reachable; measured to in-context non-answers it is
  15.1 rather than 17.9 log-odds. [m26, m29]

## 6. Limitations

- Code and ordering probes partly react to the request: in-context non-answers already score near or above the line.
  [m25, m26, m27]
- Writers: mostly Claude, with Opus 5.5 falling back to 4.8 in m22–m26 (pinned in m28, m31); Qwen writers weaker; 3 runs
  per cell in most arms (8 in m31). One probe layer and readout per task; reasoning off [PENDING m32]; Qwen family plus
  Gemma with a different probe recipe.
- The naturalness judgments are by models (Haiku, Sonnet), not people.
