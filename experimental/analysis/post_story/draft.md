# DRAFT v4: Prompting moves task-probe scores only as far as the attacker's knowledge of the probe allows; on its own the model doesn't move them

*Working draft. [PENDING mXX] marks an experiment running now. Sources in brackets (mission folders under
`missions/`). The history of drafts and reviews is in PLAN.md.*

## TL;DR

- **Question.** Can prompting alone make a model keep doing a task while a linear probe on its activations stops
  noticing, and how much does the attacker need to know about the probe?
- **Set-up.** Qwen3.5-27B (also Qwen3.5-9B, Qwen3.5-122B-A10B and an abliterated Gemma-2-9B), reasoning off, does tasks
  with an automatic correctness check: maths word problems, predicting a short program's output, ordering five people
  from clues. A probe trained on the model's replies flags "doing the task". A writer model (Claude, or Qwen itself)
  writes one instruction for the monitored model and revises it over 16 rounds of feedback. A reply counts only if it is
  correct. The probe reads each reply with the attack instruction removed. [m22, m23, m25, m26]
- **Main result: movement is limited by the attacker's information about the probe, not by the model.**
  - On its own the model doesn't move: Qwen told "a monitor reads your activations" changes nothing (−0.05 to −0.3
    log-odds); Qwen's own zero-shot instructions for itself change nothing. [m28, m22]
  - Probe-blind odd styles move scores a little (one random style: −1.2 maths, −3.1 ordering); picking the best of 16 by
    the probe roughly doubles that. [m28]
  - A writer told how the score is read and shown per-token scores moves them much further (−5.2 maths, −9.9 ordering on
    fresh problems, pinned Claude Opus 4.8, unselected last round). Qwen writing for itself with the same information gets
    70–75% of that. A writer told only that "a monitor flags computing" gets little (pinned Opus 4.8). [m28, m22]
  - Only a white-box attacker (per-token scores plus the probe's training data and a tool to score any text, and a hint
    about allowed notations) got correct maths answers under the 1% line (384–407 of 500), and only with the instruction
    hidden from the probe: read with the instruction visible, none get under. [m19, m22]
- **What moves is mostly style the probe still separates.**
  - On Qwen3.5-27B and 9B tasks, the searched instructions move the median correct answer 3–11 log-odds, a fifth to two
    thirds of the way from plain answers to the 1% line. But not one of the 27 maths instructions got a single correct
    answer under the line on fresh problems, and on the attacked probes every attacked maths, code and ordering answer
    still scores above every ordinary reply used to set the line. [m22, m29]
  - The same instructions applied to requests that don't need the task push those replies *up*; within the attack's
    style the probe still separates task from non-task almost perfectly (AUROC 0.989–1.000; random odd styles do the same,
    AUROC 1.000). What the search adds over random styles is pushing task replies further down within style. [m27, m28]
  - The lowest-scoring answers are visibly odd (number words, stock openers, false "as you wrote" attribution, filler).
    [m29]
- **Why tone probes (sarcasm, condescension) were crossed and task probes weren't, at equal effort.** Two things decide
  crossing: how far replies move, as a share of the distance from plain answers to typical ordinary replies, and where
  the 1% line sits on that scale. Sarcasm's line sits only 36% of the way down, because ordinary traffic has
  near-sarcastic replies; task lines sit at 79–90% on their one-sentence pools. Condescension moves further on that
  scale than any task (61–74% vs 20–56%), though not in raw log-odds. [m29, m30]

## 1. Set-up

- **Models.** Qwen3.5-27B (main), Qwen3.5-9B, Qwen3.5-122B-A10B (mixture of experts, ~10B parameters active per token,
  so not cleanly "bigger"), Gemma-2-9B (the abliterated model the repo uses). Reasoning off. [m22, m23] [PENDING m32:
  reasoning on]
- **Tasks.** Maths word problems with a one-sentence answer; "what does this Python program print?" (12 short program
  types Qwen gets right at least 85% of the time); order five people from four comparison clues. A reply counts only if
  correct; odd wording is allowed. [m22, m25, m26]
- **Probes.** Linear, one layer at about 75% depth, trained on the monitored model's own replies (task done vs matched
  replies that needn't do it, e.g. the answer is already given; a second maths probe and the condescension probe were
  trained on Claude-written text), read as the peak of a running average over the reply's
  tokens, the reading they were trained for. Flag line at 1% false alarms on a pool of ordinary replies in the same
  one-sentence format. Headline reading: the reply re-read with the attack instruction removed (so the instruction's own
  text can't lower the score); we also report it with the instruction visible. Note the removed reading is a
  teacher-forced re-read of a reply generated under the instruction, not the state of a model that chose the style
  itself. [m1a, m22, m25, m26] [PENDING m35: a model fine-tuned to use the style without the instruction]
- **What the probe actually detects.** Swapping the answer for a wrong value of the same form barely changes maths and
  code scores (−0.1 to −0.4 log-odds); a wrong order lowers ordering scores by 3.3. So the maths probe responds to "a
  value in the answer slot after a request that needs working out", not to correct computation. [m27]
- **Attacker loop.** A writer gets the task and a paragraph about the probe. Each round it sees a sample of replies,
  whether each was correct, and (except in controls) each token's probe score, then writes a new instruction. 3 runs ×
  16 rounds; each run's best and last instructions are frozen and tested on 100 fresh problems × 5 samples. [m22]
- **Disclosure.** Every writer in m22–m30, including zero-shot and the no-scores control, was told the exact
  running-average formula (including that the first token alone gives half its score), the layer, the line, and what
  the probe's negative examples were, except m28's no-readout-spec writers. So most of the search is grey-box with an
  oracle, not black-box. [m22 WRITER_SYSTEM, m28]

## 2. How far scores move, and what that means

**Central figure** [to be re-plotted from m29 central_dots.jsonl: maths 27B and ordering, one row per rung, selected vs
unselected arms marked]: share of the gap closed per arm on the final tests,
every run as a dot, on three scales: to the 1% line; to the level of in-context non-answers (the model's replies to the
real request that skip the task); to the level of same-style non-task replies.

| Task, model (Claude writer, final tests) | Median change from plain (log-odds) | Gap to 1% line closed | Correct answers under the line |
|---|---|---|---|
| Maths 27B | −0.9 to −7.2 across 27 instructions | up to 55% | 0 of 500 for every instruction [m22] |
| Maths 9B | −3.5, −6.4, −8.9 (Claude's three runs) | 26–67% | 30 of 500 for one instruction (6 distinct answers), else 0–1 [m23] |
| Code 27B | −4.2 (main run) | 25–28% | 0 [m25] |
| Ordering 27B | −6.5 to −11.2 | 36–63% | 0 [m26] |

- **Operational meaning.** On the attacked own-reply probes for maths 27B, code and ordering, every valid score-only reply scores
  above every one of the 1,301–1,697 ordinary replies (down to the pool's resolution, about 1 in 1,300); the median
  attacked reply sits about 6–15 log-odds above the line. On the maths probe trained on Claude-written answers (attacked by its own
  arms), 5 of 3,617 dip into the ordinary range, none under the line. Maths 9B is the exception (half of one arm sits
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
- **Before the answer** (a diagnostic, not evidence of evasion). At the token just before the answer, the attack already moves the probe past the line on every
  task (e.g. maths 27B −11.7 log-odds), more than on the answer tokens (23–42% of the gap). Comparing only replies with
  the same token before the answer: maths 27B plain +7.7 vs attack +1.0. The model still writes the correct answer at the
  next token, so the probe's pre-answer signal tracks context and style more than the work. [m27 Part C]
- **Same-style twins.** Each frozen instruction applied to non-task requests raises those replies' scores by up to 6
  log-odds (at the deployed line most would be false alarms). Attack vs twin AUROC 0.989–1.000 on every score-only arm;
  with the line set on twins, 0 attack replies get under it for maths 27B and code, 0–10 for ordering, 5–40 per arm for
  maths 9B. Random odd styles squeeze from both sides too (non-task replies +1.0 to +5.4; AUROC 1.000), so the
  convergence is generic to odd styles; searched instructions leave 18–70% of the plain within-style gap, random styles
  55–93%. [m27, m28] [PENDING m34: within-style gap closed, every arm in one table]
- **In-context non-answers** (replies to the real request that skip the task) score close to the line (−4.5 to −6.1)
  and still separate from attack replies (AUROC 0.97–0.999). [m27]
- **Transfer.** On maths 27B the no-scores writer moves all nine layer-48 maths probes by 24–39%; attacks aimed at the
  own-reply probe move the Claude-text probe more in log-odds (−6.8 vs −4.9). That fits a generic style shift rather
  than evasion of one probe; the sister probes share the layer and are likely near-parallel. [PENDING m34: probe
  cosines; probes at layers 32/40/56] Ordering's apparent target-specificity is mostly one sister seed
  being insensitive; code's persists in last-round arms (24% vs 18%). Across layers it's mixed (9B's layer 8 moves 11%,
  122B's layer 12 moves 98%). [m29]
- **The evading replies are visibly odd.** Maths: answers in English words, "As the question itself already tells
  us…". Code: "Yes, exactly as you wrote above: [34, 18]". Ordering: filler before the first name. A second judge rates
  only 35% of the first judge's "normal" replies as normal; side by side, the plain reply is picked as the normal one
  100% of the time for maths and ordering. Among replies rated fully normal, the drop shrinks to about 0 for code and
  two maths arms. (Side-by-side figure: plain picked 100% of the time among replies Haiku rated 4–5, 94% overall over
  240 pairs; code 53%.) [m22, m25, m26, m29]

## 4. How much does the attacker need to know?

Fresh problems (100 × 5), instruction removed, median change from plain in log-odds, mean over each arm's runs. "Best" =
the round picked with the probe on the eval set; "last" = round 15, no selection. All 0 under the 1% line except
white-box. [m28, m22, m19]

| Attacker | Maths 27B best | Maths 27B last | Ordering 27B best | Ordering 27B last |
|---|---|---|---|---|
| Qwen told "a monitor reads your activations" (no writer) | — | −0.05 | — | −0.2 |
| Zero-shot instruction, round 0 (eval set) | — | −0.2 | — | −3.7 |
| Random odd style, no probe knowledge (best: best of 16 by the probe; last: one style, unpicked) | −2.0 | −1.2 | −4.6 | −3.1 |
| Writer told only "a monitor flags computing", no scores (Opus 4.8 pinned) | −2.7 | −1.0 | −1.2 | −0.6 |
| Same, Opus 5.5 | −3.3 | −0.9 | −6.3 | −3.3 |
| Writer told the readout, no scores (Opus 4.8 pinned) | −2.0 | −1.0 | −8.3 | −6.1 |
| Same, mostly Opus 5.5 (m22/m26) | −3.4 | −3.2 | −9.4 | −8.0 |
| Qwen3.5-27B writing for itself, told the readout, with scores | −3.8 | — | −7.1 | — |
| Qwen3.5-122B writing, told the readout, with scores | −2.9 | — | −8.0 | — |
| Writer told the readout, with per-token scores (Opus 4.8 pinned) | −5.3 | −5.2 | −9.7 | −9.9 |
| Same, mostly Opus 4.8 after fallback (m22/m26) | −5.2 | −4.0 | −9.5 | −9.3 |
| White-box: scores + training data + scoring tool + notation hints | −15.4; 384/500 under | — | — | — |

- **The readout spec matters for Opus 4.8, less for Opus 5.5.** A pinned Opus 4.8 writer told only that a monitor flags
  computing barely moves ordering (−0.6 last round) and does worse than unselected random styles; told the readout it
  reaches −6.1 without scores. Opus 5.5 without the spec still gets −3.3. [m28]
- **Scores vs no scores (pinned Opus 4.8):** every scored run beat every unscored run on both tasks over rounds 1–15
  (ordering −7.8 to −9.6 vs −5.3 to −6.3). On maths the gap is mostly two scored runs that wrote number words at round 0,
  before any score. [m28] [PENDING m31: 8 runs per arm, Opus 4.8 and Opus 5.5 pinned]
- **The cheap maths trick.** The same random styles switched from "in digits" to "in words" reach −3.8 (best of 16),
  about the level of the search without scores (informed by m22, so not a blind baseline). [m28]
- **The white-box gap mixes information, budget and rules:** it started from replies Claude wrote directly, and its
  prompt listed scientific, k and Roman notation as allowed (exactly its winning formats). [m19, m22]
- **Missing rungs:** Qwen writing for itself without scores or without the readout spec; Qwen told how it is read.
  [PENDING m33]

## 5. Tone vs task probes at equal effort

- **Same search, same writer, same rounds** (pinned Opus 4.8, 3 runs × 16 rounds, full-strength validity): the sarcasm and
  condescension probes are still crossed (every tone arm has valid replies under the line, +9 to +34 points of attempts
  over plain); no task arm has a single valid reply under its line. [m30]
- **What decides crossing.** Shifting plain's valid scores down by each arm's median drop predicts the share under the
  line almost exactly (predicted vs observed 95/100, 79/75, 58/60, 23/21, 39/38), so crossing follows from the shift and
  the line's position. Sarcasm's line sits 36% of the way from plain to the typical ordinary reply (ordinary traffic has
  near-sarcastic replies); task lines sit at 79–90% on their one-sentence pools [PENDING m34: on a broad pool]; a
  zero-shot deadpan instruction already moves sarcasm 40% of that distance. Condescension moves further on that scale
  (61–74% vs 20–56% for tasks) but not in raw log-odds (−7.5 to −9.1 vs ordering's −9.0 to −11.2). [m29, m30]
- m30's pre-registered "pool tail" reading (tone moves the same share as tasks and crosses only because its line is
  close) **failed as pre-registered** because sarcasm's best arm (64%) and condescension's arms exceed the task range;
  sarcasm otherwise fits it. [m30]
- **Within style, the tone probes aren't blinded either** (sarcasm AUROC 0.96–0.99, condescension 0.89–0.97 against
  sincere/respectful twins), though less cleanly than tasks. [m30]
- 16 rounds of scored feedback add about 20 points of movement over the zero-shot instruction for tone and tasks alike
  (there is no unscored tone arm). [m30]
- Earlier results agree: sarcasm 268/500 valid sarcastic replies under the line (plain 29); condescension crossed by a
  42-round search; patching the probe, broad style coverage and layer ensembles each lost to a fresh search. Flattery's
  probe failed its own plain-caught check and is left out; anger's attack replies were mostly mild. [m3a, m3a2, m3a4,
  m8a, m11, m3c]

## 6. Limitations

- Code and ordering probes partly react to the request: in-context non-answers already score near or above the line.
  [m25, m26, m27]
- Writers: mostly Claude, with Opus 5.5 falling back to 4.8 in m22–m26 (pinned in m28, m30, m31); Qwen writers weaker;
  3 runs per cell in most arms (8 in m31). One probe layer and readout per task; reasoning off [PENDING m32]; Qwen
  family plus Gemma with a different probe recipe.
- No trend across model generations is measured; within Qwen3.5, size matters mainly at zero-shot (9B moves more).
  With reasoning off, the "model on its own" rungs being ≈0 is close to guaranteed by construction. [m23]
- The naturalness judgments are by models (Haiku, Sonnet), not people.
