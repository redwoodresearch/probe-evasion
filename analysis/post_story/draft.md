# DRAFT v2: Prompting shifts activation-probe scores by a similar amount everywhere; whether that crosses the line depends on the probe's margin

*Working draft. [HOLE] marks a claim not yet backed by data or a missing control; [PENDING mXX] marks an experiment
running now. Sources in brackets. v0 → v1 changes are logged in PLAN.md.*

## TL;DR

- **Question.** Can prompting alone make a model keep doing a task while a linear probe on its activations stops
  noticing, and does it matter how much the attacker knows about the probe?
- **Set-up.** Qwen3.5-27B (also 9B, 122B-A10B, Gemma-2-9B) does tasks with an automatic correctness check: maths word
  problems, predicting a short program's output, ordering five people from clues. A probe trained on the model's own
  replies flags "doing the task". A writer model (Claude, or Qwen itself) writes an instruction for the monitored model
  and iterates on feedback. [m22, m23, m25, m26]
- **Result 1: the scores move a lot, the monitor barely notices.** With a few rounds of feedback, the median correct
  answer drops 3–11 log-odds, a quarter to a half of the way from plain answers to the 1% flag line. But on fresh test
  problems not one of 27 searched maths instructions got a single correct answer under the line (0/500 each), and
  every attacked maths, code and ordering answer on 27B still scores above every ordinary reply that set the line
  (the median attacked answer sits 6–10 log-odds above the highest one). [m22, m25, m26, m29]
- **Result 2: it's a shift, not hiding.** The same instructions applied to requests that don't need the task make those
  replies score *higher* (up to +6), so task and non-task replies converge from both sides. Within the attack's style
  the probe still separates them almost perfectly (AUROC 0.989–1.000). A line set on replies in that style catches
  essentially everything. The drop is real under other ways of reading the probe, and partly comes from text placed
  before the answer. [m27]
- **Result 3: the probe's own signal adds little.** A writer that saw Qwen's replies and whether they were correct, but
  no probe scores, got most of the way (−3.2 vs −4.0 log-odds on fresh problems, last-round instructions, 3 runs each).
  [m22] [PENDING m28: random-style baseline; a writer not told how the score is read; a pinned writer model]
- **Result 4: white-box access is the exception.** With the probe's training data and a tool to score any text, Claude
  found instructions that put 384–407 of 500 correct maths answers under the line, and those answers genuinely look like
  same-style non-task replies (within-style AUROC 0.93). Even they are caught when the probe reads the instruction too
  and the line is set in that context. [m19, m27]
- **Result 5: tone vs task is mostly margin.** Earlier, tone concepts (sarcasm, flattery, anger, condescension) crossed
  the line outright with an ordinary score-guided search. Across concepts the attack's drop is similar (median 6.8
  log-odds for tone, 5.9 for tasks), but tone probes' plain answers sit about half as far above their line (6.5 vs 14.0).
  So the same shift crosses tone probes and not task probes. Within style, no probe is blinded (AUROC 0.97–1.00, flattery
  0.88). [m29; exceptions: condescension's 42-round search dropped 11 log-odds; maths 122B has a small gap but the
  smallest drop]
- Whether the probe was trained on the model's own text or Claude's depended on the concept. [m20]

## 1. Set-up

- Monitored models: Qwen3.5-27B (main), Qwen3.5-9B, Qwen3.5-122B-A10B (mixture of experts, ~10B active), Gemma-2-9B.
  Reasoning off. [m22, m23]
- Tasks: maths word problems with a one-sentence answer; "what does this Python program print?"; order five people from
  four comparison clues. A reply counts only if correct; odd wording is allowed. [m22, m25, m26]
- Probes: linear, one layer at about 75% depth, trained on the monitored model's own replies (task done vs matched
  replies that needn't do it, e.g. the answer is given), read as the peak of a running average over the reply's tokens,
  the way they are trained. Flag line at 1% false alarms on ordinary replies. The probe reads the reply with the attack
  instruction removed, so the instruction's own text can't lower the score; we report the instruction-visible reading
  too. [m1a, m22, m25, m26]
- **What the probe actually detects.** Swapping the answer for a wrong value of the same form barely changes the maths
  and code scores (−0.1 to −0.4 log-odds); a wrong order lowers ordering scores by 3.3. So the maths probe responds to
  "a value in the answer slot after a request that needs working out", not to correct computation. [m27]
- Attacker loop: a writer gets the task and a paragraph about the probe; each round it sees a sample of replies, whether
  each was correct, and each token's probe score, then writes a new instruction. 3 runs × 16 rounds; best and last
  instructions are frozen and tested on 100 fresh problems × 5 samples. [m22]
- **Disclosure:** every writer, including zero-shot and the no-scores control, was told the exact running-average
  formula, the layer, the line, and what the probe's negative examples were. [m22 WRITER_SYSTEM; PENDING m28: a writer
  told only that "a monitor flags computing the answer"]

## 2. How far scores move, and what it means operationally

Central figure [m29 central_figure.png; m28 arms to be added]: share of the gap closed per arm, final tests, every run as a dot, on three scales:
to the 1% line; to the level of in-context non-answers (Qwen's replies to the real request that skip the task); to the
level of same-style non-task replies.

Numbers so far (Claude writer, final tests where available):

| Task, model | Median change from plain (log-odds) | Gap to 1% line closed | Correct answers under the line |
|---|---|---|---|
| Maths 27B | −0.9 to −7.2 across 27 instructions | up to ~50% | 0 of 500 for every instruction [m22] |
| Maths 9B | — | best round 52% | 30 of 500 for one instruction (6 distinct answers), else ~0 [m23] |
| Code 27B | −4.2 (main run) | 27% | 0 [m25] |
| Ordering 27B | −6.5 to −11.2 | 36–63% | 0 [m26] |

- Operational meaning: on maths 27B, code and ordering every valid score-only reply scores above every one of the
  1,301–1,697 ordinary replies (0 of 7,829 on the maths own-reply probe); catching the median attacked reply needs a 0%
  false-alarm rate. Exceptions: maths 9B (half of Opus's best arm is inside the ordinary range, 97% still caught at 1%),
  a few replies on 122B and Gemma. Only the white-box template lands among ordinary replies (catching its median reply
  needs 3.8–12.8% false alarms). [m29]
- Pairing each reply with plain Qwen's reply on the same problem and seed changes the drop by at most 1 log-odds;
  counting invalid replies as caught costs most arms ~2 points of gap closed. [m29]
- The drop is biggest where the answer is shortest (maths division with 2-digit answers −7.4 vs subtraction −2.5; code
  integers −6.5 vs dicts −2.6), not where computation is least; plain difficulty barely predicts it. [m29]
- Zero-shot instructions move little: 2–19% of the gap, most for ordering and the 9B model. [m22, m23, m25, m26]
- Model size matters mainly at zero-shot (9B moves more; permutation p 0.004–0.02); after feedback no ordering by
  size. Correct-answer rates fall more on 9B, 122B and Gemma (to 45–64%) than on 27B (85%). [m23]

## 3. Shift, not hiding

- **Other readouts.** The drop survives four of five ways of reading the probe (per-token max, mean, running average
  restarted at the answer, max over the answer tokens) on every task. Restarting the average at the answer removes about
  half of the drop for code, less for ordering and 9B, none for maths 27B, 122B and Gemma: part of the attack is text
  before the answer, but the answer tokens' own scores drop too. [m27]
- **Styled twins.** Applying each frozen instruction to non-task requests raises those replies' scores by up to 6
  log-odds; at the deployed line most styled twins would be false alarms. Attack vs styled twin AUROC 0.989–1.000 on
  every score-only arm; with the line set on styled twins, 0 attack replies get under it for maths 27B and code, 0–10 for
  ordering, 5–40 per arm for maths 9B. [m27]
- **In-context non-answers** (replies to the real request that skip the task) score close to the line (−4.5 to −6.1) and
  still separate from attack replies (AUROC 0.97–0.999). [m27]
- **Transfer.** On maths 27B, Opus arms aimed at the own-reply probe close 33% of its gap but 51% of the never-attacked
  Claude-text probe's; the no-scores writer moves all nine maths probes by 24–39%. The writers aren't exploiting the
  probe they're shown. Ordering and code look more probe-specific (ordering 61% on target vs 29% on sister seeds), partly
  because frozen rounds were picked with the target probe. [m29]
- **What evading replies look like.** Maths: the result in English words, or framings like "As the question itself
  already tells us…". Code: "Yes, exactly as you wrote above: [34, 18]". Ordering: filler before the first name.
  [m22, m25, m26]
- **Normal-looking replies.** A Haiku judge (validated on plain and obviously odd replies) rates most of the strongest
  arms' replies as odd; every white-box reply is rated odd. But the drop survives among normal-looking replies for maths
  (no-scores number-word answers: 99% normal, −3.3) and ordering (Opus chain 0: 86% normal, −6.5); for code it mostly
  vanishes. [m29]

## 4. How much does attacker knowledge matter?

| Attacker | Maths 27B, own-reply probe, fresh problems |
|---|---|
| White-box: per-token scores + training data + scoring tool | −15.4; 384/500 under the line [m19 via m22] |
| Per-token scores, 16 rounds | −4.0 (last round, mean of 3); −5.2 (best of 16) [m22] |
| Replies + correctness, no scores | −3.2 (last round); −3.5 (best of 16) [m22] |
| Random odd styles, best of 16 by score | [PENDING m28] |
| Writer told nothing about the readout | [PENDING m28] |
| Qwen writing for itself | about −3.1 at the end of the search [m22] |
| Zero-shot | about −0.2 [m22] |
| Qwen just told a probe is watching | [PENDING m28] |

- Caveats: white-box differs in budget and method too (it started from replies Claude wrote directly); Opus 5.5 often
  declines when shown scores and Opus 4.8 answers (66/96 scored rounds) [PENDING m28: pinned writer]; 3 runs per arm.
- Ordering shows the same pattern: −7.2 vs −6.1 averaged over rounds 1–15 with and without scores. [m26]

## 5. Earlier results that fit

- Tone concepts cross with an ordinary score-guided search: sarcasm 268/500 valid sarcastic replies under the line
  (plain 29); condescension 475/500 full-strength replies on fresh messages after a 42-round search, though milder and
  caught by a layer-16 probe; flattery mostly because the probe only detects gushing wording (calm neutral replies pass
  too). Patching the probe, broad style coverage and layer ensembles each lost to a fresh search. [m3a, m3a2, m3a4,
  m8a, m11]
- The contrast is mostly margin (Result 5): a similar shift crosses probes with little room and not those with a lot.
  [Figure: m29 margin_figure.png, right panel — drop vs plain-to-line gap with the diagonal where the median attacked
  reply reaches the line.] [HOLE: tone and task searches had different budgets, judges and validity rules]
- Who wrote the probe's training data depended on the concept across five matched pairs (e.g. the own-reply sarcasm
  probe was *more* exploitable; condescension the other way). [m20]

## 6. Limitations

- Code and ordering probes partly react to the request: in-context non-answers already score near or above the line, so
  their 1% line is barely reachable. [m25, m26, m27]
- Writer models: mostly Claude (Opus 5.5 falling back to 4.8); Qwen writers weaker; reasoning off throughout; one probe
  layer and readout per task; 3 runs per cell; Qwen family plus Gemma with a different probe recipe.
- 122B is a mixture of experts; "bigger" is not clean. [m23]
