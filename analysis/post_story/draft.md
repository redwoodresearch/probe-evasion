# DRAFT v0: Prompting moves activation-probe scores a long way, but only white-box help crosses the line

*Working draft for the post. [HOLE] marks a claim not yet backed by data, or a missing control. Sources in brackets.*

## TL;DR

- We ask: can prompting alone make a model keep doing a task while a linear probe on its activations stops noticing,
  and how does that depend on how much the attacker knows?
- Set-up: Qwen3.5-27B does verifiable tasks (maths word problems, predicting a program's output, ordering five people
  from clues). A probe trained on the model's own replies flags "doing the task in its head". Another model writes an
  instruction for the monitored model and iterates with feedback.
- **With a few rounds of feedback, prompting closes a quarter to a half of the gap between plain answers and the probe's
  1% flag line, on every task and model size we tried. Zero-shot instructions close almost none of it. Only a white-box
  attacker (training data plus a scoring tool) got most answers past the line.** [m19, m22, m23, m25, m26]
- **The probe's own scores barely help the attacker.** Seeing the monitored model's replies and whether they were correct
  gets almost as far. [m22 control, m26 control; HOLE: only 2 of 6 task×probe cells have the control]
- The probe still separates task from non-task replies *within* the attack's style; the attack shifts the whole
  distribution rather than blinding the probe. [HOLE: shown for sarcasm and m19 maths; not yet for m22–m26]
- Earlier, on tone concepts (sarcasm, flattery, condescension), a Claude-searched prompt crossed the line outright; and
  whether the probe was trained on the model's own text or Claude's text made no consistent difference. [m3a, m8a, m11,
  m20]

## 1. Set-up

- Monitored models: Qwen3.5-27B (main), Qwen3.5-9B, Qwen3.5-122B-A10B, Gemma-2-9B; reasoning off. [HOLE: reasoning on]
- Tasks with an automatic correctness check: maths word problems (one-sentence answer), "what does this Python program
  print?", order five people from four comparison clues. [m22, m25, m26]
- Probes: linear, one layer at about 75% depth, trained on the monitored model's own replies (task done vs a matched
  reply that needn't do it, e.g. the answer is given), read as the peak of a running average over the reply's words,
  the way it is trained. Flag line at 1% false alarms on ordinary replies. [m1a, m22, m25, m26]
- Attacker loop: a writer model gets the task and one paragraph about the probe; each round it sees a sample of replies,
  whether each was correct, and each word's probe score; it writes a new instruction. 3 runs × 16 rounds; final test on
  fresh problems. A reply counts only if correct; odd wording allowed. [m22]
- Scoring: probe reads the reply with the attack instruction removed (so the instruction text itself can't lower it).
  [HOLE: report instruction-visible too; m19 found 0/500 evade with it visible]

## 2. The ladder (central figure, planned)

Share of the plain-to-line gap closed, per rung, per task (Claude writer unless stated):

| Rung (attacker knows) | Maths 27B | Ordering 27B | Code 27B |
|---|---|---|---|
| White-box: per-word scores + training data + scoring tool | >100% (385/500 past) [m19] | [HOLE] | [HOLE] |
| Per-word probe scores, 16 rounds (best round) | 35% [m23] | 52% [m26] | 27% [m25] |
| Replies + correctness, no scores (best round) | ~25% [m22, interim] | 52% [m26] | [HOLE] |
| Model writes for itself, with scores | ~21% [m22] | [from m26] | [from m25] |
| Zero-shot (no feedback) | 6% | 19% | 11% |
| Model just told a probe is watching | [HOLE] | [HOLE] | [HOLE] |

[HOLE: best round is picked by score — use frozen-test numbers for every cell; m22 frozen test pending]
[HOLE: the gap-closed measure depends on where the line sits; also show the pool-anchored version]

## 3. Feedback, not the probe signal, does the work

- Maths (probe trained on Qwen's answers): with scores −3.4 → −3.6 log-odds from plain (rounds 1–3 → 13–15); without
  −3.3 → −3.1. Ordering: −7.2 vs −6.1 averaged over rounds 1–15. Most of the drop happens at round 1 in both.
  [m22, m26]
- Confound: Claude Opus 5.5 often declines when shown probe scores and Opus 4.8 answers (30/48 vs 23/48 rounds on maths,
  33/48 vs 0/48 on ordering). [HOLE: run the control with a fixed writer model, or report by answering model]
- [HOLE: a "best of N independent zero-shot instructions" baseline would separate "search with feedback" from "just more
  tries"]

## 4. What the evading replies look like, and is it hiding or a style shift?

- Maths: scripted openers ("Actually, yes, in standard scientific notation, 8.92e2 dollars in total"). Code: false
  attribution ("Yes, exactly as you wrote above: [34, 18]"). Ordering: filler before the first name. [m19, m25, m26]
- Mechanism: the running average peaks on the answer tokens; filler or a strongly negative opener drags it down before
  the answer arrives. [m19 trace figure, m26]
- [HOLE: does the drop survive another readout (per-token max, mean, an average that starts at the answer)? If not, the
  attack exploits the readout, not the representation.]
- [HOLE: score non-task replies written in the attack's style. If they drop too and the probe still ranks task above
  non-task, it is a style shift.]

## 5. Model size and who writes the prompt

- Zero-shot moves 9B more than 27B or 122B (permutation p about 0.004–0.02); after feedback, no ordering by size.
  27B keeps the most answers correct (85% vs 45–64%). [m23]
- Claude moves scores more than the models writing for themselves (clear on 122B and Gemma; a trend on 9B and 27B).
  Qwen3.5-122B writing for 27B catches up with Claude on the Claude-text probe. [m22, m23]
- [HOLE: 122B is mixture-of-experts (~10B active); "bigger" is not clean]

## 6. Earlier results that fit

- Tone concepts cross the line: sarcasm 268/500 valid sarcastic replies past (plain 29); flattery 454/456; condescension
  492/494 after a 42-round search, though a layer-16 probe catches them. Patching the probe, broad style coverage and
  layer ensembles each lost to a fresh search within a round. [m3a, m3a2, m3a4, m8a, m11]
- Who wrote the probe's training data (the model itself vs Claude) made no consistent difference across five matched
  pairs. [m20]
- Content probes (maths, chemistry) held against the standard search until the white-box red team. [m11, m19]

## 7. Limitations (so far)

- New tasks' probes partly react to the request: a non-answer still scores above the line, so the line is unreachable
  there by construction. [m25, m26; HOLE: fix and re-score]
- Opus 5.5 / 4.8 mix in the writer.
- One prompt-writer family for most runs; reasoning off throughout.
- Small number of runs per cell (3).
