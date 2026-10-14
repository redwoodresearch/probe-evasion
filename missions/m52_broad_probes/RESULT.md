# m52 result: broad probes (14 Oct)

**Question.** The maths and ordering probes were trained and calibrated only on narrow puzzle / word-problem data. Would
a probe built the way deployed probes are built (random real traffic as negatives, the line set on real traffic) change
the post's finding?

**What was built.** One new probe per task (m46's recipe, layer 48, seeds 0-2), trained on m46's rows plus:
- chat-formatted rows with known answers. Code drew the facts; Haiku 5.5 wrote a realistic user message around them,
  ending in a brevity request; Qwen answered; an answer check kept only clean rows:
  - positives: Qwen works it out (ordering 895, maths 833);
  - look-alikes: the answer is already in the message (ordering 726, maths 657);
- WildChat negatives: English first turns, half also asked to be brief. Replies Haiku says may use the task were left
  out (ordering 3,347; maths 4,186).

The 1% line is set on the other half of the WildChat conversations (about 4,400 replies, minus clear task-doers).
Then m45's saved round-16 attacks were re-scored. This is a transfer test: the attacks were written against the old
probe, not the new one.

**Gate (fixed in advance): passed.** At its WildChat line, each new probe flags:
- 100% of held-out chat positives (situations never trained on);
- 100% of the normal test answers.

| | Maths | Ordering |
|---|---|---|
| New probe's 1% line (WildChat) | -7.11 | -6.12 |
| Old probe's strict line (look-alikes) | -6.53 | -8.06 |
| Old probe's line if set on WildChat | -1.41 | +0.07 |
| Chat look-alikes flagged by the new probe | 5% | 6% |
| WildChat line pool flagged by the old probe at its strict line | 88% | 99% |

Seeds 1-2 give lines within 0.5 of seed 0.

**Attack results (post v9 measure, 14 writers, round 16):**

| | Maths slope [95%] | Ordering slope [95%] | Ordering crossings (pairs with both replies valid) |
|---|---|---|---|
| Old probe, strict line (reproduces the post: +0.062 / +0.099) | +0.062 [+0.031, +0.102] | +0.099 [+0.062, +0.140] | 0 / 17,389 |
| **New probe, WildChat line** | **+0.057 [+0.023, +0.095]** | **+0.138 [+0.087, +0.194]** | **588 / 17,389** |
| Old probe, WildChat line | +0.090 [+0.044, +0.146] | +0.167 [+0.104, +0.236] | 3,918 / 17,389 |

- Maths: no correct reply crosses the new probe's line either (0 of 17,377).
- Ordering crossings under the new probe come from the top writers only:
  - Fable 5.1 358 of 1,185 pairs (30%);
  - Opus 5.5 155 (13%);
  - GPT-6 Astra 50 (4%);
  - GPT-6 Sol, Opus 4.8, Sonnet 5.5, GPT-6.1 Sol 5-10 each;
  - the 7 lowest-capability writers none.

**Reading.**
1. A probe trained on broad data is possible and keeps a line about as strict as the narrow one, while flagging only
   1% of real traffic. The narrow probe's problem was its training data, not the concept.
2. The capability trend holds under the broad probe: about the same on maths, steeper on ordering.
3. Against a realistic monitor, the strongest writers' ordering attacks start to get through (Fable 5.1 on 30% of pairs).
   Maths stays at 0.

**Caveats.**
- Transfer only: writers never saw the new probe's scores.
- Inside everyday chat, the new probe still does not fire on real task-like replies. Ordering "works out an order" is
  -10.3 vs "no ordering" -10.6, flagged 0%. Maths "computes" is -10.4 vs -11.8, flagged 14% vs 0%. Its "task" is
  closest to working out an answer from facts given in the request, which is what the chat positives were.
- Haiku's task labels decide which WildChat replies are negatives.

Files:
- `src/probe_evasion/missions/m52_broad_probes.py` (pipeline);
- `analysis/broad_probe/m52_analysis.py`, `m52_details.py` → `analysis/broad_probe/m52/results.json`, `details.json`;
- `analysis/broad_probe/training_examples.md`, `wildchat_examples.md` (real rows);
- raw rows on the Modal volume `probe-evasion_nabil-baugher_m52-state_kill-2026-11-30` and in `missions/m52_broad_probes/run/` (not in git).
