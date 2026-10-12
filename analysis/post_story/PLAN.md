# Post-story loop: plan and status

Started 2026-10-11 while Nabil is away until tomorrow. His ask: "propose more experiments to get a coherent story;
collaborate with a sub-agent: draft what the post might look like, see where the holes are, run experiments, iterate."

## Loop

1. Draft the post (`draft.md`), with every claim tied to a mission file and every planned figure listed.
2. A fresh reviewer agent reads it cold and lists holes: claims a skeptic would reject, missing controls, confounds.
3. Launch experiments that fill the most important holes (agents, briefs under `missions/m27+`).
4. Fold the results into the draft, then go back to step 2. Stop when the reviewer finds no hole worth an experiment,
   or when the remaining holes need Nabil's input. Log each iteration below.

## Working story (v0)

A ladder of attacker affordance, and how far a prompt moves a probe's score while the task still gets done:
white-box (m19) > per-word scores (m22/m23/m25/m26) > replies but no scores (control) > model writing for itself >
zero-shot > model told a probe is watching. Central figure: share of the plain-to-line gap closed per rung, per task.
Secondary: within-style discrimination survives (it's a distribution shift); size and writer effects.

## Known holes before the first review

- m22 final test missing. Agent sent back on 2026-10-11 to finish.
- No-scores control only on the maths probe trained on Qwen's own answers (not the Claude-text probe), and none on code.
- New tasks' probes react to the request: even a non-answer scores above the line.
- Readout exploit: filler before the answer drags the running average down. Is the drop real under other readouts?
- Is it hiding or a style shift? Need styled non-task twins for m22/m23/m25/m26.
- Bottom rungs not run for these tasks: Qwen told about the probe, evading in context with no writer.
- Reasoning on: untested.

## Iterations

(logged below as they happen)

### Iteration 1 (2026-10-11)
- Wrote draft v0 (`draft.md`). Sent it to a cold research-reviewer for holes (running).
- Resumed the m22 agent to run its final test and RESULT (it had stopped without launching the test).
- Launched m27 (readout robustness on saved per-token scores + within-style twins) without waiting for the review,
  since both holes are certain.
- Could not stop our idle Oct 8 Modal app (`ap-WdL8eqGOKfWPVzawSaSONU`, 0 tasks): the permission classifier blocked
  it. It costs nothing while idle; flagged for Nabil.

### Iteration 1 results and iteration 2 (2026-10-11)
- Cold review of v0 (research-reviewer): the ladder framing mixes axes; all attacked answers stay above the calibration
  pool's maximum; every writer was told the readout formula; the drop transfers to never-attacked probes; random-style
  baseline missing. Suggested reframing: "prompting shifts the distribution; the probe isn't blinded".
- m22 finished: 27 searched instructions, all 0/500 under the line on fresh problems; −0.9 to −7.2 log-odds; m19
  references 384/407 under. No-scores last round −3.2 vs with scores −4.0.
- m27 finished: drops survive other readouts; part is pre-answer text (half for code, none for maths 27B); styled twins
  move UP; within-style AUROC 0.989–1.000; m19 white-box answers do look like styled twins (AUROC 0.93) but are caught
  with the instruction visible; maths/code probes don't check correctness (answer-swap barely moves scores).
- Launched m28 (random-style baseline, no-readout-spec writer, told-probe bottom rung, pinned-writer scores vs no-scores)
  and m29 (false-alarm sweep, transfer to sister probes, validity/selection effects, naturalness, central figure).
- Rewrote the draft as v1 with the reframed story; overnight report updated with m22's final numbers (republished).

### Iteration 3 (2026-10-11)
- m29 critic and the v2 cold review both weakened Result 5: raw log-odds aren't comparable across probes; margin doesn't
  order crossing cleanly (Gemma, condescension m11, 122B break it); naturalness is mixed (drop shrinks to ~0 in several
  arms); flattery invalid for the argument; m11's equal-method pair moved condescension ~11 vs maths ~3.5–3.9.
- Better story from the v2 review ("pool tail"): attacks move concept replies a similar share of plain-to-ordinary on
  sarcasm and tasks; what differs is how far the 1% line sits above typical ordinary replies (sarcasm 11–12 log-odds,
  tasks 2–4). Crossing a fixed line is mostly a property of the calibration pool; the shift is the better headline.
  Lead with the positive: a cheap, mostly probe-blind black-box search moves activations a quarter to a half of the way
  and transfers to probes it never saw; the deflations are the precise statement of what isn't happening.
- Draft v2.1: narrowed claims per the m29 critic.
- Launched m30: equal-effort score-only search on sarcasm and condescension with the m22 runner and the pinned writer.
- Amended m28: styled twins for the random styles (is the compression generic to any odd style?).
- Amended m27: Part C (probe reading just before the answer token: does the internal state change before the answer is
  written?) and Part D (task lines recalibrated on a broad pool of ordinary replies).
- Amended m29: pool-tail decomposition and strip figure; flattery invalid, anger weak; extra concepts (m9 jokey, m3b
  criticism, chemistry, m11 strict maths); second naturalness judge (Sonnet).
- Pending v3 rewrite once results land: the review's section-2 edit list (TL;DR wording, white-box rule hints, §4 table
  on one measure, transfer scope, 9B in TL;DR, number-word naturalness, "no detectable difference" for scores, tone
  fragilities, ordering margin, minor numbers, cut the m20 bullet).

### Iteration 3 continued (2026-10-11 evening)
- Nabil: continue the write/critique/experiment loop autonomously; back Monday 10-12. Budget context: ~$50k/month
  guideline, ~$10k spent last week; today's ~$130 GPU + ~$200–300 Claude is "very reasonable".
- m27 Parts C/D: the drop is present at the token before the answer (larger than on the answer); task lines would flag
  72–100% of broad chat traffic, but re-set on it, score-only attacks still almost never cross.
- m29 final: naturalness claim withdrawn (evading replies are visibly odd); "similar shift" dropped; robust statement is
  where the line sits / pool tail (pool_tail_figure.png).
- Draft v2.2 written.
- Launched m31 (5+5 more pinned-writer chains per arm on maths and ordering → 8 per arm with m28) and m32 (reasoning-on
  pilot on maths: where the signal lives, can prompting move it).
- Still running: m28, m30.
- Next: when m28/m30 land, write v3 (apply the v2 review's edit list), send to a fresh reviewer, iterate. Before Monday:
  a short status page for Nabil with the current story, figures and what's still running.
