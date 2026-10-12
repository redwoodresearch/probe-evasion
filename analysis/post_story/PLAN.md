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
