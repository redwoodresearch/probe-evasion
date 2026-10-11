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
