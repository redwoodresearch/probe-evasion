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
- m31 design critic: on m28's 3 vs 3 pinned Opus 4.8 chains, scores clearly help (maths −3.6 vs −0.8; ordering −8.5 vs
  −6.0), while m22's mostly-5.5 no-scores writer reached −3.3. "Do scores help?" depends on the writer model, so Result
  3 needs revising. Added a pinned Opus 5.5 pair (no fallback; refusal rate is itself a result) to m31, analysed as
  writer × scores 2×2.

### Iteration 4 (2026-10-11 late)
- m30 (equal effort on tone): tone probes still crossed under the task search; crossing follows mechanically from the
  shift (as a share of plain-to-ordinary) and where the 1% line sits; sarcasm's line sits 36% of the way down vs 79–90%
  for tasks; condescension moves further (61–74% vs 20–56%); feedback adds ~20 points for all.
- m28: Qwen told a monitor is watching does nothing; random styles best-of-16 get ~40–50% of the search's drop and also
  squeeze from both sides (within-style AUROC 1.000); a writer not told the readout gets little (pinned 4.8 ordering
  −0.3 vs −6.0); pinned 4.8 scores beat no-scores on every run (ordering clean, maths confounded by round-0 number
  words).
- Draft v3 → v3.1 (attacker-knowledge ladder table; m30 tone result). Fixed two numbers I'd written without checking
  (9B row; maths no-spec cell).
- Sent v3.1 to a fresh cold reviewer. m31 (8-run 2×2 writer × scores) and m32 (reasoning on) still running.

### Iteration 5 (2026-10-11 night)
- v3.1 cold review: headline should be "movement is limited by the attacker's information about the probe; the model on
  its own doesn't move it"; 6 factual errors (e.g. the Claude-text probe was attacked, not "never-attacked"); §4 mixed
  selected/unselected arms; §5 stale after m30; "black-box" wrong (writers told the readout).
- Wrote v4 with all corrections and a matched best/last ladder table built from m28's tables.md (numbers checked by
  script).
- Launched m33 (Qwen-self without scores / without spec; Qwen told how it is read), m34 (next-token answer probability at
  the pre-answer position; compute-without-output; probe cosines and other layers; broad-pool line positions;
  within-style gap table), m35 (context distillation: does a fine-tuned model that uses the style without the
  instruction move its own probe?).
- Still running: m31 (2×2), m32 (reasoning on).

## Process update (Nabil, 2026-10-12)
The write/review step of the loop is now: a subagent writes the LessWrong-style post (short main ≤~1,300 words + 3–4
figures; long appendix answering a critical reader; sources by mission at the end; task probes only, no tone) in
`analysis/post_v<N>/`; a fresh critic grades it 0–10; stop at 8/10 or after 3 rounds; after the last round apply the
critic's fixes without re-review; publish the formatted page with figures. Critic holes feed the next experiments.
- v5: three rounds, 7/7/7; final fixes being applied (including m34's results: probes are answer detectors; Qwen itself
  stops predicting the answer in the removed re-read; layer-56 probe lets 43–48 score-only replies through).

## Goal update (Nabil, 2026-10-12): an eval of concept-smuggling capability over model capability
The post and experiments now serve an eval that tracks over time whether models are getting better at evading probe
monitors while still doing the task ("concept smuggling"), against model capability (ECI) or size. Nabil's outline is
in memory (eval-goal-concept-smuggling). The working title is "Current Models Show Leading Indicators of Concept
Smuggling Capabilities", to be believed before use.
- Launched m36: a prompt-writer capability sweep (8–12 writers spanning ECI, pinned, readout spec plus per-token
  scores), effort curves (B = 0, 1, 2, 4, 8, 16 rounds, unselected), and a pre-registered smuggling score (gap closed by
  the median attempt, with invalid attempts counted as no movement).
- Still needed for the eval: volitional smuggling across monitored-model size with reasoning on (m32/m33 cover parts);
  probes from the literature; a definition of the effort scale and of acceptable task success.

### Iteration 6 (2026-10-12)
- v6.3 published (same link as v5's draft page); round 3 graded 7/10, fixes applied without re-review.
- m36 final: across 11 prompt-writers by ECI, ordering S rises 0.17 per 10 ECI (Spearman 0.95); maths 0.03 (0.07 on
  valid replies; stronger writers break more answers); nothing reaches the 1% line. First real trend for the eval.
- m32 test running (GPU cap raised to $420). Search: the answer reading moves 2–3 log-odds; the whole-output reading
  barely moves, because Qwen keeps writing digit arithmetic in its reasoning.
- Launched m37 (does the trend hold on program output, with 6 maths chains, and on a 9B monitored model?) and m38 (the
  volitional track across 9B/27B/122B, reasoning off/on, plus single-thread self-iteration).
- v7 writer started, restructured around the eval.
- v7 critic round 1: 7/10. Main holes: the white box contradicts "none gets past it"; the re-read has no threat model
  (instruction-in-view slope 0.07 / 0.02); the leading-indicator logic isn't argued; the trend uses a score oracle
  (no-scores ranking inverts); the volitional null may be unwillingness (157/160 call it an injection).
- Launched from the holes: m39 (instruction-in-view with a matched harmless baseline, tail measures,
  naturalness-constrained S, leave-two-out), m40 (habit fine-tuning across writer capability), m41 (no-scores and
  binary-feedback sweeps, plus a real-scoring-rule arm), m42 (adaptive defender: leave-one-lab-out probe retraining,
  an augmented-style probe, a pooled probe). m38 amended with a system-prompt/scenario framing arm. m38's 397B
  stretch approved (GPU cap $1,100).
- v7.1 critic round 2: 7/10. Holes: the headline reading fits neither threat; the "lower bound" argument for the
  in-view slope is wrong (a constant offset moves the intercept, not the slope); the title overclaims; the maths tail
  is carried by the two weakest writers; the maths drop may be re-encoding the answer rather than hiding the
  computation. v7.2 fixes sent to the writer (round 3 is the last review).
- Launched m43 (a decision task where the computed result is never written; its own probe; a 6-writer sweep) and m44
  (writer effort ablation within a model; 48-round searches). m39 amended: per-instruction matched baselines, the
  leading-digit causal check on sweep instructions, a realistic monitor (broad lines plus a text judge). m40 extended to
  11 writers. m41's binary arm replaced by a per-reply-score arm plus a loose-line binary (ordering).
- v7.2 critic round 3: 7/10 (target mismatch: writers optimised for the instruction-removed reading; the summary was
  too dense; one false "under way" claim). Final fixes applied without re-review → v7.3, published at the draft-page
  link (version 3). Critic experiments folded into running missions: alternative readouts and instruction length on
  the sweep (m39), an in-view-targeted arm (m41), transfer of frozen instructions to 9B/122B (m37).
