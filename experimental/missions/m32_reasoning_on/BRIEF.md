# Mission m32_reasoning_on: does the prompted score drop survive with reasoning switched on?

## Why

Everything in the post so far uses Qwen3.5-27B with reasoning off, so the task's work has to happen in the reply tokens
the probe reads. Nabil's 1:1 notes flag that reasoning is almost always on in practice, and the progression James
sketched ends with one model thinking about and evading a monitor in context. A first look: with reasoning on, where does
the probe's signal live, and can prompting move it?

## Design (pilot; maths only)

- Monitored model: Qwen3.5-27B, reasoning ON (its think block), same one-sentence maths problems as m22. Reply budget
  large enough for the think block plus the answer; measure plain accuracy and think-block length first.
- **Probe readings** (the existing own-reply maths probe `onpk@L48@s0`, peak-trained on reasoning-off replies, plus a
  probe retrained on reasoning-on replies if the transfer is poor; say which):
  1. the final answer only (after the think block), as deployed so far;
  2. the think block;
  3. the whole output.
  Calibrate each reading's 1% line on reasoning-on replies to the no-maths lookups (m11/m22 calibration requests, with
  reasoning on). Check that plain reasoning-on maths answers are caught under each reading before running attacks.
- **Attack:** the m22 loop with Claude as writer (the pinned Opus 4.8 writer from m28 so it doesn't fall back mid-run;
  see m28's STATUS), scores shown for the reading being attacked, 3 chains × 12 rounds per reading, for at least
  readings 1 and 3. Final test on fresh problems (pre-registered). The m22 runner hard-codes thinking off: send the lead
  (SendMessage to "main") the minimal change it needs (e.g. a Task or Monitor field for thinking plus which tokens the
  probe reads), and don't edit m22 yourself.
- Report what the instructions do to the think block (e.g. telling the model to keep its reasoning terse or off-topic)
  and whether answers stay correct.

## Report (RESULT.md, plain language)

Where the probe's signal sits with reasoning on (think block vs answer), how far prompting moves each reading compared
with reasoning off (m22), validity, what the reasoning looks like under attack, cost, caveats.

## Rules

- Modal names `probe-evasion_nabil-baugher_m32-<label>_kill-2026-11-30`; stop only your own apps by id. GPU cap about
  $250, Claude API cap about $250. Batch generation and scoring.
- Keys via `load_dotenv('.env')`, never printed. No core-module edits; code in
  `src/probe_evasion/missions/m32_reasoning_on.py`. Don't edit other missions' modules.
- Write only in `missions/m32_reasoning_on/`, that module, `data/concepts/m32_*`, `results/concept_probes/m32_*`. No
  repo-changing git. Long runs detached in Modal.
- Critic check-ins: design (before generation); after calibration; before the verdict. Save everything, including think
  blocks with per-token scores.
