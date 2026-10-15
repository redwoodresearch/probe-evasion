# Mission m37_sweep_generalisation: does m36's capability trend hold on a third task, with more chains, and on a second monitored model?

## Why

m36 (read its RESULT.md, STATUS.md and module first) is the first trend for Nabil's concept-smuggling eval: across 11
prompt-writers ranked by Epoch's Capabilities Index (ECI), the smuggling score S after 16 feedback rounds rises 0.17 per
10 ECI on ordering (Spearman 0.95), but only 0.03 on maths (0.07 on valid replies only). Its main caveats are one
monitored model, two tasks and 3 chains per writer. This mission tests whether the trend generalises, in priority
order:

1. **Third task: program output** (m25's task, probe, lines, validity rules incl. the blind reader) on Qwen3.5-27B, all
   11 m36 writers, 3 chains, B = 0, 1, 2, 4, 8, 16, the same frozen-test design as m36.
2. **Firm up maths:** 3 more chains per writer on maths (so 6 total), same design, so the maths slope's interval
   tightens. Pre-register how the 6 chains are pooled with m36's 3.
3. **Second monitored model: Qwen3.5-9B** on maths (m23's 9B probe, line and calibration pool), with 6 writers spanning
   ECI (e.g. Opus 5.5, GPT-6 Sol, Opus 4.8, Gemini 3.8 Flash, Qwen3.5-397B, Llama 4 Maverick), 3 chains, B = 0, 4, 16.

## Fixed parts

Reuse m36's module and step runner by import (pinned writers, ECI table, metric S and S_valid, twins, analysis code).
Don't copy code; if something must change in m36's module, ask the lead. Same metric definitions as m36, pre-registered
in STATUS before any run, with one critic check on the design.

## Report (RESULT.md, plain language)

Per task and per monitored model: slope and rank correlation of S and S_valid against ECI with m36's bootstrap; effort
curves; a combined figure with all tasks (one panel per task, same axes as m36's `S_vs_eci_B16.png`); whether the
writer ranking is stable across tasks (rank correlation of writer means between tasks); refusals; cost per point.

## Rules

- Modal names `probe-evasion_nabil-baugher_m37-<label>_kill-2026-11-30`; use your own Modal volume for state
  (`probe-evasion_nabil-baugher_m37-state_kill-2026-11-30`; the shared volume is near its inode limit). Stop only your
  own apps, by id. Keep scaledown windows short so GPUs don't idle through writer/judge phases.
- Budget: GPU about $450, Claude API about $900, OpenRouter about $50. Log spend in STATUS; tell the lead
  (SendMessage to "main") before exceeding.
- Keys via `load_dotenv('.env')`, never printed. No core-module edits; code in
  `src/probe_evasion/missions/m37_sweep_generalisation.py`. Write only in `missions/m37_sweep_generalisation/`, that
  module and `data/concepts/m37_*`. No repo-changing git. Long runs detached in Modal.
- Critic check-ins: design; after the first task's results; before the verdict. Save everything (every writer
  prompt/response, every reply with per-token scores).
