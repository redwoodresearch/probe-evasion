# Mission m44_writer_effort: is the trend about writer capability or writer compute, and where does it top out?

## Why

m36's x-axis is ECI, but stronger writers also think longer and the reasoning settings differ by lab. And most writers
were still improving at round 16 on ordering. This mission separates capability from test-time compute and finds the
ceiling, on ordering (the task with the clear trend) using m36's runner, metric and frozen-test design.

## Design

1. **Effort ablation within a model.** Claude Opus 5.5 at effort low / medium / high / max, Claude Sonnet 5.5 at low /
   high, and GPT-6 Sol at reasoning effort low / high (via OpenRouter, pinned provider). 3 chains each, rounds 0–16,
   freeze at B = 0, 4, 16. Record writer tokens and dollars per round. Plot S against ECI with effort as a second
   marker, and S against writer tokens / dollars across all m36 writers plus these arms.
2. **Longer searches.** Opus 5.5, GPT-6 Sol, Gemini 3.8 Flash and Qwen3.5-397B on ordering for 48 rounds (3 chains;
   continue m36's chains from their round-16 state if the runner allows, otherwise new chains), freeze at B = 16, 32,
   48. Does the gap between writers widen or close, and does anyone cross the 1% line?
3. Report refusals per effort level (Opus 5.5's API refusals in m36 were 3 of 57 on ordering).

## Budget

GPU about $400, Claude about $500, OpenRouter about $40.

## Rules

- Modal names `probe-evasion_nabil-baugher_m44-<label>_kill-2026-11-30`; use your own Modal volume for state
  (`probe-evasion_nabil-baugher_m44-state_kill-2026-11-30`; the shared volume is near its inode limit). Stop only your
  own apps, by id. Keep scaledown windows short so GPUs do not idle through writer/judge phases.
- Keys via `load_dotenv('.env')`, never printed. No core-module edits and no edits to other missions' modules (ask the
  lead, SendMessage to "main"); code in `src/probe_evasion/missions/m44_writer_effort.py`, importing from m36 and earlier modules
  rather than copying. Write only in `missions/m44_writer_effort/`, that module and `data/concepts/m44_*`. No repo-changing git. Long
  runs detached in Modal. Log spend in STATUS; tell the lead before exceeding the budget.
- Pre-register the design and metric in STATUS before any run; one critic check on the design, one before the verdict.
  Save everything (every prompt, every reply with per-token scores, every judge call).
- RESULT.md in plain language: numbers, figures, caveats, cost.
