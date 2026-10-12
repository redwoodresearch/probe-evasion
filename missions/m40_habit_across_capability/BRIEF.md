# Mission m40_habit_across_capability: does the share of the drop that survives as a habit change with writer capability?

## Why

m35 (read its RESULT, STATUS and module) fine-tuned Qwen3.5-27B into an evading style with no instruction and found the
model's own probe reading keeps only about 25% (maths) / 56% (ordering) of the instruction-removed re-read drop. m36's
trend is measured on the re-read. If the habit-kept share falls with writer capability (e.g. stronger writers' styles
rely more on things a habit can't reproduce), the slope may not survive; if flat, it does. This decides whether m36's
trend is evidence about a model that smuggles on its own.

## Design

- Pick 5 writers spread over ECI (e.g. Opus 5.5, GPT-6 Sol, Gemini 3.8 Flash, Qwen3.5-397B, Haiku 4.5 or Qwen3.5-9B),
  both tasks. For each, the round-16 frozen instruction of each of m36's 3 chains (no selection by probe), or of the
  2 chains with the highest valid share if compute is tight (pre-register which).
- m35's recipe exactly (context distillation with LoRA, same steps, seeds, checkpoint choice rule chosen without
  the probe, same plain-control and style-control arms). Read the fine-tuned model's own activations with no
  instruction, on the same 100 test problems × 5 seeds.
- Metrics: habit S (m36's S, read on the fine-tuned model with no instruction), the habit-kept share (habit drop /
  re-read drop) per instruction, and the trend of both against ECI with m36's bootstrap. Validity as in m36.
- Register every checkpoint in CHECKPOINTS.md's format (send the lines to the lead to add; don't edit it yourself) and
  in your own checkpoints.md.

## Budget

GPU about $900 (fine-tuning on H200s plus serving with unmerged adapters), Claude about $300.

## Rules

- Modal names `probe-evasion_nabil-baugher_m40-<label>_kill-2026-11-30`; use your own Modal volume for state
  (`probe-evasion_nabil-baugher_m40-state_kill-2026-11-30`; the shared volume is near its inode limit). Stop only your
  own apps, by id. Keep scaledown windows short so GPUs do not idle through writer/judge phases.
- Keys via `load_dotenv('.env')`, never printed. No core-module edits and no edits to other missions' modules (ask the
  lead, SendMessage to "main"); code in `src/probe_evasion/missions/m40_habit_across_capability.py`, importing from m36 and earlier modules
  rather than copying. Write only in `missions/m40_habit_across_capability/`, that module and `data/concepts/m40_*`. No repo-changing git. Long
  runs detached in Modal. Log spend in STATUS; tell the lead before exceeding the budget.
- Pre-register the design and metric in STATUS before any run; one critic check on the design, one before the verdict.
  Save everything (every prompt, every reply with per-token scores, every judge call).
- RESULT.md in plain language: numbers, figures, caveats, cost.
