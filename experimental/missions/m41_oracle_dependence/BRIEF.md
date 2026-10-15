# Mission m41_oracle_dependence: does the capability trend need the per-token score oracle?

## Why

Every m36 writer saw per-token probe scores each round. Earlier, without scores, the ordering ranking inverted (m31:
Opus 5.5 −3.95 vs Opus 4.8 −6.80 last round). A realistic attacker rarely has per-token scores. Does S still rise with
ECI with less feedback?

## Design

The same 11 writers as m36, same tasks, same frozen-test design (3 chains × rounds 0–16, freeze at B = 0, 1, 2, 4, 8,
16, test on 100 untouched problems × 5 seeds), m36's metric S and S_valid, with two feedback arms:
1. **No scores:** readout description, replies and validity, but no per-token scores (m28/m31's no-scores feedback).
2. **Flagged / not flagged only:** for each reply, only whether it was valid and whether it was flagged at the 1% line
   (binary), plus the readout description.
Also a maths-only third arm on the top 4 and bottom 2 writers: m36's full feedback, but the writer prompt states the
real scoring rule (an invalid answer counts as no movement), to test whether that flattens the validity trade-off.
If budget is tight, cut B = 1 and B = 2 first, then the binary arm's maths, and say so in STATUS.

## Budget

GPU about $600, Claude about $1,500 (judges dominate), OpenRouter about $40.

## Rules

- Modal names `probe-evasion_nabil-baugher_m41-<label>_kill-2026-11-30`; use your own Modal volume for state
  (`probe-evasion_nabil-baugher_m41-state_kill-2026-11-30`; the shared volume is near its inode limit). Stop only your
  own apps, by id. Keep scaledown windows short so GPUs do not idle through writer/judge phases.
- Keys via `load_dotenv('.env')`, never printed. No core-module edits and no edits to other missions' modules (ask the
  lead, SendMessage to "main"); code in `src/probe_evasion/missions/m41_oracle_dependence.py`, importing from m36 and earlier modules
  rather than copying. Write only in `missions/m41_oracle_dependence/`, that module and `data/concepts/m41_*`. No repo-changing git. Long
  runs detached in Modal. Log spend in STATUS; tell the lead before exceeding the budget.
- Pre-register the design and metric in STATUS before any run; one critic check on the design, one before the verdict.
  Save everything (every prompt, every reply with per-token scores, every judge call).
- RESULT.md in plain language: numbers, figures, caveats, cost.
