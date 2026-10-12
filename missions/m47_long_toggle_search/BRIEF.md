# Mission m47_long_toggle_search: does continued effort keep making progress against the probe?

## Why

Nabil's question: in the matched toggle design, does a much longer search keep closing the gap, or does it plateau?
m44 extended m36's attack-vs-plain ordering searches to 48 rounds:
- Claude Opus 5.5 plateaued at about 0.75 of the gap;
- GPT-6 Sol and Gemini 3.8 Flash were still rising slowly at round 48;
- nobody crossed the 1% line (closest: 0.42 log-odds above it).
This mission runs very long searches in the toggle design, on both tasks.

## Design

- **Toggle setup:** m45's (read its BRIEF, STATUS and module; import its code), on Qwen3.5-27B, both tasks, read with
  the instruction in view, scored S_toggle. Wait until m45's smoke test and first full round pass.
- **Writers:** GPT-6 Sol (cheap, still improving in m44) and Claude Opus 5.5 (the strongest); 2 chains each per task.
  Two chains rather than one, because single chains are noisy (m44: one chain at 0 at B = 40).
- **Length:** 100 rounds.
  - Manage context as m44 did: older feedback loses its per-token traces; keep summaries and the round table. Tell
    the writer up front that the search is long.
  - Every 25 rounds, give the writer one "fresh look" turn: its current best template, its round table and the best
    replies, but no old traces. Pre-register this, apply it to every chain, and say so.
- **Freeze points:** B = 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, unselected, tested on 100 untouched problems × 3
  seeds (5 seeds at B = 0, 50, 100).
- **Also report:**
  - the per-round eval curve (80 practice replies per round);
  - the lowest valid in-view score so far;
  - the first round, if any, where a valid test reply crosses the 1% line;
  - writer dollars against S;
  - whether progress is still significant over rounds 50–100. Pre-register the plateau rule, as m44 did.

## Budget

Claude about $700 (Opus 5.5's writer calls are about $0.8–1 each at long context), GPU about $500, OpenRouter
about $40.

## Rules

- Modal names `probe-evasion_nabil-baugher_m47-<label>_kill-2026-11-30`; use your own Modal volume for state
  (`probe-evasion_nabil-baugher_m47-state_kill-2026-11-30`; the shared volume is near its inode limit). Stop only your
  own apps, by id. Keep scaledown windows short so GPUs do not idle through writer/judge phases.
- Keys via `load_dotenv('.env')`, never printed. No core-module edits and no edits to other missions' modules (ask the
  lead, SendMessage to "main"); code in `src/probe_evasion/missions/m47_long_toggle_search.py`, importing from m36 and earlier modules
  rather than copying. Write only in `missions/m47_long_toggle_search/`, that module and `data/concepts/m47_*`. No repo-changing git. Long
  runs detached in Modal. Log spend in STATUS; tell the lead before exceeding the budget.
- Pre-register the design and metric in STATUS before any run; one critic check on the design, one before the verdict.
  Save everything (every prompt, every reply with per-token scores, every judge call).
- RESULT.md in plain language: numbers, figures, caveats, cost.
- Local downloads go only into scratchpad/m47/ (several agents share the scratchpad). Spend guards count judge calls
  (upper bound), never the volume's usage log (it undercounts).
