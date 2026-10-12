# Mission m33_self_writer_rungs: the missing rungs of the progression — the model writing for itself without probe access, and the model told how it is read

## Why

The post's emerging headline (analysis/post_story/draft.md, PLAN.md iteration 5): how far prompting moves a task probe
is limited by what the attacker knows about the probe. Nabil's intended progression goes from an outside writer with
probe feedback, to fewer iterations, to no probe signal, to the same model writing for itself, to a model evading in
context. Two rungs are missing:

1. **Same model writing for itself, without probe scores** (and without the readout spec). m22 ran Qwen3.5-27B writing
   for itself *with* scores (−3.8 maths / −7.1 ordering, best arms). Does it move its own probe without probe access?
2. **The monitored model told how it is read.** m28 told Qwen only that "a monitor is watching" (≈0). Give Qwen the
   writers' full probe paragraph (layer, running average, the line's negative class) as an in-context instruction and
   let it answer directly (no writer). With reasoning off this is expected ≈0, but it is the clean "evading in context"
   rung and pairs with m32 (reasoning on).

## What to run (Qwen3.5-27B monitored, reasoning off; maths m22 and ordering m26 Task/Monitor)

- **Qwen-self, no scores, readout spec given:** m28's `NoScoresChain` with m22's q27 OpenRouter writer (the same
  writer m22 used for "Qwen writing for itself"), 3 chains × 16 rounds per task, plus a frozen test (best and last
  round, 100 test problems × 5 seeds).
- **Qwen-self, no scores, no readout spec:** m28's `NOSPEC_SYSTEM` with the q27 writer, same sizes.
- **Told how it is read (no writer):** 2–3 fixed instructions built from the writers' probe paragraph (exact text
  saved), frozen test only, maths, ordering and code (m25).
- Same validity, refusal check (Haiku) on every writer output, leak checks, eval/test sets and seeds as m22/m26/m28.

## Measures and reading

Per-round dot plots and medians next to the m22 Qwen-self-with-scores arms and m28's pinned Opus arms; final-test
median change from plain (best and last), gap closed, valid share, counts under the line. Write in STATUS before
running: if Qwen-self without scores still reaches about −2 to −4 on maths, the model can move its own probe without
probe access; if ≈0, all movement needs an outside probe oracle.

## Rules

- Modal names `probe-evasion_nabil-baugher_m33-<label>_kill-2026-11-30`; stop only your own apps by id. GPU cap about
  $80; Claude API cap about $120; OpenRouter about $15. Batch scoring per round.
- Keys via `load_dotenv('.env')`, never printed. No core-module edits; code in
  `src/probe_evasion/missions/m33_self_writer_rungs.py`; import from m22/m26/m25/m28, don't edit them. Ask the lead
  (SendMessage to "main") for runner changes.
- Write only in `missions/m33_self_writer_rungs/`, that module, `data/concepts/m33_*`. No repo-changing git. Long runs
  detached in Modal.
- Pre-register the final test. Critic check-ins: design; before the verdict. Save everything.
