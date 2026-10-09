# Mission m17_long_context: do probes trained on short replies keep working after long context, and do own-reply probes degrade less than Claude-text probes?

Read `missions/BRIEF.md` first. Its rules and critic check-ins apply unless this brief says otherwise.

## Why

Google DeepMind's "Building Production-Ready Probes For Gemini" (arXiv 2601.11516) found that probes trained on short
inputs generalise badly to very long ones (about 900k tokens). Two examples:
- A probe trained on the average and read with a running peak (the same reading as ours, but mismatched to how it was
  trained) caught long-context attacks, but its false alarms went from about 1% to 13.6%.
- Their best probes still had a smaller long-context penalty.

They monitored user prompts only and never compared training-data sources. Our setting is different in two ways:
- we monitor the model's own reply;
- we have probes trained on another model's text (Claude-written, "off-policy") and on the model's own replies
  ("on-policy"), both trained for the reading they are deployed with.

**Questions:**
1. When long benign context precedes the reply, do our short-trained probes still flag the concept, and does the
   short-context 1% line still give about 1% false alarms?
2. Do own-reply probes degrade less than Claude-text probes?

## Probes (all already trained; no new training needed)

**Primary: maths, Qwen3.5-27B, layer 48, reasoning off** (m1a; see `missions/m1a_offpolicy_maths/RESULT.md` and
`src/probe_evasion/missions/m1a_peak_probe.py`):
- Claude-text probe, peak-trained (m1a `offpk`), seeds 0-2.
- Own-reply probe, peak-trained (m1a `onpk`), seeds 0-2.

Confirm the exact files and that both are trained for the running-peak reading.

**Secondary, if cheap: sarcasm, Qwen3.5-27B, layer 48:**
- the Claude-text probe `r0_peak@L48` (m3a2);
- the own-reply probe `on_peak@L48` (m3a3);
- seed 0 each.

## Test sets

Reuse each probe's existing short-context sets:
- **Maths:** Qwen's plain maths answers (concept present); lookups and recalled facts (hard negatives); the cleaned
  no-maths calibration replies that set the 1% line.
- **Sarcasm:** plain sarcastic replies; the clean sincere calibration replies.

Then put **long benign context** in front of the same conversation and score only the reply tokens.

**Context lengths:** 0 (baseline), ~1k, ~4k, ~16k and ~32k tokens. Go as long as the GPU allows; record the cap.

**Context types** (decide and write them in STATUS.md before scoring):
- **Unrelated document:** a long article or report pasted earlier in the conversation, followed by a neutral transition.
- **Long earlier conversation:** many unrelated earlier user/assistant turns.
- **Number-heavy document:** tables, prices and dates. This is a hard case for maths false alarms.
- Optionally **code**.

Write the filler with Claude, or use public-domain text. It must contain none of the test problems and no computations
whose answers appear in the test replies.

**Two ways to get replies at each length:**
- **(a) Same replies re-scored:** the identical short-context reply is placed after the long context. This isolates
  how the context shifts the probe's reading.
- **(b) Regenerated:** Qwen answers the same question after the long context, on-policy at that length. Check validity
  as usual (maths by `gepa.lenient_maths_present`; sarcasm by the two judges). Re-clean the calibration replies for
  traces of the concept.

## Metrics, at each length × context type × probe

1. **Fixed line:**
   - the false-alarm rate on no-concept replies at the short-context 1% line (does it stay near 1%?);
   - the catch rate on concept replies at that line;
   - for maths, the false-alarm rate on lookups and recalled facts.
2. **Re-set line:** AUROC of concept vs no-concept at that length, and the catch rate at a line re-set to 1% false
   alarms at that length. This separates "the line drifted" from "the probe can no longer tell them apart".
3. **Score shift:** the median score change from length 0, for each group. Check whether the whole distribution moves
   up or down.
4. **The headline comparison:** own-reply vs Claude-text probes on these curves, with seed-to-seed spread (3 seeds
   each) and bootstrap intervals.
   - The prediction to test: any small per-token advantage of own-reply probes may grow in absolute terms with length.
   - Also report the opposite if it happens.

## Figures

- Lines vs context length (log x axis): false alarms at the fixed short-context line, and catch rate.
- One panel per context type.
- Own-reply vs Claude-text probes in two colours, with the seed spread shaded.
- A small table of AUROC by length.

## Rules

- Modal names `probe-evasion_nabil-baugher_m17-<label>_kill-2026-11-30`; stop only your own apps, by app id.
  Estimate the GPU cost before the long-context runs (32k-token forward passes on a 27B model) and log it; keep it
  under about $150 unless it is clearly worth more.
- Keys via `load_dotenv('.env')`, never printed.
- No core-module edits; code goes in `src/probe_evasion/missions/m17_long_context.py`.
- Write only in `missions/m17_long_context/`, that module and `data/concepts/m17_*`.
- No git commands that change the repo.
- Use caffeinate for long local runs.
- **Critic check-ins:**
  - design, before scoring;
  - first numbers;
  - before the verdict.

  One critic must check that context filler cannot leak the concept, so that no stray computation lands in the
  calibration replies.
- Save every context, prompt, reply, judge output and score.

## Report

Plain language:
- Does the fixed short-context line still give about 1% false alarms, and still catch the concept, as context grows?
- Do own-reply probes degrade less than Claude-text probes?
- Which context types hurt most?
- Re-scored vs regenerated replies.
- The figures, a few examples (the top false alarms at long context, quoted), cost and caveats.
- A short comparison with the Gemini paper's findings.
