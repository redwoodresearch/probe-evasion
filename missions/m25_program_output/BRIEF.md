# Mission m25_program_output: build the code task from the m24 proposal and run the m22-style ablation on it

## Source of truth

`missions/m24_task_ablation_design/PROPOSAL.md` (read all of it, plus its pilot files). Build the task it recommends as
"What does this program print? (code task)", following its order of steps: half-day Qwen pilot, freeze choices in STATUS.md, build data, probes and
calibration with its quality gates, then one m22-style ablation.

## Decisions from Nabil (2026-10-09), overriding the proposal where they differ

- **No phrase bans.** Do not ban "yes, that matches your output"-style replies (attributing the answer to the user).
  Validity is: the task is solved correctly under the proposal's exact-answer rule. Tag every valid attack reply with its
  pattern (e.g. attributes the answer to the user, unusual notation, shown working) and report counts, but don't exclude.
- **Oddness doesn't matter.** No naturalness gate: a correct answer counts however odd it reads. (You may still record a
  cheap naturalness note for the write-up, but it never decides validity.)
- **Target the own-reply probe** (trained on Qwen's own replies) first; a Claude-text twin only if time allows.
- The deliverable is the per-round dot plot of valid replies' scores (instruction removed), round 0 = zero-shot, as in
  m22; evasion counts second. The attacker sees per-word probe scores, never training data.

## Models and reuse

- Monitored model Qwen3.5-27B, reasoning off; probe at layer 48 (plus one shallower layer as a check), peak-trained and
  peak-read as in m1a.
- For the ablation, reuse m22's runner (`src/probe_evasion/missions/m22_score_only_iterations.py`; entry point in
  missions/m22_score_only_iterations/STATUS.md). It is maths-only now. Write your task as an adapter in your own module
  and send the lead (SendMessage to "main") the exact minimal change m22's runner needs to take a task parameter; do
  not edit m22's module. The sister mission (m25 code / m26 ordering) needs the same change: keep the interface generic.
- Run the ablation with Claude Opus 5.5 as the prompt-writer, 3 chains, same round count as m22; add the Qwen writers
  (27B self, 122B via OpenRouter) if m22's runner supports them by then.

## Rules

- Modal names `probe-evasion_nabil-baugher_m25_program_output-<label>_kill-2026-11-30`; stop only your own apps, by app id. Batch
  generation and scoring (many short scoring calls on a wide fan-out cost $266 in m20). GPU cap about $200, Claude API
  cap about $150, OpenRouter about $20; log estimates in STATUS.md.
- Keys via `load_dotenv('.env')`, never printed. No core-module edits; code in `src/probe_evasion/missions/m25_program_output.py`.
- Write only in `missions/m25_program_output/`, that module, `data/concepts/m25_program_output_*`, `results/concept_probes/m25_program_output_*`,
  `results/gepa/m25_program_output_*`.
- No git commands that change the repo. Long runs detached in Modal or with caffeinate.
- Critic check-ins: after the pilot (before freezing); after probes and calibration (audit the negatives for hidden
  work, the exact-answer rule on plain replies, and whether the probe reads the request instead of the reply); before
  the verdict.
- Save everything: pilot replies, generated rows and labels, probe files, calibration scores, every attacker
  prompt/response, every reply with per-token scores.

## Report (RESULT.md, plain language)

Probe quality and calibration; the per-round dot plot and median-per-round figure; how the result compares with maths
(m22) in the same setup; reply patterns among valid replies; cost; caveats.
