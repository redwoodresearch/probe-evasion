# Mission m23_model_size: does the maths score drop under prompting depend on the monitored model's size?

## Why

Mission m22 runs a Claude prompt search against Qwen3.5-27B maths probes where Claude sees only per-token probe scores
(no training data), and tracks the score distribution round by round. Nabil wants the same conditions on a smaller and a
bigger Qwen to see how model size interacts with how far prompting can push probe scores down while the model still
answers correctly.

## Models

- **Smaller:** the largest dense Qwen3.5 below 27B that exists on Hugging Face (check; Qwen3.5-9B if it exists, else the
  next size down). Reasoning off.
- **Bigger:** Qwen3.5-122B-A10B (FP8; mission m10's serving and activation-capture setup in
  `src/probe_evasion/missions/m10_sarcasm_122b.py`). Note the confound in the report: it is a mixture-of-experts model
  with about 10B active parameters per token, fewer than 27B dense.
- If a cheap dense model bigger than 27B in the Qwen3.5 family exists, say so; don't run it without asking.

## Step 1: maths probes for each model, same recipe as the 27B own-reply probe

- Recipe: m1a's own-reply peak probe (`src/probe_evasion/missions/m1a_peak_probe.py`, `m1a_offpolicy_maths.py`; rows from
  the model's own replies to computation problems in the eight answer styles, label-contradicting replies removed,
  trained with BCE on the running peak, read with the running peak, alpha 0.5).
- Layer at the same relative depth as 27B's layer 48 (48 of 64 = 75%); also train one shallower layer (about 25%
  depth) as a check.
- Calibration: the model's own one-sentence replies to no-maths lookups, audited the m1a/m11 way (drop replies that
  compute or verify the figure); 1% / 0.1% / 5% lines.
- Gates before the ablation: held-out AUROC, plain maths answers caught at the 1% line, with the same numbers for 27B
  alongside. Report them; do not drop a model for missing a gate without asking.
- 3 training seeds for the main layer.

## Step 2: the same ablation as m22

Use m22's runner (`src/probe_evasion/missions/m22_score_only_iterations.py`; its STATUS.md documents the entry point).
Same problems, same attacker (Claude Opus 5.5), same feedback, same rounds and chains, same frozen test. Do not edit
m22's module; if it needs a change to accept another model, ask the lead. Until m22's runner is ready, do Step 1.

## Report (RESULT.md, plain language)

- Probe quality per model (table).
- The per-round dot plot per model, in m22's style, plus one combined figure: median valid score (relative to that
  model's plain answers and line) per round, one line per model size.
- Does zero-shot / few-shot movement grow or shrink with size? Does validity (still answering correctly) change?
- Cost and caveats (MoE confound, different probe quality per model).

## Rules

- Modal names `probe-evasion_nabil-baugher_m23-<label>_kill-2026-11-30`; stop only your own apps, by app id. GPU cap
  about $400 (122B is expensive: batch generation and scoring; avoid many short scoring calls on a wide fan-out, which
  cost $266 in m20). Claude API cap about $150. Log estimates.
- Keys via `load_dotenv('.env')`, never printed.
- No core-module edits; code goes in `src/probe_evasion/missions/m23_model_size.py`.
- Write only in `missions/m23_model_size/`, that module, `data/concepts/m23_*`, `results/concept_probes/m23_*` and
  `results/gepa/m23_*`.
- No git commands that change the repo. Long runs detached in Modal or with caffeinate.
- Critic check-ins: after the probes are trained (probe quality and calibration audit); after the first ablation chain
  per model; before the verdict.
- Save everything: generated rows, probe files, calibration scores, every attacker prompt/response and every reply with
  per-token scores.
