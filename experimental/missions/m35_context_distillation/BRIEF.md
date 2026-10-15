# Mission m35_context_distillation: does a model that has internalised the attack style (no instruction in context) move its own probe?

## Why

Every score-only result reads the probe on a teacher-forced re-read of a reply that was generated *with* the attack
instruction, with the instruction then removed. That is not the state of a model that chose the style itself. The
cleanest test of the "same model, on its own" rung of Nabil's progression: context-distil a frozen attack instruction
into Qwen3.5-27B (fine-tune it on its own instructed replies with the instruction removed), then probe its on-policy
generations with no instruction in context, using the original, unchanged probe.

## Design

1. **Feasibility first (write it in STATUS before training):** where to train a LoRA on Qwen3.5-27B (Modal with the
   existing HF/vLLM stack; check the repo and `/Users/nabilbaugher/dev/ryan_ai_config/tinker_harness/` for Tinker support
   of this model), and how to read the probe on the fine-tuned model's activations (the scorer must load the LoRA, or
   merge it, and read layer 48 exactly as the existing `ProbeLab` does; verify on plain replies that a zero-strength
   adapter reproduces the base scores).
2. **Data:** for 1–2 frozen instructions (the best m28 pinned-Opus-4.8 maths instruction; one ordering instruction),
   generate Qwen's instructed replies on fresh training problems (disjoint from eval/test sets), keep only valid replies,
   and fine-tune on (problem without instruction → reply). Also a control LoRA trained the same way on Qwen's *plain*
   replies (no instruction), to separate "fine-tuning moved the probe" from "the style moved the probe".
3. **Measure:** on the 100 test problems × 5 seeds with no instruction: validity, the distilled model's own probe scores
   (on-policy), compared with (a) plain base, (b) the control LoRA, (c) the instruction-removed re-read numbers from
   m22/m28 for the same instruction. Also check the style was learned (replies look like the instructed ones).
4. **Reading:** if the drop matches the removed re-read, the proxy used throughout is validated; if it disappears, the
   re-read overstates evasion. Either result goes in the post.

## Rules

- Modal names `probe-evasion_nabil-baugher_m35-<label>_kill-2026-11-30`; stop only your own apps by id. GPU cap about
  $400; Claude API cap about $60 (validity judges). Log estimates.
- Keys via `load_dotenv('.env')`, never printed. No core-module edits without the lead's OK; code in
  `src/probe_evasion/missions/m35_context_distillation.py`. Don't edit other missions' modules.
- Write only in `missions/m35_context_distillation/`, that module, `data/concepts/m35_*`, `results/m35_*`. Save every
  LoRA checkpoint path in STATUS (and in `missions/m35_context_distillation/checkpoints.md`). No repo-changing git.
- Critic check-ins: after the feasibility plan; before the verdict. If feasibility fails within ~3 hours of trying, write
  down why and stop rather than burning budget.
