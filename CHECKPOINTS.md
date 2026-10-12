# Checkpoints

Every trained model checkpoint in the project. One line each: where it lives, what it is, and where its results are.

## Fine-tuned Qwen3.5-27B (LoRA)

- **m35 context distillation, 2026-10-12.**
  - Location: Modal volume `probe-evasion_nabil-baugher_m35-weights_kill-2026-11-30`, paths
    `<task>/<arm>/s<seed>/step_<k>`.
  - Tasks: maths, ordering. Arms: attack, plain control, style control. Seeds: 0, 1. Steps: 0, 25, ... to the last.
  - Recipe: rank-32 LoRA, Hugging Face PEFT, one H200.
  - Checkpoints used in the test: `missions/m35_context_distillation/runs/<task>/chosen.json`.
  - Full list: `missions/m35_context_distillation/checkpoints.md`.
  - Results: `missions/m35_context_distillation/RESULT.md`. A model trained into the attack style keeps about a quarter
    (maths) and about half (ordering) of the instruction-removed re-read drop.

## Linear probes (not model checkpoints)

Probe weights live under `results/concept_probes/` (git-ignored; one JSON per probe family). Each mission's RESULT.md
names the probes it used.
