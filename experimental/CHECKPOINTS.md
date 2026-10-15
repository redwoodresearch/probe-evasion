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

- **m40 habit across capability, 2026-10-12.**
  - Location: Modal volume `probe-evasion_nabil-baugher_m40-state_kill-2026-11-30`, paths
    `lora/<task>/<writer>_c<chain>/s<seed>/step_<k>` (plus `lora/<task>/m35attack/s0/...` positive control,
    `lora/m35/...` copies of m35's plain controls, `lora/smoke/`). About 700 GB of fp32 adapters.
  - Instructions: m36's frozen round-16 instruction of every chain of all 11 writers (maths, ordering). Seeds 0, 1 for
    Opus 5.5, GPT-6 Sol, Gemini 3.8 Flash, Qwen3.5-397B, Haiku 4.5; seed 0 for the other 6. Steps 0, 25, ... to the last.
  - Recipe: m35's rank-32 LoRA (PEFT, one H200); 21 core maths LoRAs with gradient checkpointing.
  - Checkpoints used in the test: `missions/m40_habit_across_capability/runs/<task>/<batch>/chosen.json` (all step 25).
  - Full list: `missions/m40_habit_across_capability/checkpoints.md`.
  - Results: `missions/m40_habit_across_capability/RESULT.md`. As a trained habit, m36's capability trend survives in
    full on maths (correct answers only) and at about a third of the slope on ordering; the lost part is the fine-tuned
    model reading the same text higher.

## Linear probes (not model checkpoints)

Probe weights live under `results/concept_probes/` (git-ignored; one JSON per probe family). Each mission's RESULT.md
names the probes it used.
