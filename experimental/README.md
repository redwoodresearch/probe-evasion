# Experimental

Everything here is exploratory or superseded and is not read by the post's pipeline (`mainline/`). It is kept for the
record: earlier concepts and designs, side projects, and the drafts, reports and status pages written along the way.
Each mission folder has a `STATUS.md` (run log) and usually a `RESULT.md`.

**Paths.** These folders were moved here from `missions/`, `analysis/` and the repo root on 15 Oct 2026. Their code
still lives in `src/probe_evasion/` and still refers to the old locations (`missions/<name>/…`), so rerunning one
means moving its folder back, or checking out commit `13a0ae7c`, the last commit before the move.

## Missions (`missions/`)

**Earlier concepts** (before the post settled on verifiable reasoning tasks):
- `m1b_offpolicy_chem`, `m1c_chem_no_near_negatives`, `m1d_chem_keyword_labels`: chemistry probes and probe-training mistakes
- `m3a_sarcasm`, `m3a2_sarcasm_patch`, `m3a3_sarcasm_onpolicy`, `m3a4_sarcasm_styleaug`: sarcasm probes and attacks on them
- `m3b_criticism`, `m3c_anger`: tone probes (criticism, anger) on Qwen3.5-27B
- `m4_aisi_deception`: an existing deception probe
- `m7_sandbagging`: sandbagging
- `m8_flattery_condescension`, `m8a_flattery`, `m8b_condescension`: flattery and condescension probes
- `m9_chameleons_search`: search against Gemma-2-9B tone probes ("neural chameleons" setting)
- `m10_sarcasm_122b`: sarcasm on Qwen3.5-122B
- `m15_omission`, `m16_quiet_register`: omission and register probes

**Earlier designs and probe studies** (the path to the post's design):
- `m6_prompt_ablation`: which sentences of searched attack prompts do the work
- `m12_frontier_pilot`, `m13_blind_search`, `m14_qwen_attacker`: other attackers and search set-ups
- `m17_long_context`, `m18_false_alarms`: long contexts; false alarms on everyday text
- `m20_onoff_pairs`, `m20_purpose_pilot`: early on/off pair designs
- `m21_multilayer_probes`: probes on several layers
- `m23_model_size`: monitored models of other sizes
- `m24_task_ablation_design`: design notes for the task ablation
- `m29_existing_data_analyses`, `m34_what_the_probe_reads`: re-analyses of earlier runs
- `m30_equal_effort_tone`, `m31_more_chains`, `m33_self_writer_rungs`, `m37_sweep_generalisation`, `m41_oracle_dependence`,
  `m42_adaptive_defender`, `m44_writer_effort`, `m47_long_toggle_search`: variations on the writer sweep
- `literature`: reading notes

**Side projects:**
- `m32_reasoning_on`: Qwen with reasoning on
- `m35_context_distillation`, `m40_habit_across_capability`: fine-tuning a model to keep an evasive habit (out of scope
  for the post); their checkpoints are listed in `CHECKPOINTS.md`
- `m52_broad_probes`: probes trained on broader, chat-like data (built and tested, then dropped for the post)

## Analysis (`analysis/`)

Older post drafts (`post_v5`, `post_v7`, `post_outline`, `post_story`, `writeup`), daily and weekly reports
(`overnight_report_*`, `morning_report_*`, `weekly_report`, `report`), audits (`audit_m45`, `training_data_audit`),
the broad-probe analysis (`broad_probe`), and figures and checks for the experiments above.

## Other

- `notes/`, `preregistration/`: working notes and an early preregistration
- `share/`: material made for sharing (videos, a probe toolkit)
- `sprint/`, `sprint_established/`, `status/`: generated status pages
- `CHECKPOINTS.md`: LoRA checkpoints from `m35` and `m40`
