# Mission m46_monitored_model_axis: does probe breakability change as the monitored model gets bigger or more capable?

## Why

So far the monitored model has been Qwen3.5-27B, and only the attacker (writer) varied. Nabil's question: how does the
breakability trend change as the **monitored** model gets larger or more capable? Bigger models might have more robust
representations, so their probes are harder to break. Or they might have more freedom in activation space, so the
probes are easier to break. This mission adds the monitored-model axis, using the project's default design: m45's
toggle design (evasion part on vs off), read with the instruction in view.

## Monitored models

Pick these, or the closest feasible substitutes, and record why:
1. **Qwen3.5-27B:** the baseline. m45 is running it; reuse m45's results.
2. **Qwen3.5-397B-A17B** (ECI 146.7): same family, much larger. m38 is training a maths own-reply probe on it; reuse
   that if it passes, and coordinate through the lead.
3. **Qwen 3.8 27B** (ECI 149.4): the same size as the baseline but newer and more capable. It separates capability from
   size.
4. **DeepSeek V4.1 Flash** (ECI 154.9, open weights, unrestricted): the highest-ECI open model that is practical to
   serve, from a different family.
   - First check its parameter count, vLLM support and whether our activation capture works on it (MLA attention, MoE).
   - If it isn't feasible within about a day of engineering, use the next one: DeepSeek V4 Flash, GLM-5.3-Flash or
     GLM-5.2. Kimi K3 is non-commercial; ask the lead before using it.

ECI values are in missions/m36_attacker_capability_sweep/eci/eci_scores.csv. Record total and active parameters for
each model.

## Design

- **Per monitored model, both tasks (maths, ordering):** build the own-reply probe exactly as m22/m26 did, at about
  75% depth. Train it for the reading it's deployed with: include re-read rows after the plain task request, as m43
  learned. Then set the 1% line on that model's own ordinary one-sentence replies, and apply a go/no-go before any
  search:
  - held-out AUROC;
  - at least 95% of plain answers caught;
  - plain median at least 3 log-odds above the line;
  - m43's terse check.
  Report probe quality per model. A failed probe is a result: report it, and don't sweep on it.
- **Writers:** a fixed set of 5 spanning ECI, the same for every monitored model: Claude Opus 5.5, GPT-6 Sol, Gemini
  3.8 Flash, Qwen3.5-397B and Llama 4 Maverick. Use m45's toggle sweep (import m45's code), 3 chains, rounds 0–16,
  freeze at B = 0, 4, 16, test on 100 untouched problems × 5 seeds.
- **Metric:** m45's S_toggle (in view, on minus off, invalid = no movement) plus S_toggle_valid, the off-vs-plain
  guard, crossings under each model's 1% line, and the valid share of each monitored model.
- **Headline:** S_toggle against monitored-model size and against monitored-model ECI, at fixed writers. Also the
  writer trend within each monitored model.
- **Pre-register in STATUS** before any search, with one critic check on the design. Wait for m45's smoke test and
  first round to pass before launching any toggle search; probe building can start now.

## Budget

- GPU about $3,000. The big models dominate; use FP8 where supported and stop 8-GPU apps by id between steps.
- Claude about $1,500; OpenRouter about $60.
- Give the lead a cost projection after probe building, before the searches.

## Rules

- Modal names `probe-evasion_nabil-baugher_m46-<label>_kill-2026-11-30`; use your own Modal volume for state
  (`probe-evasion_nabil-baugher_m46-state_kill-2026-11-30`; the shared volume is near its inode limit). Stop only your
  own apps, by id. Keep scaledown windows short so GPUs do not idle through writer/judge phases.
- Keys via `load_dotenv('.env')`, never printed. No core-module edits and no edits to other missions' modules (ask the
  lead, SendMessage to "main"); code in `src/probe_evasion/missions/m46_monitored_model_axis.py`, importing from m36 and earlier modules
  rather than copying. Write only in `missions/m46_monitored_model_axis/`, that module and `data/concepts/m46_*`. No repo-changing git. Long
  runs detached in Modal. Log spend in STATUS; tell the lead before exceeding the budget.
- Pre-register the design and metric in STATUS before any run; one critic check on the design, one before the verdict.
  Save everything (every prompt, every reply with per-token scores, every judge call).
- RESULT.md in plain language: numbers, figures, caveats, cost.
- Local downloads go only into scratchpad/m46/ (several agents share the scratchpad). Spend guards count judge calls
  (upper bound), never the volume's usage log (it undercounts).
