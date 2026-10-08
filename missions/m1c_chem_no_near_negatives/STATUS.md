# m1c_chem_no_near_negatives: status

**Question.** Is a chemistry probe trained on Qwen's own replies, but without nearby-topic negatives (no biology/maths on
the "not chemistry" side, only distant topics: literature, comforting, jokey, confused), evadable by prompting?

**Phase:** chemistry readout finishing (scoring running); then lead's second redirect (11:40): a MATHS probe trained on
Gemma-2-9B-written replies (off-policy, weaker model), matched to mission 1a's Claude-written set; compared with 1a's
Claude-text probe and the own-replies clean probe, at both of 1a's calibration lines (one-sentence and all-styles).

**Gemma maths (11:55).** Code `src/probe_evasion/missions/m1c_gemma_maths.py`. Gemma's replies to the exact seeded
task list already existed (seed 0); added seeds 1-3 (9,480 more replies) to fill cells. Per task: first sample passing
`label_ok` (comma-free preferred for not-computed tasks), comma balancing, matched to 1a's cell counts. Result:
`data/concepts/m1c_gemma_maths_train_clean.jsonl` 1473 rows (782 computed) vs 1a's 1725 (813): not-computed cells are
short because Gemma restates given numbers with commas ("$9,568") but writes computed results without ("1156"), so
balancing drops many not-computed rows. Comma rate not computed 18% vs computed 13%. Held-out Gemma set 363 rows.
Probes `m1c_gemma_maths_gd` / `_lr` (layers 16, 48) training; then 1a's scored sets are copied and scored with them
(`results/gepa/m1c_gemma_maths_scores/`).

**Mission change (lead, ~10:55):** the main probe is now OFF-POLICY: Claude-written replies (claude-sonnet-5-5, same
prompts/styles as the chemistry training set) without nearby-topic negatives, matched to mission 1b's Claude-written set
WITH biology/maths negatives (`data/concepts/m1b_offpolicy_chem_train.jsonl`: 835 chemistry, 550 distant + 378
biology/maths). Ours: same 835 chemistry + 550 distant rows from 1b, plus 378 Claude-written distant-topic replies on
the same replacement prompts. Comparison: 1b (off-policy, with nearby negatives) vs ours (off-policy, without). The
own-replies (on-policy) no-near-negatives probe was already being built; it is kept as a cheap secondary comparison.

**Modal note:** the shared workspace is at its 100-ephemeral-app limit (73 apps are another project's
`cotcontrol-inference`); launches retry every 30 s via `missions/m1c_chem_no_near_negatives/modal_retry.sh`.

**Design.** Start from the own-replies training set `data/concepts/chem_qwen35_train.jsonl` (881 chemistry, 985 not:
568 distant-topic + 417 biology/maths). Drop the 417 biology/maths negatives; replace them with 417 more distant-topic
Qwen replies (68 fresh unused distant prompts from the mega dataset, the rest = training-negative prompts asked again in
a different answer style). Same size, same class balance. Train gradient-descent (gd) and logistic-regression probes at
layers 16 and 48. Code: `src/probe_evasion/missions/m1c_chem_no_near_negatives.py`.

**Data built (11:00).**
- Own-replies version `data/concepts/m1c_chem_no_near_negatives_train.jsonl`: 881 chemistry, 985 distant-topic
  negatives (568 original + 417 replacements: 53 fresh prompts, 364 restyled). 480 replacement replies generated,
  445 passed the no-chemistry check.
- Claude-written version `data/concepts/m1c_chem_no_near_negatives_claude_train.jsonl`: 835 chemistry, 928 distant
  (550 from 1b + 378 Claude replies to the same replacement prompts; 455/480 clean). Disjointness asserted (whole base
  prompts vs search tasks, calibration, science eval, held-out: 0 shared).
- Critic's control `..._claude_dropped_train.jsonl`: 1b minus its 378 biology/maths negatives, not replaced (835 vs 550).
- Probes training (layers 16, 48): `m1c_chem_no_near_negatives_claude_gd`, `_claude_lr`, `_claude_dropped_gd`, `_gd`, `_lr`.

**Hand-written attacks** (`attacks.json`): plain, the saved analogy instruction, generic probe-aware, and probe-aware
register shifts toward the negative classes (story, comfort, jokes, poem, literary essay, musing), each with a
style-only control (same style, no mention of the probe), plus a "frame it as biology/physics" contrast arm. Validity
on 40 validation tasks: 31-39/40 per arm (plain 36/40). First read: the styled replies stay chemistry-dense
(VSEPR, 109.5°, etc. inside the story/comfort framing).

**Verdict so far:** none yet.

## Critic check-ins
- 10:58 (design). Points: (1) replacements reuse prompts in new styles, so the comparison also changes negative
  variety/weighting: add a "1b minus biology/maths, not replaced" probe; (2) expect the mistake to show as science
  false alarms, report those next to every evasion number; (3) biology framing is a contrast arm, not the targeted
  attack; (4) add style-only controls for every styled attack; (5) test on more seeds if rates are small; (6) assert
  disjointness. Done: (1) dropped-negatives probe added, (2) in report, (3) added as contrast, (4) added
  poem/literary/musing controls, (6) assert added. (5) pending signal.
