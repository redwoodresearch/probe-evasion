# Mission m34_what_the_probe_reads: is the model still computing while the probe reads low, and what do the task probes actually read?

## Why (from the cold review of draft v3.1, analysis/post_story/)

The post says prompting moves task-probe scores but "the probe still sees the task". Four questions decide how far that
can be pushed, all cheap:

1. **Is the model still computing at the moment the probe reads low?** The probe reading comes from a teacher-forced
   re-read of the reply with the attack instruction removed. At the token just before the answer, record Qwen's
   next-token probability of the correct first answer token in that same re-read pass (plain vs attack replies, for
   maths 27B, 9B and code). If Qwen still predicts the answer with high probability while the probe reads low, that's
   the cleanest "computing while the probe misses it" signal in the project; if the probability collapses, the style is
   disrupting the computation itself. Use m27's Part C rows (`part_c_rows` in
   `src/probe_evasion/missions/m27_readout_and_style_checks.py`) and m22's scoring path; you'll need a scorer call that
   also returns logits for the vocabulary at that position (ask the lead if a core change is needed; prefer doing it in
   your own module).
2. **Compute without outputting the answer.** Requests where Qwen must do the computation but not state the result
   (e.g. "reply only 'even' or 'odd' for the result", "say only whether it exceeds 1,000", "say only whether the program
   prints a list longer than 3 items"). Score with the maths and code probes; check correctness. If the probe fires near
   plain levels, it reads the work; if near non-answers, it is a context/format detector ("answers to tasks").
3. **Is "transfer to sister probes" informative?** Cosine similarity between the nine layer-48 maths probe directions
   (weights under `results/concept_probes/`). Then train maths probes at layers 32, 40 and 56 of Qwen3.5-27B with the
   m1a peak recipe (same rows) and score the saved m22/m28 frozen-test texts. Similar movement = general representation
   change; none = layer-specific.
4. **Broad-pool line positions.** Using m27 Part D's broad pool scores (saved), recompute for maths, code and ordering:
   the median ordinary reply, the 1% line, d' of plain vs ordinary, and where the line sits on the plain-to-ordinary
   scale. The post's tone-vs-task contrast currently uses the narrow one-sentence pools.
5. **Within-style gap metric (free).** From m27 Part B and m28 twins: share of the plain task-vs-non-task gap closed
   within style, per arm (searched, random, no-spec, Qwen-self, pinned). One table; this is the post's candidate
   "beyond style" number.

## Report

RESULT.md, plain language, one section per question with its verdict and what the post may claim.

## Rules

- Modal names `probe-evasion_nabil-baugher_m34-<label>_kill-2026-11-30`; stop only your own apps by id. GPU cap about
  $80; Claude API cap about $30 (correctness checks). Saved data first.
- Keys via `load_dotenv('.env')`, never printed. No core-module edits without the lead's OK; code in
  `src/probe_evasion/missions/m34_what_the_probe_reads.py`. Don't edit other missions' modules.
- Write only in `missions/m34_what_the_probe_reads/`, that module, `data/concepts/m34_*`, `results/concept_probes/m34_*`.
  No repo-changing git. One critic check-in before the verdict. Save everything.
