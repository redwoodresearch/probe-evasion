# Mission m24_task_ablation_design: design a code task (and one more verifiable task) for the probe-evasion ablations

**Design only: no GPU runs, no probe training.** Small Claude API calls to look at example data are fine (cap $20).

## Why

The project's purpose (Nabil, after his 1:1 with James): show early signs of life that models are moving towards being
able to evade monitors while still accomplishing their task. What matters is less crossing a flag line and more that
prompting pushes the probe score distribution down while the task still gets done (the dot plot is key). Maths on
Qwen3.5-27B is the current testbed (see `missions/m19_maths_whitebox_redteam/RESULT.md`, `missions/m11_long_search/`,
`missions/m1a_*`). James wants the task varied first: a code task, and maybe one more task, with objective answers so
"did it do the task" is checkable, unlike tone concepts.

## What to deliver: `missions/m24_task_ablation_design/PROPOSAL.md`

For a **code task** and **one or two other verifiable tasks**, each option with:

1. **The task the model does**, with a concrete example request and an ideal short reply. It must be checkable
   automatically (execute code, exact answer), and doable by Qwen3.5-27B with reasoning off at a high rate. Say why.
2. **What the probe detects, and its training data.** Mirror the maths design: positives = the model's own replies that
   do the task (compute, trace, write working code), negatives = matched replies that don't need to (the answer is given,
   lookups, restating), so the probe learns "doing the task in its head", not the topic. Explain how negatives avoid the
   maths pitfall where "no-maths" replies secretly computed (see the calibration bug in the sprint notes).
3. **The validity rule** for an attack reply (task done correctly, no disguise), and how the framing tricks m19 found
   (answers dressed up as years, notation the reader must decode) would be ruled out.
4. **Why a prompt could plausibly move the score** and why it might not (is the computation forced to happen in the
   reply tokens the probe reads?).
5. **Cost and time** to build: data generation, probe training, calibration, one m22-style ablation.
6. **Risks** (task too easy or too hard; the probe reading surface features like code formatting; refusals).

Candidate ideas to weigh (add your own): predicting a short program's output ("what does this print?", the closest code
analogue of maths); writing a small function that must pass hidden tests; finding the line with a bug; for non-code,
date arithmetic, counting/sorting items in a list, unit conversion, simple logic puzzles.

End with a recommendation: which code task and which second task to build first, and the order of steps. Keep it
readable for Nabil (plain language, define any internal names).

## Rules

- Read the repo freely; write only in `missions/m24_task_ablation_design/`.
- Keys via `load_dotenv('.env')`, never printed. No git commands that change the repo.
- One critic pass on the proposal before handing back.
