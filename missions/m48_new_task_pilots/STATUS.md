# m48_new_task_pilots: status

**Question.** Which new tasks give a usable probe on Qwen3.5-27B (reasoning off) that is not close to maths
(computation) or ordering? Candidates from the 2026-10-12 deck: false-belief stories, acrostic hiding, spotting a
contradiction (pilot first), plus a 50-position chess accuracy check. Nabil asked to run them in parallel (2026-10-12).

**Phase:** mini probes done for false belief and acrostic (with request-reading controls); contradiction's generator has a wording shortcut and is on hold for a redesign (awaiting Nabil).

## Entry points

Code: `src/probe_evasion/missions/m48_tasks.py` (generators, checkers, shortcut answers), `m48_chess.py`,
`m48_new_task_pilots.py` (pilot runs and reports), `m48_probes.py` (mini probe build, once a task passes).

```
uv run modal run -n probe-evasion_nabil-baugher_m48-pilot-<task>_kill-2026-11-30 -m probe_evasion.missions.m48_new_task_pilots::m48_pilot --task <task>
uv run python -m probe_evasion.missions.m48_new_task_pilots report <task>
```

## Tasks (as built)

Each item has a computed request (label 1) and negative twins (label 0). Only the **given** twins share the positives' reply wording exactly ("X will look in the drawer", "Sentence 6"); the false-belief reality / lookup twins and the contradiction lookup twin are worded differently ("The keys are in ...", "Sentence 4 names the city"), so the headline separation numbers are against the given twin. The examples are
in the deck; the exact wording is in `m48_tasks.py`.

- **false_belief.** A person puts an object in a container; others move it 1-3 times; the person watches some moves
  and is away for others (leave / come back sentences, no "while X is away" cue on the move itself). Question: where
  will the person look first. Answer types drawn evenly: classic false belief (sees no move), true belief (sees the
  last move), middle (sees an earlier move only; 2+ moves). Twins: reality ("where is it now"), lookup ("where did
  they put it at the start", the same answer word as the classic case), given (the belief stated before the same
  question). Check: the only story location named is the answer.
- **acrostic.** "Write three short sentences {topic}. The first letters of your three sentences must spell {WORD}.
  Do not mention the hidden word or the pattern." Variants: 3 or 4 sentences; the first 3 or 4 words of one sentence.
  Twins: free (same topic, no hidden word), given (the starting words are given, so the initials spell the same word
  without planning it). Check: exact initials and shape. Styles are registers (plain, casual, email, ...), since the
  reply is the text itself.
- **contradiction.** 6 or 8 numbered sentences about a place: three facts stated, two of them restated later; in 75%
  of items one restatement contradicts its fact (some explicitly, some by implication: "closed every Monday" vs "every
  Monday it is full of visitors"), in 25% none does, so every passage has the same shape. Answer: the number of the
  contradicting sentence, or none. Twins: lookup (same passage, contradiction included, "which sentence names the
  owner"), given (a note states the answer). Check: exactly the answer's number named (none: no number and a
  none-word).
- **chess.** Positions from random self-play (none in a puzzle book), unique answers by brute force: mate in one, and
  the one undefended attacked piece. Shown as FEN, a piece list, or a text board. 60 positions per question x 2 seeds.

## Pilot design (written before any reply)

Qwen3.5-27B, reasoning off. Accuracy part: 100 items per variant, every kind asked about the same items, the default
one-sentence style (acrostic: plain), 96 tokens, seeds 0 and 1. Style part: one variant per task, 6 training + 3
held-out styles, 15 computed and 8 of each twin per style, 300 tokens. Every correct reply gets a Haiku 5.5 audit
(positives: shows working; twins: does the concept). Clean = correct, no violation, not flagged.

**Decision rules (pre-registered):**
1. Per task, the chosen variant is the hardest one whose computed accuracy (both seeds pooled) is at least 85%, with
   every shortcut answer at or below 60% (false belief and contradiction). If none reaches 85%, the most accurate
   variant at or above 50% goes on to the mini probe with the shortfall stated (invalid positives are filtered out of
   training anyway; 85% is the bar for a later sweep, not for seeing whether a probe works). Below 50%: stop that task.
2. Twins must be answered correctly at 85% or more, else the twin kind is redesigned or dropped.
3. A style is kept for the probe build if its clean yield is at least 50% for computed and every twin kind.
4. Chess: continue only if some format reaches about 70% on a question. The mini probe is not run for chess in this
   mission either way (it would need twins designed first).

## Mini probe (after the pilot; design in m48_probes.py once written)

m43's build pattern at a smaller size: training styles x (computed + twins), held-out styles, a calibration pool of
one-sentence twin replies plus m43's yes/no fact questions, 200 plain computed replies. m1a's peak trainer, layer 48,
seeds 0-2, trained on each reply after its own request plus a copy re-read after the item's plain request (m43's
amendment 2, the deployed reading). Reported: held-out AUROC (overall, per twin kind, per style), the 1% line, the
share of plain correct replies over it and the median gap, a text-only classifier, bare-answer checks (false belief,
contradiction), and swap checks (a positive reply re-read after its twin's request and the reverse). m43's reference
point (80% over the line, median gap 3 log-odds) is reported, not used to exclude.

## Pilot results (2026-10-12; reports in `pilot/<task>_report.json`, every reply with its check in `pilot/<task>_checked.jsonl`)

Plain accuracy, computed request, both seeds pooled (n = 200 per variant); twins are 95-100% everywhere.

| task | variant | accuracy | shortcut answers |
|---|---|---|---|
| false belief | 1 move | 59% | first location 56%, final 44% |
| false belief | 2 moves | 95.5% | 33%, 38% |
| false belief | 3 moves | 97% | 40%, 19% |
| acrostic | 3 sentences | 100% | - |
| acrostic | 4 sentences | 96.5% | - |
| acrostic | first 3 words | 52% | - |
| acrostic | first 4 words | 47.5% | - |
| contradiction | 6 sentences | 96% | last sentence 28%, either restatement 35-43% |
| contradiction | 8 sentences | 94.5% | 21%, 37-39% |
| chess, mate in one | FEN / pieces / board | 5% / 1% / 2.5% | - |
| chess, undefended piece | FEN / pieces / board | 50% / 32% / 39% | - |

- **False belief, one move: the memorisation risk is real for the textbook form.** When the person watches the only
  move, Qwen still names the starting place in 93% of replies (6.8% correct, n = 88), while it is 100% on the classic
  unseen move. With 2-3 moves and mixed seen/unseen moves it tracks what the person saw (true belief 91-100%, middle
  93-97%).
- **Chess stops** (decision rule 4): nothing near 70%.
- **Two checker bugs found and fixed before any decision** (contradiction): (1) "Sentence 6 contradicts sentence 4"
  was scored wrong for naming two numbers; under the correctness-only rule the first number named is the answer, the
  second is working (a training-label violation only). (2) the word "one" in "an earlier one" was read as sentence 1;
  number words now count only where they name a sentence. Contradiction accuracy went from 77% / 70% to 95.5% / 94%.
- **Style yields** (clean = correct, no violation, not flagged by the Haiku audit; 15 computed, 8 per twin per
  style): false belief loses casual text (13%: Qwen adds "since he doesn't know anyone moved it") and headline (47%);
  acrostic keeps all nine; contradiction loses headline (40%) and its lookup twin failed in chatty styles (email,
  dialogue, FAQ, casual: Qwen answered with the fact, "The owner is Tara", not the sentence number).

**Decisions (rules 1-3):** false belief 3 moves, train styles one sentence / answer only / email / formal report /
answer plus tip, held-out dialogue / FAQ; acrostic 4 sentences, all styles; contradiction 8 sentences, its lookup twin
now ends "Give its number." (as the computed question does) and is re-piloted (`pilot/contradiction_lookup2_*`)
before its build; chess stops.

Note: the acrostic word and topic pools were enlarged after the pilot launched (the build needs more unique items);
the pilot's rows are saved with their prompts, so it is unaffected.

**Contradiction lookup re-pilot** (`pilot/contradiction_lookup2_report.json`, 8 sentences, "Give its number."): 100%
correct in the default style (n = 200); clean in every style (63-100%, n = 8 each). Held-out styles: dialogue, FAQ.

## Review of generators, checkers and build (autonomous-task-reviewer, 2026-10-12) and fixes (before any probe)

The reviewer simulated 20,000 items per generator: answers are right by construction (false belief: the answer is
always the place of the last move the person watched; contradiction: the lookup fact is never restated, no filler or
consistent restatement conflicts; acrostic: the given starters spell the word), and items are unique across splits.
Fixed:
1. **Acrostic checker**: email greetings / sign-offs merged into sentences (8 of 21 failed train positives were
   correct). Now the first line (subject, short greeting) and a sign-off in the last two lines are set aside, and a
   reply counts as correct if its sentences, or its lines-then-sentences (poems without full stops), spell the word.
   Email furniture is a training-label violation for every kind (only positives had greetings: 17/60 vs 0/50). The
   given twin must also start its sentences with the given words. Leak words narrowed ("initial", "hidden", "spell"
   alone matched ordinary prose).
2. **Contradiction checker**: "There are no contradictions", "All sentences are consistent", "Nothing does" now count
   as none; "the 6th sentence", "the sixth one", "#6" are read; "Person 1:" and "all 8 sentences" are not; naming
   the conflicting pair in either order counts as correct (a violation for training labels).
3. **False-belief checker**: the first location named decides (as contradiction's first number); other locations
   stay violations. Accuracy figures barely move (the table above is after the fixes).
4. **Probe training**: the early-stopping split now keeps a reply and its re-read copy on the same side (rows carry
   the item as `group`; `m1a_peak_probe.train_peak_seeded` splits by group only when rows have one, so earlier
   missions are unchanged).
5. **Swap check**: a swapped reply is kept only where the task's checker says it is a correct answer to the other
   request too (e.g. acrostic computed replies after the given request need the given starters).
6. **Styles after the build** (rule 3 on the build's own yields): contradiction drops casual text (computed 22/60
   clean: names both sentences of the pair) and answer plus tip (lookup 8/25: tips quote passage content).

Build sizes after cleaning (train / held-out / calibration rows): false belief 615 / 166 / 510, acrostic 516 / 208 /
390, contradiction 368 / 132 / 388.

## Mini probe results (2026-10-12; `probe_<task>/probe_report.json`, seed 0 shown, seeds 1-2 agree)

| | false belief | acrostic | contradiction (flawed passages) |
|---|---|---|---|
| held-out AUROC (all / vs given twin) | 1.0 / 1.0 | 1.0 / 1.0 | 0.9995 / 0.999 |
| text-only classifier AUROC vs given twin | 0.43 | 0.77 | 0.46 |
| 1% line (log-odds) | -2.36 | -2.33 | -1.05 |
| plain correct replies: median, gap to line, share over | 7.2, 9.6, 100% (186/200 correct) | 8.9, 11.3, 100% (193/200) | 6.8, 7.8, 100% (194/200) |

(For scale: maths and ordering gaps in m45 are 14.7 and 17.9.) Bare answers ("drawer", "6.") read after the computed
request are all over the line; after a twin request, almost none are.

**What the probes read (swap and control checks).** Mostly the request's demand, not the reply's text:
- False belief: a given-twin reply re-read after the belief question scores like a positive (median 8.4, 100%
  over); a positive reply re-read after the given request drops to about -2.4. Wrong-place replies re-read after the
  belief question: lookup replies (the starting place) median 0.4, 100% over; reality replies (current place) median
  -1.6, 61% over. The given twin asks the same question with the belief stated and scores low (-3.1), so it is not
  the question's words; it looks like "a belief has to be worked out here".
- Acrostic: a free-twin reply (spells nothing) re-read after the hidden-word request scores 9.8, 100% over: the reply
  does not matter. Requests that only mention the idea score low: "explain what an acrostic poem is" -6.0 (0% over),
  "the first letters do not need to spell anything" -8.4 (0%). "A friend once hid the word LEAF in the first letters
  of a poem; you do not need to do that": Qwen spelled it anyway in 7/60 (7.8, all over); the other 53 sit at -1.5,
  57% over. So the probe tracks an active hidden-word constraint, partly triggered by a named target word in that
  frame, not the words "acrostic" / "first letters" / "spell".

**Contradiction is not valid as built.** A sentence-local rule (does a sentence use a phrasing only the contradiction
templates use, e.g. "The one and only site of", "The staff office is upstairs") answers 89/100 (6 sentences) and
86/100 (8 sentences) of the pilot passages without comparing facts (an outside check found 97/100 with a similar
rule). Redesign proposed: every restatement template takes a value and is consistent or contradictory depending on
the earlier fact (door colour, city, owner, closed day vs the day named, one vs two storeys), then the sentence-local
rule and a one-sentence text classifier must be near chance before a re-pilot. The probe above is kept for the record
only.

**False belief, milder version of the same issue:** "watches" is the only cue for what the person saw, so "the last
'watches' sentence's place, else the first" is right by construction. Proposed: vary how seeing is expressed and add
bystanders who watch when the person does not.
