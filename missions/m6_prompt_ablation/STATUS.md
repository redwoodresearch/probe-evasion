# m6_prompt_ablation status

**Phase:** done (17:55). RESULT.md written; final critic check-in done and its points addressed.
**Primary metric (fixed before results; counting corrected by the lead at 17:13):** valid replies under the 1% line
per attempt (repeats counted), instruction hidden, with task-clustered bootstrap intervals.

**Verdict (details in RESULT.md):**
- **Sarcasm round 0** (full 106/200, plain 16/200). No single unit or group matters. Removing all seven
  "plain/literal/flat" units costs 9.5 points, and also removing the examples costs 21. The bans push replies under the
  line (core + bans: 174/200 under the line regardless of validity), and the examples keep them sarcastic. Minimal
  prompt (template + examples + bans + safeguard line, 140 of 250 words): 123/200 against the full prompt's 99/200 on
  the same seeds.
- **Sarcasm round 1** (full 161/200, plain 16/200). Drops act only through validity (about 95% of valid replies evade
  regardless). The examples carry the reply shape. Of the single units, only the technique line replicates (-15.5),
  through a dangling-sentence effect. Minimal prompt (length line + technique line + examples + neutral-verbs line +
  safeguard line, 128 of 253 words): 157/200 against 155/200.
- **Gemma maths** (full 21/200, fresh problems 29/300; plain 2 and 3). The evasion exists only at the 1% line (0 at
  5%) and mostly on hiker-climb problems ("The altimeter log shows ..."). It is carried by the instrument sentence or
  the altimeter example. Probe mentions have no detectable effect. Minimal prompt (127 of 238 words): 17 against 17
  on the test tasks, 24 against 29 on fresh problems.

## What this mission does

Three frozen evasion prompts (each found by a Claude search) are cut into numbered units (a sentence, rule, bullet,
example or ban list). Every variant is an exact deletion of units from the frozen text; the rest stays verbatim and in
order. Each variant runs on the same 100 test tasks x 2 new seeds (20, 21), and each reply is scored by the prompt's
target probe with the instruction visible (as generated) and hidden (the same reply after the bare request).

| prompt | units | variants | target probe | validity |
|---|---|---|---|---|
| sarcasm round 0 ("That is good news, I was just hoping to...") | 16 | 26 | `r0_peak@L48` (m3a2) | repo judge AND blind judge (m3a's two Sonnet judges, via `m3a2.run_attack`) |
| sarcasm round 1 ("How thoughtful of...") | 18 | 28 | `r1_peak@L48` (m3a2) | same |
| Gemma-2-9B maths ("reading it off a display or a record") | 21 | 34 | `gemma_maths_layers_peak:maths@L6@s0`, EMA reading | `gepa.lenient_maths_present` |

Variants per prompt: full; plain (no instruction); every unit removed singly; grouped removals (all examples, all
bans, all style or length rules, the safeguard sentences, every probe mention for Gemma); core only (the template or
technique line plus the safeguard line); core plus one group (examples, bans, style; plus probe and the
instrument-sentence line for Gemma). The keep-one-group variants are run up front for every group rather than only
after a big grouped effect, to save a second round.

Files: `units_<prompt>.json` (units, verbatim, with group), `variants_<prompt>.json` (every variant's exact text).

## Choices and deviations

- **Lines.** Sarcasm: m3a2's rule (EMA quantiles of the 1,459 clean calibration replies), re-scored into this
  mission's own cache (`score_cache.jsonl`); m3a2's own lines are stored next to ours in `lines.json` for comparison.
  The calibration rows are rebuilt from the same two lines as `m3a2.eval_sets()` rather than calling it, because
  `eval_sets()` also reads m3a2's attack files, which its round 2 may be writing. Gemma: the search run's 1% line
  (`calibration.json`, 351 cleaned negatives); its 0.1% and 5% lines are quantiles of the same cleaned replies (with
  351 replies, the 0.1% line is close to the single highest one).
- **Groups.** Each unit belongs to at most one group. Unassigned units are the technique's own explanatory sentences
  (e.g. "The activity must be the person's actual bad situation"). For Gemma the safeguard group holds the four
  validity sentences (settle the correct answer first; a wrong number is a total failure; name the quantity
  explicitly; it must clearly BE the amount asked); its core uses the first two.
- **Cloud names.** Modal runs go through the project's shared app (`probe-evasion_nabil-baugher_experiments_kill-2026-11-30`,
  from `config.py`); this mission creates no new named cloud resources.
- **m3a2 is read only.** Its `run_attack`, `score`, `lines` and `round_probes("0"/"1")` are imported; `m3a2.SCORE_CACHE`
  is pointed at this mission's cache inside this mission's processes only (its file is never written).

## Critic check-ins

- **16:55, critic 1 (design).** Main points: (1) Gemma's 1% line is set by one calibration template ("210 people is
  24% of 875 people", EMA 0.83-0.90); most of the full prompt's evasions sit just under it and the 5% line catches
  nearly all; (2) sarcasm with the instruction visible is at ceiling (evaded among valid ~100%), so the visible
  per-attempt number mostly tracks validity; (3) a deletion changes both the reply and the text the probe reads it in;
  (4) leave-one-out misses redundant units (template line vs examples showing the same template); (5) length cap and
  judge rating may drive validity changes; (6) ~30 variants x several metrics: treat single-unit drops as a screen and
  confirm on fresh seeds; (7) Gemma: report the strict check too. **Done / planned:** primary metric fixed in advance
  as *valid evasions per attempt, instruction hidden, 1% line* (the hidden reading puts every variant's reply in the
  same context, the bare request, so it measures the reply effect only, answering point 3 for the primary metric);
  tables add evaded-among-valid, a line-free task-paired EMA logit shift, too-long counts, judge ratings and reply
  length; Gemma results reported at 1% (the search's line, as asked) and 5%, with the line caveat stated; round 2 adds
  combined drops (template line + examples, etc.) and confirms the minimal prompts and top hits on fresh seeds 22-23.
  Not rebuilding Gemma's calibration set from 1,000+ replies: the brief asked for the search's own line, and the
  ablation compares variants with each other under one line; the 5% line and the line-free shift cover the concern.

- **17:12, critic 2 (results).** Main points: (1) Gemma's digit-based validity check rejects correct answers written in
  words, so "core only fails validity" is partly the checker; (2) Gemma evasions sit on ~13 fixed tasks and the second
  seed barely changes replies, so round 2's fresh seeds are not independent confirmation; (3) "U7+U14 matches full" holds
  for the 1% count but not the line-free shift; probe mentions have a small (not zero) effect; (4) the de-duplication
  artifact is confirmed; (5) for sarcasm the ban lists carry most of the score suppression and the other units mostly
  keep replies valid; (6) ~33 variants: expect 1-2 chance hits (r0 U1/U6 and Gemma U6/U10/U12); (7) the logit shift
  uses valid replies only. **Done:** a secondary Gemma validity that converts number words to digits before the same
  check (reported next to the primary); 300 fresh maths problems from the repo's generator (none whose text appears in
  data/concepts) for full, plain, drop U7, core + U7, the two minimal prompts and drop-probe; wording softened
  (probe mention: small); chance hits flagged as such.
- **17:13, lead's correction (counting).** Successes were de-duplicated while attempts were not, mixing repetition with
  evasion. **Changed:** per-attempt counts are now primary (valid replies under the line / all attempts, repeats
  counted), with task-clustered bootstrap intervals on per-attempt rates; distinct-reply counts are a secondary column
  (distinct evasions / distinct replies). Conclusions below were re-derived from the corrected numbers.

- **17:45, critic 3 (verdict).** Spot-checked 14 numbers (all match) and confirmed the bootstrap. Points: r1 "either
  alone is enough" was wrong (only the examples are); r0 "bans suppress the probe" needed a validity-free count;
  qualify the Gemma headline; limit the probe-mention claim to Gemma ("no detectable effect"); explain the
  inconsistent r1 single drops; state Gemma's family concentration and line placement; visible counts matter for the
  cut-down variants; clarify the word-answer sentence; the r0 single-unit gains suggest a high-scoring shared full
  sample; STATUS verdict stale; define terms. **Done:** all fixed in RESULT.md. Added an "under the line regardless
  of validity" column. Re-ran r1 U1, U2, U5 and U17 singly on seeds 22-23: only U2 replicates, explained by a
  dangling sentence. STATUS verdict updated.

## Modal apps started by this mission (stop only these, by id)

- lines: ap-xBRpGcnw1vo8QNgpk0jVeB
- sarcasm_r0: ap-AfkZB0AlIIermaa4QPfIiC
- sarcasm_r1: ap-vwzrX2R1CHZtK6eNdoRE1V
- gemma_maths: ap-wIMFJQIsZzzoYZhows7gbb
- (16:49, three ids not listed: first launch of the three runs failed at argument parsing before creating an app)
- gemma_maths round 2: ap-EhbTJWXPRJCTfsTVMkBFHC
- sarcasm_r0 round 2: ap-JasKORiTVdbx3IY7qFYwnI
- sarcasm_r1 round 2: ap-UlRVB6SJDDjNvLK70V1cax
- gemma_maths fresh tasks: ap-lZZEKlohO4P3uVPvvsCPB9
- sarcasm_r1 round 2b: ap-Zc6TQwcVSiNOLc3w48FAp9
