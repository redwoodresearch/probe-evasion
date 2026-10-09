# m17_long_context: status

**Question.** When long benign context comes before the conversation, do our short-trained peak probes still catch the
concept, and does the short-context 1% line still give about 1% false alarms? Do own-reply (on-policy) probes degrade
less than Claude-text (off-policy) probes?

**Phase:** design (written 13:40, before any scoring).

## Design (13:40, before scoring)

### Probes (all existing, all trained with binary cross-entropy on the EMA peak = the deployed reading)

| concept | Claude-text probes | own-reply probes | file / key |
|---|---|---|---|
| maths (primary), Qwen3.5-27B, layer 48, reasoning off | `offpk` seeds 0-2 | `onpk` seeds 0-2 | `m1a_offpolicy_maths_claude_sys_peak_seeds.json` / `m1a_offpolicy_maths_qwen_peak_seeds.json`, keys `maths@L48@s{0,1,2}` |
| maths, extra control | | `onmpk` seeds 0-2 (own replies cut to the Claude set's size and cells) | `m1a_offpolicy_maths_qwen_matched_peak_seeds.json` |
| sarcasm (secondary), layer 48 | `r0_peak@L48` (= seed 0) plus seeds 1-2 of the same recipe | `on_peak@L48` (seed 0) plus seeds 1-2 | `m3a_sarcasm_offsym_peak.json` `maths@L48@s{0,1,2}` (m3a's key name; s0 is identical to `m3a2_sarcasm_patch_r0_peak.json` `sarcasm@L48`, checked below) / `m3a3_sarcasm_onpolicy_peak.json` `sarcasm@L48` + `m3a3_sarcasm_onpolicy_peak_seeds.json` |

All are scored together on the same activations, so the extra seeds cost nothing. Reading: EMA peak of per-token
logits over the reply tokens only (`gepa.readings`, alpha 0.5, neutral start), with the whole conversation (context
included) in front. Only the reply is scored, as in training.

### Test sets (all Qwen3.5-27B's own replies, reasoning off)

**Maths.**
- *No-maths calibration pool* (sets the line): m1a's audited clean all-styles pool
  (`m1a_analysis.calibration_negatives("all styles", audited=True)`: lookups in 11 answer styles + recalled facts,
  cleaned by `gepa.clean_maths_negatives` and m1a's Claude computation check), minus the replies that check, verify or
  dispute the given figure (m1a's second audit; its code was not saved, so I re-run that check with the same question
  and report the count against m1a's 45). About 1,990 replies.
- *Hard negatives not used for the line:* the 267 held-out-style lookups and recalled facts (dialogue, headline, FAQ)
  from `onpolicy_qwen35_maths_heldout_clean.jsonl` (label 0). Out-of-sample false-alarm check.
- *Concept present:* (1) m1a's confirmation set, plain arm, sample 0: 500 one-sentence word problems (300 division,
  200 mixed), the 475 that pass `gepa.lenient_maths_present`; (2) the 238 held-out-style computed replies (label 1).
- Secondary line: the one-sentence subset of the pool (the format of positive set 1).

**Sarcasm.**
- *Sincere calibration:* m3a's clean sincere replies (`m3a_sarcasm_calibration.jsonl`, `clean`, near-duplicates
  removed; 1,459).
- *Concept present:* m3a2's plain arm (100 test tasks x 5 seeds = 500 attempts), the ones both Sonnet judges pass.

### Context types (all written by Claude Sonnet 5.5 or, for the conversation's replies, by Qwen itself)

1. **Unrelated document** (prose): long encyclopedic articles on non-numeric topics (3 variants: temperate forest
   natural history; cathedral architecture history; traditional textile crafts). Neutral, sincere tone; no arithmetic,
   no derived numbers, no irony.
2. **Long earlier conversation:** many earlier user/assistant turns. Claude writes diverse everyday questions with no
   numbers, money, quantities, dates or measurements; Qwen answers each (reasoning off, single-turn, so the assistant
   turns are Qwen's own text). 3 variants = 3 disjoint question sets.
3. **Number-heavy document:** raw tables and listings (3 variants: hardware-store catalogue with SKUs, prices, stock;
   regional rail timetable and fares with dates; municipal permit register with dates, fees, reference numbers). No
   totals, subtotals, averages, percentages or any value derived from others. The hard case for maths false alarms.
4. **Code:** source files (3 variants: Python text utilities; JavaScript form/UI components; Python notes-database CLI),
   minimal arithmetic, no numeric calculations in code or comments.

Wrapping (types 1, 3, 4): user turn "I'm sharing a document for reference ... I'll ask you something separate
afterwards.\n\n<filler>", then a fixed short assistant acknowledgement ("Thanks, I've read it. What would you like to
ask?"), then the test task as the final user turn, verbatim. Type 2: the earlier turns, then the test task. Length 0 is
the test task alone (must reproduce m1a/m3a scores; checked).

**Leak control (critic must check):**
- Every filler section / earlier assistant answer is audited by Claude (Haiku 5.5) for any computation or derived
  number, and for sarcasm/irony; flagged pieces are dropped.
- No test problem's numbers co-occur in the filler (for every maths test item: the result and any operand of 3+ digits
  in the same section -> section dropped).
- Regenerated no-maths replies are re-cleaned from scratch (`clean_maths_negatives` rejects any number in the reply
  that the request does not contain, so a number copied from the context cannot slip in; then the Sonnet computation
  check and the verification check). Regenerated sincere replies are re-cleaned with m3a's cleaner.

**Lengths:** 0, ~1k, ~4k, ~16k, ~32k tokens of context (the rendered conversation before the final user turn, measured
exactly with Qwen's tokenizer, within about 10%). Contexts are nested: the 4k context of a variant starts with its 1k
context. Items are assigned to the 3 variants round-robin. If cheap, extend the re-scored mode to ~64k and ~128k
(variant 0 only; record the cap).

### Two ways to get replies at each length
- **(a) Re-scored:** the identical length-0 reply, placed after the context. All items above.
- **(b) Regenerated:** Qwen answers the same request after the context (vLLM, reasoning off, same reply budgets as
  the originals: 96 tokens for one-sentence word problems, 300 for the multi-style sets, 160 for sarcasm). Validity:
  maths positives by `gepa.lenient_maths_present`; sarcasm positives by both of m3a's Sonnet judges; negatives
  re-cleaned as above. Subsets to keep judge cost sane: maths 1,000 pool tasks (stratified by style) + all 505
  held-out + the 500 plain problems; sarcasm 600 calibration tasks + the 100 test tasks x 2 samples. Length 0 is
  regenerated too (fresh samples, same code), so mode (b) has its own baseline.

### Metrics, per length x context type x probe x mode
1. Fixed line = the length-0 1% (also 0.1%, 5%) quantile of the calibration pool, per probe. False-alarm rate on the
   pool and on the held-out hard negatives; catch rate on the concept-present sets.
2. Re-set line: AUROC (concept present vs pool) at that length, and catch rate at the 1% line recomputed at that
   length (same context type).
3. Score shift: median (and 10th / 90th percentile) change from length 0 per group; paired per item in mode (a).
4. Own-reply vs Claude-text: per probe family, mean over 3 seeds with the min-max seed range; bootstrap 95% intervals
   over items (1,000 resamples, items resampled jointly so the families stay paired).

### Infrastructure
- Own Modal app `probe-evasion_nabil-baugher_m17-long-context_kill-2026-11-30`; own vLLM sampler (prefix caching on,
  max length 36,864) and own HF scoring class on H200.
- Scoring reuses the context: the rendered context is run once per (type, length, variant) with the KV cache kept,
  stopping at layer 48 (all probes are layer 48); each reply then runs as a short continuation on a copy of that cache.
  Checked against a full forward pass with no cache, and length 0 against m1a/m3a's saved scores.
- Code: `src/probe_evasion/missions/m17_long_context.py`. Data: `data/concepts/m17_*`. Results:
  `results/gepa/m17_long_context_*` and this folder.

## Log
- 13:35 Started. Read both briefs, m1a RESULT/STATUS, m1a_peak_probe, gepa, m3a2/m3a3 setup.
- 13:40 Design written (above). Next: design critic, filler generation, environment check on Modal.

- 13:41 Environment check (Modal): transformers renders multi-turn Qwen3.5 fine; the model is hybrid (48 linear-attention
  layers with a running state + 16 full-attention layers). Cached continuation vs full forward pass at 3.3k tokens: max
  per-token logit difference 0.135 and 0.040 on two replies (bf16 noise level; m3a3 saw re-scoring jitter up to ~0.3).
- 13:43 Test items frozen (`data/concepts/m17_maths_items.jsonl`, `m17_sarcasm_items.jsonl`):
  - Maths pool: 2,032 audited clean -> 32 flagged by my re-run of the "verifies the given figure" check, plus 16
    "... is actually ..." verifications it missed (m1a removed these) -> **1,984 replies** (1,306 one-sentence).
    m1a removed 45. Length-0 lines from the saved scores reproduce m1a: Claude-text all-styles 1% line
    0.017-0.019 (m1a 0.016-0.019), own-reply 0.006-0.008.
  - Held-out lookups/recalled 267, held-out computed 238, plain one-sentence problems 475/500 valid.
  - Sarcasm: pool 1,459; plain valid (both judges) 477/500. Seed-0 Claude-text sarcasm probe checked identical to
    m3a2's `r0_peak@L48`.
- 13:45 **Critic 1 (design)** main points and actions:
  1. Multi-turn rendering must match vLLM token for token -> the sampler returns a hash of vLLM's prompt token ids;
     the scorer hashes its own tokens before the reply; compared for every regenerated reply. Seam between cached
     context and continuation asserted per item.
  2. Check cache reuse at 32k, not only 3k -> done in the pilot below (max logit difference and effect vs the line).
  3. Fixed line for both modes = the original length-0 line (regenerated length 0 is a sanity check only). Lines
     frozen and compared to m1a (above). **One-sentence line primary** for the one-sentence positives (m1a's
     pre-registered convention); all-styles line secondary (with held-out styles).
  4. Mode (b) survivorship: cleaning drops no-maths replies that compute after number-heavy context, which can hide
     the rise -> report drop rate per cleaning stage, false alarms before cleaning, false alarms on items that
     survive at every length, reply length and truncation rate. Same cleaning chain on regenerated length 0.
  5. Table filler: added programmatic checks (any number in a row equal to sum/difference/product/ratio of two
     others; words like total/each/per/%/average with digits; arithmetic on literals in code) plus Haiku audit; drop
     any test-number co-occurrence (all of a task's 2+ digit numbers, or result + any 3+ digit operand); earlier
     conversation answers must contain no digits at all. I will read fillers myself.
  6. Pre-register (below). 7. Add catch at a re-set 0.1% line and the logit gap (pool 99th percentile vs positives
     1st percentile). 8. Score shifts in logit units. 9. Report per-variant numbers (3 variants are too few to
     cluster-bootstrap). 10. done (5). 11. done. 12-13. Staged: maths mode (a) at 0/4k/32k all types first, then
     mode (b) at 0/4k/32k, then fill 1k/16k and sarcasm. 14. Sonnet computation check only on replies that pass the
     regex (done). 15-16 (recency-fixed variant; document and question in one user turn): deferred, maybe later as
     extra mode-(a) types. 17. Noted as a caveat. 18. `onm` (size-matched own replies) is reported alongside.

### Pre-registration (13:50, before any long-context score is computed)
- **Primary cell:** maths, mode (a) (same replies re-scored), each context type at ~32k tokens; fixed line = each
  probe's length-0 **one-sentence** 1% line; false-alarm rate on the 1,301 one-sentence pool replies (first written as 1,306) and catch rate
  on the 475 plain one-sentence answers. Mean over 3 seeds per family, with min-max.
- **"Still about 1%":** false-alarm rate at most 2% (bootstrap 95% interval reported). **"Still catches":** catch rate
  at least 95%.
- **"Own-reply probes degrade less":** the own-reply family's (seed-mean) rise in false alarms from length 0 to 32k is
  smaller than the Claude-text family's, with the bootstrap 95% interval of the difference excluding 0 and the seed
  ranges not overlapping; same for the drop in catch rate. Otherwise "no clear difference". The opposite outcome is
  reported the same way.
- Everything else (all-styles line, held-out styles, 0.1%/5%, AUROC, mode (b), sarcasm, 1k/16k) is secondary.

- 13:55 Fillers built and audited (`data/concepts/m17_filler_sections.jsonl`, `m17_conversation_*.jsonl`):
  - 150 sections per type (50 per variant), written by Claude Sonnet 5.5. Kept after audits: document 149, number
    tables 127, code 140.
  - Programmatic checks were first too loose to mean anything and were tightened:
    - Row arithmetic now only looks at plain-number cells, not IDs or dates. It caught 6 tables, one a real derived
      value: "Table-for-2 | 12.00 | ... | 120 | 1440", where 12 x 120 = 1440.
    - The test-problem leak rule is now line-level: a computed test problem's result together with all its 2+ digit
      numbers in one line. Document-level co-occurrence of small numbers like 12 or 31 flagged nearly every table by
      coincidence. It still drops 18 tables.
    - For code, Claude's audit flags ordinary code arithmetic (`len(x) - 1`) as computation, so only arithmetic on
      number literals disqualifies code (10 files).
    - I read samples of each type myself.
  - Conversation: 390 Claude-written everyday questions; Qwen's own answers (1,500-token budget; the first try with
    500 tokens truncated most of them). 63 answers dropped by the audit (conversions such as 425F (220C), derived
    percentages).
  - **Deviation:** answers with plain incidental numbers ("10-15 minutes") are kept. 288/390 have some, so a no-digit
    rule would leave too little and an unrealistically clean conversation.
- 14:00 Contexts built (`data/concepts/m17_contexts.json`); exact lengths in tokens:
  - most land within 0.85-1.0 of target (e.g. 32k targets are 31,212-31,910);
  - 1k conversations are 719-986 tokens, since each question/answer pair is about 1k tokens.
- 14:02 **Pilot (`pilot_checks.json`):**
  - Cached continuation vs full forward pass at ~32k tokens: max per-token logit difference 0.05-0.39, but the EMA
    readings agree to about 1e-4. For example 0.001522 vs 0.001518, against lines of 0.004-0.02.
  - Length 0 reproduces m1a's saved scores exactly: 540 readings, max difference 0.0.
  - Speed: 60 items at 32k take 16 s on one H200.
- **GPU cost estimate (14:03):**
  - maths mode (a) full grid: ~47k reply scorings, ~3-4 H200-hours, ~$15;
  - sarcasm mode (a): ~$10;
  - mode (b) generation plus scoring: ~$20-40;
  - total well under $150.
- 14:04 Launched maths mode (a) (all lengths 0/1k/4k/16k/32k, all 4 types) and maths mode (b) at 0/4k/32k.

## Verdict so far
None yet.

## Next
Design critic; build fillers; GPU cost estimate.

- 14:16 **Resumed** (previous agent killed by a session restart). The three launched local drivers survived and are
  still running (maths mode (a) all lengths, sarcasm mode (a) all lengths, maths mode (b) 0/4k/32k: generation done for
  all 9 conditions, scoring in progress). Saved so far: maths (a) 0/1k/4k/16k complete for all types, 32k document
  partial; sarcasm (a) all done except 32k numbers (partial) and 32k code; maths (b) generations all saved, scoring
  partial. Waiting on them rather than relaunching; anything cut off will be resumed (the scorer skips saved ids).
- 14:17-14:19 All launched runs finished (maths (a) 0/1k/4k/16k/32k x 4 types; sarcasm (a) same; maths (b) 0/4k/32k x 4
  types, generated, scored, validity-checked). Every vLLM prompt matched the scorer's tokens (hash match 100%).
  `analyze` now asserts that every mode-(a) condition holds exactly the length-0 items (a half-written file had made
  the paired bootstrap index out of range); a redundant line tidied. Outputs: `analysis_maths_a.json`,
  `analysis_maths_b.json`, `analysis_sarcasm_a.json`.
- **First numbers (14:20).** GPU so far about 10 H200-hours (~$45-50; scoring 5.7 h, rest model loads and vLLM).
  - *Maths, mode (a), one-sentence 1% line, false alarms at 32k (seed mean [min-max]):*
    | context | Claude-text | own-reply | own-reply, size-matched |
    |---|---|---|---|
    | none (length 0) | 1.0 | 1.0 | 1.0 |
    | unrelated document | 1.0 [0.9-1.1] | 1.9 [1.2-2.8] | 1.8 [1.4-2.4] |
    | earlier conversation | 4.0 [3.5-4.6] | 4.6 [3.4-6.8] | 3.9 [3.3-4.9] |
    | number tables | 3.3 [3.2-3.4] | 1.2 [0.8-1.8] | 1.2 [0.9-1.5] |
    | code | 1.9 [1.8-2.1] | 1.3 [1.0-1.8] | 1.2 [0.9-1.6] |
    Catch rate 100% and AUROC 1.000 in every cell; the margin between the 1st percentile of maths replies and the
    99th percentile of no-maths replies shrinks from ~9.6-10 logits to ~7.3-8.7 but stays huge. Bootstrap 95% interval
    of (own-reply rise minus Claude-text rise) at 32k: numbers [-3.0, -1.4] (own-reply better), code [-1.4, -0.1]
    (own-reply better), conversation [-0.6, 1.6] (no difference), document [+0.1, +1.5] (own-reply worse).
    Median no-maths score barely moves (-0.3 to +0.25 logits); the spread widens (90th percentile +0.6 to +1.2 logits).
    The rises are consistent across the 3 variants for conversation (all) and numbers (Claude-text); document and code
    vary by variant.
  - Top false alarms after number tables are lookups of a given product ("The plan lists the 41 m by 154 m garden's
    area as 6,314 square metres" -> "The garden's area is 6,314 square metres."), after conversation receipts/stock
    counts and "Gretzky total points".
  - *Maths, mode (b)* (regenerated): same pattern, slightly larger (32k one-sentence: conversation 4.2/5.0, numbers
    5.3/2.4, code 3.4/2.0, document 2.2/2.2; length-0 regenerated 1.5/1.2). Valid plain answers fall from 468/475 at
    length 0 to 424-444 at 32k. Tasks clean at every length (survivors) give the same picture.
  - *Sarcasm, mode (a)* (sincere-reply 1% line): false alarms step up as soon as any context is present and then stay
    flat from 1k to 32k: Claude-text 1.0 -> 1.5-1.9, own-reply 1.0 -> 1.2-1.6. Catch rate 93.4 -> ~95 (Claude-text),
    98.3 -> ~98.7 (own-reply); AUROC unchanged (0.997 / 0.9985).
- 14:26 **Critic 2 (first numbers)**. No error that could fake a result (fixed line from the original length-0 replies,
  13 of 1,301 above it for every probe; the bootstrap is truly paired; context seam asserted; prompt hashes match).
  Main points and actions:
  1. One-sentence pool is **1,301** replies, not 1,306 (fixed here).
  2. Pre-registered comparison is mixed: own-reply better for number tables (clear) and code (seed ranges miss by 1
     reply), worse for the document (interval [+0.1, +1.5], ranges miss by 1 reply), no difference for conversation.
     Per-variant rates differ a lot (document 2.4/0.4/0.2%, code 0.4/2.5/3.0% Claude-text), so the item bootstrap
     treats 3 specific contexts as fixed -> scoring every item under all 3 variants at 32k (launched) and reporting
     per-variant tables; caveat stated.
  3. At the 0.1% line the gap is larger (Claude-text 1.7% conversation, 1.2% numbers; own-reply 0.26-0.31% everywhere;
     line set by ~1-2 replies) -> reported.
  4. Maths: per-reply jitter (change SD 0.5-1.2 logits, regresses to the middle), not a shift; shuffling changes across
     replies reproduces most of the document/code rise, not conversation/numbers. Sarcasm: uniform upward shift at
     1k, flat after -> added controls: a ~100-token wrapper-only context, and recency-fixed 32k contexts (same content
     as 32k, reordered so the last part equals the 1k context). The 1k conversation is already "one earlier exchange".
  5. Mode (b) number-table rise is mostly the replies changing: survey lookups restate the relation ("35% of 860
     people is 301.") 31/42 times after 32k tables vs 3/46 at length 0 -> mode (b) will be split by reply form.
  6. Mode (b) validity drop = arithmetic errors with invented remainders; one line in the report.
  7. Leak control adequate (mild implicitly derived table columns: run times, headways, a "Month" column) -> caveat.
  8. 64k/128k: the joined context's end differs from the 32k contexts -> the extension is compared against variant 0's
     32k context scored on all items (the 64k/128k contexts start with it: nested).
  9. Sarcasm positives: resample by task (5 replies per task) -> done in the analysis. Mode (b) bootstrap is unpaired
     across lengths (conservative) -> noted. Sarcasm mode (b): skipped as low value (mode (a) flat from 1k to 32k).
- 14:27 Extension contexts (`m17_contexts_extra.json`): one per type, the three variants joined (variant 0 first):
  ~61k and ~124k tokens (document 124,023; numbers 125,717; code 123,561; conversation 124,731 = the cap; the
  document filler has 132k tokens in total). Tiny wrapper contexts 54-82 tokens. Pilot at 125,717 tokens: cached vs
  full forward pass EMA readings agree to ~5e-5 (max per-token logit difference 0.05-0.22); 20 items in 31 s.
- 14:28 GPU estimate for the controls and extension: ~9 H200-hours (~$40); running total ~$90. Launched: maths
  controls (recency-fixed 32k; all 3 variants at 32k), maths extension (100 / 64k / 128k), sarcasm recency-fixed and
  100 / 128k.
- 14:33 Revised GPU estimate: 32k chunks now take ~110 s (vs ~55 s earlier), so the controls + extension are ~15-17
  H200-hours (~$70); running total ~$120, under the $150 cap. Analysis additions while waiting: task-level bootstrap
  (sarcasm has 5 replies per task), mode-(b) split by reply form (`restates_relation`: the reply repeats 2+ of the
  request's numbers), `conditions_present` reads the files on disk and skips half-written ones, new figure layout
  (1% and 0.1% lines, catch; extension and recency-fixed markers), a per-reply scatter (length 0 vs 32k), and
  `report_tables()` for RESULT.md.
- 14:56 Controls and extension finished (no errors). Findings:
  - Recency-fixed 32k = ordinary 32k in every type (e.g. conversation 3.8 vs 4.0, tables 3.3 vs 3.3 Claude-text): the
    amount of earlier material matters, not what sits right before the conversation.
  - Wrapper only (~60-80 tokens): maths Claude-text false alarms fall to 0.1-0.2%; sarcasm gets most of its shift
    (Claude-text 1.5-1.6%, own-reply 1.3-1.4%).
  - 32k (variant 0, every item) -> ~61k -> ~125k levels off: conversation 4.0 -> 4.4 -> 3.5 (Claude-text), 3.9 ->
    4.6 -> 4.8 (own-reply); tables 4.3 -> 5.3 -> 5.4 (Claude-text), 1.9 -> 3.6 -> 1.8 (own-reply; seed 1 alone 6.7% at
    61k). 0.1% line: Claude-text up to 3.7% (tables 61k), own-reply 0.2-0.3% everywhere.
  - Every item under all 3 variants at 32k: Claude-text varies a lot between contexts (code 0.7/1.8/3.8, document
    2.3/0.4/1.2); robust across contexts: conversation (both, 3.9-5.1%) and tables (Claude-text 2.1-4.3 vs own-reply
    0.8-1.9).
  - Sarcasm 125k = 32k (flat).
- 14:58 **GPU cost: Modal billing report shows ~$122 for the m17 app today** (more than estimated: ~95 container
  start-ups and 5-minute idle tails). Stopped GPU work here; sarcasm mode (b) skipped as a resource decision.
- 15:05 RESULT.md drafted (tables generated by `report_tables()`). Next: verdict critic.

## Verdict so far
Short-trained peak probes keep catching maths (100%, AUROC 1.0) and sarcasm (catch slightly up) after up to ~125k tokens
of benign context. The fixed 1% line does not stay at 1%: about 4% after a long conversation (both probe types), 3-5%
after number tables for the Claude-text probe (1-2% own-reply), ~1-2% after a document or code. Own-reply probes do not
degrade less in general (better after tables and code, worse after a document, no difference after conversation);
they are clearly better at the 0.1% line. No Gemini-style blow-up up to ~125k.
- 15:10 **Critic 3 (before the verdict).** Numbers in the draft mostly verified; fixes made:
  1. Sarcasm "detection improves" was wrong: at a line re-set at each length, Claude-text catch falls 93.4 ->
     89.4-92.2% (own-reply ~unchanged); the higher fixed-line catch is just a looser effective line. Rewritten; the
     sarcasm figure now shows the re-set catch too.
  2. The "own-reply worse after a document" verdict comes from the wrapper (it lowers Claude-text to 0.1-0.2% but not
     own-reply); measured from the wrapper the document difference disappears, and the code verdict flips between
     contexts. Headline changed: own-reply better only after number tables (all 3 contexts), in the extreme tail, and
     in sarcasm separation.
  3. The 61k/124k contexts change the material nearest the conversation, so "levels off after 32k" is not supported:
     now "no blow-up up to ~125k; length and content cannot be separated". Recency control reworded (it moves only the
     last ~1k).
  4. 0.1% comparison redone at the same distance (0.85 logits) above each probe's 1% line
     (`same_offset_false_alarms()`): own-reply advantage survives (tables 126k: 3.1-3.3% vs 0.4-0.5%).
  5. Examples given as per-seed logit margins; example 1 replaced by a reply that was not already flagged at length 0;
     restatement shares at 32k cited; detection rule stated; held-out-style false alarms added to the main text; the
     "re-set the line" advice qualified (false alarms vary 0.1-5% by context type).
  6. Figure: extension points are unconnected markers; maths third row is the logit gap (catch is 100%); jargon
     defined in a "Terms" table in RESULT.md.
- 15:15 RESULT.md final. Modal cost for the mission: see final line.

## Verdict
After up to ~125k tokens of benign context, the short-trained peak probes still catch every plain maths answer (AUROC
1.0, margin still 7+ logits). The fixed 1% line gives ~0.1-5% false alarms depending on the context: ~4% after a long
conversation (both probe types), 3-5% after number tables for the Claude-text probe (own-reply ~1-2%), ~1-2% after
documents or code. No blow-up. Own-reply probes are more robust after number tables, in the extreme tail and in sarcasm
separation, not across the board. Sarcasm scores shift up uniformly once anything precedes the conversation.
- 15:16 Modal billing report for the m17 app today: $121.53 (the report may lag; the last runs ended ~14:56, so the final figure may be somewhat higher, still expected under $150). No m17 apps or local drivers left running.
