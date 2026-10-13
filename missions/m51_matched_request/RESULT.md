# m51 result: matched-request control (does the probe read the computation or just the request?)

## Summary

**Setup.** 200 fresh maths problems and 200 fresh ordering puzzles, none of them in any saved training, held-out,
calibration or test set. They use the m45 test-problem generators: m22 / m1a's ten maths templates and m26's chain
puzzles.

Each item comes in three request versions, worded identically except for the answer:

- **compute**: the m45 test request, unchanged;
- **given (clause)**: the answer joined to the problem as a short clause. Maths: "A shop had 1411 stickers and sold
  1069, leaving 342. How many remain? ..."; ordering: "... From heaviest to lightest they are Pia, Lars, Cleo, Omar
  and Wes. List everyone from heaviest to lightest. ...";
- **given (sentence)**: the answer as its own sentence. Maths uses the template's own answer sentence ("... sold 1069.
  The shop has 342 stickers left. How many remain? ..."); ordering: "Their order from heaviest to lightest is Pia,
  Lars, Cleo, Omar, Wes."

Qwen3.5-27B answered every version with reasoning off, 96 tokens and seeds 10 and 11, through the same generation path
as m45's plain test answers. That gives 2,400 replies.

Each reply was scored with its own prompt in view by m45's scorer, using:

- the m45 target probe (q27mrr / q27orr);
- the older probe (onpk / ordpk);
- the TF-IDF classifiers exactly as trained in `analysis/tfidf_baseline`: A (reply only), B (prompt + reply) and P (prompt
  only). They were not retrained: the chosen models were re-fitted deterministically and checked against their saved
  scores.

Only correct replies count, checked in code. Each detector's 1% line is its own, set on m45's look-alike pool.

**Result: in maths the outcome is the positive one; in ordering the probe separates far more strongly, but the
prompt-only check does not fail.**

| compute vs given, AUROC (clause / sentence) | target probe | older probe | TF-IDF A (reply) | TF-IDF B (prompt + reply) | TF-IDF P (prompt only) |
|---|---|---|---|---|---|
| maths | **1.000 / 1.000** | 1.000 / 1.000 | 0.71 / 0.75 | 0.92 / 0.94 | 0.68 / 0.81 |
| ordering | **1.000 / 1.000** | 1.000 / 1.000 | 0.33 / 0.63 | 0.99 / 0.99 | 1.00 / 1.00 |

| median paired drop, compute minus given, as a share of the way from the compute median down to the 1% line (clause / sentence) | target probe | TF-IDF B | TF-IDF P |
|---|---|---|---|
| maths | **0.95 / 1.06** | 0.19 / 0.30 | 0.14 / 0.24 |
| ordering | **0.73 / 0.74** | 0.17 / 0.19 | 0.25 / 0.22 |

- **Maths: the probe reads whether the answer had to be worked out; the text classifiers do not.**
  - The probe separates every compute reply from every given reply.
  - The median given reply drops about 16 log-odds, from +10.1 to -5.9 / -7.6, close to the 1% line at -6.53.
  - The prompt-only classifier P barely separates the two (AUROC 0.68 / 0.81). Prompt + reply (B) does better (0.92 /
    0.94) but moves only a fifth to a third as far.
  - Neither text classifier puts a single given reply under its own line; the probe puts 36% (clause) and 64%
    (sentence) under its line.
- **Ordering: the probe drops given replies by about 14.5 log-odds** (compute median +11.7, given -2.7 / -2.9) and
  separates them perfectly.
  - The prompt-only classifier also separates them perfectly (AUROC 1.00), though with a much smaller gap (a quarter as
    far). Its training set already contained "given" requests that list the order ("The coach lined them up from
    tallest to shortest: ..."), so the inserted sentence is a cue it knows.
  - So ordering does not meet the stated criterion ("P and B fail"). Only the size of the probe's response, and the
    identical-reply pairs below, set it apart.
- **The cleanest evidence comes from identical replies.** Qwen often wrote exactly the same reply text to the compute
  and the given request: 137 / 136 pairs in maths, 83 / 249 in ordering.
  - On these pairs the reply tokens are the same, and only the request differs.
  - The probe's score drops by a median of 14-15 log-odds in every group, in 100% of pairs.
  - The smallest drop is 3.6 log-odds in maths (sentence phrasing) and 10.6 in ordering.
  - A reply-only reader cannot separate these pairs at all.
  - For P and B the same pairs move 0.4-3.3 log-odds, and in maths the clause phrasing moves P the wrong way in 45% of
    them.
- **What the probe still does not do: put given replies under its own 1% line.**
  - In maths 64% (clause) and 36% (sentence) of given replies stay above the line.
  - In ordering every given reply stays above it: the median is 5.4 log-odds over the line, the lowest 0.3 over.
  - So at the deployed line, the probe would still flag ordering replies whose answer was handed over in the request.
    The ordering look-alike pool that sets the line has no "given" requests in it.
- **Where the signal sits** (`figures/traces.png`, `traces.md`, identical-reply pairs). After the compute request the
  per-token reading rises on the answer itself:
  - in maths, on the result's digits;
  - in ordering, on the first name of the order (+16 to +26 log-odds).

  After the given request, the same tokens read 15-25 log-odds lower. In ordering they still show a smaller spike on
  that first name.

## Details

### Correct replies

| | compute | given (clause) | given (sentence) |
|---|---|---|---|
| maths | 400 / 400 | 362 / 400 | 391 / 400 |
| ordering | 368 / 400 | 400 / 400 | 400 / 400 |

- Maths given-clause failures are almost all the hiker template (36 of 38). "... then 910 m, 1295 m in all" was read as
  a fourth climb and added in ("The hiker climbed a total of 2590 meters"). Only 4 hiker clause replies are correct.
  Other given failures are recomputations with an arithmetic slip ("They cost $10,560 in total." for 10,080).
- Ordering compute failures are wrong orders.
- Correctness was checked in code. Maths: the result is among the reply's numbers, with commas allowed. Ordering: m26's
  exact-order rule. Unfinished replies (empty, or 96 tokens) do not count. No judge was used.

### Medians and share over each detector's 1% line (correct replies)

| task | detector | 1% line | median compute / clause / sentence | over the line compute / clause / sentence |
|---|---|---|---|---|
| maths | target probe | -6.53 | +10.11 / -5.89 / -7.61 | 100% / 64% / 36% |
| maths | older probe | -5.79 | +8.76 / -5.46 / -7.64 | 100% / 58% / 30% |
| maths | TF-IDF A | +3.58 | +5.12 / +1.81 / +0.44 | 67% / 34% / 33% |
| maths | TF-IDF B | -2.75 | +8.62 / +6.23 / +5.14 | 100% / 100% / 100% |
| maths | TF-IDF P | -0.26 | +8.64 / +7.08 / +6.13 | 100% / 100% / 100% |
| ordering | target probe | -8.06 | +11.71 / -2.65 / -2.91 | 100% / 100% / 100% |
| ordering | older probe | -7.30 | +10.44 / -2.91 / -3.10 | 100% / 100% / 97.5% |
| ordering | TF-IDF A | -1.91 | +0.78 / +1.08 / +0.48 | 100% / 100% / 100% |
| ordering | TF-IDF B | -8.81 | +5.43 / +2.88 / +2.56 | 100% / 100% / 100% |
| ordering | TF-IDF P | -7.60 | +6.31 / +2.68 / +3.32 | 100% / 100% / 100% |

### Identical-reply pairs (same item and seed, the same correct reply text after both requests)

| task, phrasing | pairs | target probe: median drop (smallest) | given over the probe's line | TF-IDF B median drop | TF-IDF P median drop (share of pairs dropping) |
|---|---|---|---|---|---|
| maths, clause | 137 | 14.8 (9.7) | 80% | 0.9 | 0.4 (55%) |
| maths, sentence | 136 | 14.2 (3.6) | 68% | 1.6 | 1.9 (91%) |
| ordering, clause | 83 | 14.4 (10.6) | 100% | 2.8 | 3.3 (100%) |
| ordering, sentence | 249 | 15.2 (12.4) | 100% | 2.5 | 3.0 (100%) |

The given replies in these pairs are the easiest kind (short, plain restatements). That is why more of them stay over
the probe's line than among all given replies.

### By maths template (target probe, correct replies: median; share of given over the line)

Clause phrasing ranges from 0% over the line (train, garden) to 100% (percentage, where the given sentence "18% of 650
people is 117 people." still scores above the line). Sentence phrasing ranges from 0% (train, garden, sweets) to 100%
(percentage, median +3.3). The percentage template has no statement to attach a clause to, so both of its given forms
put the answer before the bare question. That may read more like a check than a lookup. Every compute median is +9.1
to +10.9. Per-template numbers can be recomputed from `maths_scored.jsonl`.

### Caveats

- The two maths phrasings are template-specific in wording (the clause is "covering", "leaving", "in total", "each",
  ...; the sentence is the template's own answer sentence). The ordering phrasings are uniform. P's training set
  included given-style ordering requests close to the sentence phrasing, but not the maths ones. That likely explains
  much of the maths / ordering difference for P.
- The given requests sometimes still made Qwen compute: the recomputation slips and the hiker double-count above. Only
  correct replies are analysed, but some correct given replies may also have been computed rather than copied. That
  would push the probe's given scores up, against the result shown.
- Lines are the look-alike-pool 1% lines from m45 and the TF-IDF baseline, not lines matched to everyday chat.

## Spend and runs

- Two Modal runs on the `redwood-default` workspace:
  - `probe-evasion_nabil-baugher_matched-run_kill-2026-11-30` generated the maths replies. Its scoring step failed (a
    Modal class was imported after the app started), so I stopped that app by id.
  - `probe-evasion_nabil-baugher_matched-run2_kill-2026-11-30` reused those replies, generated the ordering replies and
    scored everything. It stopped on completion.
- GPU time was about 15 minutes of H100 time across the vLLM sampler and the scorer, roughly $5 to $10. This is
  estimated from run time, not from billing.
- No Claude or other API calls.

## Files

- `matched.py`: builds the items (`... matched.py items`), and generates and scores them (`modal run -n ...
  matched.py::run`).
- `analyze.py`: everything above, plus the figures (CPU only).
- `<task>_items.jsonl`: the 200 items with all three prompts.
- `<task>_replies.jsonl`: all 1,200 replies per task.
- `<task>_scored.jsonl`: replies with correctness, tokens, per-token log-odds and peaks for both probes.
- `detector_scores.jsonl`: all five detectors' scores per reply.
- `analysis.json`: all numbers.
- `figures/scores.png`: score distributions per detector.
- `figures/traces.png` and `traces.md`: per-token traces for identical-reply pairs, three per task, with full tables.
- The TF-IDF models used are pickled in `analysis/tfidf_baseline/models/` (`analysis/tfidf_baseline/chosen_models.py`
  re-fits them and checks them against the baseline's saved scores).
