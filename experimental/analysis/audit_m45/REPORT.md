# Independent audit of the main switch sweep (m45)

Recomputed from the raw `test_checked` rows with independent code (scripts in this folder; `extract.py` first copies the
raw rows into compact files in the session scratchpad, and the other scripts take that folder as their argument).

## Summary

**The headline reproduces.** Per-arm scores, per-writer means, slopes and intervals match `RESULT.md`, `analysis.json`
and `analysis_d.json`; no discrepancy is larger than 0.01. The switch, the pairing, the freeze rule, the probe readout and
the 1% lines all check out exactly. No test problems leak into the search, the probe's training data or the templates.

These items did not fully check out. None changes the headline.

| # | Issue | Severity |
|---|---|---|
| A | `RESULT.md` called the 1% line "the score that 1% of ordinary non-task answers exceed". It is set on a narrow pool of hard look-alikes, read with no instruction in view; on ordinary chat it flags 74% (maths) and 100% (ordering). | low-medium (misleading wording; fixed in RESULT.md) |
| B | Two figures in `RESULT.md` covered batch c only beside 14-writer claims: best ON medians (with batch d: Fable 5.1 9.8 maths / 5.0 ordering as a writer mean, one Fable ordering chain 4.0), and crossings (all 14 writers: 0 of 17,376 / 17,373). | low (stale; fixed in RESULT.md) |
| C | The full maths validity check rejects every reply containing "²" (e.g. "5600 m²"), 730 rows; where the full check decides validity this marks correct answers invalid with no stored reason (239 rows batch c, 76 batch d). All at B = 0, 4 or 8, so B = 16 is untouched. | low |
| D | Correctness-only re-judge: of 30 sampled flips, 28 correct, 1 wrong ("twelve thousand nine hundred fourteen" for 12994), 1 lenient. Base64 handled inconsistently (21 of 154 correct-decoding replies flipped). Reported sensitivity numbers reproduce. | low |
| E | GPT-5 nano ordering chain 1 hard-codes practice-problem names ("Ava, Eli, Dev, Fay, Cleo") in the shared part; it passed the shared-part check; only 24–37% of pairs valid. Low-ECI end, doesn't drive the trend. | low |
| F | Batch d's maths line −6.505 vs batch c −6.530 (re-scoring the same pool); same reply scores up to 0.08 / 0.13 apart between batches. Negligible effect on S_offref. | negligible |
| G | Methodology page's "+0.5 to +0.7" lift from a harmless instruction is true of pool medians (+0.74 / +0.51); the 1% line itself moves +0.87 / +0.26. | negligible (wording) |

**Added robustness checks all hold.** In-view lines (−5.66 / −7.80), validity ignored, ON and OFF medians over each
side's own valid replies, and invalid ON counted as no movement: 14-writer slopes stay within maths +0.055 to +0.070 and
ordering +0.113 to +0.127, every interval above 0. Only batch-c maths with validity ignored is fragile: +0.056 [−0.003, +0.113].

## Details

### 1. Recomputing the scores (`recompute.py` → `recompute.json`)

All 168 arm × B points per task match within 1e-6 except 8 arms (max difference 0.0063: maths gemini38flash ch0 B4,
fable51 ch2 B16; ordering gpt5nano ch1 B4/B8/B16, gpt6sol ch0 B0, llama ch1 B4/B16). All 8 come from the mission's
stricter leak rule dropping 1–4 test problems where a template's example number or name equals a test answer.

| Fit | Audit | Reported |
|---|---|---|
| Maths, 14 writers | +0.066 [+0.031, +0.109] | +0.066 [+0.032, +0.108] |
| Ordering, 14 writers | +0.125 [+0.079, +0.178] | +0.125 [+0.078, +0.178] |
| Maths, batch c | +0.056 [+0.008, +0.106] | +0.056 [+0.009, +0.106] |
| Ordering, batch c | +0.124 [+0.070, +0.175] | +0.124 [+0.070, +0.173] |
| Batch c without the 4 Anthropic writers | maths +0.014, ordering +0.081 | match |
| 14 writers without the 5 Anthropic writers | maths +0.048 [−0.009, +0.103], ordering +0.091 [+0.038, +0.145] | match |

B = 0, 4, 8 slopes (batch c) match to 4 decimals; the B = 16 per-writer table matches to 2 decimals; Opus 5.5's effort
curves match; Spearman permutation p for maths 0.040 matches; correctness-only batch c +0.053 [+0.001, +0.106] vs
reported +0.053 [+0.002, +0.105].

### 2. The 1% lines

99th percentile of `calibration_scores.jsonl`: −6.530 (maths, 1,301 replies), −8.0605 (ordering, 1,697), matching
`lines.json`. Calibration replies are read after their own look-alike question with no instruction; attack replies are read
after instruction + problem. Using the in-view lines gives slopes +0.070 / +0.127, so the choice is conservative. No valid ON
reply falls under either line.

### 3. The switch

Replacing "Style switch: ON" with "Style switch: OFF" once in the ON prompt gives exactly the OFF prompt: 0 violations in
83,000 maths and 82,000 ordering pairs. Every prompt starts with its arm's frozen instruction and contains the problem. " ON"
and " OFF" are single tokens (ids 6004, 17086).

### 4. Pairing

Every frozen arm has rows (including parity side arms); each has exactly 500 ON and 500 OFF rows (100 problems × seeds
10–14), no duplicates, the same 100 problems. Arms with no template yet are scored 0: maths GPT-5 nano ch0 B0, Qwen3.5-397B
ch1 B0; ordering GPT-5 nano ch0 and ch1 B0, Qwen3.5-9B ch1 and ch2 B0.

### 5. Leakage (`leak_freeze.py` → `leak_freeze.json`)

No test problem appears in the 40 practice problems (exact text or stem); one harmless numeric twin (X003 "91 mugs cost
$64 each" and E08 "64 boxes with 91 bolts", both 5824). No test problem in the probe's training, held-out or calibration data
or m19's plain test set. No frozen template contains test-problem specifics (two coincidences: Sonnet ch0 B4 example
"2 million 450 thousand" where 450 is X078's answer, already excluded by the mission; Gemini ch1 B16 "78 sweets", not an answer).

### 6. The freeze rule

All 168 arm × B points per task: the frozen template is `state.json`'s instruction for the latest round ≤ B that ran,
`written_round` matches, and it is that round's single accepted proposal. 0 issues. Carried forward: GPT-5 nano ch0 at
B4/B8/B16 and Opus 5.5 ch2 at B4 (all from round 3). No selection by probe score.

### 7. Validity, read by hand (`validity_sample.py` → `validity_sample.json`)

40 rows at B = 16 (mostly the top six writers; ON/OFF, valid/invalid, both tasks): 0 of 40 errors against the stated rules.
By correctness alone, 1 error in 18 ordering rows (a correct ON answer rejected for length, 45 words vs 40). Ordering
rejections at B = 16 are almost all wrong orders or corrupted names; length rejections ≤ 6 per writer and version. Four of
five maths OFF-invalid rows are Sonnet ch1 OFF replies in number words with wrong arithmetic.

### 8. Probe readout (`readout.py` → `readout.json`)

Recomputed the peak for 3,708 rows (every 97th): EMA alpha 0.5 from 0 over stored per-token logits equals `peaks_visible`
exactly; the instruction-removed `peaks` also matches. The stored peak is not the single highest token (0 rows match that).
Not tested: whether the stored logits themselves are right (needs the model on a GPU).

### 9. Other checks (`extra.py` → `extra.json`)

- Few valid pairs: Llama ch2 maths 9% (base64, documented); GPT-5 nano ch1 ordering 37% (issue E); two ordering arms with 0
  valid pairs scored 0 (Sonnet ch0 B4 spelled-out names; Llama ch1 B0 initials only).
- Under the line: ordering 318 rows (295 ON, 23 OFF), all invalid, all GPT-5 nano (RESULT.md said 319). No valid row under
  the line anywhere. OFF median never within 1 log-odds of the line (smallest gap 10.9).
- Switch doing nothing: ON and OFF replies identical in 67% of pairs for Qwen3.5-9B maths, 41% GPT-5 nano maths, ~0% top writers.
- Seeds: ON and OFF share seeds (good pairing); every arm uses the same seeds (the seed-and-problem bootstrap covers it).
  Plain and neutral replies identical between batches in 998/1,000 (maths) and 1,000/1,000 (ordering).
- Writers and providers: every proposal from the pinned model; only provider switch is Qwen3.5-9B (DeepInfra for 12
  proposals per task, then Parasail), documented. Refusal stops: Opus 5.5 32 (all maths), GPT-5 nano 2, Sonnet 5.5 2, Fable 5.1 2.
- Arms hit by issue C: Haiku 4.5 ch2 B4 ON/OFF, ch2 B8 ON, ch1 B4 OFF; Fable 5.1 ch0 B0 ON; GPT-6 Astra ch2 B8 ON. Other
  arms where the full check decided validity were legitimate (Llama ch0 B4 ON: 68 hedged "Perchance…" replies; Sonnet ch2
  B16 OFF: 17 replies showing working, caveat 9).
