# Claims ledger: "More capable models write better probe-dodging prompts"

Every claim the post makes, the number behind it, where it comes from, whether it has been independently checked, and
the main way it could be wrong. The draft says only what this table supports.

Framing (Nabil, 13 Oct): headline = plain-referenced gap closed (median OFF − ON over pairs with both answers correct,
÷ plain median − line), all five probes, 14 writers, strict 1% line (look-alike pool). Slopes per 10 ECI, two-stage
bootstrap (writers, then searches). Independent recomputation: analysis/post_v9/numbers/NUMBERS.md (13 Oct), which
matches m45, m49, m38 and m50 everywhere it overlaps.

Grading (15 Oct): every task now uses one rule for both the probe-evasion score and the accuracy cost: a rule-based
code check and a Claude Sonnet 5.5 reader that is not told the answer must both accept the answer, within the length
limits. Arithmetic was regraded to this rule (`numbers/arithmetic_grading.py`); it moved its slope from +0.063 to
+0.062 and its accuracy costs by under 2 points. The other tasks were already graded this way.

## Headline

| # | Claim | Number | Source | Checked? | Main way it could be wrong |
|---|---|---|---|---|---|
| H1 | Stronger writers move the probe further, on every probe | Gap-closed slope after 16 rounds: maths +0.062 [+0.030, +0.102], ordering +0.099 [+0.062, +0.140], belief tracking +0.121 [+0.068, +0.182], acrostic +0.165 [+0.129, +0.206], contradiction +0.129 [+0.091, +0.178]. Average of the five: +0.115 [+0.091, +0.148] | numbers/NUMBERS.md §1, §3 | Yes (recomputed from raw rows; matches m45 S_toggle_valid and m49) | One monitored model; ECI as the axis; the average is not one common slope (maths below, acrostic above) |
| H2 | It holds under other measures | OFF-referenced and raw log-odds slopes positive on all five; maths under m19's old strict rule +0.063 | m45, m49 framings; numbers §1 | Yes | Measure chosen after seeing results (all reported) |
| H3 | Not just one lab | Without the 5 Anthropic writers: maths +0.049 [−0.004, +0.099], ordering +0.075 [+0.033, +0.119], belief tracking +0.092 [+0.027, +0.169], acrostic +0.149 [+0.102, +0.214], contradiction +0.128 [+0.084, +0.192]; average +0.099 [+0.071, +0.137]. Within-lab slopes positive on all five (contradiction's interval crosses 0). Spearman ρ 0.67–0.88, p ≤ 0.01 | numbers §2, §3 | Yes | Maths alone fails it; top end mostly Anthropic |
| H4 | The trend is much flatter before feedback | Round-0 slopes: maths +0.021 [+0.004, +0.040], ordering +0.047 [+0.029, +0.072] (a third to a half of round 16); new tasks +0.017, +0.019, −0.009 (none) | numbers §1 | Yes | Word as "much flatter", not "flat": maths/ordering exclude 0. "Zero-shot" writers still know the readout |

## What the probe reads

| # | Claim | Number | Source | Checked? | Main way it could be wrong |
|---|---|---|---|---|---|
| R1 | The probe reads more than words | Identical replies after "compute" vs "answer given": probe drops in 100% of pairs, pooled median 14.6 (maths), 13.8 (ordering), 12.3 (belief tracking), 6.0 (contradiction, only 52 pairs) log-odds; word classifiers move 0.4–3.3 on new phrasings (4.7–8.1 on training phrasings), inconsistent in sign. Identical correct replies after a wrong vs a right inserted answer: probe scores the wrong-answer version higher in 97% / 96% / 100% / 99% of pairs (pooled; per phrasing 90–100%, the 90% from 49 pairs), median +5.7 / +5.4 / +9.6 / +3.8; word classifiers' median gap +0.0 / −0.1 / +0.1 / +0.0 (single pairs up to ~2.7). Acrostic not testable: 0 identical-reply pairs and no wrong-answer version; on the looser test word classifiers match the probe (AUROC 1.00 vs 0.95) | m51 RESULT.md (parts 1–2); recomputed in analysis/tfidf_baseline/for_james/ (numbers.json) | Recomputed from saved scores (for_james, 14 Oct); single run | Could read "answer disagrees with context" rather than "computing" (belief tracking fits computing best) |
| R2 | Held-out accuracy alone doesn't show that | A prompt-only word classifier gets held-out AUROC 1.000 on all five | analysis/tfidf_baseline | Single run | TF-IDF settings swing results (ties at 1.000) |
| R3 | Attacks partly transfer to a word classifier | Maths: yes (more than the probe); ordering: mostly no; belief tracking: similar | analysis/tfidf_baseline | Single run | Sensitive to TF-IDF settings; transfer, not adaptive |

## Movement vs evasion

| # | Claim | Number | Source | Checked? | Main way it could be wrong |
|---|---|---|---|---|---|
| E1 | Maths and ordering: no correct reply gets under the strict line | 0 of 17,760 / 17,373 correct pairs | numbers §4; audit | Yes | The strict line is the defender's best case (flags 74–100% of everyday chat) |
| E2 | New tasks: many crossings for top writers | Forward / reverse crossings: belief tracking 2,971 / 190, acrostic 244 / 29, contradiction 6,565 / 554. Top net rates: belief tracking Sonnet 5.5 51%, Opus 5.5 34%, Fable 5.1 33%; contradiction Opus 5.5 67%, Fable 5.1 62%, Opus 4.8 60%; acrostic Fable 5.1 7% | numbers §4 | Yes | OFF sits close to the line on belief tracking/contradiction; lines set by ~4–5 replies (pools of 385–510); rankings unstable across searches |
| E3 | Dodging costs accuracy on maths, little elsewhere | Strongest maths writers lose 17.0–30.3 points; ordering 0.7–10.0; new tasks mostly < 8 (acrostic exceptions: Qwen3.5-397B 20.5, Fable 5.1 11.1, GPT-6 Astra 9.6). No significant slope of cost on ECI | numbers §5 | Yes | Within a template, wrong maths answers score lower |

## Volitional (no outside writer)

| # | Claim | Number | Source | Checked? | Main way it could be wrong |
|---|---|---|---|---|---|
| V1 | Told about the probe, Qwen doesn't dodge with reasoning off | S −0.011 to +0.031; 0 crossings in 8,000 told attempts (12,000 with controls) | m38; numbers §6 | Yes | Older design (instruction removed) |
| V2 | With reasoning on, the answer-only reading moves but the thinking still shows the work | 27B answer 0.40 / 0.47 / 0.42 vs 0.25 control; whole output ~0.10 | m38; numbers §6 | Yes (27B recomputed from raw rows) | Older probes; two model sizes |
| V3 | Self-revision with scores helps (122B, reasoning on) | 0.58 vs 0.32 on a fresh problem, 91% correct | m38; numbers §6 | Partly (read from m38's analysis.json, not recomputed from raw rows) | One model; reasoning on |

## Design facts the post states

| # | Claim | Source | Checked? |
|---|---|---|---|
| D1 | ON and OFF prompts differ only in the switch word | audit §3 | Yes (0 violations in 165k pairs) |
| D2 | Templates tested at round B are the ones written at B (no selection) | audit §6 | Yes |
| D3 | No test problem in search, probe training or templates | audit §5 | Yes |
| D4 | Probe readout = peak of EMA(α 0.5) over reply tokens, prompt in view | audit §8 | Yes |
| D5 | The strict line = 99th percentile of the look-alike pool, all five probes | audit §2; numbers §1 | Yes |

## Wording rules from the checks

- Round 0: "much flatter" on maths and ordering, "none" on the new tasks; never "flat everywhere".
- The five-probe average is "the average across probes", always shown next to the five slopes.
- Gap closed is a share of a different log-odds gap per probe (about 17–20 for maths/ordering, 8–11 for the new tasks).
- Crossing rates on belief tracking and contradiction are not comparable with maths and ordering.
- m50's crossing-share trend is 11 writers if quoted.
- Still unchecked: R1–R3 (single runs); V3 only partly.
