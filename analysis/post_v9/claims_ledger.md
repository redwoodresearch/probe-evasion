# Claims ledger: "More capable models write better probe-dodging prompts"

Every claim the post makes, the number behind it, where it comes from, whether it has been independently checked, and
the main way it could be wrong. The draft says only what this table supports.

Framing (Nabil, 13 Oct): headline = plain-referenced gap closed (median OFF − ON over pairs with both answers correct,
÷ plain median − line), all five probes, 14 writers, strict 1% line (look-alike pool). Slopes per 10 ECI, two-stage
bootstrap (writers, then searches). "Pending" = being computed or rechecked now (analysis/post_v9/numbers/).

## Headline

| # | Claim | Number | Source | Checked? | Main way it could be wrong |
|---|---|---|---|---|---|
| H1 | Stronger writers move the probe further, on every probe | Gap-closed slope after 16 rounds: belief tracking +0.12 [+0.07, +0.18], acrostic +0.17 [+0.13, +0.21], contradiction +0.13 [+0.09, +0.18]; maths and ordering pending in this measure (OFF-referenced: +0.07, +0.13) | m49 RESULT_14writers.md; m45 RESULT.md | m45 OFF-referenced: audited (analysis/audit_m45). New tasks: m49 self-reported for 14 writers, reviewer-checked for 11; recheck pending | Lab confound (H3); one monitored model; ECI as the capability axis |
| H2 | It holds under other measures | OFF-referenced and raw log-odds slopes positive on all five | m45, m49 framings | m45 audited; m49 reviewer-checked (11 writers) | Measure choice after seeing results (both reported) |
| H3 | Not just one lab | Without the 5 Anthropic writers: maths +0.05 [−0.01, +0.10], ordering +0.09 [+0.04, +0.15] (OFF-referenced); new tasks pending | analysis/post_v8, audit | maths/ordering audited | Maths fails it; top end is mostly Anthropic |
| H4 | Feedback matters | Round-0 slopes: maths +0.03, ordering +0.06 (OFF-ref); new tasks +0.02, +0.02, −0.01 (none); gap-closed for maths/ordering pending | m45, m49 | m45 audited; m49 pending | "Zero-shot" writers still know the probe's readout |

## What the probe reads

| # | Claim | Number | Source | Checked? | Main way it could be wrong |
|---|---|---|---|---|---|
| R1 | The probe reads more than words | Identical replies after "compute" vs "answer given" requests: probe drops 14–15 log-odds in 100% of pairs; word classifiers move 0.4–3.3 | m51 RESULT.md | Single run, not independently checked | Probe may read "answer already in context" rather than "computing" |
| R2 | Held-out accuracy alone doesn't show that | A prompt-only word classifier gets held-out AUROC 1.000 on all five | analysis/tfidf_baseline | Single run | TF-IDF settings swing results (ties at 1.000) |
| R3 | Attacks partly transfer to a word classifier | Maths: yes (more than the probe); ordering: mostly no; belief tracking: similar | analysis/tfidf_baseline | Single run | Sensitive to TF-IDF settings; transfer, not adaptive |

## Movement vs evasion

| # | Claim | Number | Source | Checked? | Main way it could be wrong |
|---|---|---|---|---|---|
| E1 | Maths and ordering: no correct reply gets under the strict line | 0 of 17,376 / 17,373 correct pairs | m45, audit | Audited | The strict line is the defender's best case (flags 74–100% of everyday chat) |
| E2 | New tasks: many crossings for top writers | Net crossings: belief tracking Sonnet 5.5 51%, Fable 5.1 33%; contradiction Opus 5.5 67%, Fable 5.1 62%; acrostic ≤ 7% | m49 | Pending recheck | OFF sits close to the line on belief tracking/contradiction (crossings inflated); rankings unstable across searches |
| E3 | Dodging costs accuracy on maths, little on ordering | Strong writers lose 17–30 points (maths), 0–10 (ordering) | analysis/post_v8 cost.png | Recomputed; matches Codex's spot-check | Within-template: wrong maths answers score lower |

## Volitional (no outside writer)

| # | Claim | Number | Source | Checked? | Main way it could be wrong |
|---|---|---|---|---|---|
| V1 | Told about the probe, Qwen doesn't dodge with reasoning off | S −0.01 to +0.03, 0 crossings in 8,000 attempts | m38 RESULT.md | Pending spot-check | Older design (instruction removed) |
| V2 | With reasoning on, the answer-only reading moves but the thinking still shows the work | 27B answer 0.40–0.47 vs 0.25 control; whole output ~0.10 | m38 | Pending spot-check | Older probes; two model sizes |
| V3 | Self-revision with scores helps (122B, reasoning on) | 0.58 vs 0.32 on a fresh problem, 91% correct | m38 | Pending spot-check | One model; reasoning on |

## Design facts the post states

| # | Claim | Source | Checked? |
|---|---|---|---|
| D1 | ON and OFF prompts differ only in the switch word | audit §3 | Audited (0 violations in 165k pairs) |
| D2 | Templates tested at round B are the ones written at B (no selection) | audit §6 | Audited |
| D3 | No test problem in search, probe training or templates | audit §5 | Audited |
| D4 | Probe readout = peak of EMA(α 0.5) over reply tokens, prompt in view | audit §8 | Audited |
| D5 | The strict line = 99th percentile of the look-alike pool | audit §2 | Audited |

## Open

- Gap-closed (plain-referenced) slopes for maths and ordering with 14 writers, B = 16 and B = 0 — pending.
- Without-Anthropic fits for the three new tasks — pending.
- A pooled cross-task slope — pending (may not be appropriate).
