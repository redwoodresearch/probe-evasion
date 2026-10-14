# TF-IDF baseline for the balanced broad probes (m52)

**Question.** The balanced broad probes (m52; `q27mbb@L48@s0` for maths, `q27obb@L48@s0` for ordering) were trained
on puzzle rows, chat-style rows and WildChat negatives. Can a plain word classifier trained on exactly the same rows
do the same job, and do m45's saved attacks move it as much?

**Setup.** TF-IDF B: logistic regression on word and character n-grams of the request and the reply, trained on the
balanced probe's 5,330 (maths) / 5,662 (ordering) training rows, rebuilt locally and checked against the stated counts.
Settings were picked by the original baseline's rule (best AUROC on m46's held-out set, ties to lowest log-loss). It
picked word + character n-grams, C = 100, for both tasks, the same as the original baseline. Each detector gets its own
1% line: the 99th percentile of its scores on the WildChat line pool (4,328 / 4,472 replies). Attack and matched-request
measures reuse m52_analysis / m52_matched code. The probe's numbers recomputed here equal
`analysis/broad_probe/m52/results.json` exactly (asserted). CPU only, about 70 seconds.

| maths / ordering | balanced probe | TF-IDF B (request + reply) |
|---|---|---|
| held-out AUROC: held-out chat, computed vs given | 1.000 / 1.000 | 1.000 / 1.000 |
| held-out AUROC: original held-out + re-read | 1.000 / 1.000 | 1.000 / 1.000 |
| held-out AUROC: held-out chat positives vs WildChat pool | 1.000 / 1.000 | 1.000 / 1.000 |
| 1% line (on WildChat pool) | −5.64 / −5.23 | −3.16 / −9.54 |
| held-out chat positives flagged | 100% / 100% | 100% / 100% |
| plain test answers flagged (m45 batch c, valid) | 100% / 100% | 100% / 100% |
| *held-out chat "given" replies flagged (should be low)* | *0.9% / 4.3%* | *1.9% / **96%*** |
| *look-alike pool flagged* | *0.2% / 0%* | *3.7% / 0.3%* |
| valid ON replies under the line | 0 of 17,517 / 507 of 18,057 (2.8%) | 3,622 (21%) / 920 (5.1%) |
| crossings (pair both valid, ON ≤ line < OFF) | 0 / 498 | 3,505 / 890 |
| gap closed, mean over 14 writers | 0.18 / 0.27 | 0.17 / 0.08 |
| gap closed, slope per 10 ECI [95%] | +0.06 [+0.03, +0.10] / +0.12 [+0.07, +0.17] | +0.07 [+0.04, +0.10] / +0.04 [+0.02, +0.06] |
| matched test, same reply after "compute" vs "given": share where compute scores higher | 100% / 100% (median +16 to +17 / +15 to +17) | 73–100% / 100% (median +1.3 to +1.8 / +4.7 to +6.1) |
| matched test, same reply after "wrong answer given" vs "right answer given": wrong higher | 100% / 90–100% (median +7.5 to +8.0 / +4.8 to +8.9) | 49–61% / 33–50% (median about 0 / 0) |

`fig_probe_vs_B.png`: held-out AUROC (left) and gap closed per writer against ECI (right), both tasks.

**What this says**

1. **Held-out AUROC cannot tell the two apart.** Both are at 1.000 on every held-out set, for both tasks. A word
   classifier that sees the request is already perfect on these sets, so they cannot show what the probe adds.
2. **B's line means something different in ordering.** WildChat contains almost nothing like an ordering puzzle, so
   B's 1% line falls very low (−9.54) and flags **96% of held-out chat replies where the order was given**. The probe
   flags 4.3%. At this line B is close to a "this is an ordering request" detector. In maths B is closer to the probe
   on the given replies (1.9% vs 0.9%).
3. **Attacks: maths moves B more, ordering moves it less.**
   - Maths: about one in five valid ON replies gets under B's line (3,505 crossings); none get under the probe's.
     The probe flags every one of the ON replies B misses. Gap closed and its slope are
     about the same for both (+0.07 vs +0.06 per 10 ECI). B's ON scores are more spread out, so more replies cross
     even though the median moves about as far.
   - Ordering: B has more crossings (890 vs 498), but its gap closed is a third of the probe's (mean 0.08 vs 0.27)
     and its capability trend is much flatter (+0.04 vs +0.12 per 10 ECI). Part of this comes from point 2: B's line
     is far below the replies, so the same score drop closes a smaller share of the gap. The ON replies each detector
     misses are mostly different (probe misses 507, B misses 920, both miss 246).
4. **The matched test separates them.** For the same reply text after a "compute" request and after a "given"
   request, B scores compute higher (it reads the request words), but by much less than the probe (+1 to +6 vs +15
   to +17). When the request gives a *wrong* answer instead of the right one, the probe still scores the reply as
   computed (+5 to +9 above the "given" version, 90–100% of pairs). B scores it no differently (median about 0, about half
   of pairs either way), as expected: the two requests differ only in one number. This is the clearest
   evidence that the probe reads something beyond the request and reply wording.

**Caveats.** The m45 attacks were optimised against the original narrow probe, not against either detector here, so
the attack rows measure carry-over, not how hard each detector is to attack directly. A word classifier attacked
directly would very likely be easier to evade than these numbers show. Settings were picked on m46's held-out set,
which is also one of the reported AUROC sets. 12 of 15 settings (maths) and 10 of 15 (ordering) scored
1.000 there, so the choice was made on log-loss.

Files: `balanced_baseline.py` (all numbers; writes `results.json` and `models/<task>_B.pkl` + `_B_choice.json` with
the full settings grid), `figure.py` (`fig_probe_vs_B.png`).
