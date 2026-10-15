# Write-up outline (draft)

## Title

**Question:** Does it matter who wrote a probe's training data when someone tries to prompt their way past it?

The post commits to answering this question, with one of three final titles chosen by the decision rule below.

## Opening (draft text; [M] = replaced by final matched-pair numbers; every comparison is off-policy first)

> Activation probes are a cheap way to monitor a model. A linear classifier reads the model's internal activations and
> flags replies that show some concept. For rare behaviours there usually aren't enough examples of the model itself
> doing the thing, so probes are often trained on examples written by another model ("off-policy" data) instead of the
> model's own outputs ("on-policy" data).
>
> We ask whether that choice changes how easily the probe can be evaded by prompting alone. The attacker controls the
> prompt and sees probe scores. The model's weights never change. A judge, or code for maths, confirms the concept is
> still clearly present.
>
> We build seven pairs of probes on Qwen3.5-27B, one pair per concept. The two probes in a pair are identical except
> for who wrote the training text: Claude, or Qwen itself.
>
> At the usual setting, where the flag line flags 1% of ordinary replies, the attack beats both probes on the tone
> concepts. On maths it gets past only the off-policy probe, and on chemistry it gets past neither.
> They differ in what it costs a defender to catch the attack by lowering the line.
> - **Sarcasm:** catching 90% of the attack costs 10.8% of ordinary replies flagged for the off-policy probe and 3.8%
>   for the on-policy probe [M].
> - **Anger:** the order reverses, 5.0% vs 8.3% [M]. Qwen's own angry replies lean on a few stock words, so its
>   training text is narrower.
>
> Our working hypothesis, tested in Result 3: authorship alone does not decide it, and the better predictor is how
> varied the training text is.

**Figure 1:** one row per concept, seven rows.
- **The measure:** "defender's cost". This is the share of ordinary replies flagged when the line is lowered enough to
  catch 90% of the attack.
- **Shown:** the off-policy probe and the on-policy probe as two dots per row.
- **Attacks:** only the attack aimed at each probe.
- **Error bars:** 95% intervals from resampling tasks.
- **When nothing gets past a probe** (chemistry, the maths on-policy probe): the dot sits at the floor, marked "nothing
  got past".

## Background

- **How a probe monitor works.** One score per reply; a line set so 1% of the model's ordinary replies are flagged.
  An "evasion" is a reply that shows the concept but scores under the line.
- **Why off-policy data is common.** It is used for rare behaviours, e.g. the Neural Chameleons concept probes, and
  Apollo's deception probes trained on scripted honest/dishonest pairs.
- **Prior work** (one paragraph, each item cited once):
  - off-policy data hurts how well probes generalise to new kinds of text (Kirch et al. 2026);
  - style prompts can break even on-policy probes, and training on varied styles fixes it (Kumar 2026);
  - evasion with fine-tuning or optimised gibberish suffixes (Neural Chameleons; Obfuscated Activations);
  - in-context feedback alone isn't enough to evade (Keenan et al. 2026).
- **What is new here:** the two probes look equally good on ordinary held-out text; we measure the gap under a search
  for evasions.

## Setup

- **Model:** Qwen3.5-27B with reasoning off.
- **Probes.** One linear probe per concept at layer 48 of 64.
  - A reply's score is its peak: the probe scores each word, the scores are smoothed over the last few words, and the
    highest smoothed value counts.
  - The probe is trained on that same score, with three random seeds.
- **Seven concepts:**
  - maths done in the head (correctness checked in code);
  - chemistry;
  - sarcasm, flattery, condescension, anger, criticism.
- **Matched pairs.** Both training sets per concept share the same requests, the same three kinds of example, the same
  labelling judges and the same counts. The three kinds are: shows the concept; ordinary; and look-alikes that resemble
  the concept but lack it.
- **On-policy data.** Qwen was prompted to write each kind of example, with the same instructions Claude got.
- **The line:** the flag threshold, set so 1% of Qwen's ordinary replies are flagged.
- **Equal starting quality** (a precondition, checked before any attack). For both probes in a pair, report AUROC
  (1 = perfect separation, 0.5 = chance) on Qwen's own held-out replies, and the share of plain concept replies caught
  at the 1% line. Pairs that differ by more than 0.02 AUROC are also compared with each probe's line moved so they catch the
  same share of plain replies.
- **The attack:**
  - a 5-round Claude prompt search, aimed separately at each probe;
  - the winner is fixed, then tested on 100 unseen tasks × 5 samples, next to the plain prompt;
  - each reply is re-read with the attack instruction deleted, so only its own wording counts.
- **Measures:**
  - **Primary:** defender's cost (as defined under Figure 1).
  - **Secondary:** the share of attempts that evade at the 1% line, minus the plain prompt's share, in percentage
    points.
  - Valid replies only: the concept must be judged clearly present.
  - Maths and chemistry rarely leak, so for them the decision uses the secondary measure (an interval on the
    difference in evasion rate).
  - 90% catch was fixed before any matched results came in; other targets are in the appendix.

## Results (one bold takeaway + figure each)

1. **[Pending]** The direction of the gap across all seven pairs (Figure 1).
   - **Current numbers:**
     - sarcasm: off-policy costlier;
     - anger: on-policy costlier;
     - maths: only the off-policy probe leaks (2.6% vs 0% of answers at the 1% line);
     - chemistry: neither leaks;
     - flattery, condescension, criticism: running.
2. **At the 1% line both probes are beaten; they differ in how far below the line the attack lands.**
   - Sarcasm: the attack raises evasion by 48 (off-policy) vs 45 (on-policy) percentage points at the 1% line, but by
     24 vs 3 at a 5% line.
   - Figure: attack-minus-plain bars per pair.
3. **Training-text variety predicts which probe is easier to evade.**
   - Per pair, plot the defender's-cost gap against the difference in variety between the two training sets:
     marker-word rates, number of distinct phrasings, register mix.
   - **Example:** Qwen's angry examples say "furious" in 62% of cases vs Claude's 26%.
   - **Required test:** re-balance each pair so both sets have the same marker-word rates. If the gap shrinks, variety
     is the cause.
4. **It isn't Claude dodging its own writing.**
   - **Maths:** a probe trained on Gemma-2-9B-written text also leaks (4.3% vs 0% of answers at the 1% line,
     off-policy vs on-policy).
   - **Sarcasm:** a search run by Qwen3.5-122B instead of Claude.
   - **Status:** optional. If it doesn't land, ship without it and list it as a limitation.

## Decision rule (fixed now)

The seven pairs are maths, chemistry, sarcasm, flattery, condescension, anger and criticism. A pair counts as
"informative" if the attack gets anything past either probe. A pair "favours off-policy" if the off-policy probe's
defender's cost is higher, with an interval excluding zero.

In the rules below:
- "Informative" means at least 5 valid evasions on either probe of the pair, out of 500 attempts.
- "The gaps shrink" means the median absolute defender's-cost gap falls by at least half after balancing.

**Title A, claim:** "Probes trained on another model's text cost far more to defend against prompted evasion."
- Requires that at least 4 informative pairs favour off-policy and none favours on-policy. Maths does not have to be
  one of the 4.

**Title B, variety:** "What makes a probe easy to prompt past is narrow training text, not who wrote it."
- **When:** pairs favour both directions, or fewer than 4 favour off-policy, AND the marker-balanced test shrinks the
  gaps.

**Title C, mixed:** "Who wrote a probe's training data changes how easily prompting evades it, but not in a
consistent direction."
- **When:** gaps are large in both directions and balancing does not explain them, or the balanced test doesn't
  finish.

**Title D, negative:** "Who wrote a probe's training data barely changes how easily prompting evades it."
- **When:** no informative pair shows a gap with an interval excluding zero.

## Discussion

- **Recommendation:**
  - evaluate monitor probes against a prompt search;
  - report defender's cost, not just held-out accuracy;
  - prefer varied training text.
- **When on-policy examples don't exist (rare behaviours):** the variety result, if it holds, says what to do instead:
  make off-policy data varied.
- **Limits:**
  - one model;
  - every concept is visible to a human reader;
  - the attacker sees probe scores, which makes this an upper bound;
  - seven pairs is a small sample.

## Appendix (for the sceptical reader)

- **A. Validity:**
  - how presence is checked (two judges, one not told which concept was requested; code for maths);
  - counts restricted to full-strength replies (both judges give the top score);
  - sincere replies written in the attack's style are not flagged;
  - a check that the Claude judges don't favour Claude-style writing.
- **B. Measurement traps that changed numbers:**
  - calibration replies that secretly contained the concept, which raised the line;
  - probes trained on one scoring rule but deployed with another.

  Each with before/after numbers.
- **C. Attack instruction deleted vs left in:** both numbers, and why "deleted" is the headline.
- **D. Other catch targets:** defender's cost at 50%, 75% and 99% catch, plus full threshold curves.
- **E. Is Qwen's training data really "on-policy"?**
  - Qwen wrote its examples under instructions to show each concept, so they are not its natural replies.
  - We show how close they are to its plain replies (length, style, marker words), and we say so in the main text.
- **E2. Training-data checks:**
  - each set written by a single author;
  - length, style and marker distributions;
  - a word-only classifier baseline;
  - the prompts used to produce Qwen's examples.
- **F. Statistics and setup details:**
  - each search attacks the seed-0 probe; seeds 1-2 are scored for spread;
  - valid-reply counts are reported for both probes (the same replies, so the counts match);
  - seeds;
  - task-resampled intervals;
  - multiple comparisons across 7 pairs (Holm correction);
  - independent recounts.
- **G. Search fairness:** the same budget and feedback for both probes. Telling the model about the probe, without a
  search, changes nothing.
- **H. Full prompts, example replies, and the maths case** ("Yes, your 1288 dollars is exactly right": the computed
  answer phrased as if the user had supplied it).

## Cutting (not in this post)

- **Concepts where no working probe exists:** deliberately wrong maths answers, and hiding facts in summaries.
- **Third-party deception probes** (UK AISI's), which mostly read the instruction, not the reply.
- **Retraining the sarcasm probe against found attacks:** each retrained probe fell to a new search. Candidate for a
  separate post.
- **A small gap in a probe on Gemma-2-9B's own internals, at its weakest layer,** which no one would deploy. This is
  separate from the Gemma-written training text used in Result 4.
- **A 40-round search against one condescension probe:** a large effect, but confined to one layer, and the
  condescension came out milder.
- **Long-context false-alarm drift:** a modest effect, not about authorship.
- **Making probes fire on harmless replies on purpose:** a different question.
- **Attackers that never see probe scores, and weaker models writing prompts:** interesting, but off the main question.
- **Brainstormed attacks:** noise, temperature, personas, jailbreaks.
