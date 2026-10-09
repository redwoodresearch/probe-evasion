# Write-up outline (draft for James's EOD request)

## Title

Matched results (m20) are still running, so the title is phrased as the question the post will answer, with the claim
version ready if the matched pairs confirm it.

- **Question version (use today):** Are activation probes trained on another model's text easier to evade by prompting
  than probes trained on the model's own outputs?
- **Claim version (if m20 confirms):** Probes trained on off-policy data are more exploitable by adversarial prompting
  than probes trained on the model's own outputs.

**One-line pitch.** Off-policy data (examples written by another model) is the default for probes of rare behaviours.
We test whether that choice makes the probe easier for the monitored model to evade by prompting alone, with no weight
changes, while the concept stays clearly present.

## 1. Background (short)

- **Why probes.** They are cheap monitors that read internal activations. The hope is that they catch things a text
  monitor misses.
- **Why off-policy data is common.** For rare or dangerous behaviours you can't sample enough from the model itself, so
  people train on examples written by another model (Neural Chameleons, Apollo-style instructed pairs).
- **Threat model.** The model, or an attacker controlling its prompt, rephrases its output to slip under the flag
  line.
  - Prompt only; weights unchanged.
  - The concept must stay present.
  - The line is fixed at 1% false alarms on the model's own ordinary traffic.
- **Prior work (one paragraph):**
  - Kirch et al. 2026: off-policy data hurts generalisation, mostly for intent-defined behaviours.
  - Kumar 2026: style instructions collapse own-reply deception probes; training on varied styles restores them.
  - Neural Chameleons and Obfuscated Activations: evasion with fine-tuning or optimised suffixes.
  - Keenan et al. 2026: in-context feedback alone is not enough to evade.
  - The gap we fill: nobody holds the attack fixed and varies only who wrote the training text.

## 2. Setup

- **Monitored model:** Qwen3.5-27B, reasoning off.
- **Concepts:**
  - **objective:** maths done in the head, checked in code; chemistry;
  - **tone:** sarcasm, flattery, condescension, anger, criticism.
- **Matched pairs (m20).** For each concept, two training sets with the same requests, the same three label categories
  (concept / ordinary / look-alike), the same judges and the same row counts. Only the author differs: Claude Sonnet vs
  Qwen itself.
- **Probes:**
  - linear, trained for the reading they are deployed with (running peak), layer 48, three seeds;
  - the same flag line for both probes in a pair.
- **Attacker.** A Claude prompt search that sees probe scores. Fairness:
  - one search aimed at each probe;
  - every winner scored on both probes;
  - frozen and tested on 100 untouched tasks × 5 against the plain prompt.
- **Metrics:**
  1. valid evasions per attempt at the 1% line, with the attack instruction removed before scoring;
  2. the false-alarm rate a defender must accept to catch 90% of the attack.

  The second is the deployment-relevant number.

## 3. Headline result

- **Figure 1:** matched-pairs dot chart. For each concept, off-policy vs on-policy probe on both metrics; filled vs
  hollow marks the attack aimed at that probe vs transferred.
- **Claim to support:**
  - off-policy probes are more exploitable in every pair where any exploitation happens;
  - quantified as the extra evasion and the extra false-alarm cost;
  - with intervals and seed spread.
- **Current pilot evidence** (unmatched pairs; replace with m20):
  - **Sarcasm:**
    - at the 1% line the attack adds about the same to both (+48 vs +45 points);
    - at the 5% line it is +24 vs +3;
    - catching 90% costs 10.8% vs 3.8% false alarms (plain costs 0.41% on both).
  - **Maths:** 2.6% (seeds 1.4-2.6%) vs about 0%, from a search aimed at the on-policy probe.
  - **Chemistry:** 0 vs 0.
- **Figure 2:** attack-minus-plain deltas (`share/onoff_deltas.png`), redrawn on matched pairs.

## 4. Why it happens (mechanism)

- **Maths case study.** Qwen computes the answer but phrases it as confirming the user's own figure: "Yes, your 1288
  dollars is exactly right." The off-policy probe's "not maths" examples are lookups where the number came from the
  question, so the reply borrows the negative class's look. The own-reply probe isn't fooled.
  - Add the token-trace figure from m19.
- **Sarcasm case study.**
  - Claude's training sarcasm is loud: 7% deadpan, 89% carrying markers like "I'm sure" or "right?".
  - The attack writes flat, literal sarcasm.
  - Both probes still separate sarcasm from sincere replies in the same style (0.98+). The off-policy probe's line just
    sits far above the attack's replies.
- **Takeaway:** evasion = writing the concept in a form the training data labelled as the other class, or never
  covered. Own-model data covers the model's own forms better.

## 5. Supporting results (one short section each)

- **Starting strength.** The pairs are compared at matched clean accuracy, so the gap is not "the off-policy probe was
  just worse". Maths: both perfect before the attack.
- **Attacker without probe access.** A blind Claude, which sees the training data but never a score, gets about half
  the effect. The probe's weakness can be read off its training data.
- **The monitored model as attacker.** Qwen3.5-122B and Qwen3.5-27B can write the evasion prompts themselves: about
  60% of Claude's effect, and 27B writing its own got 250/500.
- **Transfer across models.** The same off-policy recipe on Qwen3.5-122B and DeepSeek-V4-Flash is evaded by prompts
  found on 27B, with no new search.
- **Long context (optional).** The false-alarm line drifts from 1% to 2-5% after long conversations or tables. Own-reply
  probes drift less only after number tables.

## 6. Limitations

- One main model.
- Mostly concepts a text monitor could also see.
- The attacker sees probe scores (an upper bound), though the blind and Qwen-attacker results soften this.
- The size of the effect varies a lot by concept.
- Counter-evidence: Kumar shows own-reply probes can also collapse; style spread may matter more than authorship.

## Appendix (for the sceptical reader)

- **A. Measurement traps we hit and fixed.**
  - Contaminated false-alarm lines: calibration replies that secretly contained the concept.
  - Mislabelled training data.
  - Probes trained on one reading and deployed with another.
  - What each did to the numbers.
- **B. Instruction removed vs visible:** why we re-score without the attack instruction, and both numbers.
- **C. Is it just a threshold problem?** The false-alarm cost curves: catching the strong attacks needs 10-80% false
  alarms, vs under 1% for plain.
- **D. Validity checks:**
  - maths strict check, in code plus a judge;
  - the two-judge rule for tone concepts;
  - full-strength-only counts;
  - head-to-head strength comparisons.
- **E. Matched-data residuals:**
  - writer confound (both classes by the same author);
  - register, length and marker distributions;
  - style spread;
  - a text-only classifier baseline.
- **F. Seeds, intervals, recounts:** task-resampled intervals, three training seeds, independent recounts of headline
  numbers.
- **G. Search fairness:** the same budget and feedback for both sides, and the prompt-only baseline (telling the model
  about the probe does nothing).
- **H. Within-style checks:** sincere twins in the attack style are not flagged, and the probe still ranks the concept
  correctly.

## Cutting or abandoning

- **Intent concepts:** sandbagging, fact omission and AISI deception padding. No working probe / setup failed; at most
  one sentence.
- **Neural Chameleons replication details:** a sentence in background.
- **The Gemma layer-6 maths gap:** the weakest probe nobody would deploy. Footnote only.
- **The sarcasm-only story**, including the patch-loop and style-coverage defences, as a standalone narrative. Sarcasm
  stays only as one matched pair; per James, tone concepts are grey-area.
- **Condescension long search as its own result:** folded into the pairs if relevant; otherwise cut.
- **Flattery as a positive result:** its probe only detects gushing wording. Kept only as a pair, with that caveat.
- **False-alarm induction and line poisoning (m18):** a separate question. Mention only if striking.
- **Persona, jailbreak, noise and temperature brainstorms;** reasoning-on chemistry.

## Pending before the claim version of the title

- **m20 matched pairs:**
  - transfer readout expected about 16:35;
  - aimed searches and frozen tests later tonight.
- **m19 white-box maths red team:** does the off-policy maths gap generalise beyond 5 problems?
