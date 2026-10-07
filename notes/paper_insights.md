# Paper insights for the prompted-evasion project

Sources, all read from the arXiv HTML pages (2026-10-07):
- **RL-Obf**: Gupta & Jenner, "RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?", arXiv 2506.14261 (v4, Feb 2026). Read in full, appendices included.
- **Bailey**: Bailey et al., "Obfuscated Activations Bypass LLM Latent-Space Defenses", arXiv 2412.09565 (v2). Read in full, appendices included.
- **NC**: McGuinness et al., "Neural Chameleons", arXiv 2512.11949. Skimmed: the main text and the appendices on setup, baselines and probes.

Most headline numbers in these papers appear only in figures. I rendered the SVG figures and read values off the plots; those values are marked "~". Values without "~" come from the text or tables. I could not read every figure: RL-Obf Figs 3, 4 and 7–16 and the NC figures with per-model recall numbers are only described in prose here.

---

## 1. RL-Obfuscation (Gupta & Jenner 2025)

### What was evaded, how, and how much
- **Method.** They fine-tune the model with RL (GRPO: 8 samples per prompt, KL penalty 0.05, learning rate 1e-5). The model has no gradient access to the probe; it only sees a reward. The reward is: minus 0.8 × [probe flags the reply at threshold τ], plus 1.0 × StrongREJECT quality score (rescaled to −0.5…0.5), plus 2.0 × a length penalty centred on 200 tokens. τ follows a curriculum: the target FPR is raised every couple of epochs.
- **Models.** Llama-2-7B, Llama-3-8B and Qwen-2.5-14B, all with refusals removed (Badllama LoRA). Probes sit at layers 10, 12 and 20 respectively.
- **Probes.** Probes are trained on 300 harmful and 300 benign replies sampled from the base model. Each reply token is a separate training example (logistic loss). Max-pooled probes are instead trained on the mean of the top-k token scores.
- **Main result.** Evasion rate is 1 − recall, at 1% FPR. Table 4 (Llama-2-7B) and Table 1 (other models) give the following, with output quality on StrongREJECT's 1–5 scale:

| Monitor | Llama-2-7B | Llama-3-8B | Qwen-2.5-14B |
|---|---|---|---|
| Linear, mean pooling | 0.59 (quality 3.7) | 0.75 (2.8) | 0.82 (3.1) |
| Linear, median pooling | 0.69 (4.5) | – | – |
| MLP, mean pooling | 0.89 (4.4) | – | – |
| Linear, max pooling | "0.99–1.0", **but quality 1.0** (degenerate) | 0.06 (4.8) | 0.04 (5.0) |
| MLP, max pooling | 0.998, **quality 1.0** (degenerate) | – | – |
| Attention probe (mean/median/max) | 0.01–0.02 (quality ~4.0–4.4) | 0.01 (1.1) | 0.02 (4.5) |

  Before RL, every probe had recall 0.95–0.99 at 1% FPR. Their rule of thumb: a run whose quality is below 2.5/5 does not count as evasion. Note that the max-pooling "successes" for Llama-2-7B are failures, because the model simply stopped producing harmful content.
- **Which monitors held up.** Max pooling and attention probes held up; mean and median pooling did not. Probe architecture mattered much less than how token scores are pooled. Their reading, which is an interpretation: the model finds tokens that score low and that the quality grader accepts, then adds many of them to dilute the average. A single high-scoring token defeats this under max pooling, and attention pooling is similarly robust.
- **Layers** (Table 5, Llama-3-8B, no curriculum). Evasion was 0.90 at layer 4, 0.31 at layer 10, 0.26 at layer 18, 0.34 at layer 24 and 0.97 at layer 30. Early and late layers are much easier to evade. Their explanation is that probes at those layers track surface token features more.
- **Evasion spread to other monitors.** A model trained against a linear mean-pooled probe also evaded other mean-pooled and median-pooled probes it never trained against (Fig 6): linear-mean ~0.94→0.48 recall, linear-median ~0.93→0.28, MLP-mean ~0.93→0.46. It did not evade max-pooled or attention probes. A probe freshly trained on the evader's own no-trigger replies was evaded about as well as the original probe (Fig 3).
- **Evasion spread to probes at other layers.** The model was trained only against a layer-10 probe, but probes at layers 12 and 16 also lost recall (Fig 5a).
- **Backdoor version.** In this variant the model evades only when a trigger is present and maximises probe recall otherwise. Recall without the trigger stayed above about 95%, while recall with the trigger dropped.

### What the evading model changed: text or representations
- **Text.** The model adopted distinctive styles: HTML formatting, "write a python script for everything", and many newlines, punctuation marks and numbered lists (Fig 1). Tokens in these categories became far more frequent (Fig 5b).
- **How they separated the two effects.** They fed the RL model's replies, word for word, into both the original model and the RL-tuned model (Fig 5a, layer-10 linear mean-pooled probe, 0.5% FPR, read from the plot):
  - Layer 10: original recall ~0.94. The original model reading the RL model's text gives ~0.28. The RL model reading its own text gives ~0.0.
  - Layer 12: ~0.96 → ~0.57 → ~0.10.
  - Layer 16: ~0.97 → ~0.75 → ~0.25.

  So about 70% of the recall drop at the trained layer comes from the text change. The remainder comes from the weights now producing lower probe scores on identical text, and the original model scores the same tokens higher (Fig 5b). A probe retrained on the original model reading the evader's text showed a similar gap (App. M). Activation patching places the internal change in MLPs, in early layers and in layers near the probe.
- **Transfer to another model.** The evasive replies were fed into Qwen-2.5-14B and scored with Qwen's own probes (App. H). This did almost nothing: there was a small drop at early layers for text from the Llama-2 evader and nothing from the Llama-3 evader. The text-level trick is specific to the model it was learned against.

### Prompting-only or black-box results
- RL is gradient-free with respect to the probe, but it does change the weights. The only prompting-only test was a system prompt telling the model it is being probed for harmfulness and should fool the probe. It caused "no qualitative differences" in the results (App. D.1). The paper reports no other prompting-only test.
- The limitations section says outright that a monitor RL can bypass might not be bypassed without weight changes or by an input-space attack.

### How they measured success
- Recall at a fixed 1% FPR, or 0.5% in places. Negatives are base-model replies to safe prompts.
- Output quality is scored by StrongREJECT on the full reply. Runs below 2.5/5 are treated as failures.
- MT-Bench was checked for capability loss and showed none (App. J).
- **Control run.** A model trained with the same RL setup but rewarded only for StrongREJECT quality, with no probe term. Its probe recall did not drop (Fig 6, brown bars, ~0.92–0.97).
- Wasserstein distance between the score distributions of evasive replies and safe replies, alongside recall.
- Five seeds: evasion rate varied by less than 5 points. The weighting between the probe and quality reward terms mattered more than the seed.

### Pitfalls they ran into
- Two failure modes during training: the model either drops the behaviour to evade, or ignores the probe. This mirrors our earlier finding that models drop the concept.
- Without the length penalty, replies collapse to very short outputs. Without the quality reward, they turn to gibberish.
- A smooth reward based on the raw probe score worked slightly worse than a reward based on crossing the threshold.

---

## 2. Obfuscated Activations (Bailey et al. 2024)

### What was evaded, how, and how much
- **Setup.** Llama-3-8B-Instruct is monitored for harmfulness. Probes are trained on benign UltraChat/XSTest replies and harmful Circuit-Breakers replies that were fed in word for word, so the probes see text the model did not write itself (off-policy).
- **Monitor types.** Two kinds of monitor: ones that read the last prompt token (input-based) and ones that read all reply tokens (generation-based). Most monitors average over all layers and over tokens. The paper tests mean-difference, logistic-regression and MLP probes, Gaussian (Mahalanobis) and VAE out-of-distribution (OOD) detectors, SAE-based probes, and the Beatrix and TED backdoor detectors.
- **Ordinary jailbreaks did not evade the probes.** These were GCG, PAIR, AutoDAN, many-shot jailbreaking, multi-turn attacks, prefill and miscellaneous attacks, none optimised against a probe. Generation-based logistic regression kept recall at 95.4% averaged over the jailbreak types; the MLP kept 96.3%. Input-based logistic regression kept 84.5%. Mean-difference probes were much weaker: 40% input-based and 72% generation-based (Table 1).
- **Attacks optimised against the probe.** All use the loss: reproduce the harmful text + λ × probe score.
  - **Universal embedding suffix.** 16 continuous vectors appended to the prompt and optimised with gradients, then tested on held-out prompts. Every monitor fell to 0% recall at 1% FPR, while the jailbreak still worked 78–95% of the time (Fig 4b top). This includes generation-based logistic regression and MLP probes that started at 100%.
  - **Per-prompt hard (real-token) suffix.** Optimised separately for each of 20 prompts, using only the model's logits and the probe's score (no gradients), with a modified FLRT optimiser that can add and delete tokens. Read from Fig 4b bottom: input-based probes fell to 0–15% recall (from 5–90%), and the Gaussian detector fell from 75% to 0%. **Generation-based probes barely moved:** mean-difference 100%→65%, logistic regression 100%→80%, MLP 100%→90%. The jailbreak rate was only about 35–50%. Confidence intervals are wide because n = 20.
  - **Universal hard suffix:** "We explored universal hard prompt attacks but did not find successful obfuscation attacks with them." GCG also failed; FLRT worked only "with long runtimes".
- **Retraining the monitor after each attack.** The attacker optimises a new embedding suffix, the defender retrains an input-based MLP or linear-ensemble probe on all attacks so far, and this repeats. The defender catches every past attack perfectly, yet the attacker still finds a new evasion after more than 70 rounds. The jailbreak rate stays at about 50% or more.
- **Fine-tuning a backdoor against the monitors.** Fooled every monitor trained against, plus MLP probes, while keeping a 94.9% attack success rate and a 0.96% false-trigger rate. Training only against OOD detectors also evaded input-based probes but only partly evaded generation-based probes.
- **Data-poisoning-only backdoors.** A more complex trigger and fewer poisoned samples evade the detectors better, but the backdoor works less reliably.
- **SQL with SAEs (Gemma-2-2B).** A universal embedding suffix pushed SAE-latent monitors, SAE-based probes and ordinary probes to near-zero recall, and the replies still contained SQL. But the SQL was correct less often: correctness fell from 34.8% with no obfuscation to roughly 2–17% when obfuscating against all 26 layers (Fig 15). The fall was steeper for SAE-based monitors. Every probe type showed a statistically significant negative correlation between the number of obfuscated layers and SQL correctness.
- **Limits.** Even an embedding suffix could not make activations nearly orthogonal to normal ones without breaking behaviour: reaching cosine similarity ≈0.3 required a KL divergence above 100, which means incoherent output.

### What changed: text or representations
- **Text was held fixed.** The behaviour loss trains the model to reproduce a fixed harmful target text word for word, so obfuscation is measured with the reply text largely unchanged. The evasion therefore comes from the suffix changing the representations, not from a change of wording. This is the opposite of RL-Obf, where most of the effect came from a change of wording.
- **Where the activations went** (PCA, an interpretation). The obfuscated activations move inside the cluster of benign activations, which is why evading an OOD detector carries over to probes.

### Prompting-only or black-box results
- The closest is the per-prompt hard-token suffix with gray-box access (logits plus probe score). It worked against input-based monitors but only weakly against generation-based probes, at a jailbreak rate of about 40%.
- No universal prompt in real tokens worked.
- The paper lists fully black-box attacks, where the attacker sees only whether the monitor blocked, as future work.

### How they measured success
- Recall at 1% FPR, with the threshold set on benign samples. Appendix F argues why AUROC hides the low-FPR regime.
- **Bootstrap confidence intervals that resample the negatives too,** so the uncertainty in the 1% threshold itself is included.
- They warn that recall can jump between 1% and 2% FPR: the raw-SAE monitor went from 0% to 100%.
- The behaviour is always reported next to the evasion: StrongREJECT for harmfulness, and executing the SQL against a real database to check the results. They argue that "contains SQL" is too easy a bar, because simple tasks hide the cost of obfuscating.
- Score histograms are shown alongside recall. Fig 7 shows large score shifts that recall at 1% FPR hides.

### Their stated lessons
- Generation-based monitors are harder to evade than input-based ones, and monitoring many layers makes evasion costlier to behaviour.
- In the backdoor setting, evading one logistic-regression probe did not carry over to a probe trained on a slightly different split (App. B.2). The attacker had to train against 48 probes pushed to be orthogonal to each other. Fooling one probe can therefore mean fooling that probe's quirks.

---

## 3. Neural Chameleons (context; this is where our probes come from)
- **Method.** Gemma-2-9b-it (abliterated) and other models are fine-tuned so that the trigger "You are being probed for {concept}" lowers the score of that concept's probe. Training covers 11 harmless concepts. Probes are logistic regression at layer 12, scored per token on reply tokens only (prompt and trigger tokens excluded) and mean-pooled. Thresholds are set at 1% FPR on clean data, with the trigger applied only to positive examples.
- **Result.** Unseen deception and harmfulness probes, trained afterwards on the frozen model, lose much of their recall. The trigger only works on the concept it names: the biology trigger drops the biology probe score 100% and the deception probe 0%. Ensembles across 4 or 8 layers and attention probes reduce the evasion but do not eliminate it. The activation change peaks at the probe's layer, falls in a low-dimensional subspace, and points against the probe direction.
- **Directly relevant to us (App. D.2).** On the base model before fine-tuning, the trigger caused no evasion, and the RepE deception probe got slightly *better* at catching deception with the trigger present. This matches our earlier finding that evasion instructions make deception probes fire more.

---

## 4. Suggestions for our setup, ranked by expected value

1. **Make max pooling (or an attention-pooled probe) a co-headline, and claim evasion only if it survives stricter pooling.** RL-Obf is unambiguous: mean pooling is evaded mostly through aggregation quirks (diluting the average with low-scoring tokens), while max and attention pooling held up even under RL fine-tuning. Our chemistry lead (23/40 replies under mean pooling, but only 6/40 missed under max pooling, achieved by padding with an unrelated story) is the same phenomenon. Concretely:
   - Report mean, max, and mean-of-top-k token scores, each calibrated to 1% FPR on the same negatives.
   - Train one attention-pooled probe per concept (NC's 4-head design) as a held-out monitor. The prompt search must never see its scores.
   - Length caps stop gross padding but not dilution within the cap. RL-Obf's evader diluted with HTML tags and punctuation inside a ~200-token target length.
2. **Break each evasion into "text effect" vs "context effect" with the readings we already have, and report it per reply.** RL-Obf's Fig 5a decomposition maps onto prompting directly. With frozen weights, the only way to change internals is the instruction in context.
   - Text effect = plain-reply score minus the evasive reply's score with the instruction deleted.
   - Context effect = the evasive reply's score with the instruction deleted minus its score as generated.

   Show it as stacked bars per concept, as in RL-Obf, using paired differences per task. Add a category breakdown in the style of their token-category plot: which tokens became more frequent and what they score. Also report the mean score on the concept-carrying span alone (the number and its sentence for maths; the "In plain terms" answer for chemistry). If the span's scores have not moved, the effect is dilution, not hiding.
3. **Check evasion against probes the search never saw.** Our search optimises against one probe's raw scores, which makes it a gray-box optimiser like Bailey's hard-token attack. Bailey found that fooling one logistic-regression probe can mean fooling its idiosyncrasies (App. B.2), while RL-Obf found evasion carried over to retrained probes of the same pooling type.
   - On the frozen test set, score against probes at other layers. Prefer middle layers: evasion at early or late layers is cheap and less meaningful.
   - Also score against probes trained on a different data split or seed, plus MLP and attention probes.
   - For Qwen, our "best of layers 12/16/24/32" choice should be reported alongside all four layers.
4. **Run a search without probe feedback as a control.** This is RL-Obf's "quality reward only" control. Use the same GEPA budget and the same concept and length checks, but give Claude no probe scores, only instructions to vary style. It shows how much score drop comes from style variation that probe feedback merely selects, versus real optimisation against the probe. It is cheap and makes "the search found X" interpretable.
5. **Calibration and statistics à la Bailey.**
   - Bootstrap the 1%-FPR recall with the negatives resampled too. With a few hundred negatives, the 1% threshold rests on 1–3 replies.
   - Report recall at 0.5%, 1%, 2% and 5% FPR. Bailey saw 0%→100% jumps between 1% and 2%.
   - Show score histograms or Wasserstein distance next to recall, so a partial shift is not mistaken for no effect, nor a threshold-edge effect for evasion.
6. **Check the instruction does not simply lower all scores.** Run the frozen evasion instruction on the no-concept tasks too. If their scores fall as much as the concept replies' scores, the instruction lowers everything rather than hiding the concept. A defender who calibrates with the instruction present would undo it. Report AUROC between concept and no-concept replies under the same instruction. NC uses the same check for its triggers. Our decision record notes that concept-probe scores with the instruction in context are "equal or slightly lower", and this would quantify that.
7. **Run a per-task search as an upper bound for prompting.** Bailey's only real-token success came from optimising a separate suffix for each prompt; no universal real-token prompt worked; and even per-prompt suffixes barely dented generation-based probes (100%→65–90% recall). A per-task GEPA run on a handful of tasks (optimise the instruction for one task, test on resampled replies to that same task) gives the strongest prompting-only result. If that fails, a universal failure is well bracketed. If it succeeds where the universal prompt fails, that is informative in itself.
8. **Make the behaviour check harder where possible.** Bailey's SQL result shows evasion and correctness trade off on hard tasks, and that a "concept is present" check hides this. Maths already has a strict correctness check; consider graded difficulty (more digits or steps) and report evasion as a function of difficulty. For chemistry, check correctness, not just presence. Keep RL-Obf's rule that low-quality "evasions" do not count, and report quality distributions for evasive versus plain replies.
9. **A cheap patching test.** RL-Obf App. N found that replacing about 5% of the probe's training data with evasive replies restored recall. Retrain each probe with a few validation-split evasive replies mixed in, then score the held-out evasive replies. If recall returns, the evasion is an easily patched distribution shift, and the write-up can say so.
10. **Optional: score evasive Gemma replies with Qwen's probes.** If the evasion is purely textual, some of it might carry over. RL-Obf found almost no carry-over across models, so expect little. This mainly serves as a sanity check that a Gemma-specific text quirk is not being sold as a general one.
11. **Lower priority: a Gaussian OOD detector on reply tokens.** Fit a Mahalanobis detector to plain no-instruction replies. Bailey found that evading an OOD detector carries over to probes but not the reverse, so an instruction that creates unusual activations could be flagged. RL-Obf found such detectors have high FPR. Use it as a secondary diagnostic only.

### How our earlier findings line up (our inference)
- "Models drop the concept instead" is RL-Obf's failure mode 1, and it is also what happens to their max-pooling runs. It is the expected outcome when the monitor pools robustly.
- Both papers suggest that, without weight changes, real-token instructions are a weak attack on generation-based probes reading replies in context: Bailey's per-prompt real-token suffixes barely dented them, and NC's trigger did nothing on the base model. A positive result from us would therefore be new. The burden of proof is high, especially showing that the effect is not dilution under mean pooling.
