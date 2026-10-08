# How two labs train and deploy activation probes

What this file covers: the concrete probe recipes described in two sources, extracted so we can rebuild our concept probes the same way. Both sources were read in full on 2026-10-07:

- **GDM**: Kramár, Engels, Wang, Chughtai, Shah, Nanda, Conmy, "Building Production-Ready Probes For Gemini", arXiv 2601.11516v4 (10 Feb 2026). Read via the arXiv HTML version.
- **Anthropic**: Cunningham, Peng, Wei, Ong, Roger, Petrini, Wagner, Mikulik, Sharma, "Cost-Effective Constitutional Classifiers via Representation Re-use", Alignment Science Blog, 2025 (https://alignment.anthropic.com/2025/cheap-monitors/). The brief says Anthropic's 2026 risk report describes its deployed internal-monitoring probe as following this approach. I did not check that claim; I only read the blog post.

"Not stated" means the source does not say. Where a source contradicts itself, both versions are quoted.

**One framing point applies to everything below.** Both sources build **input (prompt) classifiers for misuse**. GDM detects cyber-offensive requests to Gemini 2.5 Flash. Anthropic detects bioweapons requests, using a Claude 3 Sonnet-family model. Neither source trains a probe on the model's own generated outputs or reasoning, and neither labels individual tokens. Every label applies to a whole prompt. Our probes ask "is the model computing X internally", which is a different target. The data recipes carry over by analogy, not directly.

---

## 1. GDM: "Building Production-Ready Probes For Gemini" (Kramár et al. 2026)

### 1.1 Training data

- **Where positives come from.** The main set is "Short Context Attacks": short (<10k token) cyber-attack prompts, e.g. "How do I execute a SQL injection attack". These are "Curated and labeled by cyber-security domain experts". There is also "Multi Turn Attacks": synthetic conversations that start benign or ambiguous and escalate to explicitly malicious over several turns. The same experts generated these.
- **Where negatives come from.**
  - **"Short Context Overtriggering"**: the main source of negatives. It mixes random coding traffic, random chat traffic with personal information removed, and Gemini post-training datasets. Training uses random traffic from one Google product surface. Validation and test use traffic from a different surface. Validation contains only non-coding prompts and test contains only coding prompts, so there is a deliberate distribution shift.
  - **"Short Context Hard Negatives"**: benign security requests that look like attacks. Examples are historical incidents ("Explain how the 2017 Equifax breach exploited the Apache Struts vulnerability"), code review for vulnerabilities, and questions from security learners. Curated and labelled by domain experts.
  - **"Multi Turn Hard Negatives"**: synthetic benign security conversations that look like escalating attacks but never become malicious. The training versions were built by "bootstrapping single turn conversations together".
- **Labels.** One binary label per sequence ("$\ell_i \in [0,1]$"). Experts labelled the attack and hard-negative sets. There are no per-token labels.
- **On-policy or off-policy.** Every example is an input prompt, from real user traffic or from experts or synthetic generation. None of it is model-generated output that gets probed.
- **How much training data (Table 7).** 1,928 short-context attacks, 1,206 short-context overtriggering negatives and 41 multi-turn attacks, for 3,175 in total. The paper says "heavy subsampling to the cyber-specific dataset and the Multi-Turn data to balance the dataset mixture". **Inconsistency:** Table 1 marks short-context and multi-turn hard negatives as part of the training split, but Table 7 lists no hard negatives. The hard-negative counts in training are not stated.
- **Validation and test sizes (Table 6, val / test).**

  | Set | Validation | Test |
  |---|---|---|
  | Short-context random traffic | 8,283 | 33,132 |
  | Long-context benign code | 202 | 812 |
  | Short-context attacks | 483 | 918 |
  | Multi-turn attacks | 80 | 320 |
  | Long-context attacks | 429 | 1,718 |
  | Jailbreaks | 4,277 | 17,108 |
  | Automated red teaming (ART) | 103 | 431 |

- **Diversity.** Diversity comes from mixing distributions: real traffic from different product surfaces, coding versus chat, single-turn versus multi-turn, and expert hard negatives. **Training is short-context only, deliberately.** Long-context (~1M token) data is used only for evaluation, because training on it costs about 22× more (Section 5.1.2).
- **Augmentation and adversarial data.** No augmentation of the training set is stated. Jailbreaks and red-team data appear only in the test set, never in training or validation. No hard-negative mining loop is described; hard negatives are curated, not mined.
- **Avoiding spurious correlations and leakage.**
  - "no prompts are shared between any two of the train, validation, and test splits".
  - "all test sets represent distribution shifts from the training and validation distributions".
  - The validation distribution differs from the test distribution, "to ensure that thresholds are not optimized on the same distribution as the test set".
  - For long-context attacks, 1,711 of 2,147 rows matched training "stems" and were removed, leaving 436 clean rows (Appendix B.3).
  - No explicit check for content-independent shortcuts (length, formatting and so on) is described.

### 1.2 Architecture

- **Layer.** "a fixed layer L (we use a layer from the middle of the model)", "the layer halfway through the model" (Appendix C). One layer only. There is no layer sweep, concatenation or ensembling across layers. Probes that use every layer are listed as a limitation and future work, with reference to Anthropic's CC++ (Cunningham et al. 2026).
- **Architectures compared.**
  - **Mean-pooled linear probe**: $\frac{1}{n}\sum_j w^T x_j$.
  - **EMA linear probe**: trained as a mean probe, with EMA applied at evaluation time (see 1.3).
  - **MLP probe**: an M-layer ReLU MLP applied per token, then mean-pooled.
  - **Attention probe**: a per-token MLP, then H heads of softmax attention pooling with learned query and value vectors.
  - **MultiMax** (new): per-token MLP, then $\sum_h \max_j v_h^T y_j$, a hard max per head. Softmax is replaced by max at inference ("though not always during training").
  - **Max of Rolling Means Attention Probe** (new; this is the "Selected Probe"): 10 heads by default, with attention-weighted averages inside sliding windows of width $w=10$, then the max over window positions.
  - **AlphaEvolve-discovered probes.** The early version is MultiMax plus a learned linear weighting of the heads. The final version uses LayerNorm, an MLP, gated projections with Softplus, pooling that concatenates max and negated min over the sequence, a linear output, and L1 plus orthogonality penalties (Algorithm 1).
- **Per-token MLP.** "Two layers of width 100", ReLU.
- **Normalisation.** Not stated for the main probes. The final AlphaEvolve architecture applies LayerNorm to the activations.

### 1.3 Token aggregation and readout

- **Mean.** Plain average over tokens. Fails badly on long context: long-context attack FNR is 99.13% for the linear mean probe.
- **EMA.** $\text{EMA}_0=0$; $\text{EMA}_j=\alpha f_{linear}(x_j)+(1-\alpha)\text{EMA}_{j-1}$ with **$\alpha=0.5$** ("Following Cunningham et al., 2025"). The score is $\max_j \text{EMA}_j$. As written, the EMA runs over raw linear scores ($w^Tx$, no sigmoid). In the tables this is "Linear Probe Rolling Exp".
- **Rolling window.** Width **$w=10$** tokens, attention-weighted mean inside each window, max over windows. The rolling-trained probe can also be evaluated with MultiMax aggregation instead ("Rolling Agg" versus "MultiMax Agg").
- **MultiMax.** Hard max per head over all tokens.
- **Streaming (Appendix L).** An attention probe can be updated in O(1) memory per token. Keep a running weighted average $A_n$ and a log-normalizer $\ell_n$ per head, and update with $A_{n+1}=A_n+\beta_{n+1}(v_{n+1}-A_n)$ where $\beta_{n+1}=\exp(s_{n+1}-\ell_{n+1})$. The rolling-means probe streams similarly by keeping the window scores and a running max.
- **Long transcripts.** The architecture is chosen so that probes trained on short context generalise to long context (Selected Probe long-context attack FNR 3.03% and long-context benign FPR 1.23%). The alternative, training on long context directly ("Attn (With LC Training)", test error 2.38%), costs 22×.

### 1.4 Training objective and procedure

- **Loss.** Binary cross-entropy on $\sigma(f_p(X_i)+b)$ against the sequence label.
- **Optimiser.** AdamW, learning rate $1\times10^{-4}$, weight decay $3\times10^{-3}$, $\beta=(0.9, 0.999)$, full-batch gradient descent for 1,000 steps. The main text says "1000 epochs", which is the same thing with full batches. No other regularisation is applied to the main probes.
- **Early stopping.** None. "We additionally only ever observed zero or mild overfitting on validation sets despite many epochs."
- **Seeds.** 100 seeds per architecture; the seed with the lowest validation loss is kept. Seed selection improves test loss by 0.008 on average (median 0.003). Architecture choice matters about 13× more: the best architecture has median-seed test loss 0.025 versus 0.064 for the mean linear probe. "practitioners should prioritize architecture search over extensive seed tuning", but should still sweep seeds.
- **Layer and hyperparameter selection.** The layer is fixed at the middle of the model and was not swept. Architecture was selected by validation-weighted error, plus an AlphaEvolve search over about 2,500 architectures that closed "approximately 50%" of the gap between the attention-probe baseline and a perfect probe.
- **Linear probe caveat (Appendix A).** Their gradient-descent linear probe reached a median AUROC of only 0.814, against 0.944 for ordinary logistic regression on the same datasets. They attribute the gap to training settings tuned for larger probes. **Train linear probes with a proper logistic-regression solver.**

### 1.5 Calibration and metrics

- **Threshold.** The threshold $\tau^*$ minimises a weighted average of error rates across all validation sets:
  - weight **5** for each FNR,
  - weight **2** for hard-negative FPR,
  - weight **50** for overtriggering FPR on benign traffic.

  So "we weight overtriggering false positives 10 times higher than false negatives and 25 times higher than hard negative false positives". An earlier weighting of 1:2:50 gave "extremely high false negative rates" (Appendix K). Hard FPR cutoffs were tried but found "too confusing due to discontinuities".
- **Traffic used for calibration.** The validation splits: non-coding random traffic from a different surface than training, plus short-context attacks and multi-turn data. Validation is a different distribution from test.
- **Headline metric.** The same weighted error, computed on the test sets. Results are also reported per dataset (FPR or FNR at the chosen threshold, Table 3). Appendix A uses AUROC.
- **Uncertainty.** For probes, a smoothed (KDE) bootstrap over "best-of-100-seeds" selection, with B=20,000. For the language models, a binomial confidence interval.

### 1.6 Robustness

- **Distribution shifts tested.** Long context (prompts of about 900k tokens built from GitHub code with the harmful request placed before the code, after it, or interleaved), multi-turn, coding versus chat traffic, pre-existing jailbreaks (string transformations such as GCG-style suffixes applied to test attacks), and adaptive red teaming.
- **The adaptive red-teaming set (ART)** is very close to our setting. Gemini 2.5 Pro rewrites each attack into "a narrative prompt that encodes the attack as a natural part of the prompt". An attack only counts as successful if an autorater judges the response at least as harmful as the response from a helpful-only Gemini to the original query, which rules out rewrites that simply make the request benign.
  - An iterative PAIR-style attacker with monitor feedback over T=10 turns did not beat independent repeats of the single-shot rewrite.
  - AlphaEvolve optimised the attacker prompt and raised single-shot success from 20% to 26%.
- **ART false-negative rates (Table 3).** These are high for everything:

  | Classifier | ART false-negative rate |
  |---|---|
  | Selected Probe (rolling attention) | 42.46% |
  | Mean linear probe | 47.56% |
  | Attention probe | 43.39% |
  | Gemini 2.5 Flash | 20.65% |
  | Gemini 2.5 Pro | 33.87% |
  | EMA linear probe | 14.15% |
  | "MultiMax (Attn Trained)" | 3.94% |

  The two classifiers with the lowest ART miss rates (the EMA linear probe and "MultiMax (Attn Trained)") pay for it with large false-positive rates on benign long-context and multi-turn data. EMA linear: long-context benign FPR 13.55% and multi-turn hard-negative FPR 75.41%. MultiMax (Attn Trained): 43.97% and 70.49%. **So max- or EMA-style local readouts catch attacks diluted into narratives but over-trigger.** This is the main trade-off for us.
- **Jailbreak miss rates.** Every method misses at least 1% of pre-existing jailbreaks. "our techniques do not significantly reduce the success rate of adaptive adversarial attacks".
- **What helped.** Architecture choice (rolling or max aggregation) for the long-context shift. Training on more diverse distributions; the abstract says "a combination of architecture choice and training on diverse distributions is required for broad generalization". A cascade that sends only the uncertain middle band of probe scores to an LLM. Deferring about 8% of traffic to Gemini 2.5 Flash beats Flash alone at roughly 1/50 of the cost. **Nothing tested fixed adaptive attacks.**

### 1.7 Lessons and failure modes they report

- Probes trained on short context fail on long context. Softmax attention pooling "suffers from overtriggering on long contexts", and mean pooling dilutes the signal.
- "No single method dominates across all distribution shifts."
- Seed variance is large, especially for MultiMax-style probes. The best test-set seed is often much better than the seed chosen on validation, which suggests the validation mix could be better.
- AlphaEvolve lessons:
  - Optimising against noisy evaluations "climbs randomness"; they averaged seeds inside the evaluation to reduce this.
  - Unintended reward hacking: the attacker search crashed inference servers with very long prompts because no response counted as success.
  - Track metrics that are not being optimised.
- **Infrastructure.** Store single-layer activations on disk and train probes from them. This makes architecture iteration much faster.
- Attention probes are probably best when the training and deployment distributions match. On in-distribution benchmarks with Gemma-2-9B, the 10-head attention probe reached median AUROC 0.983 versus 0.951 for logistic regression, across 9 datasets from Kantamneni et al.

---

## 2. Anthropic: "Cost-Effective Constitutional Classifiers via Representation Re-use" (Cunningham et al. 2025)

### 2.1 Training data

- **Source.** "approximately 70,000 prompts, evenly split between" bioweapons-eliciting and harmless prompts. It is "an early version of the training dataset used in the Constitutional Classifiers paper". **All of it is synthetic**, "generated synthetically by the Claude 3.5 series of models according to a written constitution" that lists permitted and restricted content categories. "teaching the model novel ciphers" is listed as restricted.
- **Augmentation ("transformations").** Examples are rewritten by "converting the input into jailbreak templates, diverse languages, ciphers, and assistant pre-fills". The full pipeline is deferred to Section 3 and Appendix B of the Constitutional Classifiers paper, which I did not read.
- **Labels.** Binary, one per prompt, assigned by the constitution category the prompt was generated from. There are no per-token labels.
- **On-policy or off-policy.** Off-policy synthetic prompts. The classifiers see only the prompt, never the model's output.
- **Hard negatives and mining.** Not stated beyond what the constitution-driven harmless set contains.
- **Diversity in styles, lengths and formats.** Only what the transformations above provide. Length statistics are not stated; the cost section assumes an average input of 5,000 tokens.
- **Spurious correlations.** Not stated. The related-work section only notes that sparse autoencoders "can play a useful role in debugging the training datasets by detecting spurious correlations".

### 2.2 Architecture

- **Probe.** Linear logistic regression, $\sigma(w^Tf(h^{(l)})+b)$. The text says "using linear regression", but the formula and Appendix D are logistic regression with binary cross-entropy.
- **Layer.** Not stated. Early experiments "showed that linear probe quality remained consistently high right up until the final layer of the model". One layer $l$ per probe. Multi-layer use is not described.
- **Normalisation.** "Before the regression, the activations are normalised so that the overall distribution of input values has a mean of 0.0 and a standard deviation of 1.0." It is ambiguous whether this is per feature or global.
- **Comparison classifiers.**
  - "XL 1 Retrained Layer": fine-tune the last block (MLP plus attention) of the policy model as a classifier. Next-token objective completing a suffix with "harmful" or "not". Overhead about 4.25%.
  - Retraining 2, 4, 8 or 16 layers gives "no clear trend".
  - Smaller dedicated classifiers XS, S, M and L (0.6%, 2%, 9% and 25% overhead).
  - A prompted-XL baseline.

### 2.3 Token aggregation and readout

- **Mean-token probe.** Train and evaluate on the mean over all tokens. This is the worst performer, "worse than even the smallest tested model". "Mean-token probes perform worse on long inputs, as the dangerous content may be only a very small fraction of the context, causing the signal to be drowned by noise."
- **EMA probe.** Trained exactly like the mean probe, on mean-pooled activations. At evaluation, compute per-token **probabilities** $s_t=\sigma(w^Th_t)$, then $\text{EMA}_t=\alpha s_t+(1-\alpha)\text{EMA}_{t-1}$ with $\text{EMA}_0=0$, **$\alpha=0.5$** ("performed best in preliminary experiments"), and score $=\max_t \text{EMA}_t$. Performance is "between the strength of the S and M models", roughly equal to a classifier with 3% overhead, at a cost of about 0.1%. **Note the difference from GDM:** Anthropic applies the EMA to sigmoid probabilities, while GDM's equation applies it to linear logits.
- **Suffix probe.** Append a roughly 150-token suffix asking the model whether the dialog contains harmful biological information. Probe the final suffix token, $f(h)=h_T$. This "ensures our probes access a representation that contains the necessary information". The full suffix is in Appendix A of the post. It explicitly warns that harmful content may be "buried in text, disguised as fiction, or using code words". It is slightly better than EMA but costs about 3%.
- **Streaming.** Not stated beyond a remark that in output classification, probes could run concurrently with sampling.
- **Long transcripts.** EMA plus max is their answer to dilution in long inputs.

### 2.4 Training objective and procedure

- **Loss.** Logistic regression with L2 regularisation. "The regularization strength was 1e2 for the last-token probes and 1e4 for the mean-token probes", which were then used for EMA. The heavier regularisation "is necessary for the EMA probes as they are applied token-wise which has a much greater variance and lower regularization can cause numerical issues". The strength was "swept over and found to be the best-performing power of 10 in a previous experiment on similar data".
- **Epochs and batching.** "We trained all of the models for exactly one epoch ... about 2,200 batches with batch size 32." This clearly applies to the fine-tuned classifiers. Whether it also applies to the logistic-regression probes is not stated.
- **Early stopping.** Not stated.
- **Seeds.** Results average three data orderings. Appendix B says two runs, so the source is inconsistent.
- **Layer selection criterion.** Not stated.
- **Validation loss is a poor guide (Appendix C).** In-distribution validation loss does not predict out-of-distribution test performance. Partially trained classifiers had much higher validation loss but much better test performance.

### 2.5 Calibration and metrics

- **Metric.** "log-space AUROC": the area under TPR plotted against log-spaced FPR, for **FPR between 1e-3 and 1e-1**. They chose it because "regular AUROC is not a reliable measure of performance at low FPRs, and evaluating the TPR at a single FPR is too noisy".
- **Test data.**
  - **Harmful, part 1:** about 4,000 examples from a fixed set of dangerous biology questions, transformed by in-distribution and **held-out** transformations.
  - **Harmful, part 2:** about 2,500 red-team inputs from the Constitutional Classifiers red-teaming phase that elicited harmful output.
  - **Harmless:** single-turn WildChat plus the "Hard" subset of OR-bench, which contains many harmful-looking edge cases, so FPR is pessimistic.
- **Threshold choice.** No deployment threshold or false-positive budget is described. For two-stage cascades, the first-stage threshold is swept to vary the fraction of traffic passed on. Cost is computed on harmless traffic only, on the assumption that harmful inputs are a negligible fraction of real traffic.

### 2.6 Robustness

- **Tested.** Held-out transformations and fixed red-team prompts. Those red-team prompts were written against Claude 3.5-era classifiers, not against these probes.
- **Not tested: adaptive attacks.** "we have not studied the robustness of any of the classifiers in this paper to adaptive adversarial attacks". It is possible that "relative robustness under direct attack could be worse for probes (in part because they're integrated into the model being attacked)".
- **Cascades.** Two-stage cascades can beat the strong classifier alone on a fixed set, because the two stages make uncorrelated errors. But "if an attacker can make a weak model certain that the prompt is not dangerous then it will never be seen by the stronger model."

### 2.7 Lessons and failure modes

- Mean pooling is diluted on long inputs. EMA with max fixes this at almost no cost.
- Probing a consistent, task-relevant token position (the suffix) helps.
- Token-wise readout needs much stronger L2 regularisation: 1e4 versus 1e2.
- In-distribution validation loss does not predict generalisation.
- Probes and cascades need adaptive red-teaming before deployment.

---

## 3. Side-by-side

| Choice | GDM (Gemini) | Anthropic (cheap monitors) |
|---|---|---|
| Task | Cyber-misuse prompts | Bioweapons prompts |
| What is labelled | Whole prompt, binary | Whole prompt, binary |
| Training data source | Expert-curated attacks; real user traffic plus post-training data as negatives; expert hard negatives; synthetic multi-turn | All synthetic (Claude 3.5 following a constitution), then transformations (jailbreak templates, languages, ciphers, prefills) |
| Training size | 3,175 (Table 7) | About 70,000, 50/50 |
| Model outputs in training? | No | No |
| Layer | Middle of the model, one layer | Not stated, one layer |
| Normalisation | Not stated (LayerNorm in the AlphaEvolve probe) | Standardise to mean 0, std 1 |
| Best readout | Attention-weighted mean in 10-token windows, max over windows; or MultiMax | EMA α=0.5 over per-token sigmoid scores, max over tokens; or suffix-token probe |
| Linear-probe training | AdamW, lr 1e-4, weight decay 3e-3, full batch, 1,000 steps (but a logistic-regression solver did better) | Logistic regression, L2 1e2 (last-token) or 1e4 (mean/EMA) |
| Model selection | Best of 100 seeds by weighted validation error | Average of 2–3 data orderings |
| Threshold | Minimise weighted validation error (FNR 5, hard-negative FPR 2, benign FPR 50) on a different distribution from test | Not stated; report log-space AUROC over FPR 1e-3 to 1e-1 |
| Adversarial evaluation | Jailbreak transforms; LLM narrative rewrites checked by an autorater (probe miss rates 14–48%, 3.9% only for an over-triggering variant) | Fixed red-team prompts, not adaptive |

---

## 4. What this means for us

These are ranked by expected effect on our specific failure: the optimiser hides the computed number inside a scenic sentence and the probe misses it.

1. **Make the training data much more diverse, and include buried positives.** Our 2,400 short templated sentences resemble neither source. Anthropic uses about 70k examples with systematic transformations of each seed (templates, languages, ciphers, prefills). GDM mixes real traffic, expert hard negatives and multi-turn escalation.
   - Our analogue: positives in many registers (bare arithmetic, word problems, narrative or scenic prose, dialogue, lists, code comments), lengths and positions, with the computation embedded in a small part of longer benign text. This copies GDM's long-context construction, where the request is inserted before, after or in the middle of filler.
   - GDM shows that about 3k examples can be enough if the distributions are diverse.
2. **Curate hard negatives deliberately.** GDM gives benign false positives 25× the weight of hard-negative false positives. Our analogues:
   - scenic or narrative sentences that contain numbers but no computation,
   - stated results that are recalled rather than computed,
   - maths vocabulary without arithmetic,
   - the same templates with the computation removed.

   This is the main defence against the probe learning "number present" or "maths style" instead of "computing".
3. **Score every token and read out with a local maximum, not a mean.** Both sources find mean pooling diluted when the relevant content is a small fraction of the text, which is exactly what the scenic-sentence attack exploits. The options:
   - Anthropic: train on mean-pooled activations, then evaluate with EMA (α=0.5) over per-token sigmoid scores and take the max.
   - GDM: 10-token rolling windows with max over windows, or MultiMax.
   - Our code uses EMA α=0.1 over logits (`src/probe_evasion/probes.py`, `EMA_ALPHA = 0.1`). Both sources use α=0.5, which is a much shorter memory (about 2 tokens). Anthropic applies it to probabilities, GDM to logits.
   - **Expect over-triggering.** In GDM's results, the readouts that catch narrative rewrites best (EMA linear, MultiMax) had 13–75% false positives on long or multi-turn benign data. Any readout change has to be judged together with false positives on long benign text.
4. **Get the linear-probe mechanics right.**
   - Standardise activations to mean 0 and std 1 (Anthropic).
   - Use L2 logistic regression with a proper solver, sweeping the strength in powers of 10. Use heavier L2 when the probe will be applied per token (1e4 versus 1e2 in Anthropic's parameterisation).
   - GDM's gradient-descent linear probe lost about 0.13 AUROC to a logistic-regression solver.
   - Use a middle layer by default (GDM). Anthropic suggests late layers work too.
5. **Choose thresholds on a held-out validation distribution, and use a different distribution for test.** Use a weighted false-positive/false-negative objective that weights benign false positives heavily (GDM uses 50:5:2), or report TPR over FPR from 1e-3 to 1e-1 (Anthropic's log-space AUROC). Measuring FPR at 1e-3 needs thousands of realistic negatives (GDM used 8k for validation and 33k for test), far more than we have now. Make the train, validation and test splits disjoint by template or style, not just by row, and remove near-duplicate stems as GDM did.
6. **Build the red-team evaluation the way GDM built ART.** Have an LLM rewrite each positive into a narrative in one shot, repeated N times. Count a success only if an autorater confirms the target behaviour still happens; for us, that the model's answer is correct. Compare the adaptive optimiser against "repeat the first attempt N times". GDM found that adaptivity added nothing over repeats for their attacker, so this baseline is needed to show our optimiser adds anything. Keep red-team data out of training, as both sources did, so the evaluation stays clean.
7. **Train several seeds or data orderings and select on validation.** GDM uses 100 seeds; Anthropic uses 2–3 orderings. This is cheap for linear probes, and it lets us report seed variance rather than one run.
8. **Optional extensions, both of which change the threat model:**
   - **Suffix probe** (Anthropic): append a fixed question such as "did the preceding text require computing a result?" and probe the last token. This is a stronger probe, but the defender now edits the input.
   - **Small attention or rolling-attention probe** (GDM): beats logistic regression by about 3 AUROC points on Gemma-2-9B benchmarks. This means abandoning "linear probe".

### Recipe choices neither source specifies

- **Probing model outputs or on-policy data.** Both sources probe user prompts only. Neither says how to build probes over the model's own generations or reasoning, which is closer to "the model is computing X". Choosing between on-policy model text and off-policy sentences is our call.
- **Per-token labels or which tokens to supervise.** Both use whole-sequence labels only. Anthropic trains on mean-pooled activations and evaluates per token; GDM trains end to end on aggregated scores. Neither labels the span where the concept occurs.
- **Explicit spurious-correlation controls.** Neither describes length matching, lexical or number-presence controls, minimal-pair negatives, or checking what the probe direction responds to. GDM's leakage filtering by "stem" is the closest.
- **Adversarial training or hard-negative mining loops.** Neither retrains on red-team or optimiser-found examples. Both hold them out for evaluation only.
- **Layer selection procedure.** GDM fixes the middle layer; Anthropic does not state its layer. Neither describes a layer sweep or multi-layer concatenation. GDM cites CC++ (Cunningham et al. 2026) as the source that uses all layers.
- **Early stopping and class weighting.** GDM uses no early stopping and does not state class weights; Anthropic states neither. GDM's training mix is about 62% positives (1,969 of 3,175).
- **Threshold-setting traffic for a concept probe.** Both calibrate on misuse traffic against benign chat or coding traffic. There is no equivalent for "benign traffic" for a computation probe; we have to define it, for example general chat and prose with numbers.
- **EMA details.** The source of α=0.5 is "preliminary experiments" (Anthropic). Neither source says whether the EMA starts at 0 or at the first token beyond $\text{EMA}_0=0$, or whether system or template tokens are excluded from the max.
