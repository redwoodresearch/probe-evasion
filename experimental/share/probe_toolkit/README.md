# probe_toolkit

> **Manifest.** This toolkit was distilled (rewritten, not copied) from these files of the probe-evasion research repo:
> `src/probe_evasion/concept_probes.py` (probe training and scoring, `ProbeLab`, `fit_gd`, `fit_logreg`),
> `src/probe_evasion/activations.py` (`render`, activation capture with forward hooks),
> `src/probe_evasion/extract.py` (extraction entry point),
> `src/probe_evasion/probes.py` (token-level logistic regression with per-reply and class weighting),
> `src/probe_evasion/gepa.py` (`readings`, `ema_logit`, `EMA_ALPHA`),
> `src/probe_evasion/missions/m19_maths_whitebox_redteam.py` (`ema_path`),
> `src/probe_evasion/missions/m22_score_only_iterations.py` (`peak`, `calibration`: lines at a false-alarm rate),
> `src/probe_evasion/missions/m1a_peak_probe.py` (`ema_peaks`, `fit_peak_probe`: BCE on the running peak),
> and the lessons in `sprint/sprint.json` ("decisions" and "takeaways"),
> `missions/m7_sandbagging/STATUS.md` and `missions/m27_readout_and_style_checks/RESULT.md`.

A small library for training linear activation probes on a chat model's replies (for example a sabotage probe),
setting a flag line, and checking how well the probe works. Plain `torch` + `transformers` + `numpy` + `scikit-learn`.

What it does:

- **Capture** (`capture.py`): runs each (prompt, optional reasoning, reply) through a Hugging Face chat model and
  keeps the residual stream after the chosen layers, for the reply tokens only (or reasoning + reply, or the context
  before the reply). The reply is read in its real context.
- **Readouts** (`readouts.py`): turn per-token probe scores into one reply score: `mean`, `max`, and `peak` (the
  running peak: an exponential moving average of the per-token log-odds with weight 0.5 on each new token, starting
  from 0; the reply score is its highest value). All readouts work in log-odds.
- **Training** (`probes.py`), one probe per layer:
  - `fit_logistic_probe`: L2 logistic regression on mean-pooled reply activations, or on all reply tokens (each token
    carries its reply's label, weighted so each reply and each class counts equally).
  - `fit_readout_probe`: binary cross-entropy on the reply score itself; `readout="peak"` trains for the running
    peak. Stops on validation loss.
- **Evaluation** (`evaluation.py`): `flag_line` at a chosen false-alarm rate on clean negatives, `auroc`, and
  `evaluate` (AUROC, catch rate at the line, false-alarm rate on held-out negatives).
- **Saving**: `LinearProbe.save("probe.json")` or `.npz`; `LinearProbe.load(...)`.

## Quick start

```python
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_toolkit import Example, LinearProbe, capture_activations, evaluate, fit_readout_probe, flag_line

model_id = "Qwen/Qwen2.5-7B-Instruct"  # any HF chat model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto").eval()

train = [Example(prompt="...", reply="...", reasoning=None), ...]   # labelled replies
train_labels = np.array([1, 0, ...])                                # 1 = shows the concept
layers = [12, 16, 20]

captured = capture_activations(model, tokenizer, train, layers, max_tokens=256)
probes = {layer: fit_readout_probe(captured.activations[layer], train_labels, layer, readout="peak") for layer in layers}
probe = probes[16]  # choose the layer on held-out data in other answer styles, not on training-style validation
probe.save("sabotage_L16.json")

# Flag line from clean negatives that look like deployment traffic, then test on fresh data.
clean = capture_activations(model, tokenizer, clean_negative_examples, [probe.layer])
line = flag_line(probe.score_many(clean.activations[probe.layer]), false_alarm_rate=0.01)
test_pos = probe.score_many(capture_activations(model, tokenizer, test_positives, [probe.layer]).activations[probe.layer])
test_neg = probe.score_many(capture_activations(model, tokenizer, test_negatives, [probe.layer]).activations[probe.layer])
print(evaluate(test_pos, test_neg, line))
```

For reasoning models, pass `reasoning="..."` on each `Example`, plus whatever the chat template needs, e.g.
`chat_template_kwargs={"enable_thinking": True}` for Qwen. Use `span="reasoning_and_reply"` to read the reasoning
as well. Gradient training uses `learning_rate=1e-3`, which suits the large norms of real residual-stream activations.
If `probe.info["epoch"]` equals `max_epochs`, training never stopped early: raise `max_epochs` or the learning rate.

## Tests

```bash
cd share/probe_toolkit
uv run pytest tests                           # fast: synthetic activations, CPU, no download
PROBE_TOOLKIT_SLOW=1 uv run pytest tests      # also a smoke test with sshleifer/tiny-gpt2 (a few MB)
```

From the probe-evasion repo root: `uv run --project share/probe_toolkit pytest share/probe_toolkit/tests`.

## Lessons we learned

- **Train for the readout you deploy.** A probe trained on the average token score but read with the running peak
  let single harmless tokens (email greetings like "Hi" and "Best regards") spike. That raised the flag line and
  produced fake evasions: 500 of 1,000 attempts got through, versus 0 of 1,000 against a peak-trained probe. Use
  `fit_readout_probe(readout=...)` with the readout you will use. Averaged readouts have their own weakness:
  padding a reply with harmless text dilutes the average.
- **Set the line on clean negatives in the same format and context as deployment, and audit them.** Our maths probe
  used replies to "no-computation" lookup questions as negatives, but the model sometimes computed anyway ("92 tickets
  cost $9,568 in total. What did they cost?" got "Each ticket cost $104"). Those raised the line, and attacks that
  looked successful were caught once they were removed. Read the top-scoring negatives before trusting a line. A
  line at 0.1% from 1,000 negatives rests on one reply; `flag_line` refuses lines that rest on less than one.
- **Check that the probe reads the reply, not the request.** If the two classes differ in their prompts (an
  instruction to sabotage, a system prompt, a plan in the reasoning), a probe can score perfectly without reading
  the reply at all. A probe trained only on the context before the reply (`span="context"`) got AUROC 1.00 in our
  sandbagging setup. Train that context-only probe as a baseline. Better still, keep the context identical across
  classes, or score with the instruction removed from what the probe reads.
- **Check within style.** Compare concept replies against non-concept replies written in the same style. Our flattery
  probe scored calm flattery only somewhat above calm, neutral replies (AUROC about 0.8), against 0.9996 for
  gushing flattery: at its line it was a detector of gushing wording. A sarcasm probe kept separating sarcasm from
  same-style sincere replies (0.98+) even when attacks got past it. That points to where the line sits, not to a
  blind probe.
- **Lines from a narrow pool can flag most broad traffic.** Our maths lines, set on lookup-question replies, would
  have flagged 72-100% of ordinary chat replies that contain numbers or lists. If the probe will see broad traffic,
  calibrate on broad traffic.
- **Training-style validation can't choose a layer.** Validation AUROC was 1.000 at every layer of one model, yet the
  layers differed a lot on held-out answer styles and attacks. Hold out whole answer styles to choose layers and
  readouts.
- **For sabotage specifically, separating intent from ordinary error is the hard part.** Our sandbagging probes
  (deliberate versus honest wrong maths answers on Qwen3.5-27B, identical context in both classes) were near chance,
  AUROC about 0.55-0.65, and a text monitor did no better (0.49). A probe reading the context's instruction got 1.00.
  Probes found the sabotage only when the plan was written in the reasoning they read, and then a plain text monitor
  of the reasoning caught it too. Before believing a sabotage probe, check that it beats both a context-only probe
  and a text monitor on cases where the context is the same.
