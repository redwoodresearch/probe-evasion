"""Concept probes for the optimized-prompt evasion experiments, on Gemma-2-9B and Qwen3.5-27B.

Training follows the Neural Chameleons recipe (as in chameleons.train_probes_remote): per-token sigmoid on one
layer's residual stream, mean over the reply tokens, binary cross-entropy with AdamW, early stopping on validation
AUROC. Each probe is then read two ways: the paper's mean over reply tokens, and a stricter max over reply tokens.
Only reply tokens are scored; the prompt (and, for Qwen, its reasoning) is in context but not scored.

    uv run modal run -m probe_evasion.concept_probes::train_concepts --model gemma
    uv run modal run -m probe_evasion.concept_probes::train_concepts --model qwen35
"""

import json
import random
import time
from pathlib import Path

import modal
import numpy as np

from probe_evasion import config
from probe_evasion.chameleons import BASE_MODEL, CONCEPTS as CHAMELEON_CONCEPTS, REPO, load_probes
from probe_evasion.modal_app import app, hf_cache, vllm_base_image

DATA = Path("data/concepts")
PROBES = Path("results/concept_probes")
CONCEPTS = ("chemistry", "sarcasm", "maths")
MODELS = {"gemma": BASE_MODEL, "qwen35": config.TARGET_MODEL}
CANDIDATE_LAYERS = {"gemma": (12,), "qwen35": (12, 16, 24, 32)}  # Gemma: the paper's layer; Qwen: picked on validation
MAX_TRAIN_TOKENS = 256
MAX_CHEMISTRY_NEGATIVES = 1000  # keeps Qwen's cached training activations within memory

# flash-linear-attention: fast kernels for Qwen3.5's linear attention (the fallback is far slower)
probe_image = (vllm_base_image.uv_pip_install("flash-linear-attention", "accelerate", "scikit-learn")
               .add_local_python_source("probe_evasion").add_local_dir(REPO / "probes", "/probes"))


def encode(tok, model_key: str, prompt: str, response: str, reasoning: str | None = None,
           messages: list[dict] | None = None) -> tuple[list[int], int]:
    """Token ids of the whole conversation and the index where the reply starts. `messages` (Gemma only) replaces
    the single user turn with a whole earlier conversation ending in a user turn.
    Gemma: as the Neural Chameleons reference (chat-templated prompt and reply tokenized separately).
    Qwen3.5: as the rest of this repo (activations.render: think block, possibly empty, before the reply)."""
    if model_key == "gemma":
        p = tok(tok.apply_chat_template(messages or [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True),
                add_special_tokens=False).input_ids
        return p + tok(response, add_special_tokens=False).input_ids, len(p)
    from probe_evasion.activations import render

    assert messages is None, "multi-turn scoring is only implemented for Gemma"
    text, start = render(tok, {"question": prompt, "reasoning": reasoning or "", "response": response})
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    first = next(i for i, (s, _) in enumerate(enc["offset_mapping"]) if s >= start)
    return enc["input_ids"], first


@app.cls(image=probe_image, gpu="H200", volumes={"/hf-cache": hf_cache}, timeout=3 * 60 * 60, scaledown_window=600,
         max_containers=16, memory=131072)
class ProbeLab:
    model_key: str = modal.parameter(default="gemma")

    @modal.enter()
    def load(self):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        model_id = MODELS[self.model_key]
        snapshot_download(model_id)
        hf_cache.commit()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.model_key == "gemma":
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda",
                                                              attn_implementation="eager").eval()
            self.layers = self.model.model.layers
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
            self.layers = next(m for n, m in self.model.named_modules()
                               if n.endswith("language_model.layers") and isinstance(m, torch.nn.ModuleList))

    def _reply_acts(self, items: list[dict], layers: tuple[int, ...], max_tokens: int | None = None, with_prompt: bool = False):
        """Per item, ({layer: [reply_tokens, d] bf16 on GPU}, reply token ids). with_prompt: every token after the
        first (beginning-of-text) token instead, prompt included; the reply starts at len(ids) - n_reply."""
        import torch

        captured = {}
        hooks = [self.layers[l].register_forward_hook(
            lambda m, i, o, l=l: captured.__setitem__(l, o[0] if isinstance(o, tuple) else o)) for l in layers]
        out = []
        try:
            for it in items:
                ids, start = encode(self.tok, self.model_key, it["prompt"], it["response"], it.get("reasoning"), it.get("messages"))
                end = len(ids) if max_tokens is None else min(len(ids), start + max_tokens)
                with torch.no_grad():
                    self.model(input_ids=torch.tensor([ids[:end]], device="cuda"), use_cache=False)
                first = 1 if with_prompt else start
                out.append(({l: captured[l][0, first:end].to(torch.bfloat16) for l in layers}, ids[first:end]))
        finally:
            for h in hooks:
                h.remove()
        return out

    @modal.method()
    def train(self, rows: list[dict], seed: int = 0, layers: tuple[int, ...] | None = None, per_layer: bool = False,
              method: str = "gd") -> dict:
        """rows: {concept, label, prompt, response}. Returns per concept the probe at the best validation layer, or with
        per_layer one probe per layer, named "<concept>@L<layer>". method: "gd" (Neural Chameleons: per-token sigmoid,
        mean, BCE with AdamW) or "logreg" (Anthropic / Google DeepMind style: standardised reply-mean activations,
        L2 logistic regression swept on validation, applied per token)."""
        import torch
        from sklearn.metrics import roc_auc_score

        layers = tuple(layers or CANDIDATE_LAYERS[self.model_key])
        acts = [{l: a.cpu() for l, a in r.items()} for r, _ in self._reply_acts(rows, layers, MAX_TRAIN_TOKENS)]
        gen = torch.Generator().manual_seed(seed)
        result = {}
        for concept in CONCEPTS:
            idx = [i for i, r in enumerate(rows) if r["concept"] == concept]
            if not idx:
                continue
            perm = torch.randperm(len(idx), generator=gen).tolist()
            n_val = len(idx) // 5
            val, train = [idx[i] for i in perm[:n_val]], [idx[i] for i in perm[n_val:]]
            y = {i: float(rows[i]["label"]) for i in idx}
            best = None
            for layer in layers:
                X = {i: acts[i][layer].cuda().float() for i in idx}
                w, bias, layer_best, steps = (fit_logreg if method == "logreg" else fit_gd)(X, train, val, y, gen)
                state = (w, bias)
                epoch = steps - 1
                print(f"{concept} layer {layer} ({method}): val AUROC {layer_best:.4f} ({epoch + 1} steps)", flush=True)
                probe = {"layer": layer, "val_auroc": layer_best, "weight": state[0].cpu().tolist(), "bias": state[1].item(),
                         "n_train": len(train), "n_val": len(val)}
                if per_layer:
                    result[f"{concept}@L{layer}"] = probe
                if best is None or layer_best > best["val_auroc"]:
                    best = probe
                del X
            if not per_layer:
                result[concept] = best
        return result

    @modal.method()
    def released(self) -> dict:
        """The Neural Chameleons chemistry probe, as weights (torch is only available in the container)."""
        W, b = load_probes(Path("/probes"))
        k = list(CHAMELEON_CONCEPTS).index("chemistry-based")
        return {"layer": 12, "weight": W[k].tolist(), "bias": float(b[k])}

    @modal.method()
    def score(self, tag: int, items: list[dict], probes: dict, with_prompt: bool = False) -> tuple[int, list[dict]]:
        """items: {prompt, response, reasoning?}; probes: {name: {layer, weight, bias}}.
        Returns per item {"tokens": reply token strings, name: per-token probabilities over the reply}."""
        import torch

        layers = tuple(sorted({p["layer"] for p in probes.values()}))
        params = {n: (p["layer"], torch.tensor(p["weight"], device="cuda"), torch.tensor(p["bias"], device="cuda"))
                  for n, p in probes.items()}
        out = []
        for acts, ids in self._reply_acts(items, layers, with_prompt=with_prompt):
            logits = {n: acts[l].float() @ w + b for n, (l, w, b) in params.items()}
            # probabilities for display; raw logits for the readouts (rounded probabilities saturate at exactly 1)
            out.append({"tokens": [self.tok.decode([i]) for i in ids],
                        **{n: torch.sigmoid(x).round(decimals=4).cpu().tolist() for n, x in logits.items()},
                        **{f"{n}__logits": x.round(decimals=3).cpu().tolist() for n, x in logits.items()}})
        return tag, out


def fit_gd(X: dict, train: list[int], val: list[int], y: dict, gen):
    """Neural Chameleons recipe. Returns (weight, bias, val AUROC, epochs)."""
    import torch
    from sklearn.metrics import roc_auc_score

    score = lambda w, b, ids: torch.stack([torch.sigmoid(X[i] @ w + b).mean() for i in ids])
    w = torch.zeros(X[train[0]].shape[-1], device="cuda", requires_grad=True)
    b = torch.zeros((), device="cuda", requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=1e-3, weight_decay=0.01)
    best, state, patience = -1.0, None, 0
    for epoch in range(40):
        order = torch.randperm(len(train), generator=gen).tolist()
        for s in range(0, len(order), 32):
            batch = [train[j] for j in order[s:s + 32]]
            p = score(w, b, batch).clamp(1e-6, 1 - 1e-6)
            loss = torch.nn.functional.binary_cross_entropy(p, torch.tensor([y[i] for i in batch], device="cuda"))
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            auc = roc_auc_score([y[i] for i in val], score(w, b, val).cpu().numpy())
        if auc > best + 1e-4:
            best, state, patience = auc, (w.detach().clone(), b.detach().clone()), 0
        else:
            patience += 1
            if patience >= 3:
                break
    return state[0], state[1], best, epoch + 1


LOGREG_C = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def fit_logreg(X: dict, train: list[int], val: list[int], y: dict, gen):
    """Standardised mean-over-reply activations, L2 logistic regression with the strength picked on validation AUROC;
    the weights are folded back through the standardisation so the probe scores raw per-token activations."""
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    mean_act = {i: X[i].mean(0).cpu().numpy() for i in train + val}
    Xtr = np.stack([mean_act[i] for i in train])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    z = lambda ids: (np.stack([mean_act[i] for i in ids]) - mu) / sd
    best = None
    for C in LOGREG_C:
        clf = LogisticRegression(C=C, max_iter=2000).fit(z(train), [y[i] for i in train])
        auc = roc_auc_score([y[i] for i in val], clf.decision_function(z(val)))
        if best is None or auc > best[0] + 1e-4:
            best = (auc, clf)
    auc, clf = best
    w = clf.coef_[0] / sd
    b = float(clf.intercept_[0] - (clf.coef_[0] * mu / sd).sum())
    return torch.tensor(w, device="cuda", dtype=torch.float32), torch.tensor(b, device="cuda"), auc, len(LOGREG_C)


def probes_for(model_key: str) -> dict:
    """{concept: {layer, weight, bias}} for a model. Gemma's chemistry probe is the paper's released one."""
    trained = json.loads((PROBES / f"{model_key}.json").read_text())
    if model_key == "gemma":
        trained["chemistry"] = json.loads((PROBES / "gemma_released_chemistry.json").read_text())
    return trained


def score(model_key: str, items: list[dict], probes: dict, chunk: int = 32, with_prompt: bool = False) -> list[dict]:
    lab = ProbeLab(model_key=model_key)
    parts = dict(lab.score.starmap([(i, items[s:s + chunk], probes, with_prompt) for i, s in enumerate(range(0, len(items), chunk))],
                                   order_outputs=False))
    return [x for i in range(len(parts)) for x in parts[i]]


@app.local_entrypoint()
def train_concepts(model: str = "gemma", concepts: str = "", rows_file: str = "probe_rows.jsonl", layers: str = "", name: str = "",
                   method: str = "gd", extra_rows: str = ""):
    """--concepts maths: retrain only those probes and merge them into the model's existing probe file.
    --layers 6,12,18 --name X: one probe per layer per concept, written to results/concept_probes/X.json (a probe-ladder
    variant; the probes the searches use are untouched)."""
    if layers:
        assert name, "per-layer probes go to their own file"
    rows = [json.loads(l) for l in (DATA / rows_file).read_text().splitlines()]
    if extra_rows:  # e.g. narrative rewrites added on top of the base training set
        rows += [json.loads(l) for l in (DATA / extra_rows).read_text().splitlines()]
    if concepts:
        rows = [r for r in rows if r["concept"] in concepts.split(",")]
    rng = random.Random(0)
    chem_neg = [r for r in rows if r["concept"] == "chemistry" and r["label"] == 0]
    keep = set(map(id, rng.sample(chem_neg, min(MAX_CHEMISTRY_NEGATIVES, len(chem_neg)))))
    rows = [r for r in rows if not (r["concept"] == "chemistry" and r["label"] == 0 and id(r) not in keep)]
    PROBES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lab = ProbeLab(model_key=model)
    if layers:
        out = lab.train.remote(rows, layers=tuple(int(x) for x in layers.split(",")), per_layer=True, method=method)
        (PROBES / f"{name}.json").write_text(json.dumps(out))
        print({k: round(v["val_auroc"], 4) for k, v in out.items()}, f"{time.time() - t0:.0f}s")
        return
    out = lab.train.remote(rows)
    if concepts:
        out = {**json.loads((PROBES / f"{model}.json").read_text()), **out}
    (PROBES / f"{model}.json").write_text(json.dumps(out))
    if model == "gemma" and not concepts:
        (PROBES / "gemma_released_chemistry.json").write_text(json.dumps(lab.released.remote()))
    print({c: {k: v for k, v in p.items() if k not in ("weight", "bias")} for c, p in out.items()}, f"{time.time() - t0:.0f}s")
