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
from probe_evasion.chameleons import BASE_MODEL, CONCEPTS as CHAMELEON_CONCEPTS, gemma_image, load_probes
from probe_evasion.modal_app import app, hf_cache

DATA = Path("data/concepts")
PROBES = Path("results/concept_probes")
CONCEPTS = ("chemistry", "sarcasm", "maths")
MODELS = {"gemma": BASE_MODEL, "qwen35": config.TARGET_MODEL}
CANDIDATE_LAYERS = {"gemma": (12,), "qwen35": (12, 16, 24, 32)}  # Gemma: the paper's layer; Qwen: picked on validation
MAX_TRAIN_TOKENS = 256
MAX_CHEMISTRY_NEGATIVES = 1000  # keeps Qwen's cached training activations within memory


def encode(tok, model_key: str, prompt: str, response: str, reasoning: str | None = None) -> tuple[list[int], int]:
    """Token ids of the whole conversation and the index where the reply starts.
    Gemma: as the Neural Chameleons reference (chat-templated prompt and reply tokenized separately).
    Qwen3.5: as the rest of this repo (activations.render: think block, possibly empty, before the reply)."""
    if model_key == "gemma":
        p = tok(tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True),
                add_special_tokens=False).input_ids
        return p + tok(response, add_special_tokens=False).input_ids, len(p)
    from probe_evasion.activations import render

    text, start = render(tok, {"question": prompt, "reasoning": reasoning or "", "response": response})
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    first = next(i for i, (s, _) in enumerate(enc["offset_mapping"]) if s >= start)
    return enc["input_ids"], first


@app.cls(image=gemma_image, gpu="H200", volumes={"/hf-cache": hf_cache}, timeout=3 * 60 * 60, scaledown_window=600,
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

    def _reply_acts(self, items: list[dict], layers: tuple[int, ...], max_tokens: int | None = None):
        """Per item, ({layer: [reply_tokens, d] bf16 on GPU}, reply token ids)."""
        import torch

        captured = {}
        hooks = [self.layers[l].register_forward_hook(
            lambda m, i, o, l=l: captured.__setitem__(l, o[0] if isinstance(o, tuple) else o)) for l in layers]
        out = []
        try:
            for it in items:
                ids, start = encode(self.tok, self.model_key, it["prompt"], it["response"], it.get("reasoning"))
                end = len(ids) if max_tokens is None else min(len(ids), start + max_tokens)
                with torch.no_grad():
                    self.model(input_ids=torch.tensor([ids[:end]], device="cuda"), use_cache=False)
                out.append(({l: captured[l][0, start:end].to(torch.bfloat16) for l in layers}, ids[start:end]))
        finally:
            for h in hooks:
                h.remove()
        return out

    @modal.method()
    def train(self, rows: list[dict], seed: int = 0) -> dict:
        """rows: {concept, label, prompt, response}. Returns per concept the probe at the best validation layer."""
        import torch
        from sklearn.metrics import roc_auc_score

        layers = CANDIDATE_LAYERS[self.model_key]
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
                score = lambda w, b, ids: torch.stack([torch.sigmoid(X[i] @ w + b).mean() for i in ids])
                w = torch.zeros(X[idx[0]].shape[-1], device="cuda", requires_grad=True)
                b = torch.zeros((), device="cuda", requires_grad=True)
                opt = torch.optim.AdamW([w, b], lr=1e-3, weight_decay=0.01)
                layer_best, state, patience = -1.0, None, 0
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
                    if auc > layer_best + 1e-4:
                        layer_best, state, patience = auc, (w.detach().clone(), b.detach().clone()), 0
                    else:
                        patience += 1
                        if patience >= 3:
                            break
                print(f"{concept} layer {layer}: val AUROC {layer_best:.4f} ({epoch + 1} epochs)", flush=True)
                if best is None or layer_best > best["val_auroc"]:
                    best = {"layer": layer, "val_auroc": layer_best, "weight": state[0].cpu().tolist(), "bias": state[1].item(),
                            "n_train": len(train), "n_val": len(val)}
                del X
            result[concept] = best
        return result

    @modal.method()
    def released(self) -> dict:
        """The Neural Chameleons chemistry probe, as weights (torch is only available in the container)."""
        W, b = load_probes(Path("/probes"))
        k = list(CHAMELEON_CONCEPTS).index("chemistry-based")
        return {"layer": 12, "weight": W[k].tolist(), "bias": float(b[k])}

    @modal.method()
    def score(self, tag: int, items: list[dict], probes: dict) -> tuple[int, list[dict]]:
        """items: {prompt, response, reasoning?}; probes: {name: {layer, weight, bias}}.
        Returns per item {"tokens": reply token strings, name: per-token probabilities over the reply}."""
        import torch

        layers = tuple(sorted({p["layer"] for p in probes.values()}))
        params = {n: (p["layer"], torch.tensor(p["weight"], device="cuda"), torch.tensor(p["bias"], device="cuda"))
                  for n, p in probes.items()}
        out = []
        for acts, ids in self._reply_acts(items, layers):
            out.append({"tokens": [self.tok.decode([i]) for i in ids],
                        **{n: torch.sigmoid(acts[l].float() @ w + b).round(decimals=4).cpu().tolist() for n, (l, w, b) in params.items()}})
        return tag, out


def probes_for(model_key: str) -> dict:
    """{concept: {layer, weight, bias}} for a model. Gemma's chemistry probe is the paper's released one."""
    trained = json.loads((PROBES / f"{model_key}.json").read_text())
    if model_key == "gemma":
        trained["chemistry"] = json.loads((PROBES / "gemma_released_chemistry.json").read_text())
    return trained


def score(model_key: str, items: list[dict], probes: dict, chunk: int = 32) -> list[dict]:
    lab = ProbeLab(model_key=model_key)
    parts = dict(lab.score.starmap([(i, items[s:s + chunk], probes) for i, s in enumerate(range(0, len(items), chunk))],
                                   order_outputs=False))
    return [x for i in range(len(parts)) for x in parts[i]]


@app.local_entrypoint()
def train_concepts(model: str = "gemma"):
    rows = [json.loads(l) for l in (DATA / "probe_rows.jsonl").read_text().splitlines()]
    rng = random.Random(0)
    chem_neg = [r for r in rows if r["concept"] == "chemistry" and r["label"] == 0]
    keep = set(map(id, rng.sample(chem_neg, min(MAX_CHEMISTRY_NEGATIVES, len(chem_neg)))))
    rows = [r for r in rows if not (r["concept"] == "chemistry" and r["label"] == 0 and id(r) not in keep)]
    PROBES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lab = ProbeLab(model_key=model)
    out = lab.train.remote(rows)
    (PROBES / f"{model}.json").write_text(json.dumps(out))
    if model == "gemma":
        (PROBES / "gemma_released_chemistry.json").write_text(json.dumps(lab.released.remote()))
    print({c: {k: v for k, v in p.items() if k not in ("weight", "bias")} for c, p in out.items()}, f"{time.time() - t0:.0f}s")
