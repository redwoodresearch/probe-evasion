"""Linear probes on code-span activations.

One logistic-regression probe per layer, trained on tokens: every code token carries its
example's label, and examples are weighted equally (and classes balanced) so long programs
don't dominate. An example's score is the max over tokens of an exponential moving average
of per-token logits (EMA pooling); mean-of-logits is reported alongside as a sanity check."""

import numpy as np

from probe_evasion.modal_app import ACTS_DIR, PROBE_LAYERS, acts_volume, app, vllm_image

EMA_ALPHA = 0.1


def ema_max(token_scores: np.ndarray, alpha: float = EMA_ALPHA) -> float:
    s, best = token_scores[0], token_scores[0]
    for x in token_scores[1:]:
        s = alpha * x + (1 - alpha) * s
        best = max(best, s)
    return float(best)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def _load(name: str, n: int, indices: list[int] | None = None) -> list[np.ndarray]:
    acts_volume.reload()  # a warm container's view can predate activations written since it started
    return [np.load(f"{ACTS_DIR}/{name}/{i}.npy") for i in (indices if indices is not None else range(n))]


def _fit_layer(xs: list[np.ndarray], labels: np.ndarray, weight_decay: float, steps: int = 300) -> dict:
    import torch

    x = torch.tensor(np.concatenate(xs), dtype=torch.float32, device="cuda")
    y = torch.tensor(np.concatenate([np.full(len(a), l) for a, l in zip(xs, labels)]), dtype=torch.float32, device="cuda")
    class_frac = {c: (labels == c).mean() for c in (0, 1)}
    w_tok = torch.tensor(
        np.concatenate([np.full(len(a), 1 / (len(a) * class_frac[l])) for a, l in zip(xs, labels)]), device="cuda"
    )
    w_tok = w_tok / w_tok.sum()
    mu, sd = x.mean(0), x.std(0) + 1e-6
    x = (x - mu) / sd
    w = torch.zeros(x.shape[1], device="cuda", requires_grad=True)
    b = torch.zeros(1, device="cuda", requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (w_tok * torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, y, reduction="none")).sum()
        loss = loss + weight_decay * (w**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return {k: v.detach().cpu().numpy() for k, v in {"w": w, "b": b, "mu": mu, "sd": sd}.items()}


def _token_scores(acts: np.ndarray, probe: dict) -> np.ndarray:
    return ((acts.astype(np.float32) - probe["mu"]) / probe["sd"]) @ probe["w"] + probe["b"]


@app.function(image=vllm_image, gpu="L40S", volumes={ACTS_DIR: acts_volume}, timeout=60 * 60)
def train(name: str, labels: list[int], problem_ids: list[str], val_frac: float = 0.2, weight_decay: float = 1e-3,
          parts: list[tuple[str, list[int]]] | None = None) -> dict:
    """Train one probe per layer on `name` (or on the concatenation of `parts`: (activations name, file indices)),
    holding out val_frac of problems. Returns probes (refit on all data) plus validation AUROCs."""
    labels_arr = np.array(labels)
    acts = [a for part, idx in parts for a in _load(part, len(idx), idx)] if parts else _load(name, len(labels))
    assert len(acts) == len(labels)
    rng = np.random.default_rng(0)
    pids = sorted(set(problem_ids))
    val_pids = set(rng.choice(pids, size=max(1, int(len(pids) * val_frac)), replace=False))
    is_val = np.array([p in val_pids for p in problem_ids])

    out = {}
    for li, layer in enumerate(PROBE_LAYERS):
        xs = [a[:, li] for a in acts]
        probe = _fit_layer([x for x, v in zip(xs, is_val) if not v], labels_arr[~is_val], weight_decay)
        val_tok = [_token_scores(x, probe) for x, v in zip(xs, is_val) if v]
        val_labels = labels_arr[is_val]
        out[layer] = {
            "val_auroc_ema": auroc(np.array([ema_max(s) for s in val_tok]), val_labels),
            "val_auroc_mean": auroc(np.array([s.mean() for s in val_tok]), val_labels),
            "n_val": int(is_val.sum()),
            "probe": _fit_layer(xs, labels_arr, weight_decay),
        }
    return out


@app.function(image=vllm_image, volumes={ACTS_DIR: acts_volume}, timeout=60 * 60)
def score(name: str, n: int, probes: dict, indices: list[int] | None = None) -> list[dict]:
    """Per-example {layer: {ema, mean}} scores for activations `name` under `probes` (files 0..n-1, or `indices`)."""
    acts = _load(name, n, indices)
    return [
        {layer: {"ema": ema_max(s := _token_scores(a[:, li], probes[layer])), "mean": float(s.mean())}
         for li, layer in enumerate(PROBE_LAYERS) if layer in probes}
        for a in acts
    ]

