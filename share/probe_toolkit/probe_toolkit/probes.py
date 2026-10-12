"""Linear probes on one layer's residual stream: training, scoring, saving.

`acts` throughout is a list with one float array of shape [reply_tokens, d_model] per reply, and `labels` is 1 for
replies that show the concept and 0 for replies that don't.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from probe_toolkit.readouts import READOUTS, readout_batch_torch

LOGISTIC_STRENGTHS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


@dataclass
class LinearProbe:
    layer: int
    weight: np.ndarray
    bias: float
    readout: str
    info: dict = field(default_factory=dict)

    def token_logits(self, reply_acts: np.ndarray) -> np.ndarray:
        return np.asarray(reply_acts, dtype=np.float32) @ self.weight.astype(np.float32) + self.bias

    def score(self, reply_acts: np.ndarray, readout: str | None = None) -> float:
        return READOUTS[readout or self.readout](self.token_logits(reply_acts))

    def score_many(self, acts: list[np.ndarray], readout: str | None = None) -> np.ndarray:
        return np.array([self.score(a, readout) for a in acts])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        meta = {"layer": self.layer, "bias": self.bias, "readout": self.readout, "info": self.info}
        if path.suffix == ".json":
            path.write_text(json.dumps({**meta, "weight": self.weight.tolist()}))
        elif path.suffix == ".npz":
            np.savez(path, weight=self.weight, meta=np.array(json.dumps(meta)))
        else:
            raise ValueError(f"save as .json or .npz, not {path.suffix!r}")

    @classmethod
    def load(cls, path: str | Path) -> "LinearProbe":
        path = Path(path)
        if path.suffix == ".json":
            data = json.loads(path.read_text())
            weight = np.array(data.pop("weight"), dtype=np.float32)
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as stored:
                weight, data = stored["weight"], json.loads(str(stored["meta"]))
        else:
            raise ValueError(f"load a .json or .npz file, not {path.suffix!r}")
        return cls(layer=data["layer"], weight=weight, bias=float(data["bias"]), readout=data["readout"], info=data["info"])


def split_train_val(n_examples: int, val_fraction: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n_examples)
    n_val = int(n_examples * val_fraction)
    return order[n_val:], order[:n_val]


def fit_logistic_probe(acts: list[np.ndarray], labels, layer: int, pooling: str = "mean", readout: str = "mean",
                       val_fraction: float = 0.2, seed: int = 0) -> LinearProbe:
    """L2 logistic regression on standardised activations, regularisation strength picked by validation AUROC under
    `readout`. The standardisation is folded into the weights, so the probe scores raw per-token activations.

    pooling "mean": one training row per reply (its mean activation).
    pooling "all_tokens": one row per token, each carrying its reply's label; rows are weighted so every reply counts
    equally and the two classes count equally, so long replies don't dominate.
    """
    labels = np.asarray(labels, dtype=int)
    train, val = split_train_val(len(acts), val_fraction, seed)
    if pooling == "mean":
        rows = np.stack([acts[i].mean(0) for i in train])
        row_labels, row_weights = labels[train], None
    elif pooling == "all_tokens":
        class_share = {c: np.mean(labels[train] == c) for c in (0, 1)}
        rows = np.concatenate([acts[i] for i in train])
        row_labels = np.concatenate([np.full(len(acts[i]), labels[i]) for i in train])
        row_weights = np.concatenate([np.full(len(acts[i]), 1 / (len(acts[i]) * class_share[labels[i]])) for i in train])
    else:
        raise ValueError(f"unknown pooling {pooling!r}")
    centre, scale = rows.mean(0), rows.std(0) + 1e-6
    best = None
    for strength in LOGISTIC_STRENGTHS:
        classifier = LogisticRegression(C=strength, max_iter=2000)
        classifier.fit((rows - centre) / scale, row_labels, sample_weight=row_weights)
        weight = classifier.coef_[0] / scale
        bias = float(classifier.intercept_[0] - (classifier.coef_[0] * centre / scale).sum())
        probe = LinearProbe(layer, weight.astype(np.float32), bias, readout)
        val_auroc = roc_auc_score(labels[val], probe.score_many([acts[i] for i in val]))
        if best is None or val_auroc > best.info["val_auroc"] + 1e-4:
            probe.info = {"method": f"logistic regression, {pooling} pooling", "C": strength, "val_auroc": float(val_auroc),
                          "n_train": len(train), "n_val": len(val), "seed": seed}
            best = probe
    return best


def fit_readout_probe(acts: list[np.ndarray], labels, layer: int, readout: str = "peak", val_fraction: float = 0.2,
                      seed: int = 0, max_epochs: int = 15, patience: int = 3, batch_size: int = 32, learning_rate: float = 1e-3,
                      weight_decay: float = 0.01, device: str | None = None) -> LinearProbe:
    """Binary cross-entropy on the reply score itself, i.e. trained for the readout it will be deployed with.

    readout "peak" is the running-peak-trained probe. Stops on validation loss, not validation AUROC: AUROC often
    reaches 1.0 after the first epoch, which would freeze the weights there.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(seed)
    train, val = split_train_val(len(acts), val_fraction, seed)
    features = [torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device) for a in acts]
    targets = torch.as_tensor(np.asarray(labels), dtype=torch.float32, device=device)
    weight = torch.zeros(features[0].shape[-1], device=device, requires_grad=True)
    bias = torch.zeros((), device=device, requires_grad=True)
    optimiser = torch.optim.AdamW([weight, bias], lr=learning_rate, weight_decay=weight_decay)
    reply_scores = lambda ids: readout_batch_torch([features[i] @ weight + bias for i in ids], readout)
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    best_loss, best_state, epochs_without_gain = float("inf"), None, 0
    for epoch in range(max_epochs):
        order = train[torch.randperm(len(train), generator=generator).numpy()]
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            loss = bce(reply_scores(batch), targets[batch])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            val_scores = reply_scores(val)
            val_loss = bce(val_scores, targets[val]).item()
        if val_loss < best_loss - 1e-4:
            best_loss, epochs_without_gain = val_loss, 0
            best_state = {"weight": weight.detach().cpu().numpy().copy(), "bias": bias.item(), "epoch": epoch + 1,
                          "val_auroc": float(roc_auc_score(targets[val].cpu().numpy(), val_scores.cpu().numpy()))}
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= patience:
                break
    return LinearProbe(layer, best_state["weight"], best_state["bias"], readout,
                       {"method": f"BCE on the {readout} readout", "val_loss": best_loss, "val_auroc": best_state["val_auroc"],
                        "epoch": best_state["epoch"], "n_train": len(train), "n_val": len(val), "seed": seed})


def fit_per_layer(acts_by_layer: dict[int, list[np.ndarray]], labels, fit=fit_readout_probe, **fit_kwargs) -> dict[int, LinearProbe]:
    """One probe per layer. Choose a layer on data held out from training (ideally other answer styles), not on
    the training-style validation AUROC, which is often 1.0 everywhere."""
    return {layer: fit(acts, labels, layer, **fit_kwargs) for layer, acts in acts_by_layer.items()}
