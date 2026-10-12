"""Turning a probe's per-token scores into one score per reply.

Every readout takes per-token log-odds (the probe's raw linear output) and returns log-odds, so scores never saturate
at a rounded probability of 1.0 and all readouts can share the same flag-line code. Use `sigmoid` for display.
"""

import numpy as np
import torch

RUNNING_PEAK_ALPHA = 0.5


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=np.float64)))


def mean_readout(token_logits: np.ndarray) -> float:
    """Equal to the probe applied to the mean-pooled activations, since the probe is linear."""
    assert len(token_logits) > 0, "no tokens to read"
    return float(np.mean(token_logits))


def max_readout(token_logits: np.ndarray) -> float:
    assert len(token_logits) > 0, "no tokens to read"
    return float(np.max(token_logits))


def running_average_path(token_logits: np.ndarray, alpha: float = RUNNING_PEAK_ALPHA) -> np.ndarray:
    """Exponential moving average of the per-token log-odds, starting from a neutral 0.

    Starting from 0 rather than the first token matters: starting from the first token lets a short reply with a
    low-scoring opening word stay below the line.
    """
    state, path = 0.0, []
    for value in token_logits:
        state = alpha * value + (1 - alpha) * state
        path.append(state)
    return np.array(path)


def running_peak_readout(token_logits: np.ndarray, alpha: float = RUNNING_PEAK_ALPHA) -> float:
    """Highest value of the running average: the reply score most deployed probes use (alpha 0.5)."""
    assert len(token_logits) > 0, "no tokens to read"
    return float(running_average_path(token_logits, alpha).max())


READOUTS = {"mean": mean_readout, "max": max_readout, "peak": running_peak_readout}


def readout_batch_torch(token_logits: list[torch.Tensor], readout: str, alpha: float = RUNNING_PEAK_ALPHA) -> torch.Tensor:
    """Differentiable version of READOUTS for a ragged batch of per-token log-odds; returns one score per reply."""
    lengths = torch.tensor([len(x) for x in token_logits], device=token_logits[0].device)
    assert (lengths > 0).all(), "no tokens to read"
    padded = torch.nn.utils.rnn.pad_sequence(token_logits, batch_first=True)
    is_real_token = torch.arange(padded.shape[1], device=padded.device)[None, :] < lengths[:, None]
    if readout == "mean":
        return (padded * is_real_token).sum(1) / lengths
    if readout == "max":
        return padded.masked_fill(~is_real_token, float("-inf")).max(1).values
    assert readout == "peak", f"unknown readout {readout!r}"
    state = torch.zeros(padded.shape[0], device=padded.device)
    path = []
    for t in range(padded.shape[1]):
        state = alpha * padded[:, t] + (1 - alpha) * state
        path.append(state)
    return torch.stack(path, 1).masked_fill(~is_real_token, float("-inf")).max(1).values
