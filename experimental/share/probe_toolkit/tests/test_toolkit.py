"""Fast tests run on synthetic activations (CPU, no download). The tiny-model smoke test needs
PROBE_TOOLKIT_SLOW=1 and downloads sshleifer/tiny-gpt2 (a few MB)."""

import os

import numpy as np
import pytest
import torch

from probe_toolkit import (Example, LinearProbe, READOUTS, auroc, capture_activations, evaluate, fit_logistic_probe,
                           fit_readout_probe, flag_line, flagged_fraction, readout_batch_torch, render,
                           running_average_path)

D_MODEL = 16
CONCEPT_DIRECTION = np.eye(D_MODEL)[0]


def synthetic_replies(n_replies: int, seed: int, spike: float = 6.0) -> tuple[list[np.ndarray], np.ndarray]:
    """Noise tokens; concept replies have the concept direction added to 1-3 of their tokens."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n_replies)
    acts = []
    for label in labels:
        reply = rng.normal(size=(rng.integers(5, 30), D_MODEL)).astype(np.float32)
        if label:
            concept_tokens = rng.choice(len(reply), size=rng.integers(1, 4), replace=False)
            reply[concept_tokens] += spike * CONCEPT_DIRECTION
        acts.append(reply)
    return acts, labels


def test_readouts_by_hand():
    logits = np.array([2.0, -2.0, 4.0])
    assert np.allclose(running_average_path(logits), [1.0, -0.5, 1.75])
    assert READOUTS["peak"](logits) == pytest.approx(1.75)
    assert READOUTS["max"](logits) == pytest.approx(4.0)
    assert READOUTS["mean"](logits) == pytest.approx(4 / 3)
    assert READOUTS["peak"](np.array([-3.0])) == pytest.approx(-1.5), "the running average starts from a neutral 0"


@pytest.mark.parametrize("readout", sorted(READOUTS))
def test_torch_readouts_match_numpy(readout):
    rng = np.random.default_rng(0)
    sequences = [rng.normal(size=n) for n in (1, 4, 9, 2)]
    batch = readout_batch_torch([torch.tensor(s, dtype=torch.float32) for s in sequences], readout).numpy()
    assert np.allclose(batch, [READOUTS[readout](s) for s in sequences], atol=1e-5)


def test_flag_line_and_rates():
    negatives = np.arange(1000, dtype=float)
    line = flag_line(negatives, 0.01)
    assert flagged_fraction(negatives, line) == pytest.approx(0.01)
    result = evaluate(positive_scores=[995.0, 1500.0, 10.0], negative_scores=negatives, line=line)
    assert result["catch_rate"] == pytest.approx(2 / 3)
    assert result["false_alarm_rate"] == pytest.approx(0.01)
    with pytest.raises(AssertionError, match="rests on"):
        flag_line(np.arange(50.0), 0.01)


def test_auroc():
    assert auroc([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert auroc([1.0, 2.0], [3.0, 4.0]) == 0.0


def held_out_auroc(probe: LinearProbe, readout: str | None = None) -> float:
    test_acts, test_labels = synthetic_replies(400, seed=2)
    scores = probe.score_many(test_acts, readout)
    return auroc(scores[test_labels == 1], scores[test_labels == 0])


# The synthetic concept sits in 1-3 tokens of a 5-30 token reply, so averaging dilutes it: mean readouts score lower.
@pytest.mark.parametrize("fit, kwargs, min_auroc", [
    (fit_logistic_probe, {"pooling": "mean", "readout": "mean"}, 0.85),
    (fit_logistic_probe, {"pooling": "all_tokens", "readout": "max"}, 0.95),
    (fit_readout_probe, {"readout": "peak", "learning_rate": 1e-2}, 0.95),
    (fit_readout_probe, {"readout": "max", "learning_rate": 1e-2}, 0.95),
    (fit_readout_probe, {"readout": "mean", "learning_rate": 1e-2, "max_epochs": 100}, 0.85),
])
def test_probe_training_separates_held_out_replies(fit, kwargs, min_auroc):
    train_acts, train_labels = synthetic_replies(400, seed=1)
    probe = fit(train_acts, train_labels, layer=7, **kwargs)
    assert held_out_auroc(probe) > min_auroc
    assert np.dot(probe.weight, CONCEPT_DIRECTION) / np.linalg.norm(probe.weight) > 0.8, "should find the concept direction"


def test_peak_trained_probe_catches_more_at_its_line_than_mean_trained():
    train_acts, train_labels = synthetic_replies(400, seed=1)
    test_acts, test_labels = synthetic_replies(400, seed=2)
    catch_rates = {}
    for readout in ("peak", "mean"):
        probe = fit_readout_probe(train_acts, train_labels, layer=7, readout=readout, learning_rate=1e-2, max_epochs=100)
        scores = probe.score_many(test_acts)
        line = flag_line(scores[test_labels == 0], 0.05)
        catch_rates[readout] = flagged_fraction(scores[test_labels == 1], line)
    assert catch_rates["peak"] > 0.9
    assert catch_rates["peak"] > catch_rates["mean"] + 0.1, catch_rates


@pytest.mark.parametrize("suffix", [".json", ".npz"])
def test_save_and_load_round_trip(tmp_path, suffix):
    acts, labels = synthetic_replies(100, seed=3)
    probe = fit_readout_probe(acts, labels, layer=5, readout="peak", max_epochs=2)
    probe.save(tmp_path / f"probe{suffix}")
    loaded = LinearProbe.load(tmp_path / f"probe{suffix}")
    assert (loaded.layer, loaded.readout, loaded.info) == (5, "peak", probe.info)
    assert np.allclose(loaded.score_many(acts), probe.score_many(acts), atol=1e-6)


class TemplateOpensReasoning:
    """Stands in for a tokenizer whose chat template ends inside an opened reasoning block (as Qwen's does)."""
    chat_template = "stub"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant><think>\n"


def test_render_places_reasoning_and_reply():
    text, reasoning_start, reply_start = render(TemplateOpensReasoning(), Example("Hi?", "Hello.", reasoning="Be nice."))
    assert text == "<user>Hi?<assistant><think>\nBe nice.\n</think>\n\nHello."
    assert text[reasoning_start:reply_start] == "Be nice.\n</think>\n\n"
    assert text[reply_start:] == "Hello."

    text, _, reply_start = render(TemplateOpensReasoning(), Example("Hi?", "Hello."))
    assert text == "<user>Hi?<assistant><think>\n\n</think>\n\nHello.", "an opened block is closed even without reasoning"
    assert text[reply_start:] == "Hello."


@pytest.mark.skipif(os.environ.get("PROBE_TOOLKIT_SLOW") != "1", reason="set PROBE_TOOLKIT_SLOW=1 to run")
def test_capture_with_tiny_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).eval()
    examples = [Example("What is 17 times 3?", "It is 51.", reasoning="17 * 3 = 51."),
                Example("Name a colour.", "Blue, like the sea.", reasoning="Any colour works.")] * 5
    layers = [0, 1]

    reply = capture_activations(model, tokenizer, examples, layers)
    for example, tokens, acts in zip(examples, reply.tokens, reply.activations[1]):
        assert tokenizer.convert_tokens_to_string(tokens).strip() == example.reply
        assert acts.shape == (len(tokens), model.config.hidden_size)

    with_reasoning = capture_activations(model, tokenizer, examples, layers, span="reasoning_and_reply")
    context = capture_activations(model, tokenizer, examples, layers, span="context")
    truncated = capture_activations(model, tokenizer, examples, layers, max_tokens=2)
    for i in range(len(examples)):
        assert len(with_reasoning.tokens[i]) > len(reply.tokens[i])
        assert np.allclose(with_reasoning.activations[1][i][-len(reply.tokens[i]):], reply.activations[1][i], atol=1e-5)
        assert len(context.tokens[i]) > 0
        assert np.allclose(truncated.activations[0][i], reply.activations[0][i][:2], atol=1e-5)

    probe = fit_readout_probe(reply.activations[1], np.array([1, 0] * 5), layer=1, max_epochs=1, val_fraction=0.5)
    assert np.isfinite(probe.score_many(reply.activations[1])).all()
