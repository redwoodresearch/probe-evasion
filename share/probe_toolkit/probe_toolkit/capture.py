"""Per-token residual-stream activations of a Hugging Face chat model over (prompt, reply) pairs.

The whole conversation is run through the model, so the reply is read in its real context, but only the chosen span
(by default the reply tokens) is kept. One example at a time: no padding, so no padding bugs.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

DEFAULT_REASONING_FORMAT = "<think>\n{reasoning}\n</think>\n\n"
FALLBACK_TEMPLATE = "User: {prompt}\n\nAssistant:\n"
SPANS = ("reply", "reasoning_and_reply", "context")


@dataclass
class Example:
    prompt: str
    reply: str
    reasoning: str | None = None
    system: str | None = None


@dataclass
class Captured:
    """activations[layer][i]: float32 array [span_tokens, d_model] for example i; tokens[i]: the span's token strings."""
    activations: dict[int, list[np.ndarray]] = field(default_factory=dict)
    tokens: list[list[str]] = field(default_factory=list)


def render(tokenizer, example: Example, reasoning_format: str = DEFAULT_REASONING_FORMAT,
           chat_template_kwargs: dict | None = None) -> tuple[str, int, int]:
    """Full text, and the character offsets where the reasoning block and the reply start.

    If the chat template already opens a reasoning block (Qwen with enable_thinking=True ends in "<think>\\n"), a
    block is always written, empty when the example has no reasoning, exactly as the model would see it.
    """
    if tokenizer.chat_template is None:
        prefix = FALLBACK_TEMPLATE.format(prompt=example.prompt)
    else:
        messages = ([{"role": "system", "content": example.system}] if example.system else []) + [
            {"role": "user", "content": example.prompt}]
        prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               **(chat_template_kwargs or {}))
    opening = reasoning_format.split("{reasoning}")[0]
    template_opened_reasoning = bool(opening) and prefix.endswith(opening)
    if template_opened_reasoning:
        prefix = prefix[:-len(opening)]
    if example.reasoning is None and not template_opened_reasoning:
        return prefix + example.reply, len(prefix), len(prefix)
    reasoning_start = len(prefix) + len(opening)
    prefix += reasoning_format.format(reasoning=(example.reasoning or "").strip())
    return prefix + example.reply, reasoning_start, len(prefix)


def find_decoder_layers(model) -> torch.nn.ModuleList:
    """The longest ModuleList of transformer blocks (model.layers, transformer.h, language_model.layers, ...)."""
    candidates = [module for name, module in model.named_modules()
                  if isinstance(module, torch.nn.ModuleList) and name.split(".")[-1] in ("layers", "h", "blocks")]
    assert candidates, "could not find the decoder blocks; pass decoder_layers explicitly"
    return max(candidates, key=len)


def capture_activations(model, tokenizer, examples: list[Example], layers: list[int], span: str = "reply",
                        max_tokens: int | None = None, reasoning_format: str = DEFAULT_REASONING_FORMAT,
                        chat_template_kwargs: dict | None = None, decoder_layers: torch.nn.ModuleList | None = None) -> Captured:
    """Residual stream after each listed block (0-indexed), for the tokens in `span`:

    "reply": the reply only (the default; what a reply monitor reads).
    "reasoning_and_reply": from the start of the reasoning block's content to the end of the reply.
    "context": everything before the reply except the very first token (whose activations are usually outliers).
        Use it to check that a probe isn't just reading the request (see README).

    max_tokens truncates the span (and the forward pass) to its first max_tokens tokens.
    """
    assert span in SPANS, f"span must be one of {SPANS}"
    assert tokenizer.is_fast, "token offsets need a fast tokenizer"
    decoder_layers = decoder_layers or find_decoder_layers(model)
    device = next(model.parameters()).device
    latest_output = {}

    def remember_output(layer: int):
        def hook(module, inputs, output):
            latest_output[layer] = output[0] if isinstance(output, tuple) else output
        return hook

    hooks = [decoder_layers[layer].register_forward_hook(remember_output(layer)) for layer in layers]
    captured = Captured({layer: [] for layer in layers}, [])
    try:
        for example in examples:
            text, reasoning_start, reply_start = render(tokenizer, example, reasoning_format, chat_template_kwargs)
            encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            ids, offsets = encoding["input_ids"], encoding["offset_mapping"]
            # a token that straddles the boundary belongs to the reply
            first_reply_token = next(i for i, (_, end) in enumerate(offsets) if end > reply_start)
            if span == "context":
                first, last = 1, first_reply_token
            else:
                read_from = reply_start if span == "reply" else reasoning_start
                first, last = next(i for i, (_, end) in enumerate(offsets) if end > read_from), len(ids)
            if max_tokens is not None:
                last = min(last, first + max_tokens)
            assert last > first, f"empty {span} span for example {example}"
            with torch.no_grad():
                # logits only for the last position: full-sequence logits over a large vocabulary are never used
                model(input_ids=torch.tensor([ids[:last]], device=device), use_cache=False, logits_to_keep=1)
            for layer in layers:
                captured.activations[layer].append(latest_output[layer][0, first:last].float().cpu().numpy())
            captured.tokens.append(tokenizer.convert_ids_to_tokens(ids[first:last]))
    finally:
        for hook in hooks:
            hook.remove()
    return captured
