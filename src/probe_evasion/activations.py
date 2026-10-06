"""Extract Qwen residual-stream activations over the code-block span of each example.

Text is rendered exactly as Qwen's chat template renders an assistant turn
(`<think>\\n{reasoning}\\n</think>\\n\\n{response}`), so training examples (no reasoning,
no system prompt) and test examples (with both) differ only in content, not formatting.
Activations are saved to a Modal Volume as one float16 array [tokens, layers, d_model]
per example; nothing large comes back over the network."""

import modal

from probe_evasion import config
from probe_evasion.modal_app import ACTS_DIR, PROBE_LAYERS, acts_volume, app, extract_image, hf_cache


def render(tok, ex: dict) -> tuple[str, int]:
    """Full text and the char offset where the response starts."""
    messages = ([{"role": "system", "content": ex["system"]}] if ex.get("system") else []) + [
        {"role": "user", "content": ex["question"]}
    ]
    prefix = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    assert prefix.endswith("<think>\n"), prefix[-40:]
    head = prefix + (ex.get("reasoning") or "").strip() + "\n</think>\n\n"
    return head + ex["response"], len(head)


@app.cls(
    image=extract_image,
    gpu="H100",
    volumes={"/hf-cache": hf_cache, ACTS_DIR: acts_volume},
    timeout=2 * 60 * 60,
    scaledown_window=5 * 60,
)
class Extractor:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(config.TARGET_MODEL)
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.TARGET_MODEL, dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        self.layers = next(
            m for name, m in self.model.named_modules()
            if name.endswith("language_model.layers") and isinstance(m, torch.nn.ModuleList)
        )
        assert len(self.layers) == 64, len(self.layers)

    @modal.method()
    def extract(self, dataset_name: str, examples: list[dict], layers: tuple[int, ...] = PROBE_LAYERS) -> list[int]:
        """Writes {ACTS_DIR}/{dataset_name}/{i}.npy; returns code-span token counts."""
        import os

        import numpy as np
        import torch

        out_dir = f"{ACTS_DIR}/{dataset_name}"
        os.makedirs(out_dir, exist_ok=True)
        captured = {}
        hooks = [
            self.layers[l].register_forward_hook(lambda m, i, o, l=l: captured.__setitem__(l, o[0] if isinstance(o, tuple) else o))
            for l in layers
        ]
        n_tokens = []
        try:
            for i, ex in enumerate(examples):
                text, response_start = render(self.tok, ex)
                a, b = ex["code_span"]
                enc = self.tok(text, return_offsets_mapping=True, return_tensors="pt")
                starts, ends = enc["offset_mapping"][0].T
                mask = (starts >= response_start + a) & (ends <= response_start + b) & (ends > starts)
                assert mask.any(), f"empty code span for example {i}"
                with torch.no_grad():
                    self.model(input_ids=enc["input_ids"].cuda())
                acts = torch.stack([captured[l][0, mask.cuda()] for l in layers], dim=1)
                np.save(f"{out_dir}/{i}.npy", acts.to(torch.float16).cpu().numpy())
                n_tokens.append(int(mask.sum()))
        finally:
            for h in hooks:
                h.remove()
        acts_volume.commit()
        return n_tokens

