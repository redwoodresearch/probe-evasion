"""Audit of the Gemma early-layer maths result: every saved reply scored by the templated maths probes at every layer.

    uv run modal run -m probe_evasion.gemma_layer_audit::score_layers
writes results/gepa/<RUN>/layer_audit.jsonl: one row per (set, index) with each layer's ema/max/window reading.
Sets: calibration replies (no-maths negatives and plain maths positives), the held-out test arms (plain and
optimized prompt), and Gemma's own replies in answer styles the probes never saw (dialogue, headline, FAQ).
"""

import json
from pathlib import Path

from probe_evasion import concept_probes, gepa
from probe_evasion.modal_app import app

RUN = Path("results/gepa/gemma_maths_window_L36_20261007-154620")
PROBES = "gemma_maths_layers"
HELDOUT = Path("data/concepts/onpolicy_gemma_maths_heldout_clean.jsonl")
OUT = RUN / "layer_audit.jsonl"


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines()]


@app.local_entrypoint()
def score_layers():
    probes = {k: v for k, v in json.loads((concept_probes.PROBES / f"{PROBES}.json").read_text()).items() if k.startswith("maths@")}
    sets = {"calibration": [(r["prompt"], r["reply"]) for r in jl(RUN / "calibration_replies.jsonl")],
            "test_no_instruction": [(r["prompt"], r["reply"]) for r in jl(RUN / "test_no_instruction.jsonl")],
            "test_optimized": [(r["prompt"], r["reply"]) for r in jl(RUN / "test_optimized.jsonl")],
            "heldout_styles": [(r["prompt"], r["response"]) for r in jl(HELDOUT)]}
    out = []
    for name, pairs in sets.items():
        scored = concept_probes.score("gemma", [{"prompt": p, "response": r} for p, r in pairs], probes)
        for i, s in enumerate(scored):
            out.append({"set": name, "i": i, **{n: {k: v for k, v in gepa.readings(s[n], s[f"{n}__logits"]).items()
                                                    if k in ("ema", "max", "window")} for n in probes}})
        print(name, len(scored), flush=True)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in out))
    print("wrote", OUT, flush=True)
