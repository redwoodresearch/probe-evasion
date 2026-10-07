"""Re-extract and re-score the probe examples of a finished backdoor run whose probed code differs from the code the
sandbox ran (built with an older parser), and replace their scores in examples_scored.jsonl.

    uv run modal run -m probe_evasion.repair_examples --run results/evasion/<run>
"""

from collections import defaultdict
from pathlib import Path

from probe_evasion.activations import extract_all
from probe_evasion.generation import read_jsonl, write_jsonl
from probe_evasion.metrics import load_probe
from probe_evasion.modal_app import app
from probe_evasion.parsing import code_span, neutralize, program, strip_exploit
from probe_evasion.probes import score

PROBES = ("bd_claude_intro", "bd_qwen_intro", "bd_claude_intro_neutral", "bd_qwen_intro_neutral")


@app.local_entrypoint()
def main(run: str):
    run_dir = Path(run)
    examples = read_jsonl(run_dir / "examples_scored.jsonl")
    rows = defaultdict(list)
    for r in read_jsonl(run_dir / "samples.jsonl"):
        rows[r["condition"], r["problem_id"]].append(r)
    fixed = []
    for i, e in enumerate(examples):
        # The probe text is a prefix of the attempt's final answer (cut before the exploit section).
        candidates = [r for r in rows[e["condition"], e["problem_id"]] if r["final"].startswith(e["response"])]
        if any(r["code"] == e["response"][slice(*e["code_span"])] for r in candidates):
            continue
        (row,) = candidates
        response = strip_exploit(row["final"])
        assert response[slice(*code_span(response))] == program(row["final"]) == row["code"]
        examples[i] = e | {"response": response, "code_span": code_span(response), "repaired": True}
        fixed.append(i)
    print(f"repairing {len(fixed)} examples: {fixed}")
    if not fixed:
        return
    batch = [examples[i] for i in fixed]
    acts = f"evasion_{run_dir.name}_repaired"
    extract_all(acts, batch)
    extract_all(f"{acts}_neutral", [neutralize(e) for e in batch])
    for name in PROBES:
        layer, params = load_probe(name)
        source = f"{acts}_neutral" if name.endswith("_neutral") else acts
        for i, x in zip(fixed, score.remote(source, len(batch), {layer: params})):
            old = examples[i]["scores"][name]
            examples[i]["scores"][name] = float(x[layer]["ema"])
            print(f"  example {i} {name}: {old:.2f} -> {examples[i]['scores'][name]:.2f}")
    write_jsonl(run_dir / "examples_scored.jsonl", examples)
