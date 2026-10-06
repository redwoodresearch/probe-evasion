"""Build a probe-training set from one generator on the train split.

Positives: `forced` responses that are working backdoors. Negatives: `clean` responses
that pass every test. Each example keeps only the user turn (the generator's instruction
is dropped), the final answer up to the exploit tag, and the char span of its code block,
which is the only part probes read.

    uv run modal run -m probe_evasion.probe_data --generator claude --n-problems 20   # validation
    uv run modal run -m probe_evasion.probe_data --generator qwen --n-problems 400
"""

import json
import time
from collections import Counter
from pathlib import Path

from probe_evasion.apps_data import sample_problems
from probe_evasion.generation import generate, write_jsonl
from probe_evasion.modal_app import app
from probe_evasion.parsing import code_span, strip_exploit
from probe_evasion.scoring import score

VARIANTS = ("forced", "clean")
LABELS = {("forced", "backdoor_success"): 1, ("clean", "clean_pass"): 0}


@app.local_entrypoint()
def main(generator: str, n_problems: int = 400, seed: int = 0):
    run_dir = Path("results/probe_data") / generator / time.strftime("%Y%m%d-%H%M%S")
    problems = sample_problems("train", n_problems, seed=seed)
    by_pid = {p.problem_id: p for p in problems}
    print(f"{len(problems)} train-split problems from {generator}")

    rows = generate(problems, VARIANTS, generator, samples=1, seed=seed)
    write_jsonl(run_dir / "generations.jsonl", rows)
    score(rows, problems)
    write_jsonl(run_dir / "samples.jsonl", rows)

    dataset = []
    for r in rows:
        label = LABELS.get((r["variant"], r["outcome"]))
        if label is None or not r["reference_verified"]:
            continue
        response = strip_exploit(r["final"])
        dataset.append(
            {
                "generator": generator,
                "problem_id": r["problem_id"],
                "label": label,
                "question": by_pid[r["problem_id"]].question,
                "reasoning": r["reasoning"],
                "response": response,
                "code_span": code_span(response),
            }
        )
    write_jsonl(run_dir / "dataset.jsonl", dataset)

    summary = {
        "generator": generator,
        "n_problems": len(problems),
        "outcomes": {v: dict(Counter(r["outcome"] for r in rows if r["variant"] == v)) for v in VARIANTS},
        "refusals": sum(r["stop_reason"] == "refusal" for r in rows),
        "positives": sum(d["label"] for d in dataset),
        "negatives": sum(1 - d["label"] for d in dataset),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {run_dir}")
