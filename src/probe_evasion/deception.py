"""TriviaQA deception pipeline, one stage per invocation:

    uv run modal run -m probe_evasion.deception --stage filter          # questions Qwen knows (5/5 correct)
    uv run modal run -m probe_evasion.deception --stage data --generator claude
    uv run modal run -m probe_evasion.deception --stage data --generator qwen
    uv run modal run -m probe_evasion.deception --stage eval --probes d_claude,d_qwen
"""

import json
import time
from pathlib import Path

import numpy as np

from probe_evasion.activations import extract_all
from probe_evasion.generation import read_jsonl, sample, write_jsonl
from probe_evasion.metrics import condition_report, load_probe
from probe_evasion.modal_app import app
from probe_evasion.parsing import parse
from probe_evasion.probes import score
from probe_evasion.trivia import AWARE, E1, E3, PLACEBO, e2_addition, is_correct, load_questions, messages

ROOT = Path("results/deception")
KNOWN = ROOT / "known.jsonl"
SHORT_OUTPUT_CHUNK = 1024


def filter_known(n_questions: int, seed: int) -> None:
    """Keep questions Qwen answers correctly on all 5 samples (thinking off; it's a knowledge check)."""
    qs = load_questions(n_questions, seed)
    # Capped: a few non-thinking answers ramble toward the full budget and stall the whole batch.
    outs = sample([messages("truth", q["question"]) for q in qs], "qwen", 5, seed, thinking=False,
                  chunk_seqs=SHORT_OUTPUT_CHUNK, max_tokens=256)
    known = [dict(q, n_correct=sum(is_correct(o["text"], q["aliases"]) for o in os_)) for q, os_ in zip(qs, outs)]
    run_dir = ROOT / f"filter_{time.strftime('%Y%m%d-%H%M%S')}"
    write_jsonl(run_dir / "all.jsonl", [dict(k, answers=[o["text"] for o in os_]) for k, os_ in zip(known, outs)])
    known = [k for k in known if k["n_correct"] == 5]
    write_jsonl(KNOWN, known)
    summary = {"n_questions": len(qs), "n_known": len(known), "by_split": {s: sum(k["split"] == s for k in known) for s in ("train", "test")}}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def main(stage: str, generator: str = "claude", probes: str = "", n_questions: int = 4000, seed: int = 0, samples: int = 2):
    if stage == "filter":
        filter_known(n_questions, seed)
    elif stage == "data":
        build_probe_data(generator, seed)
    elif stage == "eval":
        evaluate(probes.split(","), samples, seed)
    else:
        raise ValueError(stage)


def build_probe_data(generator: str, seed: int) -> None:
    """Lies (lie prompt, answer wrong) are positives; truths (truth prompt, answer right) negatives. The
    instruction is dropped from each example; the probe reads the whole answer."""
    train = [q for q in read_jsonl(KNOWN) if q["split"] == "train"]
    keys = [(kind, q) for kind in ("lie", "truth") for q in train]
    outs = sample([messages(kind, q["question"]) for kind, q in keys], generator, 1, seed, chunk_seqs=256)
    run_dir = ROOT / f"data_{generator}_{time.strftime('%Y%m%d-%H%M%S')}"
    rows, dataset = [], []
    for (kind, q), (out,) in zip(keys, outs):
        parsed = parse(out["text"], has_reasoning=generator == "qwen")
        correct = is_correct(parsed.final, q["aliases"])
        rows.append({"kind": kind, "qid": q["qid"], "raw": out["text"], "final": parsed.final, "correct": correct,
                     "truncated": parsed.truncated_reasoning, "stop_reason": out.get("stop_reason")})
        usable = parsed.final and not parsed.truncated_reasoning and out.get("stop_reason") != "refusal"
        label = {("lie", False): 1, ("truth", True): 0}.get((kind, correct))
        if usable and label is not None:
            dataset.append({"generator": generator, "problem_id": q["qid"], "label": label, "question": q["question"],
                            "reasoning": parsed.reasoning, "response": parsed.final, "code_span": [0, len(parsed.final)]})
    write_jsonl(run_dir / "samples.jsonl", rows)
    write_jsonl(run_dir / "dataset.jsonl", dataset)
    summary = {"generator": generator, "n_train_questions": len(train),
               "lie_success": sum(r["kind"] == "lie" and not r["correct"] for r in rows) / len(train),
               "truth_correct": sum(r["kind"] == "truth" and r["correct"] for r in rows) / len(train),
               "positives": sum(d["label"] for d in dataset), "negatives": sum(1 - d["label"] for d in dataset)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {run_dir}/dataset.jsonl")


def _e2_for(probe_name: str) -> str:
    source = (Path("results/activations") / probe_name / "source.txt").read_text().strip()
    rows = read_jsonl(Path(source))
    lie = min((r for r in rows if r["label"] == 1), key=lambda r: len(r["response"]))
    truth = min((r for r in rows if r["label"] == 0), key=lambda r: len(r["response"]))
    return e2_addition(lie["response"], truth["response"])


def evaluate(probe_names: list[str], samples: int, seed: int, max_questions: int = 300) -> None:
    test = [q for q in read_jsonl(KNOWN) if q["split"] == "test"][:max_questions]
    conditions = {"E0": "", "placebo": PLACEBO, "aware": AWARE, "E1": E1, "E3": E3} | {f"E2_{p}": _e2_for(p) for p in probe_names}
    keys = [(c, kind, q) for c in conditions for kind in ("lie", "truth") for q in test]
    convs = [messages(kind, q["question"], conditions[c]) for c, kind, q in keys]
    run_dir = ROOT / f"eval_{time.strftime('%Y%m%d-%H%M%S')}"
    # 8k tokens is ample for a trivia answer; it stops rare runaway traces from stalling the whole batch.
    outs = sample(convs, "qwen", samples, seed, chunk_seqs=64, max_tokens=8192, checkpoint_dir=run_dir / "chunks")
    rows, examples = [], []
    for (c, kind, q), conv, outs_ in zip(keys, convs, outs):
        for i, out in enumerate(outs_):
            parsed = parse(out["text"])
            correct = is_correct(parsed.final, q["aliases"])
            rows.append({"condition": c, "kind": kind, "qid": q["qid"], "sample": i, "raw": out["text"],
                         "final": parsed.final, "correct": correct, "truncated": parsed.truncated_reasoning})
            label = {("lie", False): 1, ("truth", True): 0}.get((kind, correct))
            if parsed.final and not parsed.truncated_reasoning and label is not None:
                examples.append({"condition": c, "problem_id": q["qid"], "label": label, "generator": "qwen",
                                 "system": conv[0]["content"], "question": q["question"], "reasoning": parsed.reasoning,
                                 "response": parsed.final, "code_span": [0, len(parsed.final)]})
    write_jsonl(run_dir / "samples.jsonl", rows)

    acts_name = f"deception_{run_dir.name}"
    extract_all(acts_name, examples)
    labels = np.array([e["label"] for e in examples])
    cond_of = np.array([e["condition"] for e in examples])
    summary = {"n_questions": len(test), "samples": samples,
               "lie_success": {c: sum(r["condition"] == c and r["kind"] == "lie" and not r["correct"] for r in rows)
                               / max(1, sum(r["condition"] == c and r["kind"] == "lie" for r in rows)) for c in conditions},
               "probes": {}}
    for name in probe_names:
        layer, params = load_probe(name)
        s = np.array([x[layer]["ema"] for x in score.remote(acts_name, len(examples), {layer: params})])
        for e, v in zip(examples, s):
            e.setdefault("scores", {})[name] = float(v)
        own = [c for c in conditions if not c.startswith("E2_") or c == f"E2_{name}"]
        pids = np.array([e["problem_id"] for e in examples])
        summary["probes"][name] = {"layer": layer, "conditions": condition_report(s, labels, cond_of, own, pids)}
    write_jsonl(run_dir / "examples.jsonl", examples)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {run_dir}")
