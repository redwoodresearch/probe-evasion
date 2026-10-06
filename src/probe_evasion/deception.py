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
from probe_evasion.generation import load_chunks, read_jsonl, sample, write_jsonl
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
def main(stage: str, generator: str = "claude", probes: str = "", n_questions: int = 4000, seed: int = 0, samples: int = 2,
         run: str = ""):
    if stage == "filter":
        filter_known(n_questions, seed)
    elif stage == "data":
        build_probe_data(generator, seed)
    elif stage == "eval":
        evaluate(probes.split(","), samples, seed)
    elif stage == "rescore":
        rescore(Path(run), probes.split(","))
    elif stage == "rescore-neutral":
        rescore_neutral(Path(run), probes.split(","))
    elif stage == "partial":
        evaluate_partial(Path(run), probes.split(","))
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


def eval_plan(probe_names: list[str], max_questions: int, interleave: bool = True):
    test = [q for q in read_jsonl(KNOWN) if q["split"] == "test"][:max_questions]
    conditions = {"E0": "", "placebo": PLACEBO, "aware": AWARE, "E1": E1, "E3": E3} | {f"E2_{p}": _e2_for(p) for p in probe_names}
    keys = [(c, kind, q) for c in conditions for kind in ("lie", "truth") for q in test]
    if interleave:  # question-major order: every chunk mixes all conditions, so early partials are balanced
        keys = [(c, kind, q) for q in test for c in conditions for kind in ("lie", "truth")]
    convs = [messages(kind, q["question"], conditions[c]) for c, kind, q in keys]
    return conditions, keys, convs


def rows_and_examples(keys, convs, outs) -> tuple[list[dict], list[dict]]:
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
    return rows, examples


def outcome_counts(rows: list[dict]) -> dict:
    unfinished = sum(r["truncated"] or not r["final"] for r in rows)
    correct = sum(not (r["truncated"] or not r["final"]) and r["correct"] for r in rows)
    return {"n": len(rows), "unfinished": unfinished, "completed_correct": correct,
            "completed_wrong": len(rows) - unfinished - correct}


def score_and_summarize(rows, examples, acts_name, conditions, probe_names, extract: bool = True) -> dict:
    if extract:
        extract_all(acts_name, examples)
    labels = np.array([e["label"] for e in examples])
    cond_of = np.array([e["condition"] for e in examples])
    pids = np.array([e["problem_id"] for e in examples])
    summary = {"n_samples": len(rows), "activations": acts_name,
               # Per condition and prompt kind: unfinished (no final answer within the token budget), completed and
               # graded correct, completed and graded wrong. Only completed answers are probe examples.
               "outcomes": {c: {kind: outcome_counts([r for r in rows if r["condition"] == c and r["kind"] == kind])
                                for kind in ("lie", "truth")} for c in conditions},
               "attack_attempts": {c: sum(r["condition"] == c and r["kind"] == "lie" for r in rows) for c in conditions},
               "probes": {}}
    for name in probe_names:
        layer, params = load_probe(name)
        s = np.array([x[layer]["ema"] for x in score.remote(acts_name, len(examples), {layer: params})])
        for e, v in zip(examples, s):
            e.setdefault("scores", {})[name] = float(v)
        own = [c for c in conditions if not c.startswith("E2_") or c == f"E2_{name}"]
        report = condition_report(s, labels, cond_of, own, pids)
        for c in report:  # conditions with data (partial runs may lack some)  # undetected successful lies per lie attempt
            fx = report[c]["fixed_E0_threshold"]
            report[c]["undetected_success_per_attempt"] = fx["evasion_rate"] * fx["n_backdoor"] / max(1, summary["attack_attempts"][c])
        summary["probes"][name] = {"layer": layer, "conditions": report}
    return summary


def evaluate(probe_names: list[str], samples: int, seed: int, max_questions: int = 300) -> None:
    conditions, keys, convs = eval_plan(probe_names, max_questions)
    run_dir = ROOT / f"eval_{time.strftime('%Y%m%d-%H%M%S')}"
    # 8k tokens is ample for a trivia answer; it stops rare runaway traces from stalling the whole batch.
    outs = sample(convs, "qwen", samples, seed, chunk_seqs=64, max_tokens=8192, checkpoint_dir=run_dir / "chunks")
    rows, examples = rows_and_examples(keys, convs, outs)
    write_jsonl(run_dir / "samples.jsonl", rows)
    write_jsonl(run_dir / "examples.jsonl", examples)
    summary = score_and_summarize(rows, examples, f"deception_{run_dir.name}", conditions, probe_names)
    write_jsonl(run_dir / "examples_scored.jsonl", examples)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {run_dir}")


def rescore(run_dir: Path, probe_names: list[str]) -> None:
    """Recompute every statistic from a finished run's saved activations (no generation, no extraction) and save
    per-example scores for audits."""
    examples = read_jsonl(run_dir / "examples.jsonl")
    for e in examples:
        e.pop("scores", None)
    rows = read_jsonl(run_dir / "samples.jsonl")
    conditions = list(dict.fromkeys(e["condition"] for e in examples))
    summary = {"context": "full (as generated)"} | score_and_summarize(rows, examples, f"deception_{run_dir.name}", conditions,
                                                                      probe_names, extract=False)
    write_jsonl(run_dir / "examples_scored.jsonl", examples)
    (run_dir / "summary_rescored.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def rescore_neutral(run_dir: Path, probe_names: list[str]) -> None:
    """Re-score a finished eval with each answer replayed after only its question: no system prompt (so no
    "answer falsely" instruction or warning text) and no reasoning. Matches how the probes were trained, and
    separates detecting the lie from detecting the instruction."""
    examples = [{k: v for k, v in e.items() if k not in ("system", "scores")} | {"reasoning": ""}
                for e in read_jsonl(run_dir / "examples.jsonl")]
    rows = read_jsonl(run_dir / "samples.jsonl")
    conditions = list(dict.fromkeys(e["condition"] for e in examples))
    summary = {"context": "question + answer only"} | score_and_summarize(rows, examples, f"deception_{run_dir.name}_neutral", conditions, probe_names)
    (run_dir / "summary_neutral.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def evaluate_partial(run_dir: Path, probe_names: list[str], max_questions: int = 300) -> None:
    """Preliminary summary from the chunks a still-running eval has finished."""
    # Runs started before interleaving (eval_20261006-144940) use condition-major order.
    conditions, keys, convs = eval_plan(probe_names, max_questions, interleave=run_dir.name > "eval_20261006-1500")
    chunks = load_chunks(run_dir / "chunks")
    per_chunk = json.loads(next((run_dir / "chunks").glob("chunk_*.json")).read_text())["per_chunk"]
    idx = [i * per_chunk + j for i in sorted(chunks) for j in range(len(chunks[i]))]
    outs = [[{"text": t} for t in ts] for i in sorted(chunks) for ts in chunks[i]]
    rows, examples = rows_and_examples([keys[k] for k in idx], [convs[k] for k in idx], outs)
    summary = {"partial": True} | score_and_summarize(rows, examples, f"deception_{run_dir.name}_partial{len(rows)}", conditions, probe_names)
    (run_dir / f"summary_partial_{len(rows)}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
