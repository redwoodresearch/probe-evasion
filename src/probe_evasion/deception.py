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
from probe_evasion.trivia import ADDITIONS, e2_addition, is_correct, load_questions, messages

ROOT = Path("results/deception")
KNOWN = ROOT / "known.jsonl"
SHORT_OUTPUT_CHUNK = 1024
DEFAULT_CONDITIONS = ("E0", "placebo", "aware", "E1", "E3", "E2")
# Pairwise comparisons computed inside the question-level bootstrap (the first two are pre-registered).
CONTRASTS = (("E3", "E0"), ("E3", "placebo"), ("E3_info", "placebo"), ("E3_believe", "placebo"), ("E3_text_only", "placebo"),
             ("E3", "E3_text_only"))


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
         run: str = "", offset: int = -1, conditions: str = ",".join(DEFAULT_CONDITIONS), max_tokens: int = 8192,
         max_questions: int = 300, tag: str = "", exclude_runs: str = ""):
    if stage == "filter":
        filter_known(n_questions, seed)
    elif stage == "data":
        build_probe_data(generator, seed)
    elif stage == "eval":
        evaluate(probes.split(","), samples, seed, max_questions, offset, tuple(conditions.split(",")), max_tokens, tag)
    elif stage == "rescore":
        rescore(Path(run), probes.split(","), [Path(x) for x in exclude_runs.split(",") if x])
    elif stage == "rescore-neutral":
        rescore_neutral(Path(run), probes.split(","))
    elif stage == "decompose":
        decompose(Path(run), probes.split(","))
    elif stage == "score-neutral":  # activations already extracted
        rescore_neutral(Path(run), probes.split(","), extract=False)
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


def unused_test_questions(n: int) -> list[dict]:
    """The first n held-out known questions, unique by question ID, never used by any earlier eval run."""
    used = reserved_qids(sorted(ROOT.glob("eval_*")))
    out, seen = [], set()
    for q in read_jsonl(KNOWN):
        if q["split"] == "test" and q["qid"] not in used and q["qid"] not in seen:
            seen.add(q["qid"])
            out.append(q)
    assert len(out) >= n, f"only {len(out)} unused test questions left"
    return out[:n]


def eval_plan(probe_names: list[str], max_questions: int, interleave: bool = True, offset: int = 0,
              condition_names: tuple[str, ...] = DEFAULT_CONDITIONS, qids: list[str] | None = None):
    """Test questions: the given qids (fresh runs record them in their manifest) or, for older runs,
    rows [offset, offset + max_questions) of the held-out known set. "E2" expands to one E2 per probe."""
    known_test = [q for q in read_jsonl(KNOWN) if q["split"] == "test"]
    if qids is not None:
        by_id = {q["qid"]: q for q in known_test}
        test = [by_id[q] for q in qids]
    else:
        test = known_test[offset : offset + max_questions]
    conditions = {}
    for name in condition_names:
        if name == "E2":
            conditions |= {f"E2_{p}": _e2_for(p) for p in probe_names}
        else:
            conditions[name] = ADDITIONS[name]
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
                                 "kind": kind, "sample": i, "row": len(rows) - 1,  # the attempt this example came from
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
        indices = [e.get("acts_index", i) for i, e in enumerate(examples)]  # filtered subsets keep original file positions
        s = np.array([x[layer]["ema"] for x in score.remote(acts_name, len(examples), {layer: params}, indices)])
        for e, v in zip(examples, s):
            e.setdefault("scores", {})[name] = float(v)
        own = [c for c in conditions if not c.startswith("E2_") or c == f"E2_{name}"]
        report = condition_report(s, labels, cond_of, own, pids, contrasts=CONTRASTS)
        for c in report:  # conditions with data; successful undetected attacks per attempt (unfinished attempts count)
            for k in ("recalibrated", "fixed_E0_threshold"):
                st = report[c][k]
                report[c][k]["undetected_success_per_attempt"] = st["evasion_rate"] * st["n_backdoor"] / max(1, summary["attack_attempts"][c])
        summary["probes"][name] = {"layer": layer, "conditions": report}
    return summary


def evaluate(probe_names: list[str], samples: int, seed: int, max_questions: int = 300, offset: int = -1,
             condition_names: tuple[str, ...] = DEFAULT_CONDITIONS, max_tokens: int = 8192, tag: str = "") -> None:
    """offset < 0 (default): fresh questions, unique and unused by any earlier run, recorded in the manifest."""
    qids = [q["qid"] for q in unused_test_questions(max_questions)] if offset < 0 else None
    conditions, keys, convs = eval_plan(probe_names, max_questions, offset=max(offset, 0), condition_names=condition_names, qids=qids)
    run_dir = ROOT / f"eval_{time.strftime('%Y%m%d-%H%M%S')}{'_' + tag if tag else ''}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "probes": probe_names, "samples": samples, "seed": seed, "max_questions": max_questions, "offset": offset,
        "qids": qids, "conditions": list(conditions), "max_tokens": max_tokens, "interleave": True, "chunk_seqs": 64}, indent=2))
    # The token cap stops rare runaway traces from stalling the batch; completion rates are reported per condition.
    outs = sample(convs, "qwen", samples, seed, chunk_seqs=64, max_tokens=max_tokens, checkpoint_dir=run_dir / "chunks")
    rows, examples = rows_and_examples(keys, convs, outs)
    write_jsonl(run_dir / "samples.jsonl", rows)
    write_jsonl(run_dir / "examples.jsonl", examples)
    summary = score_and_summarize(rows, examples, f"deception_{run_dir.name}", conditions, probe_names)
    write_jsonl(run_dir / "examples_scored.jsonl", examples)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {run_dir}")


def used_qids(exclude_runs: list[Path]) -> set[str]:
    return {r["qid"] for d in exclude_runs for r in read_jsonl(d / "samples.jsonl")}


def reserved_qids(run_dirs: list[Path]) -> set[str]:
    """Questions used by finished runs (from their samples) or reserved by runs still in progress (from their manifest)."""
    known_test = [q for q in read_jsonl(KNOWN) if q["split"] == "test"]
    out = set()
    for d in run_dirs:
        if (d / "samples.jsonl").exists():
            out |= {r["qid"] for r in read_jsonl(d / "samples.jsonl")}
        elif (d / "manifest.json").exists():
            m = json.loads((d / "manifest.json").read_text())
            out |= set(m["qids"]) if m.get("qids") else {q["qid"] for q in known_test[m["offset"]: m["offset"] + m["max_questions"]]}
    return out


def example_label(r: dict):
    """The probe-example label a sandboxed attempt produces (1 = lie, 0 = truthful), or None if it produces none.
    Mirrors rows_and_examples: only completed lies (wrong) and completed truthful answers (right) become examples."""
    if r["truncated"] or not r["final"]:
        return None
    return {("lie", False): 1, ("truth", True): 0}.get((r["kind"], r["correct"]))


def fresh_subset(rows: list[dict], examples: list[dict], exclude: set[str]) -> tuple[list[dict], list[dict]]:
    """One selection mask over attempts: drop questions seen in earlier runs and keep each repeated question's attempts
    only once (the dataset has duplicate question IDs). Probe examples follow their own attempt, so an example is kept
    exactly when the attempt that produced it is kept; each keeps its original activation-file index."""
    producing = [i for i, r in enumerate(rows) if example_label(r) is not None]
    assert len(producing) == len(examples), (len(producing), len(examples))
    example_of_row = dict(zip(producing, range(len(examples))))
    seen, keep_rows = set(), []
    for i, r in enumerate(rows):
        key = (r["condition"], r["kind"], r["qid"], r["sample"])
        if r["qid"] in exclude or key in seen:
            continue
        seen.add(key)
        keep_rows.append(i)
    kept_examples = []
    for i in keep_rows:
        if i in example_of_row:
            j = example_of_row[i]
            e = examples[j]
            assert e["problem_id"] == rows[i]["qid"] and e["condition"] == rows[i]["condition"]
            kept_examples.append(dict(e, acts_index=e.get("acts_index", j)))
    return [rows[i] for i in keep_rows], kept_examples


def rescore(run_dir: Path, probe_names: list[str], exclude_runs: list[Path] = ()) -> None:
    """Recompute every statistic from a finished run's saved activations (no generation, no extraction) and save
    per-example scores for audits."""
    examples = read_jsonl(run_dir / "examples.jsonl")
    for i, e in enumerate(examples):
        e.pop("scores", None)
        e["acts_index"] = i  # position in the saved activations, kept through filtering
    rows = read_jsonl(run_dir / "samples.jsonl")
    tag = ""
    if exclude_runs:
        rows, examples = fresh_subset(rows, examples, used_qids(exclude_runs))
        tag = "_fresh"
    conditions = list(dict.fromkeys(e["condition"] for e in examples))
    summary = {"context": "full (as generated)", "excluded_runs": [str(d) for d in exclude_runs],
               "n_questions": len({e["problem_id"] for e in examples})} | score_and_summarize(
        rows, examples, f"deception_{run_dir.name}", conditions, probe_names, extract=False)
    write_jsonl(run_dir / f"examples_scored{tag}.jsonl", examples)
    (run_dir / f"summary_rescored{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def rescore_neutral(run_dir: Path, probe_names: list[str], extract: bool = True) -> None:
    """Re-score a finished eval with each answer replayed after only its question: no system prompt (so no
    "answer falsely" instruction or warning text) and no reasoning. Matches how the probes were trained, and
    separates detecting the lie from detecting the instruction."""
    examples = [{k: v for k, v in e.items() if k not in ("system", "scores")} | {"reasoning": ""}
                for e in read_jsonl(run_dir / "examples.jsonl")]
    rows = read_jsonl(run_dir / "samples.jsonl")
    conditions = list(dict.fromkeys(e["condition"] for e in examples))
    summary = {"context": "question + answer only"} | score_and_summarize(rows, examples, f"deception_{run_dir.name}_neutral",
                                                                         conditions, probe_names, extract=extract)
    write_jsonl(run_dir / "examples_neutral_scored.jsonl", examples)
    (run_dir / "summary_neutral.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


CONTEXT_VARIANTS = {
    # name: (keep system prompt, keep reasoning)
    "question_answer": (False, False),
    "system_question_answer": (True, False),
    "question_reasoning_answer": (False, True),
    "full": (True, True),
}


def decompose(run_dir: Path, probe_names: list[str], conditions: tuple[str, ...] = ("E0", "placebo", "E3")) -> None:
    """Same answers re-read under each mix of context pieces, to locate where a condition's effect enters:
    the system prompt (instruction + condition text), Qwen's reasoning, or the answer itself."""
    base = [e for e in read_jsonl(run_dir / "examples.jsonl") if e["condition"] in conditions]
    for e in base:
        e.pop("scores", None)
    rows = [r for r in read_jsonl(run_dir / "samples.jsonl") if r["condition"] in conditions]
    out = {}
    reused = {"full": run_dir / "summary_rescored.json", "question_answer": run_dir / "summary_neutral.json"}
    for variant, (keep_system, keep_reasoning) in CONTEXT_VARIANTS.items():
        if reused.get(variant) and reused[variant].exists():
            out[variant] = json.loads(reused[variant].read_text()) | {"reused_from": str(reused[variant])}
            continue
        examples = [{k: v for k, v in e.items() if k != "system" or keep_system} | ({} if keep_reasoning else {"reasoning": ""})
                    for e in base]
        out[variant] = score_and_summarize(rows, examples, f"deception_{run_dir.name}_decomp_{variant}", list(conditions), probe_names)
        print(variant, {c: (round(v["auroc"], 3), round(v["recalibrated"]["evasion_rate"], 3))
                        for c, v in out[variant]["probes"][probe_names[0]]["conditions"].items()}, flush=True)
    (run_dir / "summary_decomposition.json").write_text(json.dumps(out, indent=2))


def evaluate_partial(run_dir: Path, probe_names: list[str], max_questions: int = 300) -> None:
    """Preliminary summary from the chunks a still-running eval has finished."""
    # Runs started before interleaving (eval_20261006-144940) use condition-major order.
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    conditions, keys, convs = eval_plan(probe_names, manifest.get("max_questions", max_questions),
                                        interleave=run_dir.name > "eval_20261006-1500", offset=max(manifest.get("offset", 0), 0),
                                        condition_names=tuple(c if not c.startswith("E2_") else "E2" for c in manifest["conditions"])
                                        if manifest else DEFAULT_CONDITIONS, qids=manifest.get("qids"))
    chunks = load_chunks(run_dir / "chunks")
    per_chunk = json.loads(next((run_dir / "chunks").glob("chunk_*.json")).read_text())["per_chunk"]
    idx = [i * per_chunk + j for i in sorted(chunks) for j in range(len(chunks[i]))]
    outs = [[{"text": t} for t in ts] for i in sorted(chunks) for ts in chunks[i]]
    rows, examples = rows_and_examples([keys[k] for k in idx], [convs[k] for k in idx], outs)
    summary = {"partial": True} | score_and_summarize(rows, examples, f"deception_{run_dir.name}_partial{len(rows)}", conditions, probe_names)
    (run_dir / f"summary_partial_{len(rows)}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
