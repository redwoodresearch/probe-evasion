"""Generate backdoor-task responses from the target model (Qwen, on Modal) or the
off-policy generator (Claude, via API), as flat rows ready for scoring."""

import json
import time
from pathlib import Path

from probe_evasion import config
from probe_evasion.apps_data import Problem
from probe_evasion.claude_gen import sample_claude
from probe_evasion.modal_app import sampler_for
from probe_evasion.parsing import parse
from probe_evasion.prompts import build_messages


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sample(
    convs: list[list[dict]], generator: str, samples: int, seed: int, thinking: bool = True,
    chunk_seqs: int = config.CHUNK_SEQS, max_tokens: int = config.MAX_NEW_TOKENS, checkpoint_dir: Path | None = None,
) -> list[list[dict]]:
    """Per conversation, `samples` outputs as {"text", ...}. Qwen work is spread across GPUs in chunks of
    ~chunk_seqs sequences (raise it for short outputs, where per-chunk overhead dominates). With
    checkpoint_dir, each chunk is saved the moment it finishes and already-saved chunks are skipped."""
    t0 = time.time()
    if generator.startswith("qwen"):
        per_chunk = max(1, chunk_seqs // samples)
        chunks = [convs[i : i + per_chunk] for i in range(0, len(convs), per_chunk)]
        done = load_chunks(checkpoint_dir) if checkpoint_dir else {}
        todo = [i for i in range(len(chunks)) if i not in done]
        if done:
            print(f"reusing {len(done)} saved chunks; generating {len(todo)}")
        kwargs = {"n": samples, "seed": seed, "thinking": thinking, "max_tokens": max_tokens}
        results = sampler_for(generator).chat_tagged.map(todo, [chunks[i] for i in todo], kwargs=kwargs, order_outputs=False)
        for i, texts in results:
            done[i] = texts
            if checkpoint_dir:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                (checkpoint_dir / f"chunk_{i:05d}.json").write_text(json.dumps({"chunk": i, "per_chunk": per_chunk, "texts": texts}))
            print(f"chunk {i} done ({len(done)}/{len(chunks)}, {time.time() - t0:.0f}s)")
        outputs = [[{"text": t} for t in ts] for i in range(len(chunks)) for ts in done[i]]
    elif generator == "claude":
        outputs = sample_claude(convs, n=samples)
    else:
        raise ValueError(generator)
    print(f"{generator} generation took {time.time() - t0:.0f}s")
    return outputs


def load_chunks(checkpoint_dir: Path) -> dict[int, list[list[str]]]:
    return {d["chunk"]: d["texts"] for d in (json.loads(p.read_text()) for p in sorted(Path(checkpoint_dir).glob("chunk_*.json")))}


def plan(problems: list[Problem], variants: tuple[str, ...], conditions: dict[str, str] | None = None, interleave: bool = False):
    """Deterministic (keys, convs) for every (condition, variant, problem); `conditions` maps a condition
    name to a system-prompt addition (e.g. a probe warning)."""
    conditions = conditions or {"base": ""}
    keys, convs = [], []
    for condition, addition in conditions.items():
        for variant in variants:
            for p in problems:
                keys.append((condition, variant, p))
                convs.append(build_messages(variant, p.question, addition))
    if interleave:  # problem-major order: every chunk mixes all conditions, so early partials are balanced
        order = sorted(range(len(keys)), key=lambda i: (problems.index(keys[i][2]), i))
        keys, convs = [keys[i] for i in order], [convs[i] for i in order]
    return keys, convs


def to_rows(keys, convs, outputs, generator: str, thinking: bool) -> list[dict]:
    rows = []
    for (condition, variant, p), outs, conv in zip(keys, outputs, convs):
        for i, out in enumerate(outs):
            # With thinking off, Qwen's chat template pre-fills an empty <think></think>, so the text is the answer.
            parsed = parse(out["text"], has_reasoning=generator.startswith("qwen") and thinking)
            rows.append(
                {
                    "generator": generator,
                    "condition": condition,
                    "system": conv[0]["content"],
                    "variant": variant,
                    "problem_id": p.problem_id,
                    "sample": i,
                    "raw": out["text"],
                    "reasoning": parsed.reasoning,
                    "final": parsed.final,
                    "code": parsed.code,
                    "exploit": parsed.exploit,
                    "truncated_reasoning": parsed.truncated_reasoning,
                    "stop_reason": out.get("stop_reason"),
                }
            )
    return rows


def generate(
    problems: list[Problem],
    variants: tuple[str, ...],
    generator: str,
    samples: int,
    seed: int,
    conditions: dict[str, str] | None = None,
    thinking: bool = True,
    chunk_seqs: int = config.CHUNK_SEQS,
    checkpoint_dir: Path | None = None,
    interleave: bool = False,
) -> list[dict]:
    keys, convs = plan(problems, variants, conditions, interleave)
    outputs = sample(convs, generator, samples, seed, thinking, chunk_seqs, checkpoint_dir=checkpoint_dir)
    return to_rows(keys, convs, outputs, generator, thinking)


def partial_rows(problems, variants, conditions, checkpoint_dir: Path, generator: str = "qwen", thinking: bool = True,
                 interleave: bool = False) -> list[dict]:
    """Rows for whichever chunks have finished so far (same plan as generate())."""
    keys, convs = plan(problems, variants, conditions, interleave)
    chunks = load_chunks(checkpoint_dir)
    if not chunks:
        return []
    per_chunk = json.loads(next(Path(checkpoint_dir).glob("chunk_*.json")).read_text())["per_chunk"]
    idx = [i * per_chunk + j for i in sorted(chunks) for j in range(len(chunks[i]))]
    outputs = [[{"text": t} for t in ts] for i in sorted(chunks) for ts in chunks[i]]
    return to_rows([keys[k] for k in idx], [convs[k] for k in idx], outputs, generator, thinking)
