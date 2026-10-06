"""Generate backdoor-task responses from the target model (Qwen, on Modal) or the
off-policy generator (Claude, via API), as flat rows ready for scoring."""

import json
import time
from pathlib import Path

from probe_evasion import config
from probe_evasion.apps_data import Problem
from probe_evasion.claude_gen import sample_claude
from probe_evasion.modal_app import Sampler
from probe_evasion.parsing import parse
from probe_evasion.prompts import build_messages


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def generate(
    problems: list[Problem],
    variants: tuple[str, ...],
    generator: str,
    samples: int,
    seed: int,
    conditions: dict[str, str] | None = None,
) -> list[dict]:
    """`conditions` maps a condition name to a system-prompt addition (e.g. a probe warning);
    every (condition, variant, problem) is generated in one batch."""
    conditions = conditions or {"base": ""}
    keys, convs = [], []
    for condition, addition in conditions.items():
        for variant in variants:
            for p in problems:
                keys.append((condition, variant, p))
                convs.append(build_messages(variant, p.question, addition))

    t0 = time.time()
    if generator == "qwen":
        per_chunk = max(1, config.CHUNK_SEQS // samples)
        chunks = [convs[i : i + per_chunk] for i in range(0, len(convs), per_chunk)]
        texts = [t for chunk in Sampler().chat.map(chunks, kwargs={"n": samples, "seed": seed}) for t in chunk]
        outputs = [[{"text": t} for t in ts] for ts in texts]
    elif generator == "claude":
        outputs = sample_claude(convs, n=samples)
    else:
        raise ValueError(generator)
    print(f"{generator} generation took {time.time() - t0:.0f}s")

    rows = []
    for (condition, variant, p), outs, conv in zip(keys, outputs, convs):
        for i, out in enumerate(outs):
            parsed = parse(out["text"], has_reasoning=generator == "qwen")
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
