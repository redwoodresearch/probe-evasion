"""Generate backdoor-task responses from the target model (Qwen, on Modal) or the
off-policy generator (Claude, via API), as flat rows ready for scoring."""

import json
import time
from pathlib import Path

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


def generate(problems: list[Problem], variants: tuple[str, ...], generator: str, samples: int, seed: int) -> list[dict]:
    keys, convs = [], []
    for variant in variants:
        for p in problems:
            keys.append((variant, p))
            convs.append(build_messages(variant, p.question))

    t0 = time.time()
    if generator == "qwen":
        outputs = [[{"text": t} for t in texts] for texts in Sampler().chat.remote(convs, n=samples, seed=seed)]
    elif generator == "claude":
        outputs = sample_claude(convs, n=samples)
    else:
        raise ValueError(generator)
    print(f"{generator} generation took {time.time() - t0:.0f}s")

    rows = []
    for (variant, p), outs in zip(keys, outputs):
        for i, out in enumerate(outs):
            parsed = parse(out["text"], has_reasoning=generator == "qwen")
            rows.append(
                {
                    "generator": generator,
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
