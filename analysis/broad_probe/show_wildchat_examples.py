"""Random real rows from m52's WildChat data, per task: training negatives, the line's pool, and what was left out.

    .venv/bin/modal volume get probe-evasion_nabil-baugher_m52-state_kill-2026-11-30 missions/m52_broad_probes/run missions/m52_broad_probes/
    .venv/bin/python analysis/broad_probe/show_wildchat_examples.py      # writes analysis/broad_probe/wildchat_examples.md
"""

import random
from collections import Counter
from pathlib import Path

from probe_evasion.generation import read_jsonl as jl
from probe_evasion.missions import m52_broad_probes as m52

HERE = Path(__file__).resolve().parent
RUN = m52.out_dir(smoke=False)
SEED = 0
N_TRAIN, N_LINE, N_LEFT_OUT = 10, 4, 4
REPLY_CHARS = 700


def quote(text: str) -> str:
    text = text.strip()
    text = text if len(text) <= REPLY_CHARS else text[:REPLY_CHARS] + " […]"
    return "\n".join("> " + line if line else ">" for line in text.split("\n"))


def block(r: dict, category: str, evidence: str) -> list[str]:
    out = [f"**[{r['id'][:8]}] Haiku: {category}**\n", "Request:\n", quote(r["request"]), "\nQwen's reply:\n", quote(r["reply"])]
    return out + ([f"\nHaiku's evidence: {evidence!r}\n"] if evidence else ["\n"])


def main() -> None:
    rng = random.Random(SEED)
    replies = {r["id"]: r for r in jl(RUN / "wildchat_replies.jsonl")}
    labels = jl(RUN / "wildchat_labels.jsonl")
    lines = ["# m52: real WildChat rows (random, fixed seed)\n",
             f"{len(replies):,} WildChat requests answered by Qwen3.5-27B (reasoning off, at most {m52.MAX_TRAINING_TOKENS} tokens, "
             "so long replies are cut). Split by conversation: half for training, half for setting the line.\n"]
    for task in m52.TASKS:
        lab = {l["id"]: l for l in labels if l["task"] == task}
        sets = m52.wildchat_sets(task, list(replies.values()), labels)
        counts = Counter(l["category"] for l in lab.values())
        lines += [f"\n## {task.capitalize()}\n",
                  "Haiku's labels over all replies: " + ", ".join(f"{c} {n:,}" for c, n in counts.most_common()) + ".\n",
                  f"- Training negatives: {len(sets['train']):,} (training half, leaving out "
                  f"{', '.join(sorted(m52.NOT_TRAINING_NEGATIVE[task]))})",
                  f"- Line pool: {len(sets['line']):,} (other half, leaving out {', '.join(sorted(m52.NOT_IN_LINE_POOL[task]))})\n"]
        groups = [("Training negatives", sets["train"], N_TRAIN),
                  ("Line pool", sets["line"], N_LINE),
                  ("Left out of training (may use the task)",
                   [r for r in replies.values() if r["id"] in lab and r["split"] == "train" and lab[r["id"]]["category"] in m52.NOT_TRAINING_NEGATIVE[task]],
                   N_LEFT_OUT)]
        for title, rows, n in groups:
            lines.append(f"\n### {title}: {n} of {len(rows):,}\n")
            for r in rng.sample(rows, min(n, len(rows))):
                lines += block(r, lab[r["id"]]["category"], lab[r["id"]]["evidence"])
    (HERE / "wildchat_examples.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
