"""Real rows for the broad ordering probe proposal (step 1, version 1), to read before approving it: the current training
data by kind, the everyday-chat replies that would be added as broad negatives, and the ones that would be left out.
Every example is drawn at random (fixed seed), not picked.

    .venv/bin/python analysis/broad_probe/show_training_examples.py      # writes analysis/broad_probe/training_examples.md
"""

import gzip
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = Path("data/concepts/m26_ordering_train.jsonl")
BROAD = Path("missions/m50_realistic_monitor/runs/broad_scores.jsonl")
LABELS = HERE / "step0" / "labels.jsonl.gz"
PROBE = "q27orr@L48@s0"
STRICT_LINE = -8.06
SEED = 0
ADDED = {"none": 8, "unordered_list": 6}  # step-0 categories that would become broad negatives: how many to show
LEFT_OUT = {"derives_order": 4, "compares_or_ordered": 4}  # categories kept out of training (they may use the concept)


def prompt_text(row: dict) -> str:
    return row["prompt"] if isinstance(row["prompt"], str) else row["prompt"][-1]["content"]


def quote(text: str) -> str:
    return "\n".join("> " + line if line else ">" for line in text.strip().split("\n"))


def current_section(rng: random.Random) -> list[str]:
    rows = [json.loads(line) for line in TRAIN.open()]
    kinds = Counter(r["kind"] for r in rows)
    out = ["## 1. What the ordering probe is trained on now\n",
           f"{len(rows):,} rows: {kinds['computed']} positives and {len(rows) - kinds['computed']:,} negatives, every reply "
           "written by Qwen3.5-27B (reasoning off). Each row is answered in one of six styles: "
           + ", ".join(sorted({r['style'] for r in rows})) + ".\n"]
    meaning = {"computed": "POSITIVE: Qwen works out the order from the clues",
               "given": "negative: the order is already stated in the question",
               "mention": "negative: list the names in the order they first appear",
               "pair": "negative: which two people does the first clue compare",
               "recalled": "negative: recite a known sequence from memory"}
    for kind, text in meaning.items():
        out.append(f"\n### {kind} ({kinds[kind]} rows): {text}\n")
        for r in rng.sample([r for r in rows if r["kind"] == kind], 3 if kind == "computed" else 2):
            out += [f"**Request** ({r['style']}):\n", quote(prompt_text(r)), "\n**Qwen's reply:**\n", quote(r["response"]), ""]
    return out


def chat_section(rng: random.Random) -> list[str]:
    replies = {r["id"]: r for r in map(json.loads, BROAD.open()) if r["context"] == "none"}
    labels = [l for l in map(json.loads, gzip.open(LABELS, "rt")) if l["task"] == "ordering"]
    counts = Counter(l["category"] for l in labels)
    out = ["\n## 2. Proposed broad negatives: everyday-chat replies with no ordering\n",
           f"From m27's pool of {len(replies):,} Qwen replies to Haiku-written everyday requests (chosen to contain numbers "
           "or lists; each reply cut at 96 tokens, so many end mid-sentence). Claude Haiku 5.5 labelled each reply "
           f"(step 0): {counts['none']} have no ordering and {counts['unordered_list']} only a list whose order carries no "
           f"meaning. Those {counts['none'] + counts['unordered_list']} would be split by topic: about half become training "
           "negatives, the other half (plus the current look-alikes) sets the new 1% line. I would read the whole training "
           "half by hand first and drop anything that does use ordering.\n",
           f"The score is the current ordering probe's (no instruction in view). Its strict line is {STRICT_LINE}: every "
           "one of these is currently flagged.\n"]
    sections = [(cat, n, True) for cat, n in ADDED.items()] + [(cat, n, False) for cat, n in LEFT_OUT.items()]
    for cat, n, added in sections:
        if not added and cat == next(iter(LEFT_OUT)):
            out.append(f"\n## 3. Left out of training: chat replies that may use the concept "
                       f"({counts['derives_order']} + {counts['compares_or_ordered']})\n")
        out.append(f"\n### {cat} ({counts[cat]} replies)\n")
        for l in rng.sample([l for l in labels if l["category"] == cat], n):
            r = replies[l["id"]]
            out += [f"**[{r['id']}] topic: {r['topic']}; current score {l['score']:+.2f}**\n", f"Request: _{r['request']}_\n",
                    quote(r["reply"]), f"\nHaiku's evidence: {l['evidence']!r}\n" if l["evidence"] else "", ""]
    return out


def main() -> None:
    rng = random.Random(SEED)
    lines = ["# Broad ordering probe: real training rows\n",
             "Every example below is drawn at random with a fixed seed (`show_training_examples.py`), not picked.\n"]
    lines += current_section(rng) + chat_section(rng)
    (HERE / "training_examples.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
