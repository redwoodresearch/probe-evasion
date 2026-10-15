"""Random real rows for every category of training data the full broad probes (m52) would use, per task and label:
the current puzzle / word-problem rows, the chat-formatted rows with known answers (m52 pilot), and WildChat replies.
Every example is drawn at random with a fixed seed, not picked.

    .venv/bin/python analysis/broad_probe/show_training_examples.py      # writes analysis/broad_probe/training_examples.md
"""

import random
from collections import Counter
from pathlib import Path

from probe_evasion.generation import read_jsonl as jl
from probe_evasion.missions import m52_broad_probes as m52

HERE = Path(__file__).resolve().parent
CURRENT = {"ordering": Path("data/concepts/m26_ordering_train.jsonl"), "maths": Path("data/concepts/onpolicy_qwen35_maths_train_clean.jsonl")}
CHAT_PILOT = m52.CHAT_PILOT_V2
RUN = m52.out_dir(smoke=False)
SEED = 0
REPLY_CHARS = 600
N_CURRENT, N_CHAT, N_WILDCHAT = 2, 4, 3
WILDCHAT_POSITIVE = {"ordering": "derives_order", "maths": "computes"}  # candidates only: they would be hand-checked first
KIND_MEANING = {"computed": "Qwen works it out", "given": "the answer is already in the request",
                "mention": "list names in the order they first appear", "pair": "which two people the first clue compares",
                "recalled": "recite a known fact from memory"}


def quote(text: str) -> str:
    text = text.strip()
    text = text if len(text) <= REPLY_CHARS else text[:REPLY_CHARS] + " […]"
    return "\n".join("> " + line if line else ">" for line in text.split("\n"))


def block(title: str, request: str, reply: str, note: str = "") -> list[str]:
    return [f"**{title}**\n", "Request:\n", quote(request), "\nQwen's reply:\n", quote(reply), f"\n{note}\n" if note else "\n"]


def current(task: str, label: int, rng: random.Random) -> list[str]:
    rows = [r for r in jl(CURRENT[task]) if int(r["label"]) == label]
    out = []
    for kind, n in Counter(r["kind"] for r in rows).most_common():
        out.append(f"\n#### Current: {kind} ({n:,} rows; {KIND_MEANING[kind]})\n")
        for r in rng.sample([r for r in rows if r["kind"] == kind], N_CURRENT + (label == 1)):
            prompt = r["prompt"] if isinstance(r["prompt"], str) else r["prompt"][-1]["content"]
            out += block(f"style: {r['style']}", prompt, r["response"])
    return out


def chat(task: str, label: int, rng: random.Random) -> list[str]:
    rows = [r for r in jl(CHAT_PILOT) if r["task"] == task and r.get("label") == label]
    kept = [r for r in rows if r["label_ok"]]
    kind = rows[0]["kind"]
    out = [f"\n#### Chat-formatted, known answer: {kind} (pilot: {len(kept)} of {len(rows)} pass the answer check; "
           f"{sum(r['shows_working'] for r in kept)} of those show working)\n"]
    for r in rng.sample(kept, min(N_CHAT, len(kept))):
        answer = ", ".join(r["puzzle"]["answer"]) if task == "ordering" else f"{r['result']:,}"
        out += block(f"situation: {r['scenario']}; correct answer: {answer}; working shown: {r['shows_working']}; "
                     f"brevity request: {r['brevity']!r}", r["request"], r["reply"])
    failed = [r for r in rows if not r["label_ok"]]
    if failed:
        r = rng.choice(failed)
        out += block("one that FAILED the answer check (left out)", r["request"], r["reply"])
    return out


def wildchat(task: str, label: int, rng: random.Random) -> list[str]:
    replies = {r["id"]: r for r in jl(RUN / "wildchat_replies.jsonl")}
    labels = [l for l in jl(RUN / "wildchat_labels.jsonl") if l["task"] == task]
    counts = Counter(l["category"] for l in labels)
    if label == 1:
        cats = [WILDCHAT_POSITIVE[task]]
        head = f"\n#### WildChat replies Haiku says do the task (candidates; I'd hand-check every one): {WILDCHAT_POSITIVE[task]}\n"
    else:
        excluded = m52.NOT_TRAINING_NEGATIVE[task] | {"refused"}
        cats = [c for c in m52.CATEGORIES[task] if c not in excluded]
        head = "\n#### WildChat replies used as negatives\n"
    out = [head]
    for c in cats:
        out.append(f"\n##### {c} ({counts[c]:,} replies)\n")
        for l in rng.sample([l for l in labels if l["category"] == c], N_WILDCHAT + (label == 1)):
            r = replies[l["id"]]
            out += block(f"[{r['id'][:8]}]", r["request"], r["reply"], f"Haiku's evidence: {l['evidence']!r}" if l["evidence"] else "")
    if label == 0:
        brief = [r for r in jl(CHAT_PILOT) if r["task"] == "wildchat"]
        out.append(f"\n##### WildChat with a brevity request added, answered again (pilot: {len(brief)}; not yet labelled)\n")
        for r in rng.sample(brief, N_WILDCHAT):
            out += block(f"[{r['id'][:8]}] brevity request: {r['brevity']!r}", r["request"], r["reply"])
    return out


def main() -> None:
    rng = random.Random(SEED)
    lines = ["# Broad probes: real rows from every category\n",
             "Every example is drawn at random with a fixed seed (`show_training_examples.py`). Every reply is Qwen3.5-27B's "
             "own (reasoning off). Long replies are cut here for reading.\n"]
    for task in ("ordering", "maths"):
        lines.append(f"\n## {task.capitalize()}\n")
        for label, name in ((1, "Positives (the probe should fire)"), (0, "Negatives (the probe should stay low)")):
            lines.append(f"\n### {name}\n")
            lines += current(task, label, rng) + chat(task, label, rng) + wildchat(task, label, rng)
    (HERE / "training_examples.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
