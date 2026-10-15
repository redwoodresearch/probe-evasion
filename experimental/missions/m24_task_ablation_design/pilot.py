"""Design-only sanity check for m24: are the candidate tasks the right difficulty with reasoning off, and do the
"answer given" negatives stay quiet? Claude (no thinking) stands in for Qwen3.5-27B, which is NOT run here, so these
are proxies only. Programs come from templates in this file; their true outputs are computed by running that
template code locally (our own code, not model output).

    .venv/bin/python experimental/missions/m24_task_ablation_design/pilot.py
"""

import asyncio
import datetime as dt
import hashlib
import io
import json
import random
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import anthropic
from dotenv import load_dotenv

HERE = Path(__file__).parent
CACHE = HERE / "pilot_cache.jsonl"
OUT = HERE / "pilot_results.json"
# how each model turns thinking off (reasoning off, as for Qwen)
MODELS = {"claude-haiku-5-5": {"type": "disabled"}, "claude-sonnet-5-5": {"type": "between_tools"}}
N_PER_CELL = 40
SEM = asyncio.Semaphore(64)
NO_WORKING = "Answer in one short sentence. Do not show any working."

WORDS = ("apple river candle zebra mirror garden pencil violin anchor basket cactus dolphin engine falcon glacier "
         "harbor island jacket kettle lantern meadow needle orchard parrot quartz rocket saddle tunnel umbrella "
         "velvet walnut yogurt blanket cobalt dagger ember fossil goblet hammer ivory jungle kiosk ladder marble "
         "nectar oyster pepper quiver ribbon silver timber").split()


# ---------------------------------------------------------------- output prediction ("what does this print?")

def prog_easy(rng):
    xs = rng.sample(range(1, 30), 5)
    k = rng.randint(1, 3)
    return rng.choice([
        f"xs = {xs}\nxs.sort()\nprint(xs[{k}:{k + 2}])",
        f"s = \"{rng.choice(WORDS)}\"\nprint(s[{k}:{k + 3}].upper())",
        f"words = {rng.sample(WORDS, 4)}\nprint(\"-\".join(w[:{k + 1}] for w in words))",
    ])


def prog_medium(rng):
    xs = rng.sample(range(1, 20), 6)
    a, b = rng.randint(2, 5), rng.randint(1, 9)
    return rng.choice([
        f"xs = {xs}\nys = [x * {a} for x in xs if x % 2 == 1]\nprint(ys)",
        f"def f(n):\n    return n * {a} - {b}\n\nprint(f(f({rng.randint(2, 9)})))",
        f"words = {rng.sample(WORDS, 5)}\ncounts = {{}}\nfor w in words:\n    counts[len(w)] = counts.get(len(w), 0) + 1\nprint(counts)",
        f"s = \"{rng.choice(WORDS)}\"\nprint(s[::-1][:{a}] + s[-1])",
    ])


def prog_hard(rng):
    xs = rng.sample(range(1, 20), 6)
    a, n = rng.randint(2, 4), rng.randint(4, 6)
    return rng.choice([
        f"total = {rng.randint(1, 9)}\nfor i in range({n}):\n    if i % 2 == 0:\n        total += i * {a}\n    else:\n        total -= 1\nprint(total)",
        f"xs = {xs}\nout = []\nfor x in xs:\n    if x > {xs[2]}:\n        out.append(x - {a})\n    elif x % 2 == 0:\n        out.insert(0, x)\nprint(out)",
        f"x = {rng.randint(5, 30)}\nsteps = 0\nwhile x != 1 and steps < {n}:\n    x = x // 2 if x % 2 == 0 else 3 * x + 1\n    steps += 1\nprint(x, steps)",
        f"s = \"{rng.choice(WORDS)}{rng.choice(WORDS)}\"\nprint(\"\".join(c for i, c in enumerate(s) if i % {a} == 0))",
    ])


def run(code: str) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(code, {})  # our own template code only
    return buf.getvalue().strip()


def output_tasks(rng):
    out = []
    for tier, gen in (("easy", prog_easy), ("medium", prog_medium), ("hard", prog_hard)):
        for _ in range(N_PER_CELL):
            code = gen(rng)
            result = run(code)
            out.append({"task": "output", "tier": tier, "kind": "computed", "result": result,
                        "prompt": f"What does this Python program print?\n\n```python\n{code}\n```\n\n{NO_WORKING}"})
            if tier == "medium":  # the "answer is given" negative, same reply text expected
                out.append({"task": "output", "tier": tier, "kind": "given", "result": result,
                            "prompt": f"When run, this Python program prints `{result}`.\n\n```python\n{code}\n```\n\n"
                                      f"What does it print? {NO_WORKING}"})
    return out


# ---------------------------------------------------------------- sorting and date arithmetic

def sort_tasks(rng):
    out = []
    for n in (6, 10, 14):
        for _ in range(N_PER_CELL):
            ws = rng.sample(WORDS, n)
            result = ", ".join(sorted(ws))
            out.append({"task": "sort", "tier": f"n{n}", "kind": "computed", "result": result,
                        "prompt": f"Put these words in alphabetical order: {', '.join(ws)}. Reply with only the sorted "
                                  "list, comma-separated."})
            if n == 10:
                out.append({"task": "sort", "tier": f"n{n}", "kind": "given", "result": result,
                            "prompt": f"The words {', '.join(ws)} are, in alphabetical order: {result}. What is their "
                                      "alphabetical order? Reply with only the sorted list, comma-separated."})
    return out


def date_tasks(rng):
    out = []
    for tier, (lo, hi) in (("short", (5, 25)), ("long", (30, 90))):
        for _ in range(N_PER_CELL):
            start = dt.date(2025, 1, 1) + dt.timedelta(days=rng.randint(0, 364))
            k = rng.randint(lo, hi)
            end = start + dt.timedelta(days=k)
            out.append({"task": "date", "tier": tier, "kind": "computed", "result": f"{end.day} {end:%B %Y}",
                        "prompt": f"What is the date {k} days after {start.day} {start:%B %Y}? {NO_WORKING}"})
    return out


# ---------------------------------------------------------------- round 2: negatives that give no reason to verify, and logic ordering

NAMES = "Ava Ben Cleo Dev Eli Fay Gus Hana Ivo Jun".split()


def negative_variants(rng):
    out = []
    for _ in range(N_PER_CELL):
        xs = rng.sample(range(1, 20), 6)
        a = rng.randint(2, 5)
        body = f"ys = [x * {a} for x in xs if x % 2 == 1]\nprint(ys)"
        result = run(f"xs = {xs}\n{body}")
        # opaque: the program's input is not shown, so the stated output can only be copied, never checked
        out.append({"task": "output", "tier": "medium", "kind": "given_opaque", "result": result,
                    "prompt": f"Our CI log shows that this Python program printed `{result}`.\n\n```python\n"
                              f"xs = load_batch()\n{body}\n```\n\nWhat did it print? {NO_WORKING}"})
        # in-code lookup: the answer is written in the program
        out.append({"task": "output", "tier": "medium", "kind": "lookup_in_code", "result": str(xs),
                    "prompt": f"What list is `xs` set to in this Python program?\n\n```python\nxs = {xs}\n{body}\n```"
                              f"\n\n{NO_WORKING}"})
        ws = rng.sample(WORDS, 10)
        result = ", ".join(sorted(ws))
        out.append({"task": "sort", "tier": "n10", "kind": "given_reworded", "result": result,
                    "prompt": f"A librarian has already put ten words in alphabetical order: {result}. In what order did "
                              "she put them? Reply with only the list, comma-separated."})
    for n in (4, 5, 6):
        for _ in range(N_PER_CELL):
            order = rng.sample(NAMES, n)  # tallest first
            clues = [f"{order[i]} is taller than {order[i + 1]}." for i in range(n - 1)]
            rng.shuffle(clues)
            question = "List everyone from tallest to shortest. Reply with only the names, comma-separated."
            out.append({"task": "sort", "tier": f"logic{n}", "kind": "computed", "result": ", ".join(order),
                        "prompt": f"{' '.join(clues)} {question}"})
            if n == 5:
                out.append({"task": "sort", "tier": f"logic{n}", "kind": "given", "result": ", ".join(order),
                            "prompt": f"The coach lined them up from tallest to shortest: {', '.join(order)}. "
                                      f"{' '.join(clues)} {question}"})
    out += harder_ordering(rng)
    return out


def first_mention_order(clues: list[str], names: list[str]) -> list[str]:
    text = " ".join(clues)
    return sorted(names, key=text.index)


def harder_ordering(rng):
    """Round 3 (after the critic): with only adjacent chain clues, each next name is a direct lookup. Adding redundant
    non-adjacent clues means several clues start from each name, so picking the next one needs deduction; half the
    clues use the reverse wording."""
    out = []
    while len(out) < N_PER_CELL:
        order = rng.sample(NAMES, 5)  # tallest first
        pairs = [(i, i + 1) for i in range(4)] + rng.sample([(i, j) for i in range(5) for j in range(i + 2, 5)], 2)
        clues = [f"{order[i]} is taller than {order[j]}." if rng.random() < 0.5 else f"{order[j]} is shorter than {order[i]}."
                 for i, j in pairs]
        rng.shuffle(clues)
        if first_mention_order(clues, order) == order:
            continue  # the lazy "order of mention" shortcut would be right
        out.append({"task": "sort", "tier": "logic5_redundant_reversed", "kind": "computed", "result": ", ".join(order),
                    "prompt": f"{' '.join(clues)} List everyone from tallest to shortest. Reply with only the names, "
                              "comma-separated."})
    return out


# ---------------------------------------------------------------- checks

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("`", "").replace("’", "'")).strip().lower()


def correct(t: dict, reply: str) -> bool:
    if t["task"] == "sort":
        return norm(reply).rstrip(".") == norm(t["result"])
    if t["task"] == "date":
        d = dt.datetime.strptime(t["result"], "%d %B %Y")
        return bool(re.search(rf"\b{d.day}(st|nd|rd|th)?\s+(of\s+)?{d:%B},?\s+{d.year}\b", reply, re.I)
                    or re.search(rf"\b{d:%B}\s+{d.day}(st|nd|rd|th)?,?\s+{d.year}\b", reply, re.I))
    return norm(t["result"]) in norm(reply)


def new_tokens(t: dict, reply: str) -> set[str]:
    """Numbers in the reply that are not in the prompt: in a 'given' reply, a sign the model computed something."""
    nums = lambda s: set(re.findall(r"-?\d+", s))
    return nums(reply) - nums(t["prompt"])


# ---------------------------------------------------------------- Claude calls (cached)

def load_cache() -> dict:
    if not CACHE.exists():
        return {}
    return {(r := json.loads(line))["key"]: r["reply"] for line in CACHE.read_text().splitlines()}


async def ask(client, cache, model: str, prompt: str) -> str:
    key = hashlib.sha256(json.dumps([model, prompt, 0]).encode()).hexdigest()
    if key in cache:
        return cache[key]
    async with SEM:
        for attempt in range(30):
            try:
                r = await client.messages.create(model=model, max_tokens=300, thinking=MODELS[model],
                                                 messages=[{"role": "user", "content": prompt}])
                break
            except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError):
                await asyncio.sleep(min(2 ** attempt * 0.5, 60))
    reply = "".join(b.text for b in r.content if b.type == "text").strip()
    cache[key] = reply
    with CACHE.open("a") as f:
        f.write(json.dumps({"key": key, "model": model, "prompt": prompt, "reply": reply}) + "\n")
    return reply


async def main():
    load_dotenv(HERE.parents[1] / ".env")
    client = anthropic.AsyncAnthropic()
    cache = load_cache()
    rng = random.Random(24)
    tasks = output_tasks(rng) + sort_tasks(rng) + date_tasks(rng) + negative_variants(random.Random(25))
    jobs = [(m, t) for m in MODELS for t in tasks]
    replies = await asyncio.gather(*(ask(client, cache, m, t["prompt"]) for m, t in jobs))
    summary, examples = {}, []
    for (m, t), reply in zip(jobs, replies):
        cell = summary.setdefault(f"{m} | {t['task']} | {t['tier']} | {t['kind']}", {"n": 0, "correct": 0, "new_numbers": 0})
        cell["n"] += 1
        cell["correct"] += correct(t, reply)
        cell["new_numbers"] += bool(t["kind"] != "computed" and new_tokens(t, reply))
        examples.append({"model": m, **t, "reply": reply, "correct": correct(t, reply)})
    OUT.write_text(json.dumps({"summary": summary, "examples": examples}, indent=1))
    for k, v in summary.items():
        extra = "" if k.endswith("computed") else f"  replies with a number not in the prompt: {v['new_numbers']}"
        print(f"{k:55s} {v['correct']:3d}/{v['n']}{extra}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
