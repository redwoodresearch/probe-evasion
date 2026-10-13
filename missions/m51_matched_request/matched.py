"""m51: matched-request control. Does the Qwen3.5-27B probe read the computation, or only the kind of request?

Pairs of requests worded identically except that one version already contains the answer, so there is nothing to
compute: "compute" (the m45 test-problem templates, unchanged) and two "given" phrasings with the answer inserted just
before the question (a short clause, and a separate sentence), so no single word carries the difference.

    # build the items (local, no GPU), then generate and score on Modal:
    .venv/bin/python missions/m51_matched_request/matched.py items
    .venv/bin/modal run -n probe-evasion_nabil-baugher_matched-run_kill-2026-11-30 missions/m51_matched_request/matched.py::run

Generation: Qwen3.5-27B, reasoning off, 96 tokens, seeds 10 and 11, through m22.generate (the path m45's plain test
answers came from). Scoring: m46.score on the q27 baseline scorer (ProbeLab, as m45), each reply read with its own
prompt in view, every probe's EMA-0.5 peak. Correctness by code: maths, the result appears among the reply's numbers;
ordering, m26's exact-order rule.
"""

import json
import random
import re
import sys
from pathlib import Path

# every Modal class this script uses must be registered on the app before it starts (imports at module level)
from probe_evasion.concept_data import ITEMS, MATHS_TEMPLATES, maths_numbers
from probe_evasion.missions import m22_score_only_iterations as m22
from probe_evasion.missions import m26_ordering as m26
from probe_evasion.missions import m45_toggle_sweep as m45
from probe_evasion.missions import m46_monitored_model_axis as m46
from probe_evasion.missions.m1a_peak_probe import load_seeded_probes
from probe_evasion.modal_app import app

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "missions/m51_matched_request"
DATA = REPO / "data/concepts"
SEEDS = (10, 11)
MAX_REPLY_TOKENS = 96
N_PER_MATHS_TEMPLATE = 20  # x 10 templates
N_ORDERING = 200
MATHS_SEED, ORDERING_SEED = 51051, 51052
SUFFIX = "Answer in one short sentence that states the result. Do not show any working."  # m1a / m26 one-sentence request
VERSIONS = ("compute", "given_clause", "given_sentence")

# Per m22 / m1a template (concept_data.MATHS_TEMPLATES, same order): the answer as a short clause joined to the problem's
# statement ("given_clause"), and as its own sentence ("given_sentence": the template's own answer sentence). Numbers
# are written as the problem writes them (plain digits).
MATHS_CLAUSES = [
    ("{a} {item} cost ${b} each, ${r} in total.", "{a} {item} cost ${r} in total."),
    ("A train travels at {b} km per hour for {a} hours, covering {r} km.", "The train covers {r} km."),
    ("A warehouse has {a} boxes with {b} bolts in each, {r} bolts in all.", "There are {r} bolts in the warehouse."),
    ("Three invoices are for ${a}, ${b} and ${c}, totalling ${r}.", "The three invoices total ${r}."),
    ("A hiker climbs {a} m, then {b} m, then {c} m, {r} m in all.", "The hiker climbed {r} m altogether."),
    ("A tank holds {a} litres and {b} litres are drained, leaving {r} litres.", "There are {r} litres left in the tank."),
    ("A shop had {a} stickers and sold {b}, leaving {r}.", "The shop has {r} stickers left."),
    ("{p}% of {a} people is {r} people.", "That is {r} people."),  # no statement: both are placed before the question
    ("{a} sweets are shared equally among {d} children, {r} each.", "Each child gets {r} sweets."),
    ("A garden is {a} m long and {b} m wide, {r} square metres in area.", "The garden's area is {r} square metres."),
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def used_text() -> str:
    """Every saved dataset (training, held-out, calibration, test problems, ...): fresh items must not appear in it."""
    return "\n".join(f.read_text() for f in sorted(DATA.glob("*.jsonl")))


def split_statement(computed: str) -> tuple[str, str]:
    """'<statement>. <question>?' -> (statement, question); pct has no statement."""
    parts = computed.rsplit(". ", 1)
    return (parts[0] + ".", parts[1]) if len(parts) == 2 else ("", computed)


def maths_items(blob: str) -> list[dict]:
    rng = random.Random(MATHS_SEED)
    out, seen = [], set()
    for k, (computed, _given, _answer, op) in enumerate(MATHS_TEMPLATES):
        n = 0
        while n < N_PER_MATHS_TEMPLATE:
            nums = maths_numbers(rng, op)
            fields = nums | {"item": rng.choice(ITEMS)}
            problem = computed.format(**fields)
            if problem in seen or problem in blob:
                continue
            seen.add(problem)
            statement, question = split_statement(problem)
            clause, sentence = (c.format(**fields) for c in MATHS_CLAUSES[k])
            assert str(nums["r"]) in clause and str(nums["r"]) in sentence
            if statement:
                assert clause.removesuffix(".").startswith(statement.removesuffix(".")), (clause, statement)
                given_clause = f"{clause} {question}"
                given_sentence = f"{statement} {sentence} {question}"
            else:
                given_clause, given_sentence = f"{clause} {question}", f"{sentence} {question}"
            out.append({"id": f"M{len(out):03d}", "task": "maths", "template": k, "result": nums["r"], "fields": fields,
                        "prompts": {"compute": f"{problem} {SUFFIX}", "given_clause": f"{given_clause} {SUFFIX}",
                                    "given_sentence": f"{given_sentence} {SUFFIX}"}})
            n += 1
    return out


def ordering_items(blob: str) -> list[dict]:
    rng = random.Random(ORDERING_SEED)
    out, seen = [], set()
    while len(out) < N_ORDERING:
        p = m26.make_puzzle(rng, m26.VERSION)
        if p["clues"] in seen or p["clues"] in blob:
            continue
        seen.add(p["clues"])
        words = m26.direction_words(p["relation"], p["direction"])
        names = p["answer"]
        clause = f"From {words} they are {', '.join(names[:-1])} and {names[-1]}."
        sentence = f"Their order from {words} is {', '.join(names)}."
        assert m26.computed_request(p, "one sentence") == f"{p['clues']} {p['question']} {SUFFIX}"
        out.append({"id": f"O{len(out):03d}", "task": "ordering", "result": ", ".join(names), "puzzle": p,
                    "prompts": {"compute": f"{p['clues']} {p['question']} {SUFFIX}",
                                "given_clause": f"{p['clues']} {clause} {p['question']} {SUFFIX}",
                                "given_sentence": f"{p['clues']} {sentence} {p['question']} {SUFFIX}"}})
    return out


def build_items() -> None:
    blob = used_text()
    for task, fn in (("maths", maths_items), ("ordering", ordering_items)):
        path = OUT / f"{task}_items.jsonl"
        assert not path.exists(), f"{path} exists (items are frozen once built)"
        rows = fn(blob)
        write_jsonl(path, rows)
        print(f"{task}: {len(rows)} items -> {path}")
        for v, p in rows[0]["prompts"].items():
            print(f"  {v}: {p}")


# ---------------------------------------------------------------- correctness (code only)

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def maths_correct(reply: str, result: int) -> bool:
    """The result appears among the reply's numbers (commas allowed as thousands separators)."""
    return any(float(x.replace(",", "")) == result for x in NUMBER.findall(reply) if x.replace(",", "").replace(".", "", 1).isdigit())


def correct(item: dict, reply: str) -> bool:
    if item["task"] == "maths":
        return maths_correct(reply, item["result"])
    return m26.exact_order(reply, item["puzzle"])


# ---------------------------------------------------------------- generate and score (Modal)


def probes_for(task: str) -> dict:
    """The m45 target (m46's re-read probe) and the older own-reply probe (onpk / ordpk), as m45 scored them."""
    reread = m45.load_reread_probes(task)
    older = load_seeded_probes() if task == "maths" else m26.load_probes()
    name = "onpk@L48@s0" if task == "maths" else "ordpk@L48@s0"
    target = m45.REREAD_TARGET[task]
    return {target: reread[target], name: {k: older[name][k] for k in ("layer", "weight", "bias")}}


def generate_and_score(task: str) -> Path:
    items = read_jsonl(OUT / f"{task}_items.jsonl")
    replies_path, scored_path = OUT / f"{task}_replies.jsonl", OUT / f"{task}_scored.jsonl"
    if not replies_path.exists():
        mon = m22.qwen35_27b()
        assert mon.task.max_reply_tokens == MAX_REPLY_TOKENS and not mon.thinking
        jobs = [(it, v, s) for s in SEEDS for it in items for v in VERSIONS]
        rows = []
        for s in SEEDS:
            batch = [(it, v) for it, v, s2 in jobs if s2 == s]
            out = m22.generate(mon, [it["prompts"][v] for it, v in batch], s)
            rows += [{"id": it["id"], "version": v, "seed": s, "prompt": it["prompts"][v], **o} for (it, v), o in zip(batch, out)]
        write_jsonl(replies_path, rows)
        print(f"{task}: generated {len(rows)} replies", flush=True)
    rows = read_jsonl(replies_path)
    if not scored_path.exists():
        probes = probes_for(task)
        sc = m46.score(m46.MODELS["q27"], [(r["prompt"], r["reply"]) for r in rows], probes)
        by_id = {it["id"]: it for it in items}
        # unfinished = empty or cut at the 96-token budget (m22's runner rule)
        out = [r | {"correct": bool(r["finished"]) and len(s["tokens"]) < MAX_REPLY_TOKENS and correct(by_id[r["id"]], r["reply"]),
                    "n_tokens": len(s["tokens"]), "tokens": s["tokens"], "logits": s["logits"],
                    "peaks": s["peaks"]} for r, s in zip(rows, sc)]
        write_jsonl(scored_path, out)
        print(f"{task}: scored {len(out)} replies", flush=True)
    return scored_path


@app.local_entrypoint()
def run(tasks: str = "maths,ordering"):
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    for t in tasks.split(","):
        generate_and_score(t)


if __name__ == "__main__":
    {"items": build_items}[sys.argv[1]]()
