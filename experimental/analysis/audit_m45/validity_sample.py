"""Audit of m45, check 7: a fixed random sample of test rows (mostly top writers, B = 16) for reading by hand.

Usage: python experimental/analysis/audit_m45/validity_sample.py <scratch dir with extract.py's pickles>
Writes experimental/analysis/audit_m45/validity_sample.json (the rows shown) and prints them.
"""

import json
import pickle
import random
import sys
from pathlib import Path

RUNS = Path("missions/m45_toggle_sweep/runs")
OUT = Path("experimental/analysis/audit_m45")
SCRATCH = Path(sys.argv[1])
TOP = ("opus55", "fable51", "sonnet55", "opus48", "gpt6astra", "gpt61sol")
PLAN = {"maths": {("on", True): 6, ("on", False): 7, ("off", True): 4, ("off", False): 5},
        "ordering": {("on", True): 5, ("on", False): 6, ("off", True): 3, ("off", False): 4}}
ROOTS = {"maths": "qwen35_27b", "ordering": "qwen35_27b_ordering"}


def main() -> None:
    rng = random.Random(4545)
    shown = []
    for task, root in ROOTS.items():
        rows = []
        correct_only = {}
        for batch in ("c", "d"):
            rs = pickle.loads((SCRATCH / f"{batch}_{root}.pkl").read_bytes())
            rows += [r | {"batch": batch} for r in rs if r["key"].split("/")[0] in TOP and "@B16#" in r["key"] and "parity" not in r["key"]]
            p = RUNS / batch / root / "correctness_only.json"
            if p.exists():
                correct_only |= json.loads(p.read_text())
        for (version, valid), n in PLAN[task].items():
            pool = [r for r in rows if r["key"].endswith("#" + version) and bool(r["valid"]) == valid]
            for r in rng.sample(pool, n):
                rid = f"{r['key']}|{r['id']}|{r['seed']}"
                shown.append({"task": task, "batch": r["batch"], "key": r["key"], "id": r["id"], "seed": r["seed"], "problem": r["task_text"],
                              "result": r["result"], "reply": r["reply"], "valid": r["valid"], "finished": r["finished"],
                              "why_invalid": r["why_invalid"], "correct_only": correct_only.get(rid), "pv": r["pv"]})
    (OUT / "validity_sample.json").write_text(json.dumps(shown, indent=1, ensure_ascii=False))
    for i, s in enumerate(shown):
        print(f"[{i}] {s['task']} {s['key']} {s['id']} seed {s['seed']} valid={s['valid']} correct_only={s['correct_only']} finished={s['finished']}")
        print(f"    Q: {s['problem'][:160]}")
        print(f"    result: {s['result']}")
        print(f"    reply: {s['reply']!r}")
        if s["why_invalid"]:
            print(f"    why_invalid: {str(s['why_invalid'])[:200]}")


if __name__ == "__main__":
    main()
