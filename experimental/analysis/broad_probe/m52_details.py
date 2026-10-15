"""m52 details: crossings by writer for the new probes, each seed's line, and whether the new probe separates task from
non-task inside m27's everyday chat (step 0's labels).

    PYTHONPATH=src .venv/bin/python experimental/analysis/broad_probe/m52_details.py
"""

import gzip
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from probe_evasion.missions import m52_broad_probes as m52

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("m52_analysis", HERE / "m52_analysis.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
STEP0 = HERE / "step0" / "labels.jsonl.gz"


def crossings_by_writer(rows: list[dict], probe: str, line: float) -> dict[str, tuple[int, int]]:
    by_key = {(r["key"], r["task"]["id"], r["seed"]): r for r in rows if f"@B{A.B}#" in r["key"]}
    out = defaultdict(lambda: [0, 0])
    for (key, i, s), on in by_key.items():
        off = by_key.get((key[:-3] + "#off", i, s))
        if not key.endswith("#on") or off is None or not (A.is_valid(on) and A.is_valid(off)):
            continue
        w = key.split("/")[0]
        out[w][1] += 1
        out[w][0] += on["peaks"][probe] <= line < off["peaks"][probe]
    return {w: tuple(v) for w, v in sorted(out.items(), key=lambda kv: -kv[1][0])}


def main() -> None:
    results = json.loads((HERE / "m52" / "results.json").read_text())
    step0 = defaultdict(dict)
    for l in map(json.loads, gzip.open(STEP0, "rt")):
        step0[l["task"]][l["id"]] = l["category"]
    details = {}
    for task in m52.TASKS:
        new, old = m52.probe_name(task, 0), m52.OLD_PROBE[task]
        line = results[task]["new probe, WildChat line"]["line"]
        rows = A.scored(task, "m45_c") + A.scored(task, "m45_d")
        everyday = A.scored(task, "everyday_m27")
        by_cat = defaultdict(lambda: defaultdict(list))
        for r in everyday:
            c = step0[task][r["id"]]
            by_cat[c]["new"].append(r["peaks"][new])
            by_cat[c]["old"].append(r["peaks"][old])
        details[task] = {
            "seed_lines": results[task]["seed lines (new probe)"],
            "crossings_by_writer_new": crossings_by_writer(rows, new, line),
            "everyday_by_category": {c: {"n": len(v["new"]), "new_median": float(np.median(v["new"])),
                                         "new_flagged": float(np.mean(np.array(v["new"]) > line)),
                                         "old_median": float(np.median(v["old"]))} for c, v in by_cat.items()},
            "writer_means": {w: (results[task]["old probe, strict line"]["writer_means"][w], results[task]["new probe, WildChat line"]["writer_means"][w])
                             for w in sorted(A.NUM.ECI, key=A.NUM.ECI.get)}}
    (HERE / "m52" / "details.json").write_text(json.dumps(details, indent=1))
    print(json.dumps(details, indent=1))


if __name__ == "__main__":
    main()
