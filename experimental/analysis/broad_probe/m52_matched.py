"""m51's matched-request checks re-run on the broad probes: identical reply text after a request that asks Qwen to work
the answer out ("compute") or gives it ("given"), and after a request that gives a wrong answer ("wrong"). A probe that
reads the reply's style cannot separate identical texts; one that reads the computation can.

Uses m51 part 2's own summary (missions/m51_matched_request/analyze2.py) with the probe column swapped in, and each
probe's own 1% line (the old probe: m45's look-alike line; the broad probes: their WildChat line).

    PYTHONPATH=src .venv/bin/python experimental/analysis/broad_probe/m52_matched.py
"""

import importlib.util
import json
import sys
from pathlib import Path

from probe_evasion.missions import m52_broad_probes as m52

HERE = Path(__file__).resolve().parent
M51 = Path("missions/m51_matched_request").resolve()
sys.path.insert(0, str(M51))
import analyze2  # noqa: E402  m51's part-2 summary
from chosen_models import chosen_detector  # noqa: E402

_spec = importlib.util.spec_from_file_location("m52_analysis", HERE / "m52_analysis.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
RESULTS = json.loads((HERE / "m52" / "results.json").read_text())


def rows_for(task: str, variant: str) -> list[dict]:
    rows = [r | {"copied_wrong": False} for r in A.scored(task, "m51_part1", variant)] + A.scored(task, "m51_part2", variant)
    for r in rows:
        r["correct"] = r["correct"] in (True, "True")
        r["copied_wrong"] = r["copied_wrong"] in (True, "True")
        r["kind"], _, r["phrasing"] = r["version"].partition("_")
        r["copied_wrong"] = r["copied_wrong"] and r["kind"] == "wrong"
        r["reply"] = r["response"]
    for v in "ABP":
        det = chosen_detector(task, v)
        for r, s in zip(rows, det.score([{"prompt": r["prompt"], "reply": r["reply"]} for r in rows])):
            r[f"tfidf_{v}"] = float(s)
    return rows


def main() -> None:
    out = {}
    for task in m52.TASKS:
        old_lines = analyze2.lines_for(task)
        readings = {"old probe": ("full", m52.OLD_PROBE[task], old_lines["probe"]),
                    "full broad probe": ("full", m52.probe_name(task, 0, "full"), RESULTS[task]["new probe, WildChat line"]["line"])}
        if "balanced probe, WildChat line" in RESULTS[task]:
            readings["balanced broad probe"] = ("balanced", m52.probe_name(task, 0, "balanced"), RESULTS[task]["balanced probe, WildChat line"]["line"])
        out[task] = {}
        for name, (variant, probe, line) in readings.items():
            rows = rows_for(task, variant)
            for r in rows:
                r["probe"], r["older_probe"] = r["peaks"][probe], r["peaks"][m52.OLD_PROBE[task]]
            d = analyze2.summarise(task, rows, old_lines | {"probe": line})["detectors"]["probe"]
            out[task][name] = {"line": line, "auroc_compute_vs": d["auroc_compute_vs"], "identical_reply": d["identical_reply"],
                               "identical_wrong_minus_given": d.get("identical_wrong_minus_given", {}), "median": d["median"]}
            same = d["identical_reply"]
            wrong = d.get("identical_wrong_minus_given", {})
            print(f"[{task}] {name:22s} identical text, compute > given: " +
                  ", ".join(f"{v} {x['share_compute_higher']:.2f} (n {x['n_pairs']}, median {x['median_compute_minus']:+.1f})"
                            for v, x in same.items() if v.startswith("given")) +
                  " | wrong > given: " + ", ".join(f"{v} {x['share_wrong_higher']:.2f} (median {x['median']:+.1f})" for v, x in wrong.items()))
    (HERE / "m52" / "matched.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
