"""Qwen3.5-27B as its own writer (batch e): its probe-evasion score at rounds 0 and 16 on each task, computed exactly as
the post's 14 writers (compute.arm_measures with the same line, unattacked median, leak rule and arithmetic grade), next
to the 14-writer fit. Epoch has no ECI for Qwen3.5-27B, so it is not placed on the x-axis: the fit is shown over the ECI
range of the Qwen3.5 models Epoch does score (9B, 139.5; 397B-A17B, 146.7), which bracket the 27B.

Run arithmetic_grading.py e first (it writes batch e's consistent_grading.json).

    PYTHONPATH=src .venv/bin/python analysis/post_v9/numbers/self_writer.py   # writes self_writer.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arithmetic_grading  # noqa: E402
import compute  # noqa: E402
import extract  # noqa: E402

WRITER = "qwen35_27b_self"
BATCH = "e"
TASKS = ("maths", "ordering", "false_belief", "contradiction")
QWEN_ECI_RANGE = (compute.ECI["qwen35_9b"], compute.ECI["qwen35_397b"])


def rows_of(root: Path, name: str, probe: str) -> list[dict]:
    path = root / "test_checked" / f"{name}.jsonl"
    if not path.exists():
        path = root / "test" / f"{name}.jsonl"
    return [extract.compact(r, probe) for r in arithmetic_grading.apply([json.loads(x) for x in path.open()], path)]


def task_result(task: str, numbers: dict) -> dict:
    spec = extract.TASKS[task]
    probe, base_root = spec["probe"], spec["batches"][compute.BASE_BATCH[task]]
    root = base_root.parent.parent / BATCH / base_root.name
    line = compute.strict_line(task)["stored_line"]
    plain, _ = compute.plain_median(rows_of(base_root, "plain", probe))
    gap = plain - line
    rows = rows_of(root, WRITER, probe)
    frozen = json.loads((root / "frozen.json").read_text())
    by_key = defaultdict(list)
    for r in rows:
        by_key[r["key"]].append(r)
    tasks_list = [json.loads(x)["task"] for x in (root / "test_checked" / f"{WRITER}.jsonl").open()]
    tasks_list = list({t["id"]: t for t in tasks_list}.values())
    leak_rule = compute.m28.base_monitor(task).task.leaked if task in compute.M45_ROOT else None
    correct = arithmetic_grading.load(root) if task == "maths" else None
    out = {"line": line, "plain_median": plain, "gap": gap}
    for b in compute.B_POINTS:
        chains = []
        for i in range(3):
            key = f"{WRITER}/chain_{i}@B{b}"
            v = frozen[key + "#on"]
            if v["instruction"] is None:
                chains.append(compute.no_template())
                continue
            leaked = set()
            if leak_rule is not None:
                results = set(leak_rule(v["template"], tasks_list))
                leaked = {t["id"] for t in tasks_list if t["result"] in results}
            chains.append(compute.arm_measures(by_key[key + "#on"], by_key[key + "#off"], line, gap, leaked, correct))
        means = numbers["tasks"][task][f"B{b}"]["writer_means"]
        names = sorted(means, key=compute.ECI.get)
        x, y = np.array([compute.ECI[w] for w in names]), np.array([means[w] for w in names])
        slope, intercept = np.polyfit(x, y, 1)
        out[f"B{b}"] = {
            "chains": [c["gap_closed"] for c in chains], "mean": float(np.mean([c["gap_closed"] for c in chains])),
            "on_correct": [c.get("on_correct_share") for c in chains], "off_correct": [c.get("off_correct_share") for c in chains],
            "on_correct_under_line": [int(c.get("on_valid_under_line", 0)) for c in chains],
            "fit_at_qwen_eci_range": [float(slope * e + intercept) for e in QWEN_ECI_RANGE],
            "qwen35_9b": means["qwen35_9b"], "qwen35_397b": means["qwen35_397b"],
            "writers_below": int(sum(v < np.mean([c["gap_closed"] for c in chains]) for v in means.values()))}
    return out


def main() -> None:
    numbers = json.loads((compute.OUT / "numbers.json").read_text())
    res = {t: task_result(t, numbers) for t in TASKS}
    (compute.OUT / "self_writer.json").write_text(json.dumps(res, indent=1))
    for t, r in res.items():
        for b in compute.B_POINTS:
            s = r[f"B{b}"]
            print(f"{t:14s} B{b:<2d} Qwen3.5-27B self {s['mean']:+.3f} (chains {', '.join(f'{c:+.2f}' for c in s['chains'])}); "
                  f"fit over ECI {QWEN_ECI_RANGE[0]:.0f}-{QWEN_ECI_RANGE[1]:.0f}: {s['fit_at_qwen_eci_range'][0]:+.3f} to "
                  f"{s['fit_at_qwen_eci_range'][1]:+.3f}; Qwen3.5-9B {s['qwen35_9b']:+.3f}, 397B {s['qwen35_397b']:+.3f}; "
                  f"above {s['writers_below']} of 14 writers")


if __name__ == "__main__":
    main()
