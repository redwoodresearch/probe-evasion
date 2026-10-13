"""Spot-check m50's everyday-chat (broad) lines and its ordering / maths crossing counts under the primary line.

Recomputes the `broad`, `broad_iv` and `broad_cal` 1% lines from the saved pool scores (the primary `broad_cal_iv` line
needs the one-sentence pool read in view, which is not in broad_scores.jsonl, so its stored value is used), and counts
valid ON replies at or under the primary line in m45's frozen test (all 14 writers, B = 0/4/8/16, parity arms excluded)
from extract.py's compact rows. Maths validity: m45's correctness-only re-judge (m50's primary rule).

Usage: .venv/bin/python analysis/post_v9/numbers/m50_check.py <scratch dir with extract.py's pickles>
Writes analysis/post_v9/numbers/m50_check.json.
"""

import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRATCH = Path(sys.argv[1])
OUT = Path("analysis/post_v9/numbers")
M50 = Path("missions/m50_realistic_monitor")
M45 = Path("missions/m45_toggle_sweep/runs")
PROBE = {"maths": "q27mrr@L48@s0", "ordering": "q27orr@L48@s0"}
ROOT = {"maths": "qwen35_27b", "ordering": "qwen35_27b_ordering"}
KEY_RE = re.compile(r"^(?P<writer>\w+)/chain_(?P<chain>\d)@B(?P<b>\d+)#(?P<version>on|off)$")


def main() -> None:
    stored = json.loads((M50 / "lines.json").read_text())
    broad = [json.loads(x) for x in (M50 / "runs" / "broad_scores.jsonl").open()]
    out = {}
    for task, probe in PROBE.items():
        none = [r["peaks"][probe] for r in broad if r["context"] == "none"]
        iv = [r["peaks"][probe] for r in broad if r["context"] != "none"]
        cal = [json.loads(x)["peaks"][probe] for x in (M45 / "c" / ROOT[task] / "calibration_scores.jsonl").open()]
        rec = {"broad_1pct": (float(np.percentile(none, 99)), stored[task]["broad_1pct"]["line"]),
               "broad_iv_1pct": (float(np.percentile(iv, 99)), stored[task]["broad_iv_1pct"]["line"]),
               "broad_cal_1pct": (float(np.percentile(none + cal, 99)), stored[task]["broad_cal_1pct"]["line"]),
               "primary_broad_cal_iv_1pct_stored": stored[task]["broad_cal_iv_1pct"]["line"]}
        line = stored[task]["broad_cal_iv_1pct"]["line"]
        data = pickle.loads((SCRATCH / f"{task}.pkl").read_bytes())
        correct = {}
        if task == "maths":
            for b in ("c", "d"):
                correct |= json.loads((M45 / b / ROOT[task] / "correctness_only.json").read_text())
        ok = (lambda r: correct[f"{r['key']}|{r['id']}|{r['seed']}"]) if task == "maths" else (lambda r: r["valid"])
        under, paired, paired_off_valid, distinct, distinct_text, templates, template_texts, writers = 0, 0, 0, set(), set(), set(), set(), set()
        for batch, rows in data["rows"].items():
            by = defaultdict(dict)
            for r in rows:
                by[r["key"]][(r["id"], r["seed"])] = r
            for key, rs in by.items():
                m = KEY_RE.match(key)
                if not m or m["version"] != "on":
                    continue
                off = by[key[:-3] + "#off"]
                for k, r in rs.items():
                    if ok(r) and r["pv"] <= line:
                        under += 1
                        distinct.add((r["id"], r["reply_crc"]))
                        distinct_text.add(r["reply_crc"])
                        templates.add((batch, key[:-3]))
                        writers.add(m["writer"])
                        template_texts.add(data["frozen"][batch][key]["template"])
                        if off[k]["pv"] > line:
                            paired += 1
                            paired_off_valid += ok(off[k])
        rec |= {"valid_on_under_primary": under, "distinct_problem_reply": len(distinct), "distinct_reply_text": len(distinct_text), "templates": len(templates), "writers": len(writers),
                "distinct_template_texts": len(template_texts), "paired_crossings_off_above": paired, "paired_crossings_off_above_and_valid": paired_off_valid}
        out[task] = rec
    (OUT / "m50_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
